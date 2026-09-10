"""Deterministic llama.cpp transport checks; no server or GPU required."""

from __future__ import annotations

import json
from typing import Any

import requests

from claim_evidence import Settings
from claim_evidence.model_client import ModelClient, ModelError
from claim_evidence.models import Adjudication, FactExtraction, VisualVerification


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


class Response:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> Any:
        return self.payload


class LlamaSession:
    def __init__(
        self,
        *,
        chat_replies: list[str] | None = None,
        embedding_rows: list[dict[str, Any]] | None = None,
        vision: bool = True,
        context: int = 16384,
    ) -> None:
        self.chat_replies = list(chat_replies or [])
        self.embedding_rows = embedding_rows
        self.vision = vision
        self.context = context
        self.posts: list[tuple[str, dict[str, Any], dict[str, str] | None]] = []
        self.gets: list[tuple[str, dict[str, str] | None]] = []

    def post(
        self,
        url: str,
        json: dict[str, Any],  # noqa: A002
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> Response:
        self.posts.append((url, json, headers))
        if url.endswith("/v1/embeddings"):
            rows = self.embedding_rows
            if rows is None:
                rows = [
                    {"index": index, "embedding": [float(index + 1)] * 8}
                    for index in reversed(range(len(json["input"])))
                ]
            return Response({"object": "list", "data": rows})
        if url.endswith("/v1/chat/completions"):
            if not self.chat_replies:
                raise AssertionError("unexpected llama.cpp chat call")
            return Response(
                {"choices": [{"message": {"content": self.chat_replies.pop(0)}}]}
            )
        raise AssertionError(f"unexpected URL {url}")

    def get(
        self,
        url: str,
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> Response:
        self.gets.append((url, headers))
        if url.endswith("/health"):
            return Response({"status": "ok"})
        if url.endswith("/v1/models"):
            return Response(
                {
                    "data": [
                        {"id": "embed-model"},
                        {"id": "chat-model"},
                        {"id": "vision-model"},
                    ]
                }
            )
        if url.endswith("/props"):
            return Response(
                {
                    "default_generation_settings": {"n_ctx": self.context},
                    "modalities": {"vision": self.vision},
                }
            )
        raise AssertionError(f"unexpected URL {url}")


def settings(**overrides: Any) -> Settings:
    values = {
        "model_backend": "llamacpp",
        "embed_model": "embed-model",
        "chat_model": "chat-model",
        "vision_model": "vision-model",
        "embed_dimensions": 8,
        "llamacpp_base_url": "https://chat.example.test:8443",
        "llamacpp_embed_base_url": "https://embed.example.test:8443",
        "llamacpp_vision_base_url": "https://vision.example.test:8443",
        "llamacpp_api_key": "secret-key",
    }
    return Settings(**{**values, **overrides})


def test_role_routing_wire_format_and_authentication() -> None:
    session = LlamaSession(
        chat_replies=[
            json.dumps({"verdict": "supported", "rationale": "matches"}),
            json.dumps(
                {
                    "result": "illegible",
                    "reason_code": "figures_not_legible",
                    "reason": "unreadable",
                }
            ),
        ]
    )
    client = ModelClient(settings(), session)
    vectors = client.embed(["first", "second"])
    check(vectors[0][0] == 1.0 and vectors[1][0] == 2.0, "embedding indices restore input order")
    result = client.structured(Adjudication, "system", "claim", model="override-chat")
    check(result.verdict == "supported", "structured response is validated")
    client.vision(VisualVerification, "system", "inspect", b"\x89PNG")

    embed_url, embed_payload, embed_headers = session.posts[0]
    check(embed_url.startswith("https://embed.example.test:8443"), "embedding role uses its URL")
    check(embed_payload["encoding_format"] == "float", "OpenAI-compatible embedding payload used")
    check(embed_headers == {"Authorization": "Bearer secret-key"}, "bearer key sent")

    chat_url, chat_payload, _ = session.posts[1]
    check(chat_url.startswith("https://chat.example.test:8443"), "chat role uses its URL")
    check(chat_payload["model"] == "override-chat", "explicit model override uses chat")
    check(chat_payload["response_format"]["schema"]["title"] == "Adjudication", "schema is constrained")
    check("format" not in chat_payload and "options" not in chat_payload, "Ollama-only fields omitted")

    vision_url, vision_payload, _ = session.posts[2]
    check(vision_url.startswith("https://vision.example.test:8443"), "vision role uses its URL")
    parts = vision_payload["messages"][1]["content"]
    check(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"), "PNG is embedded")


def test_invalid_structured_output_gets_one_retry() -> None:
    session = LlamaSession(
        chat_replies=["not json", json.dumps({"facts": []})]
    )
    result = ModelClient(settings(), session).structured(
        FactExtraction, "system", "passage"
    )
    check(result.facts == [], "valid retry is returned")
    check(len(session.posts) == 2, "exactly one correction retry is made")
    check(
        "did not match the required schema"
        in session.posts[1][1]["messages"][-1]["content"],
        "retry explains the validation failure",
    )


def test_invalid_embedding_indices_and_values_are_rejected() -> None:
    invalid_rows = (
        [
            {"index": 0, "embedding": [1.0] * 8},
            {"index": 0, "embedding": [2.0] * 8},
        ],
        [{"index": 0, "embedding": [float("nan")] * 8}],
        [{"index": 0, "embedding": [0.0] * 8}],
        [{"index": True, "embedding": [1.0] * 8}],
        [{"index": -1, "embedding": [1.0] * 8}],
        [{"index": 1, "embedding": [1.0] * 8}],
        [{"index": 0, "embedding": ["bad"] * 8}],
        [{"index": 0, "embedding": [1.0] * 4}],
    )
    inputs = (["a", "b"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"])
    for rows, batch in zip(invalid_rows, inputs):
        try:
            ModelClient(settings(), LlamaSession(embedding_rows=rows)).embed(batch)
        except ModelError:
            continue
        raise AssertionError(f"invalid embedding response was accepted: {rows!r}")
    check(True, "invalid indices, non-finite values, and zero vectors are rejected")


def test_malformed_envelopes_and_http_errors_do_not_retry() -> None:
    class BrokenSession(LlamaSession):
        def __init__(self, payload: Any, status: int = 200) -> None:
            super().__init__()
            self.payload = payload
            self.status = status

        def post(
            self,
            url: str,
            json: dict[str, Any],  # noqa: A002
            timeout: float,
            headers: dict[str, str] | None = None,
        ) -> Response:
            self.posts.append((url, json, headers))
            return Response(self.payload, self.status)

    for session in (
        BrokenSession({"choices": [42]}),
        BrokenSession({"choices": [{"message": {"content": 42}}]}),
        BrokenSession({}, 500),
    ):
        try:
            ModelClient(settings(), session).structured(
                FactExtraction, "system", "passage"
            )
        except ModelError as exc:
            check(
                "secret-key" not in str(exc) and "https://" not in str(exc),
                "malformed response errors expose no key or URL",
            )
        else:
            raise AssertionError("a malformed or failed response was accepted")
        check(len(session.posts) == 1, "transport and envelope failures are not retried")


def test_vision_retry_preserves_the_image() -> None:
    session = LlamaSession(
        chat_replies=[
            "not json",
            json.dumps(
                {
                    "result": "illegible",
                    "reason_code": "figures_not_legible",
                }
            ),
        ]
    )
    result = ModelClient(settings(), session).vision(
        VisualVerification, "system", "inspect", b"\x89PNG"
    )
    first_image = session.posts[0][1]["messages"][1]["content"][1]["image_url"]
    second_image = session.posts[1][1]["messages"][1]["content"][1]["image_url"]
    check(result.result == "illegible", "vision uses the valid correction")
    check(first_image == second_image, "vision retry preserves the image payload")


def test_health_checks_distinct_origins_and_capabilities() -> None:
    config = settings(llamacpp_vision_base_url=None)
    session = LlamaSession()
    reachable, models, problems = ModelClient(config, session).health()
    check(reachable and not problems, "healthy llama.cpp servers are ready")
    check(all(model.available for model in models), "all configured aliases are available")
    check(len(session.gets) == 6, "three diagnostics run once for each distinct origin")
    check(
        all(headers == {"Authorization": "Bearer secret-key"} for _, headers in session.gets),
        "diagnostic requests are authenticated",
    )

    _, models, problems = ModelClient(config, LlamaSession(vision=False)).health()
    vision = next(model for model in models if model.role == "vision")
    check(not vision.available and any("projector" in problem for problem in problems), "missing projector is reported")

    _, models, problems = ModelClient(config, LlamaSession(context=8192)).health()
    check(
        not next(model for model in models if model.role == "chat").available
        and any("context" in problem for problem in problems),
        "context mismatch makes chat unavailable",
    )

    class MissingAliasSession(LlamaSession):
        def get(
            self,
            url: str,
            timeout: float,
            headers: dict[str, str] | None = None,
        ) -> Response:
            response = super().get(url, timeout, headers)
            if url.endswith("/v1/models"):
                return Response({"data": [{"id": "embed-model"}]})
            return response

    _, models, problems = ModelClient(config, MissingAliasSession()).health()
    check(
        not next(model for model in models if model.role == "chat").available
        and any("chat model" in problem for problem in problems),
        "missing alias is reported",
    )

    class LoadingSession(LlamaSession):
        def get(
            self,
            url: str,
            timeout: float,
            headers: dict[str, str] | None = None,
        ) -> Response:
            self.gets.append((url, headers))
            return Response({"status": "loading"}, 503)

    reachable, models, problems = ModelClient(config, LoadingSession()).health()
    check(not reachable and not any(model.available for model in models), "503/loading is not ready")
    check(
        "secret-key" not in str(problems) and "https://" not in str(problems),
        "health problems expose no keys or URLs",
    )


def test_identity_and_configuration_are_provider_qualified_and_secret_safe() -> None:
    config = settings()
    identity = ModelClient(config, LlamaSession()).model_identity("embed-model")
    check(identity["provider"] == "llamacpp", "identity names the provider")
    check(identity["reproducibility"] == "tag_only", "alias is not treated as a digest")
    check(config.model_identifier("embed-model") == "llamacpp:embed-model", "stored identifier is qualified")
    check("secret-key" not in repr(config), "API key is excluded from settings repr")

    for bad in (
        "",
        "ftp://host",
        "https://host/v1",
        "https://user:pass@host",
        "https://:8080",
        "https://host:not-a-port",
    ):
        try:
            settings(llamacpp_base_url=bad)
        except Exception:
            continue
        raise AssertionError(f"invalid server origin accepted: {bad}")
    check(True, "invalid llama.cpp server origins are rejected")

    for overrides in (
        {"model_backend": "unknown"},
        {"llamacpp_api_key": "bad\nkey"},
        {"request_timeout": float("inf")},
        {"embed_dimensions": True},
    ):
        try:
            settings(**overrides)
        except Exception:
            continue
        raise AssertionError(f"invalid llama.cpp setting accepted: {list(overrides)}")
    check(True, "backend, key, and numeric settings are validated")


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
