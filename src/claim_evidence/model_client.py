"""HTTP client for the supported local model servers."""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import time
from numbers import Real
from typing import Any, Iterable, Sequence, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .config import Settings
from .errors import IndexNotReadyError
from .models import ModelHealth

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
THINK_ENV = "CLAIM_EVIDENCE_OLLAMA_THINK"


class OllamaError(RuntimeError):
    """The model server was unreachable or returned unusable output."""


# The neutral name preserves the original exception object's public class name.
ModelError = OllamaError


def require_compatible_embeddings(
    settings: Settings, rows: Sequence[dict[str, Any]]
) -> None:
    """Refuse vectors created by another provider, model, or dimension."""
    expected = settings.model_identifier(settings.embed_model)
    for row in rows:
        if row["embed_model"] != expected or int(row["embed_dim"]) != settings.embed_dimensions:
            raise IndexNotReadyError(
                f"document {row['document_id']} was indexed with incompatible embeddings; "
                "re-ingest it using the selected model backend"
            )


_UNSUPPORTED_KEYWORDS = frozenset({"pattern", "format", "contentEncoding"})


def gbnf_safe_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Remove JSON Schema keywords unsupported by both server grammars."""
    def prune(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: prune(v) for k, v in node.items() if k not in _UNSUPPORTED_KEYWORDS}
        if isinstance(node, list):
            return [prune(v) for v in node]
        return node

    return prune(schema.model_json_schema())


def request_think() -> bool | None:
    """Optional Ollama thinking override; unset preserves the model default."""
    raw = os.getenv(THINK_ENV)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{THINK_ENV} must be true or false")


class ModelClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str] | None:
        key = self.settings.llamacpp_api_key
        return {"Authorization": f"Bearer {key}"} if key else None

    def _post(
        self,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        *,
        authenticated: bool = False,
    ) -> dict[str, Any]:
        started = time.monotonic()
        logger.debug(
            "model request started",
            extra={
                "event": "model_request_started",
                "endpoint": path,
                "model": payload.get("model"),
                "provider": self.settings.model_backend,
            },
        )
        try:
            kwargs: dict[str, Any] = {
                "json": payload,
                "timeout": self.settings.request_timeout,
            }
            if authenticated and self._headers():
                kwargs["headers"] = self._headers()
            response = self.session.post(f"{base_url}{path}", **kwargs)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("response is not an object")
            logger.debug(
                "model request completed",
                extra={
                    "event": "model_request_completed",
                    "endpoint": path,
                    "model": payload.get("model"),
                    "provider": self.settings.model_backend,
                    "duration_seconds": round(time.monotonic() - started, 3),
                },
            )
            return result
        except requests.RequestException as exc:
            logger.error(
                "model request failed",
                extra={
                    "event": "model_request_failed",
                    "endpoint": path,
                    "model": payload.get("model"),
                    "provider": self.settings.model_backend,
                    "error_type": type(exc).__name__,
                    "duration_seconds": round(time.monotonic() - started, 3),
                },
            )
            if self.settings.model_backend == "ollama":
                detail = getattr(getattr(exc, "response", None), "text", "") or ""
                url = f"{base_url}{path}"
                raise ModelError(
                    f"{url} failed: {exc}{f' -- {detail[:400]}' if detail else ''}"
                ) from exc
            raise ModelError(f"llama.cpp request to {path} failed") from exc
        except (ValueError, TypeError) as exc:
            logger.error(
                "model returned invalid JSON",
                extra={
                    "event": "model_response_non_json",
                    "endpoint": path,
                    "model": payload.get("model"),
                    "provider": self.settings.model_backend,
                    "duration_seconds": round(time.monotonic() - started, 3),
                },
            )
            label = f"{base_url}{path}" if self.settings.model_backend == "ollama" else path
            raise ModelError(f"{label} returned invalid JSON") from exc

    def model_identity(self, model: str) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "name": model,
            "digest": None,
            "reproducibility": "tag_only",
        }
        if self.settings.model_backend == "llamacpp":
            identity["provider"] = "llamacpp"
            return identity
        try:
            data = self._post(
                self.settings.ollama_base_url, "/api/show", {"model": model}
            )
        except ModelError:
            return identity
        digest = _first_digest(data)
        if digest:
            identity["digest"] = digest
            identity["reproducibility"] = "digest_pinned"
        return identity

    def embed(self, inputs: Sequence[str]) -> list[list[float]]:
        if not inputs:
            return []
        if self.settings.model_backend == "ollama":
            data = self._post(
                self.settings.ollama_base_url,
                "/api/embed",
                {"model": self.settings.embed_model, "input": list(inputs)},
            )
            vectors = data.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(inputs):
                count = len(vectors) if isinstance(vectors, list) else 0
                raise ModelError(
                    f"/api/embed returned {count} vectors for {len(inputs)} inputs"
                )
        else:
            data = self._post(
                self.settings.llamacpp_url("embed"),
                "/v1/embeddings",
                {
                    "model": self.settings.embed_model,
                    "input": list(inputs),
                    "encoding_format": "float",
                },
                authenticated=True,
            )
            rows = data.get("data")
            if not isinstance(rows, list) or len(rows) != len(inputs):
                count = len(rows) if isinstance(rows, list) else 0
                raise ModelError(
                    f"/v1/embeddings returned {count} vectors for {len(inputs)} inputs"
                )
            ordered: list[Any] = [None] * len(inputs)
            seen: set[int] = set()
            for row in rows:
                if not isinstance(row, dict):
                    raise ModelError("/v1/embeddings returned an invalid item")
                index = row.get("index")
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index >= len(inputs)
                    or index in seen
                ):
                    raise ModelError("/v1/embeddings returned invalid indices")
                seen.add(index)
                ordered[index] = row.get("embedding")
            if seen != set(range(len(inputs))):
                raise ModelError("/v1/embeddings returned incomplete indices")
            vectors = ordered
        return [self._fit_dimension(vector) for vector in vectors]

    def _fit_dimension(self, vector: Any) -> list[float]:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ModelError("embedding server returned an invalid vector")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise ModelError("embedding server returned a non-finite vector")
        expected = self.settings.embed_dimensions
        if len(vector) < expected:
            raise ModelError(
                f"{self.settings.embed_model} returned dimension {len(vector)}, "
                f"which is shorter than the configured {expected}; "
                "set CLAIM_EVIDENCE_EMBED_DIMENSIONS to match the model"
            )
        fitted = [float(value) for value in vector[:expected]]
        if not any(fitted):
            raise ModelError("embedding server returned a zero vector")
        return fitted

    def embed_batched(self, inputs: Sequence[str]) -> Iterable[list[list[float]]]:
        size = self.settings.embed_batch_size
        for start in range(0, len(inputs), size):
            yield self.embed(inputs[start : start + size])

    def structured(
        self,
        schema: type[T],
        system: str,
        user: str,
        *,
        images: Sequence[bytes] = (),
        model: str | None = None,
    ) -> T:
        return self._structured(
            schema, system, user, images=images, model=model, role="chat"
        )

    def _structured(
        self,
        schema: type[T],
        system: str,
        user: str,
        *,
        images: Sequence[bytes],
        model: str | None,
        role: str,
    ) -> T:
        selected_model = model or (
            self.settings.vision_model if role == "vision" else self.settings.chat_model
        )
        if self.settings.model_backend == "ollama":
            message: dict[str, Any] = {"role": "user", "content": user}
            if images:
                message["images"] = [base64.b64encode(img).decode("ascii") for img in images]
            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": [{"role": "system", "content": system}, message],
                "format": gbnf_safe_schema(schema),
                "stream": False,
                "options": {"temperature": 0, "num_ctx": self.settings.num_ctx},
            }
            think = request_think()
            if think is not None:
                payload["think"] = think
            base_url, path, authenticated = self.settings.ollama_base_url, "/api/chat", False
        else:
            content: str | list[dict[str, Any]] = user
            if images:
                content = [{"type": "text", "text": user}]
                content.extend(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(image).decode("ascii")
                        },
                    }
                    for image in images
                )
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                "response_format": {
                    "type": "json_object",
                    "schema": gbnf_safe_schema(schema),
                },
                "stream": False,
                "temperature": 0,
            }
            base_url, path, authenticated = (
                self.settings.llamacpp_url(role),
                "/v1/chat/completions",
                True,
            )

        last_error: Exception | None = None
        for attempt in range(2):
            data = self._post(
                base_url, path, payload, authenticated=authenticated
            )
            try:
                if self.settings.model_backend == "ollama":
                    message = data.get("message")
                else:
                    choices = data.get("choices")
                    first = choices[0] if isinstance(choices, list) and choices else None
                    message = first.get("message") if isinstance(first, dict) else None
                if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                    raise ModelError(f"{path} returned an invalid response envelope")
                answer = message["content"]
                result = schema.model_validate_json(answer)
                logger.debug(
                    "structured response validated",
                    extra={
                        "event": "structured_response_validated",
                        "schema": schema.__name__,
                        "model": selected_model,
                        "provider": self.settings.model_backend,
                        "attempt": attempt + 1,
                    },
                )
                return result
            except ModelError:
                raise
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "structured response failed validation; retrying"
                    if attempt == 0
                    else "structured response failed validation after retry",
                    extra={
                        "event": "structured_response_invalid",
                        "schema": schema.__name__,
                        "model": selected_model,
                        "provider": self.settings.model_backend,
                        "attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                    },
                )
                payload["messages"] = [
                    *payload["messages"][:2],
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            "That response did not match the required schema: "
                            f"{exc}. Reply with valid JSON only."
                        ),
                    },
                ]
        raise ModelError(
            f"{selected_model} failed schema {schema.__name__} twice: {last_error}"
        )

    def vision(
        self, schema: type[T], system: str, user: str, image: bytes
    ) -> T:
        return self._structured(
            schema,
            system,
            user,
            images=[image],
            model=self.settings.vision_model,
            role="vision",
        )

    def health(self) -> tuple[bool, list[ModelHealth], list[str]]:
        if self.settings.model_backend == "ollama":
            installed = self._ollama_models()
            reachable = installed is not None
            problems = [] if reachable else ["ollama is not reachable"]
            models = []
            for role, name in self._roles():
                available = bool(installed and name in installed)
                models.append(ModelHealth(role=role, name=name, available=available))
                if installed is not None and not available:
                    problems.append(f"{role} model {name} is not pulled")
            return reachable, models, problems

        origins = {self.settings.llamacpp_url(role) for role, _ in self._roles()}
        snapshots = {origin: self._llamacpp_snapshot(origin) for origin in origins}
        reachable = all(snapshot[0] for snapshot in snapshots.values())
        problems: list[str] = []
        models: list[ModelHealth] = []
        for role, name in self._roles():
            ready, aliases, props = snapshots[self.settings.llamacpp_url(role)]
            available = ready and name in aliases
            if ready and name not in aliases:
                problems.append(f"{role} model {name} is unavailable")
            if available and role == "vision":
                modalities = props.get("modalities") if isinstance(props, dict) else None
                has_vision = (
                    bool(modalities.get("vision"))
                    if isinstance(modalities, dict)
                    else "vision" in modalities if isinstance(modalities, list) else False
                )
                if not has_vision:
                    available = False
                    problems.append("vision model has no loaded multimodal projector")
            if available and role in {"chat", "vision"}:
                generation = props.get("default_generation_settings") if isinstance(props, dict) else None
                context = generation.get("n_ctx") if isinstance(generation, dict) else None
                if context != self.settings.num_ctx:
                    available = False
                    problems.append(
                        f"{role} server context does not match configured {self.settings.num_ctx}"
                    )
            models.append(ModelHealth(role=role, name=name, available=available))
        if not reachable:
            problems.insert(0, "llama.cpp is not reachable")
        return reachable, models, problems

    def _roles(self) -> tuple[tuple[str, str], ...]:
        return (
            ("embed", self.settings.embed_model),
            ("chat", self.settings.chat_model),
            ("vision", self.settings.vision_model),
        )

    def _ollama_models(self) -> set[str] | None:
        try:
            response = self.session.get(
                f"{self.settings.ollama_base_url}/api/tags", timeout=10
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
                return None
            return {
                model["name"]
                for model in payload["models"]
                if isinstance(model, dict) and isinstance(model.get("name"), str)
            }
        except (requests.RequestException, ValueError, TypeError):
            return None

    def _llamacpp_snapshot(self, origin: str) -> tuple[bool, set[str], dict[str, Any]]:
        health = self._get(origin, "/health")
        ready = isinstance(health, dict) and health.get("status") == "ok"
        listing = self._get(origin, "/v1/models") if ready else None
        props = self._get(origin, "/props") if ready else None
        data = listing.get("data") if isinstance(listing, dict) else None
        aliases = {
            row["id"]
            for row in data or []
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        } if isinstance(data, list) else set()
        return ready, aliases, props if isinstance(props, dict) else {}

    def _get(self, origin: str, path: str) -> dict[str, Any] | None:
        try:
            kwargs: dict[str, Any] = {"timeout": 10}
            if self._headers():
                kwargs["headers"] = self._headers()
            response = self.session.get(f"{origin}{path}", **kwargs)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except (requests.RequestException, ValueError, TypeError):
            return None


def _first_digest(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    candidates = [
        data.get("digest"),
        (data.get("details") or {}).get("digest")
        if isinstance(data.get("details"), dict)
        else None,
        (data.get("model_info") or {}).get("digest")
        if isinstance(data.get("model_info"), dict)
        else None,
    ]
    for value in candidates:
        if isinstance(value, str):
            text = value.split(":")[-1].strip().lower()
            if len(text) >= 32 and all(character in "0123456789abcdef" for character in text):
                return text
    return None


__all__ = [
    "ModelClient",
    "ModelError",
    "THINK_ENV",
    "gbnf_safe_schema",
    "request_think",
    "require_compatible_embeddings",
]
