"""A requests.Session stand-in so tests need no GPU and no live model."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable


class FakeResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> Any:
        return self._payload


class FakeSession:
    """Routes ``/api/embed`` to a deterministic hash embedding and ``/api/chat``
    to a queue of scripted replies."""

    def __init__(
        self,
        dimensions: int = 8,
        chat_replies: list[str] | None = None,
        embed_hook: Callable[[list[str]], Any] | None = None,
        chat_router: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
        model_digests: dict[str, str] | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.chat_replies = list(chat_replies or [])
        self.embed_hook = embed_hook
        # Model name -> digest. A model absent from this map answers without
        # one, which is how the tag-only path is exercised.
        self.model_digests = dict(model_digests or {})
        # Keyed by the response schema's title, so a test does not have to
        # predict how many chat calls a pipeline makes or in what order.
        self.chat_router = chat_router or {}
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, json: dict[str, Any], timeout: float) -> FakeResponse:  # noqa: A002
        payload = json
        self.requests.append((url, payload))
        if url.endswith("/api/embed"):
            inputs = payload["input"]
            if self.embed_hook is not None:
                override = self.embed_hook(inputs)
                if override is not None:
                    return FakeResponse({"embeddings": override})
            return FakeResponse({"embeddings": [self.vector(t) for t in inputs]})
        if url.endswith("/v1/embeddings"):
            inputs = payload["input"]
            vectors = self.embed_hook(inputs) if self.embed_hook is not None else None
            if vectors is None:
                vectors = [self.vector(text) for text in inputs]
            return FakeResponse(
                {
                    "data": [
                        {"index": index, "embedding": vector}
                        for index, vector in enumerate(vectors)
                    ]
                }
            )
        if url.endswith("/api/show"):
            digest = self.model_digests.get(payload.get("model", ""))
            return FakeResponse({"digest": digest} if digest else {})
        if url.endswith("/api/chat"):
            title = (payload.get("format") or {}).get("title")
            handler = self.chat_router.get(title)
            if handler is not None:
                return FakeResponse({"message": {"content": reply(handler(payload))}})
            if not self.chat_replies:
                raise AssertionError(f"unexpected chat call for schema {title}")
            return FakeResponse({"message": {"content": self.chat_replies.pop(0)}})
        if url.endswith("/v1/chat/completions"):
            title = ((payload.get("response_format") or {}).get("schema") or {}).get(
                "title"
            )
            handler = self.chat_router.get(title)
            if handler is not None:
                content = reply(handler(payload))
            elif self.chat_replies:
                content = self.chat_replies.pop(0)
            else:
                raise AssertionError(f"unexpected chat call for schema {title}")
            return FakeResponse({"choices": [{"message": {"content": content}}]})
        raise AssertionError(f"unexpected url {url}")

    def get(self, url: str, timeout: float) -> FakeResponse:
        """No tag listing by default, so health() sees Ollama as unreachable.

        Subclass and override to simulate a reachable daemon.
        """
        import requests

        raise requests.ConnectionError(f"fake session has no endpoint for {url}")

    def prompt_of(self, payload: dict[str, Any]) -> str:
        return payload["messages"][1]["content"]

    def vector(self, text: str) -> list[float]:
        """Stable pseudo-embedding: same text always yields the same vector."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [
            digest[i % len(digest)] / 255.0 - 0.5 for i in range(self.dimensions)
        ]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]


def reply(payload: Any) -> str:
    return json.dumps(payload)
