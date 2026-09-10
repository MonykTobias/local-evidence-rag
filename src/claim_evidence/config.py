"""Environment-driven settings.

Every knob is a plain environment variable so the package can run against a
managed PostgreSQL and a remote model server without a config file.
"""

from __future__ import annotations

import os
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ValidationError

DEFAULT_DATABASE_URL = (
    "postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence"
)
DEFAULT_DATABASE_CONNECT_TIMEOUT = 10.0
DEFAULT_BUILD_STALE_MINUTES = 60.0
# Every structured call this package makes is bounded: one evidence passage for
# fact extraction, at most 15 passages of 800 characters for adjudication. The
# model's 64k default buys nothing for prompts that size and costs KV-cache
# memory that would otherwise hold model layers on the GPU.
DEFAULT_NUM_CTX = 16384
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_LLAMACPP_EMBED_BASE_URL = "http://127.0.0.1:8081"
DEFAULT_EMBED_MODEL = "qwen3-embedding:4b"
DEFAULT_EMBED_DIMENSIONS = 1024
DEFAULT_CHAT_MODEL = "hf.co/unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL"
DEFAULT_VISION_MODEL = DEFAULT_CHAT_MODEL

# One shared file that says "the local application is running". The frontend
# creates it at startup and removes it on exit; the destructive reset refuses
# while it exists. A fixed temp-directory path rather than something derived
# from a checkout, so the CLI and the app agree without being configured to.
DEFAULT_APP_MARKER = str(Path(tempfile.gettempdir()) / "claim_evidence_app.running")


@dataclass(frozen=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dimensions: int = DEFAULT_EMBED_DIMENSIONS
    chat_model: str = DEFAULT_CHAT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    embed_batch_size: int = 32
    request_timeout: float = 600.0
    # Bounded on purpose: every frontend call opens its own connection, so an
    # unreachable host must fail in seconds rather than on the OS network
    # timeout with the browser spinning.
    database_connect_timeout: float = DEFAULT_DATABASE_CONNECT_TIMEOUT
    # How long a version may sit in 'building' with no recorded progress before
    # health calls it interrupted. Conservative on purpose: a 494-page ingest
    # with narrative facts is legitimately slow, and calling live work dead is
    # worse than reporting a dead build late.
    build_stale_minutes: float = DEFAULT_BUILD_STALE_MINUTES
    # Context window for chat and vision requests. Embeddings are unaffected.
    num_ctx: int = DEFAULT_NUM_CTX
    # Empty unless CE_ENVIRONMENT is set. Destructive operations require the
    # exact value "development", so an unset environment is never a development
    # one by accident.
    environment: str = ""
    app_marker: str = DEFAULT_APP_MARKER
    model_backend: str = "ollama"
    llamacpp_base_url: str = DEFAULT_LLAMACPP_BASE_URL
    llamacpp_embed_base_url: str = DEFAULT_LLAMACPP_EMBED_BASE_URL
    llamacpp_vision_base_url: str | None = None
    llamacpp_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Validated for every construction path, not just from_env(), and the
        # message names the variable rather than echoing the database URL.
        for variable, value in (
            ("CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT", self.database_connect_timeout),
            ("CLAIM_EVIDENCE_BUILD_STALE_MINUTES", self.build_stale_minutes),
            ("CLAIM_EVIDENCE_REQUEST_TIMEOUT", self.request_timeout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValidationError(f"{variable} must be a positive number")
        for variable, value in (
            ("CLAIM_EVIDENCE_NUM_CTX", self.num_ctx),
            ("CLAIM_EVIDENCE_EMBED_DIMENSIONS", self.embed_dimensions),
            ("CLAIM_EVIDENCE_EMBED_BATCH_SIZE", self.embed_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"{variable} must be a positive integer")

        if not isinstance(self.model_backend, str):
            raise ValidationError("CLAIM_EVIDENCE_MODEL_BACKEND must be ollama or llamacpp")
        backend = self.model_backend.strip().lower()
        if backend not in {"ollama", "llamacpp"}:
            raise ValidationError("CLAIM_EVIDENCE_MODEL_BACKEND must be ollama or llamacpp")
        object.__setattr__(self, "model_backend", backend)

        key = self.llamacpp_api_key
        if key is not None:
            if not isinstance(key, str) or "\r" in key or "\n" in key:
                raise ValidationError("CLAIM_EVIDENCE_LLAMACPP_API_KEY contains an invalid newline")
            object.__setattr__(self, "llamacpp_api_key", key or None)

        if backend == "llamacpp":
            for field_name, variable in (
                ("llamacpp_base_url", "CLAIM_EVIDENCE_LLAMACPP_BASE_URL"),
                ("llamacpp_embed_base_url", "CLAIM_EVIDENCE_LLAMACPP_EMBED_BASE_URL"),
            ):
                object.__setattr__(self, field_name, _server_origin(getattr(self, field_name), variable))
            if self.llamacpp_vision_base_url is not None:
                object.__setattr__(
                    self,
                    "llamacpp_vision_base_url",
                    _server_origin(
                        self.llamacpp_vision_base_url,
                        "CLAIM_EVIDENCE_LLAMACPP_VISION_BASE_URL",
                    ),
                )

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.environ.get(
                "CLAIM_EVIDENCE_DATABASE_URL", DEFAULT_DATABASE_URL
            ),
            database_connect_timeout=_positive_float(
                "CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT",
                DEFAULT_DATABASE_CONNECT_TIMEOUT,
            ),
            build_stale_minutes=_positive_float(
                "CLAIM_EVIDENCE_BUILD_STALE_MINUTES", DEFAULT_BUILD_STALE_MINUTES
            ),
            num_ctx=_positive_int("CLAIM_EVIDENCE_NUM_CTX", DEFAULT_NUM_CTX),
            environment=os.environ.get("CE_ENVIRONMENT", "").strip(),
            app_marker=os.environ.get("CLAIM_EVIDENCE_APP_MARKER", DEFAULT_APP_MARKER),
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL
            ).rstrip("/"),
            embed_model=os.environ.get(
                "CLAIM_EVIDENCE_EMBED_MODEL", DEFAULT_EMBED_MODEL
            ),
            embed_dimensions=_positive_int(
                "CLAIM_EVIDENCE_EMBED_DIMENSIONS", DEFAULT_EMBED_DIMENSIONS
            ),
            chat_model=os.environ.get("CLAIM_EVIDENCE_CHAT_MODEL", DEFAULT_CHAT_MODEL),
            vision_model=os.environ.get(
                "CLAIM_EVIDENCE_VISION_MODEL", DEFAULT_VISION_MODEL
            ),
            embed_batch_size=_positive_int(
                "CLAIM_EVIDENCE_EMBED_BATCH_SIZE", 32
            ),
            request_timeout=_positive_float("CLAIM_EVIDENCE_REQUEST_TIMEOUT", 600.0),
            model_backend=os.environ.get("CLAIM_EVIDENCE_MODEL_BACKEND", "ollama"),
            llamacpp_base_url=os.environ.get(
                "CLAIM_EVIDENCE_LLAMACPP_BASE_URL", DEFAULT_LLAMACPP_BASE_URL
            ),
            llamacpp_embed_base_url=os.environ.get(
                "CLAIM_EVIDENCE_LLAMACPP_EMBED_BASE_URL",
                DEFAULT_LLAMACPP_EMBED_BASE_URL,
            ),
            llamacpp_vision_base_url=os.environ.get(
                "CLAIM_EVIDENCE_LLAMACPP_VISION_BASE_URL"
            ),
            llamacpp_api_key=os.environ.get("CLAIM_EVIDENCE_LLAMACPP_API_KEY"),
        )

    @property
    def index_fingerprint_parts(self) -> tuple[str, str]:
        """Model identity that invalidates an existing index when it changes."""
        return (self.model_identifier(self.embed_model), str(self.embed_dimensions))

    def model_identifier(self, model: str) -> str:
        """Provider-qualified identity stored with vectors and audit rows."""
        return model if self.model_backend == "ollama" else f"llamacpp:{model}"

    def llamacpp_url(self, role: str) -> str:
        if role == "embed":
            return self.llamacpp_embed_base_url
        if role == "vision":
            return self.llamacpp_vision_base_url or self.llamacpp_base_url
        return self.llamacpp_base_url


def _positive_float(variable: str, default: float) -> float:
    raw = os.environ.get(variable)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        # Never echo the value: an operator can paste a URL into the wrong
        # variable, and the error goes to a browser.
        raise ValidationError(f"{variable} must be a positive number") from None


def _positive_int(variable: str, default: int) -> int:
    raw = os.environ.get(variable)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValidationError(f"{variable} must be a positive integer") from None


def _server_origin(value: str, variable: str) -> str:
    """A server origin, not an endpoint or credential-bearing URL."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{variable} must be an absolute HTTP or HTTPS origin")
    text = value.strip().rstrip("/")
    try:
        parsed = urlsplit(text)
        parsed.port
    except ValueError:
        raise ValidationError(
            f"{variable} must be an absolute HTTP or HTTPS origin"
        ) from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValidationError(f"{variable} must be an absolute HTTP or HTTPS origin")
    return text


__all__ = [
    "DEFAULT_BUILD_STALE_MINUTES",
    "DEFAULT_DATABASE_CONNECT_TIMEOUT",
    "DEFAULT_DATABASE_URL",
    "DEFAULT_NUM_CTX",
    "Settings",
]
