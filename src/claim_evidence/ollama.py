"""Backward-compatible imports for the original Ollama-only client."""

from .model_client import (
    ModelClient as OllamaClient,
    ModelError as OllamaError,
    THINK_ENV,
    gbnf_safe_schema,
    request_think,
)

__all__ = [
    "OllamaClient",
    "OllamaError",
    "THINK_ENV",
    "gbnf_safe_schema",
    "request_think",
]
