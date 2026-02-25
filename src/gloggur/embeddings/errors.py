from __future__ import annotations

from dataclasses import dataclass


def _provider_remediation(provider: str) -> list[str]:
    if provider == "openai":
        return [
            "Set OPENAI_API_KEY in the environment.",
            "Confirm the configured model is available (GLOGGUR_OPENAI_MODEL).",
        ]
    if provider == "gemini":
        return [
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.",
            "Confirm the configured model is available (GLOGGUR_GEMINI_MODEL).",
        ]
    if provider == "local":
        return [
            "Install local embedding dependencies (`pip install -e '.[local]'`).",
            "Use fallback mode only when appropriate (GLOGGUR_LOCAL_FALLBACK=1).",
        ]
    return [
        "Verify embedding provider configuration values and required credentials.",
        "Retry once configuration and environment variables are corrected.",
    ]


@dataclass
class EmbeddingProviderError(RuntimeError):
    """Structured provider error for stable CLI output and tests."""

    provider: str
    operation: str
    detail: str
    remediation: list[str]

    def __str__(self) -> str:
        return (
            f"Embedding provider failure [{self.provider}] during "
            f"'{self.operation}': {self.detail}"
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "error": {
                "type": "embedding_provider_error",
                "provider": self.provider,
                "operation": self.operation,
                "detail": self.detail,
                "remediation": self.remediation,
            }
        }


def wrap_embedding_error(
    exc: Exception,
    *,
    provider: str,
    operation: str,
) -> EmbeddingProviderError:
    """Convert arbitrary provider exceptions into a structured provider error."""
    if isinstance(exc, EmbeddingProviderError):
        return exc
    return EmbeddingProviderError(
        provider=provider or "unknown",
        operation=operation,
        detail=f"{type(exc).__name__}: {exc}",
        remediation=_provider_remediation(provider or "unknown"),
    )


def format_embedding_error_message(error: EmbeddingProviderError) -> str:
    """Create stable human-readable stderr output for provider failures."""
    lines = [str(error), "Remediation:"]
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(error.remediation, start=1))
    return "\n".join(lines)
