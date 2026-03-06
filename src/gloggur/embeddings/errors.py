"""Structured error types and helpers for embedding provider failures.

Provides ``EmbeddingProviderError``, a stable dataclass exception used by all
provider adapters, plus helpers for converting arbitrary exceptions into that
form and formatting human-readable output for stderr.
"""

from __future__ import annotations

from dataclasses import dataclass


def _provider_remediation(provider: str) -> list[str]:
    """Return actionable remediation steps for a given embedding provider.

    Returns a provider-specific list of steps the user or agent should take to
    resolve credential or configuration issues.  Falls back to generic guidance
    for unrecognised provider names.
    """
    if provider == "openai":
        return [
            "Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY in the environment.",
            "When routing via OpenRouter, set OPENAI_BASE_URL or "
            "GLOGGUR_OPENROUTER_BASE_URL if a non-default endpoint is required.",
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
            "Ensure the local sentence-transformers model is installed and loadable.",
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
        """Return a single-line summary suitable for log output and exception chains."""
        return (
            f"Embedding provider failure [{self.provider}] during "
            f"'{self.operation}': {self.detail}"
        )

    def to_payload(self) -> dict[str, object]:
        """Return a machine-readable error payload for JSON CLI outputs."""
        error_code = "embedding_provider_error"
        return {
            "error": {
                "type": error_code,
                "code": error_code,
                "provider": self.provider,
                "operation": self.operation,
                "detail": self.detail,
                "remediation": self.remediation,
            },
            "failure_codes": [error_code],
            "failure_guidance": {
                error_code: list(self.remediation),
            },
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
