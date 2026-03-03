from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_test_embedding_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep integration tests deterministic/offline unless a test overrides provider explicitly."""
    monkeypatch.setenv("GLOGGUR_EMBEDDING_PROVIDER", "test")
