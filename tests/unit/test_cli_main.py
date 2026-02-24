from __future__ import annotations

import click
import pytest

from gloggur.cli import main as cli_main
from gloggur.cli.main import _profile_reindex_reason
from gloggur.indexer.cache import CacheRecoveryError


def test_profile_reindex_reason_no_metadata_and_no_profile() -> None:
    """No index metadata/profile should not force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=False,
        cached_profile=None,
        expected_profile="local:model-a",
    )
    assert reason is None


def test_profile_reindex_reason_unknown_profile_with_metadata() -> None:
    """Index metadata without cached profile should force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile=None,
        expected_profile="local:model-a",
    )
    assert reason == "cached embedding profile is unknown"


def test_profile_reindex_reason_profile_changed() -> None:
    """Mismatched cached/expected profile should force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile="local:model-a",
        expected_profile="local:model-b",
    )
    assert reason == "embedding profile changed (cached=local:model-a, current=local:model-b)"


def test_profile_reindex_reason_profile_matches() -> None:
    """Matching profile should not force reindex."""
    reason = _profile_reindex_reason(
        metadata_present=True,
        cached_profile="local:model-a",
        expected_profile="local:model-a",
    )
    assert reason is None


def test_create_cache_manager_wraps_cache_recovery_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrecoverable cache corruption should be raised as a user-facing ClickException."""

    class BrokenCacheManager:
        def __init__(self, _config: object) -> None:
            raise CacheRecoveryError("recovery failed")

    monkeypatch.setattr(cli_main, "CacheManager", BrokenCacheManager)
    with pytest.raises(click.ClickException, match="recovery failed"):
        cli_main._create_cache_manager("/tmp/gloggur-cache")
