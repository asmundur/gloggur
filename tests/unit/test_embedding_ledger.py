from __future__ import annotations

import tempfile

from gloggur.indexer.embedding_ledger import EmbeddingLedger, embedding_text_hash


def test_embedding_ledger_round_trips_vectors_and_isolates_profiles() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    ledger = EmbeddingLedger(cache_dir)

    hash_a = embedding_text_hash("alpha")
    hash_b = embedding_text_hash("beta")
    ledger.upsert_vectors(
        embedding_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
        record_kind="chunk",
        entries=[
            (hash_a, [0.1, 0.2]),
            (hash_b, [0.3, 0.4]),
        ],
    )

    loaded = ledger.get_vectors(
        embedding_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
        record_kind="chunk",
        text_hashes=[hash_a, hash_b],
    )

    assert loaded == {
        hash_a: [0.1, 0.2],
        hash_b: [0.3, 0.4],
    }
    assert (
        ledger.get_vectors(
            embedding_profile="openai:text-embedding-3-large|embed_graph_edges=0",
            record_kind="chunk",
            text_hashes=[hash_a, hash_b],
        )
        == {}
    )


def test_embedding_ledger_clear_removes_persisted_artifacts() -> None:
    cache_dir = tempfile.mkdtemp(prefix="gloggur-cache-")
    ledger = EmbeddingLedger(cache_dir)
    hash_a = embedding_text_hash("alpha")
    ledger.upsert_vectors(
        embedding_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
        record_kind="edge",
        entries=[(hash_a, [0.5, 0.6])],
    )

    ledger.clear()

    assert (
        ledger.get_vectors(
            embedding_profile="gemini:gemini-embedding-001|embed_graph_edges=0",
            record_kind="edge",
            text_hashes=[hash_a],
        )
        == {}
    )
