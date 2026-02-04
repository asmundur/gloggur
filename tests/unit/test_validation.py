from __future__ import annotations

from gloggur.embeddings.base import EmbeddingProvider
from gloggur.models import Symbol
from gloggur.validation.docstring_validator import validate_docstrings


def _symbol(
    *,
    symbol_id: str,
    name: str,
    kind: str = "function",
    signature: str | None = None,
    docstring: str | None = None,
) -> Symbol:
    return Symbol(
        id=symbol_id,
        name=name,
        kind=kind,
        file_path="sample.py",
        start_line=1,
        end_line=2,
        signature=signature,
        docstring=docstring,
        body_hash="abc123",
        language="python",
    )


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed_text(self, text: str) -> list[float]:
        lowered = text.lower()
        if "file" in lowered:
            return [1.0, 0.0]
        if "add" in lowered or "sum" in lowered:
            return [0.0, 1.0]
        return [0.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        return 2


def test_validation_reports_missing_docstring() -> None:
    symbol = _symbol(symbol_id="s1", name="add", signature="def add(a, b):", docstring=None)
    reports = validate_docstrings([symbol])
    assert len(reports) == 1
    assert reports[0].warnings == ["Missing docstring"]


def test_validation_includes_private_symbols() -> None:
    symbol = _symbol(symbol_id="s2", name="_internal", signature="def _internal():", docstring=None)
    reports = validate_docstrings([symbol])
    assert len(reports) == 1
    assert reports[0].warnings == ["Missing docstring"]


def test_validation_flags_low_semantic_similarity() -> None:
    symbol = _symbol(
        symbol_id="s3",
        name="compute",
        signature="def compute(a, b):",
        docstring="Read a file from disk.",
    )
    code_texts = {symbol.id: "def compute(a, b):\n    return a + b"}
    reports = validate_docstrings(
        [symbol],
        code_texts=code_texts,
        embedding_provider=FakeEmbeddingProvider(),
        semantic_threshold=0.5,
    )
    assert len(reports) == 1
    warnings = reports[0].warnings
    assert any("Low semantic similarity" in warning for warning in warnings)
    assert reports[0].semantic_score is not None
