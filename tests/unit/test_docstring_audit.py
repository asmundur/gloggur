from __future__ import annotations

from gloggur.embeddings.base import EmbeddingProvider
from gloggur.models import Symbol
from gloggur.audit.docstring_audit import (
    audit_docstrings,
    _assess_symbol,
    _compute_semantic_scores,
)


def _symbol(
    *,
    symbol_id: str,
    name: str,
    kind: str = "function",
    signature: str | None = None,
    docstring: str | None = None,
    language: str = "python",
) -> Symbol:
    """Create a sample Symbol for audit tests."""
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
        language=language,
    )


class FakeEmbeddingProvider(EmbeddingProvider):
    """Fake embedding provider for semantic similarity tests."""
    def embed_text(self, text: str) -> list[float]:
        """Return a deterministic embedding vector based on text."""
        lowered = text.lower()
        if "file" in lowered:
            return [1.0, 0.0]
        if "add" in lowered or "sum" in lowered:
            return [0.0, 1.0]
        return [0.0, 0.0]

    def embed_batch(self, texts):
        """Batch embed by delegating to embed_text."""
        return [self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        """Return the fake embedding dimension."""
        return 2


# ---------------------------------------------------------------------------
# Existing coverage (must remain green)
# ---------------------------------------------------------------------------

def test_audit_reports_missing_docstring() -> None:
    """Missing docstrings should emit warnings."""
    symbol = _symbol(symbol_id="s1", name="add", signature="def add(a, b):", docstring=None)
    reports = audit_docstrings([symbol])
    assert len(reports) == 1
    assert reports[0].warnings == ["Missing docstring"]


def test_audit_includes_private_symbols() -> None:
    """Private symbols are still inspected for docstrings."""
    symbol = _symbol(symbol_id="s2", name="_internal", signature="def _internal():", docstring=None)
    reports = audit_docstrings([symbol])
    assert len(reports) == 1
    assert reports[0].warnings == ["Missing docstring"]


def test_audit_flags_low_semantic_similarity() -> None:
    """Low semantic similarity should be flagged."""
    symbol = _symbol(
        symbol_id="s3",
        name="compute",
        signature="def compute(a, b):",
        docstring="Read a file from disk.",
    )
    code_texts = {symbol.id: "def compute(a, b):\n    return a + b"}
    reports = audit_docstrings(
        [symbol],
        code_texts=code_texts,
        embedding_provider=FakeEmbeddingProvider(),
        semantic_threshold=0.5,
        semantic_min_chars=0,
    )
    assert len(reports) == 1
    warnings = reports[0].warnings
    assert any("Low semantic similarity" in warning for warning in warnings)
    assert reports[0].semantic_score is not None


# ---------------------------------------------------------------------------
# score_metadata explainability
# ---------------------------------------------------------------------------

def test_score_metadata_present_when_scored() -> None:
    """score_metadata should be populated with threshold and kind info when scored."""
    symbol = _symbol(
        symbol_id="m1",
        name="compute",
        docstring="Read a file from disk.",
    )
    code_texts = {symbol.id: "def compute(a, b):\n    return a + b"}
    reports = audit_docstrings(
        [symbol],
        code_texts=code_texts,
        embedding_provider=FakeEmbeddingProvider(),
        semantic_threshold=0.5,
        semantic_min_chars=0,
        semantic_min_code_chars=0,
    )
    assert len(reports) == 1
    meta = reports[0].score_metadata
    assert meta is not None
    assert meta["symbol_kind"] == "function"
    assert meta["threshold_applied"] == 0.5
    assert meta["scored"] is True
    assert meta["score_value"] is not None


def test_score_metadata_absent_for_non_inspected_kind() -> None:
    """score_metadata should be None for symbol kinds not audited."""
    symbol = _symbol(symbol_id="m2", name="MY_CONST", kind="variable", docstring="A constant.")
    reports = audit_docstrings([symbol])
    # Variable kinds are not inspected at all; no report is created.
    assert len(reports) == 0


def test_score_metadata_none_for_missing_docstring() -> None:
    """score_metadata should be None when the symbol lacks a docstring."""
    symbol = _symbol(symbol_id="m3", name="my_func", docstring=None)
    reports = audit_docstrings([symbol])
    assert len(reports) == 1
    assert reports[0].score_metadata is None


# ---------------------------------------------------------------------------
# Kind-aware threshold overrides
# ---------------------------------------------------------------------------

def test_kind_threshold_suppresses_class_warning() -> None:
    """Class-level threshold override should suppress borderline class warnings."""
    # Docstring "A file handler." -> [1.0, 0.0]; code body contains "add" -> [0.0, 1.0]
    # Cosine similarity = 0.0, which is < global threshold 0.2 but >= class threshold 0.0
    class_sym = _symbol(
        symbol_id="k1",
        name="FileHandler",
        kind="class",
        docstring="A file handler.",
    )
    code_texts = {class_sym.id: "class FileHandler:\n    def add(self, x): return x"}
    # With global threshold 0.2 and class override 0.0, the warning should be suppressed.
    reports = audit_docstrings(
        [class_sym],
        code_texts=code_texts,
        embedding_provider=FakeEmbeddingProvider(),
        semantic_threshold=0.2,
        semantic_min_chars=0,
        semantic_min_code_chars=0,
        kind_thresholds={"class": 0.0},
    )
    assert len(reports) == 1
    assert reports[0].warnings == []
    meta = reports[0].score_metadata
    assert meta is not None
    assert meta["threshold_applied"] == 0.0


def test_kind_threshold_override_reflected_in_warning_message() -> None:
    """The warning message should show the kind-specific threshold, not the global one."""
    func_sym = _symbol(
        symbol_id="k2",
        name="read_file",
        kind="function",
        docstring="Adds two numbers.",
    )
    code_texts = {func_sym.id: "def read_file(path):\n    return open(path).read()"}
    # Docstring "Adds two numbers." -> [0.0, 1.0]; code body has "file" -> [1.0, 0.0]
    # Cosine similarity = 0.0, below function override threshold of 0.5.
    reports = audit_docstrings(
        [func_sym],
        code_texts=code_texts,
        embedding_provider=FakeEmbeddingProvider(),
        semantic_threshold=0.2,
        semantic_min_chars=0,
        semantic_min_code_chars=0,
        kind_thresholds={"function": 0.5},
    )
    assert len(reports) == 1
    warnings = reports[0].warnings
    assert any("threshold=0.500" in w for w in warnings)
    meta = reports[0].score_metadata
    assert meta is not None
    assert meta["threshold_applied"] == 0.5


def test_no_kind_threshold_uses_global() -> None:
    """When kind_thresholds is None, the global threshold applies."""
    sym = _symbol(
        symbol_id="k3",
        name="compute",
        kind="function",
        docstring="Read a file from disk.",
    )
    code_texts = {sym.id: "def compute(a, b):\n    return a + b"}
    reports = audit_docstrings(
        [sym],
        code_texts=code_texts,
        embedding_provider=FakeEmbeddingProvider(),
        semantic_threshold=0.5,
        semantic_min_chars=0,
        semantic_min_code_chars=0,
        kind_thresholds=None,
    )
    assert len(reports) == 1
    meta = reports[0].score_metadata
    assert meta is not None
    assert meta["threshold_applied"] == 0.5


# ---------------------------------------------------------------------------
# semantic_min_code_chars filtering
# ---------------------------------------------------------------------------

def test_skip_scoring_for_short_code_body() -> None:
    """Symbols whose code body is shorter than min_code_chars should be skipped."""
    sym = _symbol(
        symbol_id="c1",
        name="noop",
        docstring="A file reader.",
    )
    # Code body after stripping docstring is just "pass" (4 chars).
    code_texts = {sym.id: 'def noop():\n    """A file reader."""\n    pass'}
    reports = audit_docstrings(
        [sym],
        code_texts=code_texts,
        embedding_provider=FakeEmbeddingProvider(),
        semantic_threshold=0.5,
        semantic_min_chars=0,
        semantic_min_code_chars=20,  # require at least 20 chars
    )
    assert len(reports) == 1
    # No warning since scoring was skipped.
    assert reports[0].warnings == []
    meta = reports[0].score_metadata
    assert meta is not None
    assert meta["scored"] is False
    assert "skip_reason" in meta
    assert "code_body_too_short" in meta["skip_reason"]


def test_zero_min_code_chars_scores_all() -> None:
    """With semantic_min_code_chars=0, no code body is skipped on length alone."""
    sym = _symbol(
        symbol_id="c2",
        name="noop",
        docstring="Read a file from disk.",  # -> [1.0, 0.0]
    )
    # Code "return x" is short but contains "add"-like content... use "return add(x)"
    code_texts = {sym.id: "def noop():\n    return sum([1])"}
    reports = audit_docstrings(
        [sym],
        code_texts=code_texts,
        embedding_provider=FakeEmbeddingProvider(),
        semantic_threshold=0.5,
        semantic_min_chars=0,
        semantic_min_code_chars=0,
    )
    assert len(reports) == 1
    meta = reports[0].score_metadata
    assert meta is not None
    assert meta["scored"] is True


# ---------------------------------------------------------------------------
# _assess_symbol unit tests
# ---------------------------------------------------------------------------

def test_assess_symbol_returns_no_metadata_for_unsupported_kind() -> None:
    """Non-inspected kinds return empty warnings and None metadata."""
    sym = _symbol(symbol_id="a1", name="CONST", kind="variable", docstring="A value.")
    warnings, meta = _assess_symbol(sym, semantic_score=None, semantic_threshold=0.2)
    assert warnings == []
    assert meta is None


def test_assess_symbol_returns_no_metadata_on_missing_docstring() -> None:
    """Missing-docstring branch returns None metadata."""
    sym = _symbol(symbol_id="a2", name="my_func", docstring=None)
    warnings, meta = _assess_symbol(sym, semantic_score=None, semantic_threshold=0.2)
    assert warnings == ["Missing docstring"]
    assert meta is None


def test_assess_symbol_skip_reason_propagated() -> None:
    """skip_reason from scoring step should appear in score_metadata."""
    sym = _symbol(symbol_id="a3", name="my_func", docstring="Does stuff.")
    warnings, meta = _assess_symbol(
        sym,
        semantic_score=None,
        semantic_threshold=0.2,
        skip_reason="code_body_too_short (len=5 < min_code=30)",
    )
    assert warnings == []
    assert meta is not None
    assert meta["skip_reason"] == "code_body_too_short (len=5 < min_code=30)"
    assert meta["scored"] is False


# ---------------------------------------------------------------------------
# _compute_semantic_scores unit tests (min_code_chars parameter)
# ---------------------------------------------------------------------------

def test_compute_scores_skips_short_code_bodies() -> None:
    """Symbols with short code bodies are recorded in skip_reasons."""
    sym = _symbol(symbol_id="cs1", name="trivial", docstring="Does something.")
    code_texts = {sym.id: "def trivial():\n    pass"}
    scores, skip_reasons = _compute_semantic_scores(
        [sym],
        code_texts=code_texts,
        embedding_provider=FakeEmbeddingProvider(),
        min_chars=0,
        max_chars=4000,
        min_code_chars=30,
    )
    assert sym.id not in scores
    assert sym.id in skip_reasons
    assert "code_body_too_short" in skip_reasons[sym.id]


def test_compute_scores_skips_short_docstrings() -> None:
    """Symbols with docstrings shorter than min_chars are skipped."""
    sym = _symbol(symbol_id="cs2", name="x", docstring="Hi.")  # 3 chars
    code_texts = {sym.id: "def x():\n    return 42 + additional_long_code_body_to_pass_min"}
    scores, skip_reasons = _compute_semantic_scores(
        [sym],
        code_texts=code_texts,
        embedding_provider=FakeEmbeddingProvider(),
        min_chars=10,
        max_chars=4000,
        min_code_chars=0,
    )
    assert sym.id not in scores
    assert sym.id in skip_reasons
    assert "docstring_too_short" in skip_reasons[sym.id]


# ---------------------------------------------------------------------------
# Config calibration defaults
# ---------------------------------------------------------------------------

def test_default_config_kind_thresholds() -> None:
    """Calibrated GloggurConfig defaults include kind-aware thresholds."""
    from gloggur.config import GloggurConfig
    cfg = GloggurConfig()
    assert cfg.docstring_semantic_threshold == 0.10
    assert cfg.docstring_semantic_min_code_chars == 30
    thresholds = cfg.docstring_semantic_kind_thresholds
    assert thresholds is not None
    assert thresholds.get("class") == 0.05
    assert thresholds.get("interface") == 0.05


def test_config_kind_thresholds_overridable() -> None:
    """kind_thresholds can be overridden via config overrides dict."""
    from gloggur.config import GloggurConfig
    cfg = GloggurConfig.load(overrides={"docstring_semantic_kind_thresholds": {"class": 0.15}})
    assert cfg.docstring_semantic_kind_thresholds == {"class": 0.15}


def test_config_min_code_chars_overridable() -> None:
    """docstring_semantic_min_code_chars can be overridden."""
    from gloggur.config import GloggurConfig
    cfg = GloggurConfig.load(overrides={"docstring_semantic_min_code_chars": 50})
    assert cfg.docstring_semantic_min_code_chars == 50
