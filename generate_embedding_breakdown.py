from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
if (THIS_FILE.parent / "src" / "gloggur").exists():
    PROJECT_ROOT = THIS_FILE.parent
    IMPORT_ROOT = PROJECT_ROOT / "src"
elif (THIS_FILE.parent / "gloggur").exists():
    PROJECT_ROOT = THIS_FILE.parent
    IMPORT_ROOT = PROJECT_ROOT
else:
    PROJECT_ROOT = THIS_FILE.parents[1]
    IMPORT_ROOT = PROJECT_ROOT / "src" if (PROJECT_ROOT / "src" / "gloggur").exists() else PROJECT_ROOT
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

from gloggur.byte_spans import LineByteSpanIndex  # noqa: E402
from gloggur.config import GloggurConfig  # noqa: E402
from gloggur.embeddings.factory import create_embedding_provider  # noqa: E402
from gloggur.indexer.indexer import Indexer  # noqa: E402
from gloggur.models import Symbol, SymbolChunk  # noqa: E402
from gloggur.parsers.registry import ParserRegistry  # noqa: E402


def _default_output_path(source_path: Path) -> Path:
    """Return the default markdown output path for a source file."""
    slug = source_path.stem.replace("_", "-")
    return PROJECT_ROOT / "docs" / f"semantic-search-embedding-breakdown-{slug}.md"


def _load_config(config_path: str | None, embedding_provider: str | None) -> GloggurConfig:
    """Load repository config with optional embedding-provider override."""
    overrides: dict[str, object] | None = None
    if embedding_provider:
        overrides = {"embedding_provider": embedding_provider}
    return GloggurConfig.load(path=config_path, overrides=overrides)


def _project_relative(path: Path) -> str:
    """Return a stable repository-relative path when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _provider_runtime(provider: Any, vectors: list[list[float]]) -> dict[str, Any]:
    """Return runtime details for the active embedding provider."""
    info: dict[str, Any] = {
        "provider_class": provider.__class__.__name__,
        "provider_module": provider.__class__.__module__,
    }
    model_name = getattr(provider, "model_name", None) or getattr(provider, "model", None)
    if model_name:
        info["model_name"] = model_name
    if hasattr(provider, "_use_fallback"):
        info["fallback_mode"] = bool(provider._use_fallback)
    if hasattr(provider, "_fallback_marker"):
        marker = Path(provider._fallback_marker)
        info["fallback_marker"] = str(marker)
        info["fallback_marker_exists"] = marker.exists()
    dimension = len(vectors[0]) if vectors else 0
    if not dimension and hasattr(provider, "get_dimension"):
        dimension = int(provider.get_dimension())
    info["embedding_dimension"] = dimension
    return info


def _build_symbol_chunks(
    config: GloggurConfig,
    *,
    source_key: str,
    source: str,
    symbols: list[Symbol],
) -> list[SymbolChunk]:
    """Build chunk rows using the same implementation as the live indexer."""
    indexer = object.__new__(Indexer)
    indexer.config = config
    span_index = LineByteSpanIndex.from_bytes(source.encode("utf8"))
    return Indexer._build_symbol_chunks(
        indexer,
        path=source_key,
        source=source,
        symbols=symbols,
        commit="",
        span_index=span_index,
    )


def _chunk_payload(
    chunk: SymbolChunk,
    symbol: Symbol | None,
    vector: list[float],
    preview: int,
) -> dict[str, Any]:
    """Build one report payload entry for an embedding chunk."""
    return {
        "chunk_id": chunk.chunk_id,
        "symbol_id": chunk.symbol_id,
        "name": symbol.name if symbol is not None else chunk.symbol_id,
        "kind": symbol.kind if symbol is not None else None,
        "fqname": symbol.fqname if symbol is not None else None,
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "start_byte": chunk.start_byte,
        "end_byte": chunk.end_byte,
        "chunk_part_index": chunk.chunk_part_index,
        "chunk_part_total": chunk.chunk_part_total,
        "signature": symbol.signature if symbol is not None else None,
        "docstring": symbol.docstring if symbol is not None else None,
        "body_hash": symbol.body_hash if symbol is not None else None,
        "language": chunk.language,
        "chunk_text": chunk.text,
        "embedding_dimension": len(vector),
        "embedding_preview_first_n": [round(value, 6) for value in vector[:preview]],
        "embedding_vector": vector,
    }


def _render_markdown(
    source_path: str,
    config: GloggurConfig,
    runtime: dict[str, Any],
    chunks: list[dict[str, Any]],
    preview: int,
    include_vectors: bool,
) -> str:
    """Render the embedding breakdown markdown document."""
    lines: list[str] = []
    lines.append("# Semantic Search Embedding Breakdown")
    lines.append("")
    lines.append(f"Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"Source file: `{source_path}`")
    lines.append("")
    lines.append("## Embedding Setup")
    lines.append("")
    lines.append(f"- `embedding_profile`: `{config.embedding_profile()}`")
    lines.append(f"- `provider_class`: `{runtime['provider_class']}`")
    lines.append(f"- `provider_module`: `{runtime['provider_module']}`")
    if "model_name" in runtime:
        lines.append(f"- `model_name`: `{runtime['model_name']}`")
    lines.append(f"- `embedding_dimension`: `{runtime['embedding_dimension']}`")
    if "fallback_mode" in runtime:
        lines.append(f"- `fallback_mode`: `{runtime['fallback_mode']}`")
    if "fallback_marker" in runtime:
        lines.append(f"- `fallback_marker`: `{runtime['fallback_marker']}`")
    if "fallback_marker_exists" in runtime:
        lines.append(f"- `fallback_marker_exists`: `{runtime['fallback_marker_exists']}`")
    lines.append("")
    lines.append("## Chunking Rule")
    lines.append("")
    lines.append(
        "The report uses the same chunk builder as the live indexer. A symbol may produce one"
        " or more chunks when its body exceeds the configured chunk-size budget."
    )
    lines.append("")
    lines.append("```text")
    lines.append("chunk_text = join_non_empty([")
    lines.append("  fqname/kind/file/line header,")
    lines.append("  signature block,")
    lines.append("  imports-in-scope block (when present),")
    lines.append("  bounded body slice")
    lines.append("])")
    lines.append("```")
    lines.append("")
    lines.append(f"## Chunks ({len(chunks)})")
    lines.append("")
    for index, chunk in enumerate(chunks, start=1):
        kind_suffix = f" ({chunk['kind']})" if chunk.get("kind") else ""
        lines.append(f"### Chunk {index}: `{chunk['name']}`{kind_suffix}")
        lines.append("")
        lines.append("- Annotations:")
        annotation_keys = [
            "chunk_id",
            "symbol_id",
            "name",
            "kind",
            "fqname",
            "file_path",
            "start_line",
            "end_line",
            "start_byte",
            "end_byte",
            "chunk_part_index",
            "chunk_part_total",
            "signature",
            "docstring",
            "body_hash",
            "language",
            "embedding_dimension",
        ]
        for key in annotation_keys:
            lines.append(f"  - `{key}`: `{chunk[key]}`")
        lines.append(
            f"  - `embedding_preview_first_{preview}`: " f"`{chunk['embedding_preview_first_n']}`"
        )
        lines.append("")
        lines.append("- `chunk_text`:")
        lines.append("")
        lines.append("```python")
        lines.append(chunk["chunk_text"])
        lines.append("```")
        if include_vectors:
            lines.append("")
            lines.append("- `embedding_vector`:")
            lines.append("")
            lines.append("```json")
            lines.append(str(chunk["embedding_vector"]))
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    """Generate a markdown report that shows embedding chunk inputs for one file."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a semantic-search embedding breakdown markdown report " "for one source file."
        )
    )
    parser.add_argument(
        "source_file",
        type=str,
        help="Supported source file to inspect (e.g. .py, .ts).",
    )
    parser.add_argument("--output", type=str, default=None, help="Output markdown file path.")
    parser.add_argument("--config", type=str, default=None, help="Optional config file path.")
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default=None,
        choices=["local", "openai", "gemini", "test"],
        help="Override configured embedding provider.",
    )
    parser.add_argument(
        "--vector-preview",
        type=int,
        default=8,
        help="How many leading vector values to include per chunk.",
    )
    parser.add_argument(
        "--include-full-vectors",
        action="store_true",
        help="Include full embedding vectors in the markdown report.",
    )
    args = parser.parse_args()

    if args.vector_preview <= 0:
        raise SystemExit("--vector-preview must be > 0")

    source_file = Path(args.source_file).resolve()
    if not source_file.exists() or not source_file.is_file():
        raise SystemExit(f"Source file not found: {source_file}")

    output_path = Path(args.output).resolve() if args.output else _default_output_path(source_file)
    config = _load_config(args.config, args.embedding_provider)

    source_key = _project_relative(source_file)
    parser_entry = ParserRegistry().get_parser_for_path(source_key)
    if parser_entry is None:
        raise SystemExit(f"Unsupported file extension for parser registry: {source_file}")

    source = source_file.read_text(encoding="utf8")
    symbols = parser_entry.parser.extract_symbols(source_key, source)
    chunks = _build_symbol_chunks(config, source_key=source_key, source=source, symbols=symbols)
    chunk_texts = [chunk.text for chunk in chunks]

    provider = create_embedding_provider(config)
    vectors = provider.embed_batch(chunk_texts) if chunk_texts else []
    if len(vectors) != len(chunks):
        raise SystemExit(
            f"Embedding provider returned {len(vectors)} vectors for {len(chunks)} chunks."
        )

    symbol_by_id = {symbol.id: symbol for symbol in symbols}
    chunk_payloads = [
        _chunk_payload(chunk, symbol_by_id.get(chunk.symbol_id), vector, args.vector_preview)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    runtime = _provider_runtime(provider, vectors)
    markdown = _render_markdown(
        source_path=source_key,
        config=config,
        runtime=runtime,
        chunks=chunk_payloads,
        preview=args.vector_preview,
        include_vectors=args.include_full_vectors,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf8")
    print(f"Wrote embedding breakdown: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
