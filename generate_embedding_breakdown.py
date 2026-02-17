from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
if (THIS_FILE.parent / "gloggur").exists():
    PROJECT_ROOT = THIS_FILE.parent
else:
    PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gloggur.config import GloggurConfig  # noqa: E402
from gloggur.embeddings.factory import create_embedding_provider  # noqa: E402
from gloggur.indexer.indexer import Indexer  # noqa: E402
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


def _chunk_payload(
    symbol: Any, lines: list[str], vector: list[float], preview: int
) -> dict[str, Any]:
    """Build one report payload entry for a symbol and its embedding."""
    snippet_start = max(0, symbol.start_line - 1)
    snippet_end = min(len(lines), snippet_start + 3)
    chunk_text = Indexer._symbol_text(symbol, lines)
    return {
        "id": symbol.id,
        "name": symbol.name,
        "kind": symbol.kind,
        "file_path": symbol.file_path,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
        "signature": symbol.signature,
        "docstring": symbol.docstring,
        "body_hash": symbol.body_hash,
        "language": symbol.language,
        "snippet_start_line": snippet_start + 1,
        "snippet_end_line": snippet_end,
        "chunk_text": chunk_text,
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
    lines.append("One embedding chunk is produced per extracted symbol. The chunk text is:")
    lines.append("")
    lines.append("```text")
    lines.append(
        "chunk_text = join_non_empty([signature, docstring, "
        "3_line_snippet_from_start_line])"
    )
    lines.append("```")
    lines.append("")
    lines.append(f"## Per-Symbol Chunks ({len(chunks)})")
    lines.append("")
    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"### Chunk {index}: `{chunk['name']}` ({chunk['kind']})")
        lines.append("")
        lines.append("- Annotations:")
        annotation_keys = [
            "id",
            "name",
            "kind",
            "file_path",
            "start_line",
            "end_line",
            "signature",
            "docstring",
            "body_hash",
            "language",
            "snippet_start_line",
            "snippet_end_line",
            "embedding_dimension",
        ]
        for key in annotation_keys:
            lines.append(f"  - `{key}`: `{chunk[key]}`")
        lines.append(
            f"  - `embedding_preview_first_{preview}`: "
            f"`{chunk['embedding_preview_first_n']}`"
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
    """Generate a markdown report that shows embedding chunk inputs per symbol."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a semantic-search embedding breakdown markdown report "
            "for one source file."
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
        choices=["local", "openai", "gemini"],
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
    lines = source.splitlines()
    chunk_texts = [Indexer._symbol_text(symbol, lines) for symbol in symbols]

    provider = create_embedding_provider(config)
    vectors = provider.embed_batch(chunk_texts) if chunk_texts else []
    if len(vectors) != len(symbols):
        raise SystemExit(
            f"Embedding provider returned {len(vectors)} vectors for {len(symbols)} symbols."
        )

    chunks = [
        _chunk_payload(symbol, lines, vector, args.vector_preview)
        for symbol, vector in zip(symbols, vectors, strict=True)
    ]

    runtime = _provider_runtime(provider, vectors)
    markdown = _render_markdown(
        source_path=source_key,
        config=config,
        runtime=runtime,
        chunks=chunks,
        preview=args.vector_preview,
        include_vectors=args.include_full_vectors,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf8")
    print(f"Wrote embedding breakdown: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
