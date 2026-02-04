from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import click

from gloggur.config import GloggurConfig
from gloggur.embeddings.factory import create_embedding_provider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.indexer import Indexer
from gloggur.models import ValidationFileMetadata
from gloggur.parsers.registry import ParserRegistry
from gloggur.search.hybrid_search import HybridSearch
from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from gloggur.validation.docstring_validator import validate_docstrings


@click.group()
def cli() -> None:
    """Gloggur CLI for indexing, search, and docstring validation."""


def _emit(payload: Dict[str, object], as_json: bool) -> None:
    """Print payload as JSON or raw text."""
    if as_json:
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(payload)


def _load_config(config_path: Optional[str]) -> GloggurConfig:
    """Load configuration from file/env."""
    return GloggurConfig.load(path=config_path)


def _hash_content(source: str) -> str:
    """Hash content to detect changes."""
    return hashlib.sha256(source.encode("utf8")).hexdigest()


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--embedding-provider", type=str, default=None)
def index(path: str, config_path: Optional[str], as_json: bool, embedding_provider: Optional[str]) -> None:
    """Index a repository or file and emit counts."""
    overrides = {}
    if embedding_provider:
        overrides["embedding_provider"] = embedding_provider
    config = _load_config(config_path)
    if overrides:
        config = GloggurConfig.load(path=config_path, overrides=overrides)
    click.echo("Indexing...", err=True)
    cache = CacheManager(CacheConfig(config.cache_dir))
    vector_store = VectorStore(VectorStoreConfig(config.cache_dir))
    embedding = create_embedding_provider(config) if config.embedding_provider else None
    indexer = Indexer(
        config=config,
        cache=cache,
        parser_registry=ParserRegistry(),
        embedding_provider=embedding,
        vector_store=vector_store,
    )
    if os.path.isdir(path):
        result = indexer.index_repository(path)
    else:
        count = indexer.index_file(path) or 0
        result = {
            "indexed_files": 1 if count else 0,
            "indexed_symbols": count,
            "skipped_files": 0 if count else 1,
            "duration_ms": 0,
        }
        _emit(result, as_json)
        return
    _emit(result.__dict__, as_json)


@cli.command()
@click.argument("query", type=str)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--kind", type=str, default=None)
@click.option("--file", "file_path", type=str, default=None)
@click.option("--top-k", type=int, default=10)
@click.option("--stream", is_flag=True, default=False)
def search(
    query: str,
    config_path: Optional[str],
    as_json: bool,
    kind: Optional[str],
    file_path: Optional[str],
    top_k: int,
    stream: bool,
) -> None:
    """Search indexed symbols with optional filters."""
    config = _load_config(config_path)
    embedding = create_embedding_provider(config)
    vector_store = VectorStore(VectorStoreConfig(config.cache_dir))
    metadata_store = MetadataStore(MetadataStoreConfig(config.cache_dir))
    searcher = HybridSearch(embedding, vector_store, metadata_store)
    filters = {}
    if kind:
        filters["kind"] = kind
    if file_path:
        filters["file"] = file_path
    result = searcher.search(query, filters=filters, top_k=top_k)
    if stream and as_json:
        for item in result["results"]:
            click.echo(json.dumps(item))
        return
    _emit(result, as_json)


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--force", is_flag=True, default=False, help="Revalidate even if unchanged since last run.")
def validate(path: str, config_path: Optional[str], as_json: bool, force: bool) -> None:
    """Run docstring validation and emit warnings/reports."""
    config = _load_config(config_path)
    cache = CacheManager(CacheConfig(config.cache_dir))
    parser_registry = ParserRegistry()
    embedding = create_embedding_provider(config) if config.embedding_provider else None
    symbols = []
    code_texts: Dict[str, str] = {}
    processed_files: List[Tuple[str, str]] = []
    skipped_files = 0
    validated_files = 0
    paths = [path]
    if os.path.isdir(path):
        paths = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in config.excluded_dirs]
            for filename in files:
                full_path = os.path.join(root, filename)
                if any(full_path.endswith(ext) for ext in config.supported_extensions):
                    paths.append(full_path)
    for file_path in paths:
        try:
            with open(file_path, "r", encoding="utf8") as handle:
                source = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        content_hash = _hash_content(source)
        if not force:
            existing = cache.get_validation_file_metadata(file_path)
            if existing and existing.content_hash == content_hash:
                skipped_files += 1
                continue
        parser_entry = parser_registry.get_parser_for_path(file_path)
        if not parser_entry:
            continue
        file_symbols = parser_entry.parser.extract_symbols(file_path, source)
        lines = source.splitlines()
        for symbol in file_symbols:
            snippet_start = max(0, symbol.start_line - 1)
            snippet_end = min(len(lines), symbol.end_line)
            code_texts[symbol.id] = "\n".join(lines[snippet_start:snippet_end])
        symbols.extend(file_symbols)
        processed_files.append((file_path, content_hash))
        validated_files += 1
    reports = validate_docstrings(
        symbols,
        code_texts=code_texts,
        embedding_provider=embedding,
        semantic_threshold=config.docstring_semantic_threshold,
        semantic_min_chars=config.docstring_semantic_min_chars,
        semantic_max_chars=config.docstring_semantic_max_chars,
    )
    for report in reports:
        cache.set_validation_warnings(report.symbol_id, report.warnings)
    for file_path, content_hash in processed_files:
        cache.upsert_validation_file_metadata(
            ValidationFileMetadata(path=file_path, content_hash=content_hash)
        )
    warning_reports = [report for report in reports if report.warnings]
    payload = {
        "path": path,
        "warnings": [report.__dict__ for report in warning_reports],
        "total": len(warning_reports),
        "reports": [report.__dict__ for report in reports],
        "reports_total": len(reports),
        "validated_files": validated_files,
        "skipped_files": skipped_files,
    }
    _emit(payload, as_json)


@cli.command()
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
def status(config_path: Optional[str], as_json: bool) -> None:
    """Show index statistics and metadata."""
    config = _load_config(config_path)
    cache = CacheManager(CacheConfig(config.cache_dir))
    metadata = cache.get_index_metadata()
    payload = {
        "cache_dir": config.cache_dir,
        "metadata": metadata.model_dump(mode="json") if metadata else None,
        "total_symbols": len(cache.list_symbols()),
    }
    _emit(payload, as_json)


@cli.command("clear-cache")
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
def clear_cache(config_path: Optional[str], as_json: bool) -> None:
    """Clear the index cache."""
    config = _load_config(config_path)
    cache = CacheManager(CacheConfig(config.cache_dir))
    cache.clear()
    vector_store = VectorStore(VectorStoreConfig(config.cache_dir))
    vector_store.clear()
    _emit({"cleared": True, "cache_dir": config.cache_dir}, as_json)


if __name__ == "__main__":
    cli()
