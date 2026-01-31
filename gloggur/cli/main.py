from __future__ import annotations

import json
import os
from typing import Dict, Optional

import click

from gloggur.config import GloggurConfig
from gloggur.embeddings.factory import create_embedding_provider
from gloggur.indexer.cache import CacheConfig, CacheManager
from gloggur.indexer.indexer import Indexer
from gloggur.parsers.registry import ParserRegistry
from gloggur.search.hybrid_search import HybridSearch
from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig
from gloggur.validation.docstring_validator import validate_docstrings


@click.group()
def cli() -> None:
    """Gloggur: symbol-level, incremental codebase indexer."""


def _emit(payload: Dict[str, object], as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(payload)


def _load_config(config_path: Optional[str]) -> GloggurConfig:
    return GloggurConfig.load(path=config_path)


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--embedding-provider", type=str, default=None)
def index(path: str, config_path: Optional[str], as_json: bool, embedding_provider: Optional[str]) -> None:
    """Index a repository or file."""
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
    """Search for code patterns with filters."""
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
def validate(path: str, config_path: Optional[str], as_json: bool) -> None:
    """Run docstring validation."""
    config = _load_config(config_path)
    cache = CacheManager(CacheConfig(config.cache_dir))
    parser_registry = ParserRegistry()
    symbols = []
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
        parser_entry = parser_registry.get_parser_for_path(file_path)
        if not parser_entry:
            continue
        symbols.extend(parser_entry.parser.extract_symbols(file_path, source))
    reports = validate_docstrings(symbols)
    for report in reports:
        cache.set_validation_warnings(report.symbol_id, report.warnings)
    payload = {
        "path": path,
        "warnings": [report.__dict__ for report in reports],
        "total": len(reports),
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
        "metadata": metadata.model_dump() if metadata else None,
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
