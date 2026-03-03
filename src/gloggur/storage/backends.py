from __future__ import annotations

from dataclasses import dataclass

from gloggur.adapters.registry import AdapterRegistry, adapter_module_override, instantiate_adapter
from gloggur.config import GloggurConfig
from gloggur.storage.metadata_store import MetadataStore, MetadataStoreConfig
from gloggur.storage.vector_store import VectorStore, VectorStoreConfig


class StorageBackend:
    """Storage backend factory for vector + metadata stores."""

    backend_id: str

    def create_vector_store(self, cache_dir: str, *, load_existing: bool = True):
        """Create a vector backend instance."""
        raise NotImplementedError

    def create_metadata_store(self, cache_dir: str):
        """Create a metadata backend instance."""
        raise NotImplementedError


@dataclass
class SQLiteFaissStorageBackend(StorageBackend):
    """Default storage backend wrapping current SQLite + FAISS implementations."""

    backend_id: str = "sqlite_faiss"

    def create_vector_store(self, cache_dir: str, *, load_existing: bool = True) -> VectorStore:
        return VectorStore(VectorStoreConfig(cache_dir), load_existing=load_existing)

    def create_metadata_store(self, cache_dir: str) -> MetadataStore:
        return MetadataStore(MetadataStoreConfig(cache_dir))


_STORAGE_BACKENDS = AdapterRegistry[StorageBackend]("gloggur.storage_backends")
_STORAGE_BACKENDS.register_builtin("sqlite_faiss", lambda: SQLiteFaissStorageBackend())


def create_storage_backend(config: GloggurConfig) -> StorageBackend:
    """Create storage backend from config/entrypoints/module override."""
    backend_id = config.storage_backend()
    module_override = adapter_module_override(
        config.adapters if isinstance(config.adapters, dict) else None,
        category="storage",
        adapter_id=backend_id,
    )
    factory = _STORAGE_BACKENDS.resolve_factory(backend_id, module_path_override=module_override)
    backend = instantiate_adapter(factory)
    if not isinstance(backend, StorageBackend):
        if not all(
            hasattr(backend, method) for method in ("create_vector_store", "create_metadata_store")
        ):
            raise RuntimeError(
                f"Storage backend '{backend_id}' is invalid ({type(backend).__name__})."
            )
    return backend


def list_storage_backends() -> list[dict[str, object]]:
    """Return discoverable storage backend descriptors."""
    return _STORAGE_BACKENDS.available()
