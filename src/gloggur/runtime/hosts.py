from __future__ import annotations

from dataclasses import dataclass

from gloggur.adapters.registry import AdapterRegistry, adapter_module_override, instantiate_adapter
from gloggur.config import GloggurConfig
from gloggur.watch.service import WatchService


class RuntimeHostAdapter:
    """Runtime host abstraction for watch/bootstrap lifecycle integration."""

    host_id: str

    def build_watch_service(
        self,
        *,
        config,
        embedding_provider,
        cache,
        vector_store,
        parser_registry,
    ):
        """Create the runtime watch service."""
        raise NotImplementedError


@dataclass
class PythonLocalRuntimeHost(RuntimeHostAdapter):
    """Default runtime host backed by local Python process orchestration."""

    host_id: str = "python_local"

    def build_watch_service(
        self,
        *,
        config,
        embedding_provider,
        cache,
        vector_store,
        parser_registry,
    ) -> WatchService:
        return WatchService(
            config=config,
            embedding_provider=embedding_provider,
            cache=cache,
            vector_store=vector_store,
            parser_registry=parser_registry,
        )


_RUNTIME_HOSTS = AdapterRegistry[RuntimeHostAdapter]("gloggur.runtime_hosts")
_RUNTIME_HOSTS.register_builtin("python_local", lambda: PythonLocalRuntimeHost())


def create_runtime_host(config: GloggurConfig) -> RuntimeHostAdapter:
    """Resolve runtime host adapter from config/entrypoints/module overrides."""
    host_id = config.runtime_host()
    module_override = adapter_module_override(
        config.adapters if isinstance(config.adapters, dict) else None,
        category="runtime",
        adapter_id=host_id,
    )
    factory = _RUNTIME_HOSTS.resolve_factory(host_id, module_path_override=module_override)
    host = instantiate_adapter(factory)
    if not isinstance(host, RuntimeHostAdapter):
        if not hasattr(host, "build_watch_service"):
            raise RuntimeError(f"Runtime host '{host_id}' is invalid ({type(host).__name__}).")
    return host


def list_runtime_hosts() -> list[dict[str, object]]:
    """Return discoverable runtime host adapter descriptors."""
    return _RUNTIME_HOSTS.available()
