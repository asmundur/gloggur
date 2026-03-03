from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class AdapterResolutionError(RuntimeError):
    """Raised when an adapter id cannot be resolved to a callable factory."""


class AdapterValidationError(RuntimeError):
    """Raised when an adapter factory returns an invalid object."""


@dataclass(frozen=True)
class _AdapterFactory:
    """One adapter factory registration with source provenance."""

    adapter_id: str
    factory: Callable[..., T]
    source: str
    callable_path: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


class AdapterRegistry(Generic[T]):
    """Resolve adapter factories from builtins, entry points, and module paths."""

    def __init__(self, entrypoint_group: str) -> None:
        self._entrypoint_group = entrypoint_group
        self._builtins: dict[str, _AdapterFactory] = {}
        self._aliases: dict[str, str] = {}
        self._entrypoints_loaded = False

    def register_builtin(
        self,
        adapter_id: str,
        factory: Callable[..., T],
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Register one built-in adapter factory."""
        normalized = adapter_id.strip()
        if not normalized:
            raise ValueError("adapter_id must not be empty")
        callable_name = getattr(factory, "__name__", factory.__class__.__name__)
        descriptor = _AdapterFactory(
            adapter_id=normalized,
            factory=factory,
            source="builtin",
            callable_path=f"{factory.__module__}:{callable_name}",
            aliases=aliases,
        )
        self._builtins[normalized] = descriptor
        self._aliases[normalized] = normalized
        for alias in aliases:
            if alias:
                self._aliases[alias] = normalized

    def available(self) -> list[dict[str, object]]:
        """Return deterministic adapter descriptors for operator discovery."""
        self._load_entrypoints_once()
        rows: list[dict[str, object]] = []
        for adapter_id in sorted(self._builtins):
            descriptor = self._builtins[adapter_id]
            rows.append(
                {
                    "id": descriptor.adapter_id,
                    "source": descriptor.source,
                    "callable": descriptor.callable_path,
                    "aliases": list(descriptor.aliases),
                }
            )
        return rows

    def resolve_factory(
        self,
        adapter_id: str,
        *,
        module_path_override: str | None = None,
    ) -> Callable[..., T]:
        """Resolve adapter factory from module override or registry lookup."""
        if module_path_override:
            return self._load_module_factory(module_path_override)
        self._load_entrypoints_once()
        canonical = self._aliases.get(adapter_id, adapter_id)
        descriptor = self._builtins.get(canonical)
        if descriptor is None:
            available = ", ".join(sorted(self._builtins))
            raise AdapterResolutionError(
                f"Unknown adapter '{adapter_id}' for group '{self._entrypoint_group}'. "
                f"Available: [{available}]"
            )
        return descriptor.factory

    def _load_entrypoints_once(self) -> None:
        """Load entry points exactly once and merge them into local registrations."""
        if self._entrypoints_loaded:
            return
        self._entrypoints_loaded = True
        try:
            eps = metadata.entry_points()
            selected = eps.select(group=self._entrypoint_group)  # type: ignore[attr-defined]
        except Exception:
            selected = []
        for entry_point in selected:
            try:
                factory = entry_point.load()
            except Exception:
                continue
            if not callable(factory):
                continue
            adapter_id = str(entry_point.name).strip()
            if not adapter_id or adapter_id in self._builtins:
                continue
            descriptor = _AdapterFactory(
                adapter_id=adapter_id,
                factory=factory,
                source="entrypoint",
                callable_path=f"{entry_point.module}:{entry_point.attr}",
            )
            self._builtins[adapter_id] = descriptor
            self._aliases[adapter_id] = adapter_id

    @staticmethod
    def _load_module_factory(module_path: str) -> Callable[..., T]:
        """Load `module:callable` factory for local/explicit adapter overrides."""
        module_name, sep, attr_name = module_path.partition(":")
        if not sep or not module_name or not attr_name:
            raise AdapterResolutionError(
                "Adapter override must use 'module:callable' format, "
                f"received '{module_path}'."
            )
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise AdapterResolutionError(
                f"Failed importing adapter module '{module_name}': {exc}"
            ) from exc
        if not hasattr(module, attr_name):
            raise AdapterResolutionError(
                f"Adapter callable '{attr_name}' was not found in module '{module_name}'."
            )
        factory = getattr(module, attr_name)
        if not callable(factory):
            raise AdapterResolutionError(
                f"Adapter target '{module_path}' is not callable."
            )
        return factory


def adapter_module_override(
    adapter_map: dict[str, object] | None,
    *,
    category: str,
    adapter_id: str,
) -> str | None:
    """Resolve optional module override path for one adapter id/category."""
    if not adapter_map:
        return None
    category_map = adapter_map.get(category)
    if not isinstance(category_map, dict):
        return None
    value = category_map.get(adapter_id)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def instantiate_adapter(factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Instantiate one adapter via callable factory with deterministic errors."""
    try:
        return factory(*args, **kwargs)
    except TypeError:
        # Support class-style factories that only accept config as positional.
        if args and not kwargs:
            return factory(args[0])
        raise
