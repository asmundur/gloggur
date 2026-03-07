from __future__ import annotations

from dataclasses import dataclass

import pytest

import gloggur.adapters.registry as registry_module
from gloggur.adapters.registry import (
    AdapterRegistry,
    AdapterResolutionError,
    adapter_module_override,
    instantiate_adapter,
)


def test_register_builtin_supports_alias_lookup_and_available_descriptors() -> None:
    registry = AdapterRegistry[object]("gloggur.tests")

    def factory() -> object:
        return object()

    registry.register_builtin("alpha", factory, aliases=("a", "alias"))

    assert registry.resolve_factory("a") is factory
    assert registry.resolve_factory("alias") is factory
    assert registry.available() == [
        {
            "id": "alpha",
            "source": "builtin",
            "callable": f"{factory.__module__}:{factory.__name__}",
            "aliases": ["a", "alias"],
        }
    ]


def test_register_builtin_rejects_empty_ids() -> None:
    registry = AdapterRegistry[object]("gloggur.tests")

    with pytest.raises(ValueError, match="must not be empty"):
        registry.register_builtin("  ", lambda: object())


def test_resolve_factory_unknown_adapter_includes_available_ids() -> None:
    registry = AdapterRegistry[object]("gloggur.tests")
    registry.register_builtin("alpha", lambda: object())

    with pytest.raises(AdapterResolutionError) as error:
        registry.resolve_factory("missing")

    assert "Unknown adapter 'missing'" in str(error.value)
    assert "Available: [alpha]" in str(error.value)


def test_load_entrypoints_once_ignores_invalid_duplicate_and_noncallable_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = AdapterRegistry[object]("gloggur.tests")
    registry.register_builtin("alpha", lambda: object())
    calls = {"entry_points": 0}

    def beta_factory() -> object:
        return object()

    @dataclass
    class FakeEntryPoint:
        name: str
        module: str
        attr: str
        loader: object

        def load(self) -> object:
            if isinstance(self.loader, Exception):
                raise self.loader
            return self.loader

    class FakeEntryPoints:
        def select(self, *, group: str) -> list[FakeEntryPoint]:
            assert group == "gloggur.tests"
            return [
                FakeEntryPoint("alpha", "pkg", "alpha_factory", lambda: object()),
                FakeEntryPoint("beta", "pkg", "beta_factory", beta_factory),
                FakeEntryPoint("broken", "pkg", "broken_factory", RuntimeError("boom")),
                FakeEntryPoint("not-callable", "pkg", "value", object()),
            ]

    def fake_entry_points() -> FakeEntryPoints:
        calls["entry_points"] += 1
        return FakeEntryPoints()

    monkeypatch.setattr(registry_module.metadata, "entry_points", fake_entry_points)

    first = registry.available()
    second = registry.available()

    assert calls["entry_points"] == 1
    assert [row["id"] for row in first] == ["alpha", "beta"]
    assert first == second


def test_load_module_factory_validates_format_imports_and_callables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModule:
        existing = staticmethod(lambda: "ok")
        value = "not-callable"

    monkeypatch.setattr(registry_module.importlib, "import_module", lambda name: FakeModule())

    assert AdapterRegistry._load_module_factory("fake_module:existing")() == "ok"

    with pytest.raises(AdapterResolutionError, match="module:callable"):
        AdapterRegistry._load_module_factory("broken-format")
    with pytest.raises(AdapterResolutionError, match="was not found"):
        AdapterRegistry._load_module_factory("fake_module:missing")
    with pytest.raises(AdapterResolutionError, match="is not callable"):
        AdapterRegistry._load_module_factory("fake_module:value")


def test_load_module_factory_wraps_import_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registry_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(RuntimeError(f"cannot import {name}")),
    )

    with pytest.raises(AdapterResolutionError) as error:
        AdapterRegistry._load_module_factory("fake_module:factory")

    assert "Failed importing adapter module 'fake_module'" in str(error.value)


def test_adapter_module_override_requires_string_mapping_values() -> None:
    assert (
        adapter_module_override(
            {"coverage_importers": {"json": " custom.module:factory "}},
            category="coverage_importers",
            adapter_id="json",
        )
        == "custom.module:factory"
    )
    assert adapter_module_override(None, category="coverage_importers", adapter_id="json") is None
    assert (
        adapter_module_override(
            {"coverage_importers": []}, category="coverage_importers", adapter_id="json"
        )
        is None
    )
    assert (
        adapter_module_override(
            {"coverage_importers": {"json": 3}},
            category="coverage_importers",
            adapter_id="json",
        )
        is None
    )


def test_instantiate_adapter_retries_positional_factory_after_type_error() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def factory(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise TypeError("retry with positional config")
        return "ok"

    assert instantiate_adapter(factory, "cfg") == "ok"
    assert calls == [(("cfg",), {}), (("cfg",), {})]
