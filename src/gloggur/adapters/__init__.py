from __future__ import annotations

from gloggur.adapters.contracts import (
    CoverageImporterAdapter,
    MetadataBackend,
    ParserAdapter,
    RuntimeHost,
    VectorBackend,
)
from gloggur.adapters.registry import (
    AdapterRegistry,
    AdapterResolutionError,
    AdapterValidationError,
)

__all__ = [
    "AdapterRegistry",
    "AdapterResolutionError",
    "AdapterValidationError",
    "CoverageImporterAdapter",
    "MetadataBackend",
    "ParserAdapter",
    "RuntimeHost",
    "VectorBackend",
]
