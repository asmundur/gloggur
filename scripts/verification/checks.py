from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional as TypingOptional, Tuple, Union


@dataclass
class CheckResult:
    """Result of schema or output checks."""
    ok: bool
    message: str
    details: TypingOptional[Dict[str, object]] = None

    @staticmethod
    def success(message: str = "ok", details: TypingOptional[Dict[str, object]] = None) -> "CheckResult":
        """Create a successful check result."""
        return CheckResult(ok=True, message=message, details=details)

    @staticmethod
    def failure(message: str, details: TypingOptional[Dict[str, object]] = None) -> "CheckResult":
        """Create a failed check result."""
        return CheckResult(ok=False, message=message, details=details)


class Optional:
    """Optional wrapper for schema elements."""
    def __init__(self, schema: "SchemaType") -> None:
        """Store the wrapped schema."""
        self.schema = schema

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"Optional({self.schema!r})"


class Range:
    """Numeric range constraint for schema values."""
    def __init__(
        self,
        min_value: TypingOptional[Union[int, float]] = None,
        max_value: TypingOptional[Union[int, float]] = None,
        value_type: Union[type, Tuple[type, ...]] = (int, float),
    ) -> None:
        """Initialize range bounds and allowed value types."""
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("min_value cannot be greater than max_value")
        self.min_value = min_value
        self.max_value = max_value
        self.value_type = value_type

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"Range(min_value={self.min_value!r}, max_value={self.max_value!r}, value_type={self.value_type!r})"


SchemaType = Union[type, Tuple[type, ...], Dict[str, Any], List[Any], Optional, Range]


class Checks:
    """Static helpers for checking schema outputs."""
    logger = logging.getLogger(__name__)
    @staticmethod
    def check_index_output(output: Dict[str, object]) -> CheckResult:
        """Check index command output schema."""
        schema = {
            "indexed_files": int,
            "indexed_symbols": int,
            "skipped_files": int,
            "duration_ms": (int, float),
        }
        return Checks.check_json_structure(output, schema)

    @staticmethod
    def check_search_output(output: Dict[str, object]) -> CheckResult:
        """Check search command output schema."""
        schema = {
            "query": str,
            "results": list,
            "metadata": {
                "total_results": int,
                "search_time_ms": int,
            },
        }
        base = Checks.check_json_structure(output, schema)
        if not base.ok:
            return base
        results = output.get("results", [])
        return Checks.check_similarity_scores(results)

    @staticmethod
    def check_similarity_scores(results: List[Dict[str, object]]) -> CheckResult:
        """Check similarity score ranges in search results."""
        invalid: List[Dict[str, object]] = []
        result_schema = {"similarity_score": Range(min_value=0.0, max_value=1.0)}
        for idx, result in enumerate(results):
            check = Checks.check_json_structure(result, result_schema)
            if not check.ok:
                invalid.append({"index": idx, "reason": check.message})
        if invalid:
            Checks.logger.warning("Similarity score check failed for %d results", len(invalid))
            return CheckResult.failure("Invalid similarity scores", {"errors": invalid})
        return CheckResult.success("Similarity scores within bounds")

    @staticmethod
    def check_json_structure(data: Dict[str, object], schema: Dict[str, SchemaType]) -> CheckResult:
        """Check a nested JSON-like dict against a schema.

        Schema examples:
            {"count": int, "scores": [int]}
            {"score": (int, float)}  # Union types via tuple
            {"elapsed_ms": Range(min_value=0.0)}
            {"meta": Optional({"status": str})}
            {"score": Optional(Range(0.0, 1.0))}
            {
                "indexed_files": int,
                "indexed_symbols": int,
                "duration_ms": Range(min_value=0.0),
            }
        """
        errors: List[Dict[str, Any]] = []

        def _format_expected_type(expected_type: SchemaType) -> str:
            """Format the expected type for error messages."""
            if isinstance(expected_type, Optional):
                return f"optional {_format_expected_type(expected_type.schema)}"
            if isinstance(expected_type, Range):
                return f"range({_format_expected_type(expected_type.value_type)})"
            if isinstance(expected_type, tuple):
                return " or ".join(t.__name__ for t in expected_type)
            if isinstance(expected_type, type):
                return expected_type.__name__
            return str(expected_type)

        def _preview_value(value: object) -> str:
            """Return a short preview string for a value."""
            if value is None:
                return ""
            if isinstance(value, str):
                if len(value) > 50:
                    return f"'{value[:50]}...'"
                return f"'{value}'"
            if isinstance(value, bool):
                return str(value)
            if isinstance(value, (int, float)):
                return str(value)
            if isinstance(value, dict):
                return f"dict with {len(value)} keys"
            if isinstance(value, list):
                return f"list with {len(value)} items"
            preview = repr(value)
            if len(preview) > 50:
                return preview[:50] + "..."
            return preview

        def check(value: object, expected: SchemaType, path: str) -> None:
            """Check a value against the schema."""
            optional_expected = False
            if isinstance(expected, Optional):
                optional_expected = True
                if value is None:
                    return
                expected = expected.schema
            if value is None:
                if isinstance(expected, dict):
                    errors.append(
                        {
                            "path": path,
                            "message": "expected dict, got None (NoneType)",
                            "expected": "dict",
                            "actual": "NoneType",
                            "value_preview": "",
                        }
                    )
                    return
                if isinstance(expected, list):
                    errors.append(
                        {
                            "path": path,
                            "message": "expected list, got None (NoneType)",
                            "expected": "list",
                            "actual": "NoneType",
                            "value_preview": "",
                        }
                    )
                    return
                errors.append(
                    {
                        "path": path,
                        "message": "expected non-null value, got None",
                        "expected": "optional value" if optional_expected else "non-null value",
                        "actual": "NoneType",
                        "value_preview": "",
                    }
                )
                return
            if isinstance(expected, Range):
                if not isinstance(value, expected.value_type):
                    expected_label = _format_expected_type(expected)
                    if optional_expected:
                        expected_label = _format_expected_type(Optional(expected))
                    errors.append(
                        {
                            "path": path,
                            "message": "type mismatch",
                            "expected": expected_label,
                            "actual": type(value).__name__,
                            "value_preview": _preview_value(value),
                        }
                    )
                    return
                if expected.min_value is not None and value < expected.min_value:
                    errors.append(
                        {
                            "path": path,
                            "message": f"value {value} below minimum {expected.min_value}",
                            "expected": f">= {expected.min_value}",
                            "actual": value,
                            "value_preview": _preview_value(value),
                        }
                    )
                    return
                if expected.max_value is not None and value > expected.max_value:
                    errors.append(
                        {
                            "path": path,
                            "message": f"value {value} above maximum {expected.max_value}",
                            "expected": f"<= {expected.max_value}",
                            "actual": value,
                            "value_preview": _preview_value(value),
                        }
                    )
                return
            if isinstance(expected, dict):
                if not isinstance(value, dict):
                    expected_label = "dict"
                    if optional_expected:
                        expected_label = "optional dict"
                    errors.append(
                        {
                            "path": path,
                            "message": "type mismatch",
                            "expected": expected_label,
                            "actual": type(value).__name__,
                            "value_preview": _preview_value(value),
                        }
                    )
                    return
                for key, sub_schema in expected.items():
                    if isinstance(sub_schema, Optional) and key not in value:
                        continue
                    if key not in value:
                        errors.append(
                            {
                                "path": f"{path}.{key}",
                                "message": "missing field",
                                "expected": "present",
                                "actual": "missing",
                                "value_preview": "",
                            }
                        )
                        continue
                    check(value[key], sub_schema, f"{path}.{key}")
            elif isinstance(expected, list):
                if not isinstance(value, list):
                    expected_label = "list"
                    if optional_expected:
                        expected_label = "optional list"
                    errors.append(
                        {
                            "path": path,
                            "message": "type mismatch",
                            "expected": expected_label,
                            "actual": type(value).__name__,
                            "value_preview": _preview_value(value),
                        }
                    )
                    return
                if len(expected) == 0:
                    return
                item_schema = expected[0]
                for idx, item in enumerate(value):
                    check(item, item_schema, f"{path}[{idx}]")
            else:
                if isinstance(expected, tuple) and all(isinstance(t, type) for t in expected):
                    if not isinstance(value, expected):
                        errors.append(
                            {
                                "path": path,
                                "message": "type mismatch",
                                "expected": _format_expected_type(expected) if not optional_expected else _format_expected_type(Optional(expected)),
                                "actual": type(value).__name__,
                                "value_preview": _preview_value(value),
                            }
                        )
                    return
                if not isinstance(value, expected):
                    errors.append(
                        {
                            "path": path,
                            "message": "type mismatch",
                            "expected": _format_expected_type(expected) if not optional_expected else _format_expected_type(Optional(expected)),
                            "actual": type(value).__name__,
                            "value_preview": _preview_value(value),
                        }
                    )

        for key, expected in schema.items():
            if isinstance(expected, Optional) and key not in data:
                continue
            if key not in data:
                errors.append(
                    {
                        "path": key,
                        "message": "missing field",
                        "expected": "present",
                        "actual": "missing",
                        "value_preview": "",
                    }
                )
                continue
            check(data[key], expected, key)

        if errors:
            Checks.logger.debug("Schema check failed with %d errors", len(errors))
            categories: Dict[str, List[Dict[str, Any]]] = {}
            for error in errors:
                categories.setdefault(error["message"], []).append(error)
            summary_parts = [f"{len(items)} {name}" for name, items in categories.items()]
            summary = f"Schema check failed ({len(errors)} errors: {', '.join(summary_parts)})"
            lines = [summary]
            for name, items in categories.items():
                lines.append(f"{name.title()}:")
                for idx, item in enumerate(items, start=1):
                    preview = item.get("value_preview") or ""
                    preview_text = f" ({preview})" if preview else ""
                    if item["message"] == "missing field":
                        lines.append(f"{idx}. {item['path']}: missing field")
                    else:
                        lines.append(
                            f"{idx}. {item['path']}: expected {item['expected']}, got {item['actual']}{preview_text}"
                        )
            return CheckResult.failure("\n".join(lines), {"errors": errors})
        return CheckResult.success("Schema check passed")

    @staticmethod
    def optional(schema: SchemaType) -> Optional:
        """Wrap a schema as optional, e.g. Checks.optional(int)."""
        return Optional(schema)

    @staticmethod
    def check_database_symbols(db_path: str, expected_min: int) -> CheckResult:
        """Check the symbol table count meets a minimum."""
        if not os.path.exists(db_path):
            return CheckResult.failure("Database not found", {"db_path": db_path})
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()
                total = int(row[0]) if row else 0
        except sqlite3.Error as exc:
            return CheckResult.failure("Database query failed", {"error": str(exc)})
        if total < expected_min:
            return CheckResult.failure(
                "Symbol count below expected minimum",
                {"expected_min": expected_min, "actual": total},
            )
        return CheckResult.success("Database symbols checked", {"actual": total})

    @staticmethod
    def check_cache_exists(cache_dir: str) -> CheckResult:
        """Ensure the cache directory and database exist."""
        if not os.path.isdir(cache_dir):
            return CheckResult.failure("Cache directory missing", {"cache_dir": cache_dir})
        db_path = os.path.join(cache_dir, "index.db")
        if not os.path.exists(db_path):
            return CheckResult.failure("Cache database missing", {"db_path": db_path})
        return CheckResult.success("Cache exists", {"db_path": db_path})
