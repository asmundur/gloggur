from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ValidationResult:
    ok: bool
    message: str
    details: Optional[Dict[str, object]] = None

    @staticmethod
    def success(message: str = "ok", details: Optional[Dict[str, object]] = None) -> "ValidationResult":
        return ValidationResult(ok=True, message=message, details=details)

    @staticmethod
    def failure(message: str, details: Optional[Dict[str, object]] = None) -> "ValidationResult":
        return ValidationResult(ok=False, message=message, details=details)


SchemaType = Union[type, Tuple[type, ...], Dict[str, Any], List[Any]]


class Validators:
    @staticmethod
    def validate_index_output(output: Dict[str, object]) -> ValidationResult:
        schema = {
            "indexed_files": int,
            "indexed_symbols": int,
            "skipped_files": int,
            "duration_ms": (int, float),
        }
        return Validators.validate_json_structure(output, schema)

    @staticmethod
    def validate_search_output(output: Dict[str, object]) -> ValidationResult:
        schema = {
            "query": str,
            "results": list,
            "metadata": {
                "total_results": int,
                "search_time_ms": int,
            },
        }
        base = Validators.validate_json_structure(output, schema)
        if not base.ok:
            return base
        results = output.get("results", [])
        return Validators.validate_similarity_scores(results)

    @staticmethod
    def validate_similarity_scores(results: List[Dict[str, object]]) -> ValidationResult:
        invalid: List[Dict[str, object]] = []
        for idx, result in enumerate(results):
            score = result.get("similarity_score")
            if not isinstance(score, (int, float)):
                invalid.append({"index": idx, "reason": "missing similarity_score"})
                continue
            if score < 0 or score > 1:
                invalid.append({"index": idx, "reason": f"out of range: {score}"})
        if invalid:
            return ValidationResult.failure("Invalid similarity scores", {"errors": invalid})
        return ValidationResult.success("Similarity scores valid")

    @staticmethod
    def validate_json_structure(data: Dict[str, object], schema: Dict[str, SchemaType]) -> ValidationResult:
        errors: List[str] = []

        def check(value: object, expected: SchemaType, path: str) -> None:
            if isinstance(expected, dict):
                if not isinstance(value, dict):
                    errors.append(f"{path} expected dict")
                    return
                for key, sub_schema in expected.items():
                    if key not in value:
                        errors.append(f"{path}.{key} missing")
                        continue
                    check(value[key], sub_schema, f"{path}.{key}")
            elif isinstance(expected, list):
                if not isinstance(value, list):
                    errors.append(f"{path} expected list")
                    return
                if len(expected) == 0:
                    return
                item_schema = expected[0]
                for idx, item in enumerate(value):
                    check(item, item_schema, f"{path}[{idx}]")
            else:
                if not isinstance(value, expected):
                    errors.append(f"{path} expected {expected}")

        for key, expected in schema.items():
            if key not in data:
                errors.append(f"{key} missing")
                continue
            check(data[key], expected, key)

        if errors:
            return ValidationResult.failure("Schema validation failed", {"errors": errors})
        return ValidationResult.success("Schema validation passed")

    @staticmethod
    def check_database_symbols(db_path: str, expected_min: int) -> ValidationResult:
        if not os.path.exists(db_path):
            return ValidationResult.failure("Database not found", {"db_path": db_path})
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()
                total = int(row[0]) if row else 0
        except sqlite3.Error as exc:
            return ValidationResult.failure("Database query failed", {"error": str(exc)})
        if total < expected_min:
            return ValidationResult.failure(
                "Symbol count below expected minimum",
                {"expected_min": expected_min, "actual": total},
            )
        return ValidationResult.success("Database symbols validated", {"actual": total})

    @staticmethod
    def check_cache_exists(cache_dir: str) -> ValidationResult:
        if not os.path.isdir(cache_dir):
            return ValidationResult.failure("Cache directory missing", {"cache_dir": cache_dir})
        db_path = os.path.join(cache_dir, "index.db")
        if not os.path.exists(db_path):
            return ValidationResult.failure("Cache database missing", {"db_path": db_path})
        return ValidationResult.success("Cache exists", {"db_path": db_path})
