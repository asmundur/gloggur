from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from gloggur.models import Symbol


@dataclass
class DocstringReport:
    symbol_id: str
    warnings: List[str]


def validate_docstrings(symbols: List[Symbol]) -> List[DocstringReport]:
    reports: List[DocstringReport] = []
    for symbol in symbols:
        warnings = _validate_symbol(symbol)
        if warnings:
            reports.append(DocstringReport(symbol_id=symbol.id, warnings=warnings))
    return reports


def _validate_symbol(symbol: Symbol) -> List[str]:
    warnings: List[str] = []
    if symbol.kind not in {"function", "class", "interface"}:
        return warnings
    if symbol.name.startswith("_"):
        return warnings
    if not symbol.docstring:
        warnings.append("Missing docstring")
        return warnings
    style = _detect_style(symbol.docstring)
    if not style:
        warnings.append("Unknown docstring style")
    if symbol.signature:
        signature_params = _extract_signature_params(symbol.signature)
        doc_params = _extract_docstring_params(symbol.docstring)
        missing = [param for param in signature_params if param not in doc_params]
        if missing:
            warnings.append(f"Missing docstring params: {', '.join(missing)}")
    return warnings


def _detect_style(docstring: str) -> str | None:
    if "Args:" in docstring or "Returns:" in docstring:
        return "google"
    if re.search(r"Parameters\n[-=]+", docstring):
        return "numpy"
    return None


def _extract_signature_params(signature: str) -> List[str]:
    match = re.search(r"\((.*?)\)", signature)
    if not match:
        return []
    params = []
    for raw in match.group(1).split(","):
        name = raw.strip().split(":")[0].split("=")[0].strip()
        if name and name != "self":
            params.append(name)
    return params


def _extract_docstring_params(docstring: str) -> List[str]:
    params = []
    for line in docstring.splitlines():
        line = line.strip()
        if line.startswith(":param"):
            parts = line.split()
            if len(parts) >= 2:
                params.append(parts[1].rstrip(":"))
        if re.match(r"^\w+\s*\(.*\):", line):
            params.append(line.split("(")[0].strip())
        if re.match(r"^\w+\s*:\s", line):
            params.append(line.split(":")[0].strip())
    return list(dict.fromkeys(params))
