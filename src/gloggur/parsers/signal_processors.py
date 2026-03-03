from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from tree_sitter import Node

from gloggur.models import Signal


@dataclass
class SignalProcessingOutcome:
    """Outcome payload returned by one signal processor."""

    signals: list[Signal] = field(default_factory=list)
    kind_override: str | None = None
    attributes: dict[str, object] = field(default_factory=dict)


class ParserSignalProcessor:
    """Interface for parser-side signal processors."""

    def process(
        self,
        *,
        language: str,
        node: Node,
        name: str,
        kind: str,
        source: str,
    ) -> SignalProcessingOutcome:
        """Process one AST symbol node and return optional signal enrichments."""
        raise NotImplementedError


def _walk(node: Node) -> Iterable[Node]:
    """Yield depth-first traversal for one syntax subtree."""
    yield node
    for child in node.children:
        yield from _walk(child)


class PythonTestingSignalProcessor(ParserSignalProcessor):
    """Emit Python test/fixture/invariant signals from tree-sitter nodes."""

    def process(
        self,
        *,
        language: str,
        node: Node,
        name: str,
        kind: str,
        source: str,
    ) -> SignalProcessingOutcome:
        if language != "python":
            return SignalProcessingOutcome()

        outcome = SignalProcessingOutcome()
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type != "decorator":
                    continue
                text = source[child.start_byte : child.end_byte]
                if "fixture" in text:
                    outcome.kind_override = "fixture"
                    outcome.signals.append(
                        Signal(
                            type="python.fixture",
                            payload={"decorator": text.strip()},
                            source="python_testing_signal_processor",
                        )
                    )
                    break

        if kind == "function" and name.startswith("test_"):
            outcome.signals.append(
                Signal(
                    type="test.implicit_contract",
                    payload={"text": name[5:].replace("_", " ").strip()},
                    source="python_testing_signal_processor",
                )
            )

        for child in _walk(node):
            if child.type != "assert_statement":
                continue
            expression = source[child.start_byte : child.end_byte].strip()
            if expression.startswith("assert "):
                expression = expression[7:].strip()
            outcome.signals.append(
                Signal(
                    type="code.invariant",
                    payload={"expression": expression},
                    source="python_testing_signal_processor",
                )
            )
        return outcome


class CallGraphSignalProcessor(ParserSignalProcessor):
    """Emit static call graph signals from syntax nodes."""

    def process(
        self,
        *,
        language: str,
        node: Node,
        name: str,
        kind: str,
        source: str,
    ) -> SignalProcessingOutcome:
        _ = (language, name, kind)
        signals: list[Signal] = []
        seen: set[str] = set()
        for child in _walk(node):
            if child.type not in {"call", "call_expression"}:
                continue
            if child.named_child_count <= 0:
                continue
            target_node = child.named_children[0]
            target = source[target_node.start_byte : target_node.end_byte].strip()
            if not target or target in seen:
                continue
            seen.add(target)
            signals.append(
                Signal(
                    type="code.call",
                    payload={"target": target},
                    source="call_graph_signal_processor",
                )
            )
        return SignalProcessingOutcome(signals=signals)


class SerializationBoundarySignalProcessor(ParserSignalProcessor):
    """Emit serialization-boundary signals based on names and API call hints."""

    _KEYWORDS = (
        "serialize",
        "deserialize",
        "to_dict",
        "from_dict",
        "to_json",
        "from_json",
        "parse",
    )

    def process(
        self,
        *,
        language: str,
        node: Node,
        name: str,
        kind: str,
        source: str,
    ) -> SignalProcessingOutcome:
        _ = (language, kind)
        signals: list[Signal] = []
        lowered = name.lower()
        if any(keyword in lowered for keyword in self._KEYWORDS):
            signals.append(
                Signal(
                    type="boundary.serialization",
                    payload={"detector": "name_keyword"},
                    source="serialization_boundary_signal_processor",
                )
            )

        for child in _walk(node):
            if child.type not in {"call", "call_expression"}:
                continue
            if child.named_child_count <= 0:
                continue
            target_node = child.named_children[0]
            target = source[target_node.start_byte : target_node.end_byte].strip()
            if "json.dump" in target or "json.load" in target:
                signals.append(
                    Signal(
                        type="boundary.serialization",
                        payload={"detector": "json_call", "target": target},
                        source="serialization_boundary_signal_processor",
                    )
                )
                break
        return SignalProcessingOutcome(signals=signals)


def default_signal_processors() -> list[ParserSignalProcessor]:
    """Return default built-in signal processors in deterministic order."""
    return [
        PythonTestingSignalProcessor(),
        CallGraphSignalProcessor(),
        SerializationBoundarySignalProcessor(),
    ]


def project_legacy_fields(
    signals: list[Signal],
) -> tuple[list[str], list[str], bool, str | None]:
    """Project legacy symbol fields from normalized signal payloads."""
    invariants: list[str] = []
    calls: list[str] = []
    is_serialization_boundary = False
    implicit_contract: str | None = None

    for signal in signals:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        if signal.type == "code.invariant":
            expression = payload.get("expression")
            if isinstance(expression, str) and expression:
                invariants.append(expression)
        elif signal.type == "code.call":
            target = payload.get("target")
            if isinstance(target, str) and target and target not in calls:
                calls.append(target)
        elif signal.type == "boundary.serialization":
            is_serialization_boundary = True
        elif signal.type == "test.implicit_contract" and implicit_contract is None:
            text = payload.get("text")
            if isinstance(text, str) and text:
                implicit_contract = text

    return invariants, calls, is_serialization_boundary, implicit_contract
