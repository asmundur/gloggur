from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_TOP_K = 5
DEFAULT_EVIDENCE_MIN_CONFIDENCE = 0.0
DEFAULT_EVIDENCE_MIN_ITEMS = 1
DEFAULT_MIN_PASS_RATE = 0.8
MAX_RETRY_TOP_K = 64


@dataclass(frozen=True)
class EvalCase:
    """One deterministic eval case for the reference agent loop."""

    case_id: str
    query: str
    expected_symbol: str


@dataclass
class LoopResult:
    """Structured final output for one reference agent loop run."""

    ok: bool
    status: str
    attempts_used: int
    retry_performed: bool
    final_query: str
    final_top_k: int
    logs: List[Dict[str, object]]
    result_count: int
    top_symbol: Optional[str]
    top_symbol_id: Optional[str]
    validation: Optional[Dict[str, object]]
    failure: Optional[Dict[str, object]] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "status": self.status,
            "attempts_used": self.attempts_used,
            "retry_performed": self.retry_performed,
            "final_query": self.final_query,
            "final_top_k": self.final_top_k,
            "logs": self.logs,
            "result_count": self.result_count,
            "top_symbol": self.top_symbol,
            "top_symbol_id": self.top_symbol_id,
            "validation": self.validation,
            "failure": self.failure,
        }


def _default_eval_cases() -> List[EvalCase]:
    """Return deterministic built-in eval suite with at least ten representative cases."""
    return [
        EvalCase("case_01_add", "add numbers token", "add_numbers"),
        EvalCase("case_02_subtract", "subtract numbers token", "subtract_numbers"),
        EvalCase("case_03_multiply", "multiply numbers token", "multiply_numbers"),
        EvalCase("case_04_divide", "divide numbers token", "divide_numbers"),
        EvalCase("case_05_capitalize", "capitalize words token", "capitalize_words"),
        EvalCase("case_06_slugify", "slugify words token", "slugify_words"),
        EvalCase("case_07_count_vowels", "count vowels token", "count_vowels"),
        EvalCase("case_08_reverse", "reverse words token", "reverse_words"),
        EvalCase("case_09_parse_int", "parse int token", "parse_int_safe"),
        EvalCase("case_10_format_currency", "format currency token", "format_currency"),
    ]


def _build_fixture_repo(repo: Path) -> None:
    """Create deterministic fixture repo for eval runs."""
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "math_ops.py").write_text(
        "def add_numbers(left: int, right: int) -> int:\n"
        '    """add numbers token"""\n'
        "    return left + right\n\n"
        "def subtract_numbers(left: int, right: int) -> int:\n"
        '    """subtract numbers token"""\n'
        "    return left - right\n\n"
        "def multiply_numbers(left: int, right: int) -> int:\n"
        '    """multiply numbers token"""\n'
        "    return left * right\n\n"
        "def divide_numbers(left: int, right: int) -> float:\n"
        '    """divide numbers token"""\n'
        "    if right == 0:\n"
        '        raise ValueError("division by zero")\n'
        "    return left / right\n",
        encoding="utf8",
    )
    (repo / "string_ops.py").write_text(
        "def capitalize_words(text: str) -> str:\n"
        '    """capitalize words token"""\n'
        "    return \" \".join(part.capitalize() for part in text.split())\n\n"
        "def slugify_words(text: str) -> str:\n"
        '    """slugify words token"""\n'
        "    return \"-\".join(part.lower() for part in text.split())\n\n"
        "def count_vowels(text: str) -> int:\n"
        '    """count vowels token"""\n'
        "    return sum(1 for char in text.lower() if char in \"aeiou\")\n\n"
        "def reverse_words(text: str) -> str:\n"
        '    """reverse words token"""\n'
        "    return \" \".join(reversed(text.split()))\n\n"
        "def parse_int_safe(raw: str) -> int:\n"
        '    """parse int token"""\n'
        "    return int(raw.strip())\n\n"
        "def format_currency(amount: float) -> str:\n"
        '    """format currency token"""\n'
        "    return f\"${amount:0.2f}\"\n",
        encoding="utf8",
    )


def _truncate(value: str, limit: int = 400) -> str:
    """Truncate potentially large process output snippets."""
    if len(value) <= limit:
        return value
    return value[:limit] + "...(truncated)"


def _parse_json_payload(raw: str) -> Optional[Dict[str, object]]:
    """Parse JSON payload from command output including prefixed log lines."""
    payload = raw.strip()
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = payload.find("{")
    if start < 0:
        return None
    try:
        parsed = json.loads(payload[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _broaden_query(query: str) -> str:
    """Return deterministic one-step retry query broadening."""
    base = query.strip()
    if not base:
        return "function symbol"
    if " function" not in base:
        return f"{base} function"
    if " symbol" not in base:
        return f"{base} symbol"
    return f"{base} detail"


def _expand_top_k(current_top_k: int) -> int:
    """Expand top-k deterministically for bounded retry attempts."""
    if current_top_k < 1:
        raise ValueError("current_top_k must be >= 1")
    expanded = max(current_top_k + 1, current_top_k * 2)
    if expanded > MAX_RETRY_TOP_K:
        expanded = MAX_RETRY_TOP_K
    return expanded


def execute_reference_loop(
    *,
    query: str,
    top_k: int,
    max_retries: int,
    timeout_seconds: float,
    search_json: Callable[[str, int], Dict[str, object]],
) -> LoopResult:
    """Execute decide/act/validate/stop loop with bounded retry and fail-closed semantics."""
    if not query.strip():
        return LoopResult(
            ok=False,
            status="failed",
            attempts_used=0,
            retry_performed=False,
            final_query=query,
            final_top_k=top_k,
            logs=[{"step": "stop", "attempt": 0, "outcome": "failed", "reason_code": "agent_query_invalid"}],
            result_count=0,
            top_symbol=None,
            top_symbol_id=None,
            validation=None,
            failure={
                "code": "agent_query_invalid",
                "detail": "Query must be non-empty.",
                "remediation": "Pass a non-empty --query value.",
            },
        )
    if top_k < 1:
        return LoopResult(
            ok=False,
            status="failed",
            attempts_used=0,
            retry_performed=False,
            final_query=query,
            final_top_k=top_k,
            logs=[{"step": "stop", "attempt": 0, "outcome": "failed", "reason_code": "agent_top_k_invalid"}],
            result_count=0,
            top_symbol=None,
            top_symbol_id=None,
            validation=None,
            failure={
                "code": "agent_top_k_invalid",
                "detail": "--top-k must be >= 1.",
                "remediation": "Set --top-k to a positive integer.",
            },
        )

    started = time.perf_counter()
    attempt = 0
    current_query = query
    current_top_k = top_k
    logs: List[Dict[str, object]] = []
    retry_performed = False

    while True:
        elapsed = time.perf_counter() - started
        if elapsed > timeout_seconds:
            logs.append(
                {
                    "step": "stop",
                    "attempt": attempt,
                    "outcome": "failed",
                    "reason_code": "agent_loop_timeout",
                }
            )
            return LoopResult(
                ok=False,
                status="failed",
                attempts_used=attempt,
                retry_performed=retry_performed,
                final_query=current_query,
                final_top_k=current_top_k,
                logs=logs,
                result_count=0,
                top_symbol=None,
                top_symbol_id=None,
                validation=None,
                failure={
                    "code": "agent_loop_timeout",
                    "detail": f"Loop exceeded timeout budget ({timeout_seconds}s).",
                    "remediation": "Increase --timeout-seconds or reduce retries/top-k.",
                },
            )

        attempt += 1
        logs.append(
            {
                "step": "decide",
                "attempt": attempt,
                "query": current_query,
                "top_k": current_top_k,
                "decision": "search",
            }
        )
        try:
            payload = search_json(current_query, current_top_k)
        except subprocess.TimeoutExpired as exc:
            logs.append({"step": "stop", "attempt": attempt, "outcome": "failed", "reason_code": "agent_search_timeout"})
            return LoopResult(
                ok=False,
                status="failed",
                attempts_used=attempt,
                retry_performed=retry_performed,
                final_query=current_query,
                final_top_k=current_top_k,
                logs=logs,
                result_count=0,
                top_symbol=None,
                top_symbol_id=None,
                validation=None,
                failure={
                    "code": "agent_search_timeout",
                    "detail": f"Search command timed out: {exc}",
                    "remediation": "Increase timeout budget or reduce query complexity.",
                },
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logs.append({"step": "stop", "attempt": attempt, "outcome": "failed", "reason_code": "agent_search_failed"})
            return LoopResult(
                ok=False,
                status="failed",
                attempts_used=attempt,
                retry_performed=retry_performed,
                final_query=current_query,
                final_top_k=current_top_k,
                logs=logs,
                result_count=0,
                top_symbol=None,
                top_symbol_id=None,
                validation=None,
                failure={
                    "code": "agent_search_failed",
                    "detail": f"Search command failed unexpectedly: {type(exc).__name__}: {exc}",
                    "remediation": "Inspect search command stderr/stdout and fix command/runtime errors.",
                },
            )

        results = payload.get("results")
        validation = payload.get("validation")
        if not isinstance(results, list) or not isinstance(validation, dict):
            logs.append(
                {
                    "step": "stop",
                    "attempt": attempt,
                    "outcome": "failed",
                    "reason_code": "agent_search_payload_invalid",
                }
            )
            return LoopResult(
                ok=False,
                status="failed",
                attempts_used=attempt,
                retry_performed=retry_performed,
                final_query=current_query,
                final_top_k=current_top_k,
                logs=logs,
                result_count=0,
                top_symbol=None,
                top_symbol_id=None,
                validation=None,
                failure={
                    "code": "agent_search_payload_invalid",
                    "detail": "Search payload missing required results/validation structures.",
                    "remediation": "Run search with --with-evidence-trace --validate-grounding and verify JSON schema.",
                },
            )

        logs.append(
            {
                "step": "act",
                "attempt": attempt,
                "query": current_query,
                "top_k": current_top_k,
                "result_count": len(results),
            }
        )
        logs.append(
            {
                "step": "validate",
                "attempt": attempt,
                "passed": bool(validation.get("passed")),
                "reason_code": validation.get("reason_code"),
            }
        )

        top_symbol = None
        top_symbol_id = None
        if results and isinstance(results[0], dict):
            first = results[0]
            symbol_raw = first.get("symbol")
            symbol_id_raw = first.get("symbol_id")
            top_symbol = str(symbol_raw) if isinstance(symbol_raw, str) else None
            top_symbol_id = str(symbol_id_raw) if isinstance(symbol_id_raw, str) else None

        passed = bool(validation.get("passed"))
        if passed:
            logs.append({"step": "stop", "attempt": attempt, "outcome": "grounded"})
            return LoopResult(
                ok=True,
                status="grounded",
                attempts_used=attempt,
                retry_performed=retry_performed,
                final_query=current_query,
                final_top_k=current_top_k,
                logs=logs,
                result_count=len(results),
                top_symbol=top_symbol,
                top_symbol_id=top_symbol_id,
                validation=validation,
            )

        if attempt <= max_retries:
            retry_performed = True
            next_query = _broaden_query(current_query)
            next_top_k = _expand_top_k(current_top_k)
            logs.append(
                {
                    "step": "stop",
                    "attempt": attempt,
                    "outcome": "retry",
                    "next_query": next_query,
                    "next_top_k": next_top_k,
                }
            )
            current_query = next_query
            current_top_k = next_top_k
            continue

        logs.append({"step": "stop", "attempt": attempt, "outcome": "ungrounded"})
        return LoopResult(
            ok=False,
            status="ungrounded",
            attempts_used=attempt,
            retry_performed=retry_performed,
            final_query=current_query,
            final_top_k=current_top_k,
            logs=logs,
            result_count=len(results),
            top_symbol=top_symbol,
            top_symbol_id=top_symbol_id,
            validation=validation,
            failure={
                "code": "agent_grounding_failed",
                "detail": f"Grounding failed after {attempt} attempt(s): {validation.get('reason_code')}",
                "remediation": "Broaden query scope, increase top-k, or lower evidence thresholds with explicit policy.",
            },
        )


def _build_eval_summary(case_results: List[Dict[str, object]], *, min_pass_rate: float) -> Dict[str, object]:
    """Build deterministic eval summary and threshold decision."""
    total = len(case_results)
    passed = sum(1 for case in case_results if bool(case.get("passed")))
    failed = total - passed
    pass_rate = 0.0 if total == 0 else (passed / total)
    ok = pass_rate >= min_pass_rate
    return {
        "ok": ok,
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 4),
        "required_pass_rate": min_pass_rate,
    }


def _render_markdown(payload: Dict[str, object]) -> str:
    """Render compact markdown for human-friendly CI logs."""
    lines: List[str] = []
    lines.append(f"# Reference Agent {str(payload.get('mode', 'run')).title()} Report")
    lines.append("")
    lines.append(f"- ok: `{payload.get('ok')}`")
    if "summary" in payload:
        summary = payload["summary"]
        if isinstance(summary, dict):
            lines.append(f"- total_cases: `{summary.get('total_cases')}`")
            lines.append(f"- passed: `{summary.get('passed')}`")
            lines.append(f"- failed: `{summary.get('failed')}`")
            lines.append(f"- pass_rate: `{summary.get('pass_rate')}`")
            lines.append(f"- required_pass_rate: `{summary.get('required_pass_rate')}`")
    if "result" in payload:
        result = payload["result"]
        if isinstance(result, dict):
            lines.append(f"- status: `{result.get('status')}`")
            lines.append(f"- attempts_used: `{result.get('attempts_used')}`")
            lines.append(f"- final_query: `{result.get('final_query')}`")
            lines.append(f"- top_symbol: `{result.get('top_symbol')}`")
    return "\n".join(lines).rstrip() + "\n"


class ReferenceAgentHarness:
    """Minimal reference loop + eval harness with deterministic failure contracts."""

    def __init__(
        self,
        *,
        repo: Optional[Path],
        keep_artifacts: bool,
        timeout_seconds: float,
        max_retries: int,
        top_k: int,
        evidence_min_confidence: float,
        evidence_min_items: int,
        min_pass_rate: float,
    ) -> None:
        self._external_repo = repo.resolve() if repo is not None else None
        self._keep_artifacts = keep_artifacts
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._top_k = top_k
        self._evidence_min_confidence = evidence_min_confidence
        self._evidence_min_items = evidence_min_items
        self._min_pass_rate = min_pass_rate
        self._workspace_dir: Optional[Path] = None
        self._repo_dir: Optional[Path] = None
        self._cache_dir: Optional[Path] = None

    @property
    def repo_dir(self) -> Path:
        if self._repo_dir is None:
            raise RuntimeError("repo dir not initialized")
        return self._repo_dir

    @property
    def cache_dir(self) -> Path:
        if self._cache_dir is None:
            raise RuntimeError("cache dir not initialized")
        return self._cache_dir

    def _setup(self) -> None:
        if self._external_repo is not None:
            if not self._external_repo.exists():
                raise RuntimeError(f"provided repo does not exist: {self._external_repo}")
            self._repo_dir = self._external_repo
            self._workspace_dir = Path(tempfile.mkdtemp(prefix="gloggur-agent-runtime-"))
        else:
            self._workspace_dir = Path(tempfile.mkdtemp(prefix="gloggur-agent-eval-"))
            self._repo_dir = self._workspace_dir / "repo"
            _build_fixture_repo(self._repo_dir)
        self._cache_dir = self._workspace_dir / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup(self) -> None:
        if self._workspace_dir is None:
            return
        if self._keep_artifacts:
            return
        shutil.rmtree(self._workspace_dir, ignore_errors=True)

    def _env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["GLOGGUR_CACHE_DIR"] = str(self.cache_dir)
        env["GLOGGUR_EMBEDDING_PROVIDER"] = "test"
        return env

    def _run_cli_json(
        self,
        args: List[str],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, object]:
        timeout = self._timeout_seconds if timeout_seconds is None else timeout_seconds
        command = [sys.executable, "-m", "gloggur.cli.main", *args]
        completed = subprocess.run(
            command,
            cwd=str(self.repo_dir),
            env=self._env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Command exited non-zero ({completed.returncode}): {' '.join(command)} | "
                f"stdout={_truncate(completed.stdout.strip())} stderr={_truncate(completed.stderr.strip())}"
            )
        payload = _parse_json_payload(completed.stdout) or _parse_json_payload(completed.stderr)
        if payload is None:
            raise RuntimeError(
                f"Command produced invalid JSON: {' '.join(command)} | "
                f"stdout={_truncate(completed.stdout.strip())} stderr={_truncate(completed.stderr.strip())}"
            )
        return payload

    def _ensure_index(self) -> None:
        payload = self._run_cli_json(["index", str(self.repo_dir), "--json"])
        indexed_symbols_raw = payload.get("indexed_symbols", 0)
        try:
            indexed_symbols = int(indexed_symbols_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"index payload has invalid indexed_symbols: {indexed_symbols_raw!r}") from exc
        if indexed_symbols <= 0:
            raise RuntimeError("index produced zero symbols; reference loop cannot run grounded retrieval")

    def _search_json(self, query: str, top_k: int) -> Dict[str, object]:
        return self._run_cli_json(
            [
                "search",
                query,
                "--json",
                "--top-k",
                str(top_k),
                "--disable-bounded-requery",
                "--with-evidence-trace",
                "--validate-grounding",
                "--evidence-min-confidence",
                str(self._evidence_min_confidence),
                "--evidence-min-items",
                str(self._evidence_min_items),
            ]
        )

    def run_single(self, query: str) -> Dict[str, object]:
        """Run one reference loop for an explicit query."""
        self._setup()
        try:
            self._ensure_index()
            result = execute_reference_loop(
                query=query,
                top_k=self._top_k,
                max_retries=self._max_retries,
                timeout_seconds=self._timeout_seconds,
                search_json=self._search_json,
            )
            payload = {
                "mode": "run",
                "ok": result.ok,
                "result": result.as_dict(),
            }
            if not result.ok and result.failure:
                payload["failure"] = result.failure
            return payload
        finally:
            self._cleanup()

    def run_eval(self) -> Dict[str, object]:
        """Run deterministic eval suite against fixture repo and fail below threshold."""
        self._setup()
        try:
            self._ensure_index()
            cases = _default_eval_cases()
            case_results: List[Dict[str, object]] = []
            for case in cases:
                loop = execute_reference_loop(
                    query=case.query,
                    top_k=self._top_k,
                    max_retries=self._max_retries,
                    timeout_seconds=self._timeout_seconds,
                    search_json=self._search_json,
                )
                passed = loop.ok and loop.top_symbol == case.expected_symbol
                case_results.append(
                    {
                        "case_id": case.case_id,
                        "query": case.query,
                        "expected_symbol": case.expected_symbol,
                        "passed": passed,
                        "loop_status": loop.status,
                        "attempts_used": loop.attempts_used,
                        "top_symbol": loop.top_symbol,
                        "failure": loop.failure,
                    }
                )

            summary = _build_eval_summary(case_results, min_pass_rate=self._min_pass_rate)
            payload = {
                "mode": "eval",
                "ok": bool(summary.get("ok")),
                "summary": summary,
                "cases": case_results,
            }
            if not payload["ok"]:
                payload["failure"] = {
                    "code": "agent_eval_threshold_failed",
                    "detail": (
                        f"Pass rate {summary['pass_rate']} below required {summary['required_pass_rate']}"
                    ),
                    "remediation": "Inspect failing case outputs and adjust retrieval/validation configuration.",
                }
            return payload
        finally:
            self._cleanup()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal reference agent loop and eval harness.")
    parser.add_argument("--mode", choices=("run", "eval"), default="eval")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--evidence-min-confidence",
        type=float,
        default=DEFAULT_EVIDENCE_MIN_CONFIDENCE,
    )
    parser.add_argument("--evidence-min-items", type=int, default=DEFAULT_EVIDENCE_MIN_ITEMS)
    parser.add_argument("--min-pass-rate", type=float, default=DEFAULT_MIN_PASS_RATE)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args(argv)

    if args.mode == "run" and (not isinstance(args.query, str) or not args.query.strip()):
        parser.error("--query is required when --mode run")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be > 0")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.evidence_min_confidence < 0.0 or args.evidence_min_confidence > 1.0:
        parser.error("--evidence-min-confidence must be between 0.0 and 1.0")
    if args.evidence_min_items < 1:
        parser.error("--evidence-min-items must be >= 1")
    if args.min_pass_rate < 0.0 or args.min_pass_rate > 1.0:
        parser.error("--min-pass-rate must be between 0.0 and 1.0")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    harness = ReferenceAgentHarness(
        repo=args.repo,
        keep_artifacts=bool(args.keep_artifacts),
        timeout_seconds=float(args.timeout_seconds),
        max_retries=int(args.max_retries),
        top_k=int(args.top_k),
        evidence_min_confidence=float(args.evidence_min_confidence),
        evidence_min_items=int(args.evidence_min_items),
        min_pass_rate=float(args.min_pass_rate),
    )

    try:
        if args.mode == "run":
            payload = harness.run_single(str(args.query))
        else:
            payload = harness.run_eval()
    except Exception as exc:  # pragma: no cover - defensive fallback
        payload = {
            "mode": args.mode,
            "ok": False,
            "failure": {
                "code": "agent_harness_runtime_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "remediation": "Inspect runtime environment and command logs; fix setup/runtime errors before retrying.",
            },
        }

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(payload), end="")
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
