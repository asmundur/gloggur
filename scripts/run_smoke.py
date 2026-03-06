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
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"


@dataclass(frozen=True)
class StageSpec:
    """Definition for one smoke stage."""

    name: str
    failure_code: str
    remediation: str


@dataclass
class StageFailure(RuntimeError):
    """Structured stage failure with deterministic code and remediation."""

    code: str
    remediation: str
    detail: str
    context: Optional[Dict[str, object]] = None

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass
class StageResult:
    """Result for a smoke stage."""

    name: str
    status: str
    duration_ms: int
    failure_code: Optional[str] = None
    remediation: Optional[str] = None
    detail: Optional[str] = None
    context: Optional[Dict[str, object]] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "failure_code": self.failure_code,
            "remediation": self.remediation,
            "detail": self.detail,
            "context": self.context or {},
        }


STAGE_SPECS: List[StageSpec] = [
    StageSpec(
        name="index",
        failure_code="smoke_index_failed",
        remediation="Run `gloggur index <repo> --json` and fix index/runtime errors before retrying.",
    ),
    StageSpec(
        name="watch_incremental",
        failure_code="smoke_watch_incremental_failed",
        remediation="Run `gloggur watch status --json` and inspect watch state/log files, then retry.",
    ),
    StageSpec(
        name="resume_status",
        failure_code="smoke_resume_status_failed",
        remediation="Check `gloggur status --json` resume fields and reindex if resume_decision is not resume_ok.",
    ),
    StageSpec(
        name="search",
        failure_code="smoke_search_failed",
        remediation="Run `gloggur search <query> --json` and verify ContextPack v2 `summary` + `hits`.",
    ),
    StageSpec(
        name="inspect",
        failure_code="smoke_inspect_failed",
        remediation="Run `gloggur inspect <repo> --json` and verify inspect summary payload fields.",
    ),
]


def _truncate(value: str, limit: int = 400) -> str:
    """Truncate potentially large command output for deterministic payloads."""
    if len(value) <= limit:
        return value
    return value[:limit] + "...(truncated)"


def _parse_json_payload(raw: str) -> Optional[Dict[str, object]]:
    """Parse JSON payload from command output."""
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start < 0:
        return None
    try:
        parsed = json.loads(raw[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _execute_stage_plan(
    specs: List[StageSpec],
    runner: Callable[[StageSpec], StageResult],
) -> Tuple[List[StageResult], Optional[StageResult]]:
    """Execute stages in order and mark downstream stages as not_run after first failure."""
    results: List[StageResult] = []
    failed_result: Optional[StageResult] = None
    for spec in specs:
        if failed_result is not None:
            results.append(
                StageResult(
                    name=spec.name,
                    status="not_run",
                    duration_ms=0,
                    failure_code="blocked_by_prior_stage_failure",
                    remediation="Fix the previous failed stage and rerun the smoke harness.",
                    detail=f"Blocked by {failed_result.name} ({failed_result.failure_code})",
                    context={"blocked_by_stage": failed_result.name},
                )
            )
            continue
        result = runner(spec)
        results.append(result)
        if result.status == "failed":
            failed_result = result
    return results, failed_result


class SmokeHarness:
    """Run end-to-end smoke workflow with deterministic stage diagnostics."""

    def __init__(self, repo: Optional[Path], keep_artifacts: bool, timeout_seconds: float) -> None:
        self._external_repo = repo
        self._keep_artifacts = keep_artifacts
        self._timeout_seconds = timeout_seconds
        self._workspace_dir: Optional[Path] = None
        self._repo_dir: Optional[Path] = None
        self._cache_dir: Optional[Path] = None
        self._runtime_dir: Optional[Path] = None
        self._watch_config: Optional[Path] = None
        self._watch_target = "smoke_target.py"
        self._updated_phrase = "after smoke watch update phrase"
        self._watch_started = False

    @property
    def repo_dir(self) -> Path:
        if self._repo_dir is None:
            raise RuntimeError("smoke repo not initialized")
        return self._repo_dir

    @property
    def cache_dir(self) -> Path:
        if self._cache_dir is None:
            raise RuntimeError("smoke cache not initialized")
        return self._cache_dir

    @property
    def runtime_dir(self) -> Path:
        if self._runtime_dir is None:
            raise RuntimeError("smoke runtime dir not initialized")
        return self._runtime_dir

    @property
    def watch_config(self) -> Path:
        if self._watch_config is None:
            raise RuntimeError("smoke watch config not initialized")
        return self._watch_config

    def _setup(self) -> None:
        """Prepare fixture repository and isolated runtime paths."""
        if self._external_repo is not None:
            self._repo_dir = self._external_repo.resolve()
            if not self._repo_dir.exists():
                raise StageFailure(
                    code="smoke_index_failed",
                    remediation="Pass an existing --repo path or omit --repo to use fixture generation.",
                    detail=f"Provided repo path does not exist: {self._repo_dir}",
                    context={"repo": str(self._repo_dir)},
                )
            self._workspace_dir = Path(tempfile.mkdtemp(prefix="gloggur-smoke-runtime-"))
        else:
            self._workspace_dir = Path(tempfile.mkdtemp(prefix="gloggur-smoke-"))
            self._repo_dir = self._workspace_dir / "repo"
            self._repo_dir.mkdir(parents=True, exist_ok=True)
            target = self._repo_dir / self._watch_target
            target.write_text(
                "def smoke_target() -> str:\n"
                '    """before smoke watch update phrase"""\n'
                '    return "before"\n\n'
                "def smoke_helper() -> int:\n"
                "    return 1\n",
                encoding="utf8",
            )
        self._cache_dir = self._workspace_dir / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_dir = self._workspace_dir / "runtime"
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        self._watch_config = self._workspace_dir / ".gloggur.yaml"

    def _cleanup(self) -> None:
        """Remove temporary artifacts unless debugging is requested."""
        if self._workspace_dir is None:
            return
        if self._keep_artifacts:
            return
        shutil.rmtree(self._workspace_dir, ignore_errors=True)

    def _env(self) -> Dict[str, str]:
        """Build deterministic environment for smoke commands."""
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(SOURCE_ROOT)
            if not existing_pythonpath
            else f"{SOURCE_ROOT}{os.pathsep}{existing_pythonpath}"
        )
        env["GLOGGUR_CACHE_DIR"] = str(self.cache_dir)
        env["GLOGGUR_EMBEDDING_PROVIDER"] = "test"
        env["WATCHFILES_FORCE_POLLING"] = "1"
        env["WATCHFILES_POLL_DELAY_MS"] = "50"
        env["GLOGGUR_WATCH_STATE_FILE"] = str(self.runtime_dir / "watch_state.json")
        env["GLOGGUR_WATCH_PID_FILE"] = str(self.runtime_dir / "watch.pid")
        env["GLOGGUR_WATCH_LOG_FILE"] = str(self.runtime_dir / "watch.log")
        return env

    def _run_cli_json(
        self,
        args: List[str],
        *,
        spec: StageSpec,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, object]:
        """Run CLI command and return JSON payload or deterministic stage failure."""
        timeout = self._timeout_seconds if timeout_seconds is None else timeout_seconds
        cmd = [sys.executable, "-m", "gloggur.cli.main", *args]
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.repo_dir),
                env=self._env(),
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail=f"Command timed out after {timeout}s: {' '.join(cmd)}",
                context={
                    "command": cmd,
                    "stdout": _truncate((exc.stdout or "").strip()),
                    "stderr": _truncate((exc.stderr or "").strip()),
                },
            ) from exc
        if completed.returncode != 0:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail=f"Command exited non-zero ({completed.returncode}): {' '.join(cmd)}",
                context={
                    "command": cmd,
                    "exit_code": completed.returncode,
                    "stdout": _truncate(completed.stdout.strip()),
                    "stderr": _truncate(completed.stderr.strip()),
                },
            )
        payload = _parse_json_payload(completed.stdout)
        if payload is None:
            payload = _parse_json_payload(completed.stderr)
        if payload is None:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail=f"Command produced invalid/missing JSON output: {' '.join(cmd)}",
                context={
                    "command": cmd,
                    "stdout": _truncate(completed.stdout.strip()),
                    "stderr": _truncate(completed.stderr.strip()),
                },
            )
        return payload

    def _read_watch_state(self) -> Dict[str, object]:
        """Read the watch state file directly for low-latency polling."""
        state_path = self.runtime_dir / "watch_state.json"
        if not state_path.exists():
            return {}
        try:
            payload = json.loads(state_path.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_watch_target_update(self, revision: int) -> None:
        """Write an updated smoke target revision to trigger incremental watch indexing."""
        target = self.repo_dir / self._watch_target
        target.write_text(
            "def smoke_target() -> str:\n"
            f'    """{self._updated_phrase}"""\n'
            '    return "after"\n\n'
            "def smoke_helper() -> int:\n"
            f"    return {revision}\n",
            encoding="utf8",
        )

    def _run_stage(self, spec: StageSpec) -> StageResult:
        """Run one stage and return structured pass/fail result."""
        start = time.perf_counter()
        try:
            if spec.name == "index":
                context = self._stage_index(spec)
            elif spec.name == "watch_incremental":
                context = self._stage_watch_incremental(spec)
            elif spec.name == "resume_status":
                context = self._stage_resume_status(spec)
            elif spec.name == "search":
                context = self._stage_search(spec)
            elif spec.name == "inspect":
                context = self._stage_inspect(spec)
            else:
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail=f"Unknown stage requested: {spec.name}",
                )
            duration_ms = int((time.perf_counter() - start) * 1000)
            return StageResult(name=spec.name, status="passed", duration_ms=duration_ms, context=context)
        except StageFailure as failure:
            duration_ms = int((time.perf_counter() - start) * 1000)
            return StageResult(
                name=spec.name,
                status="failed",
                duration_ms=duration_ms,
                failure_code=failure.code,
                remediation=failure.remediation,
                detail=failure.detail,
                context=failure.context or {},
            )
        except Exception as exc:  # pragma: no cover - defensive
            duration_ms = int((time.perf_counter() - start) * 1000)
            return StageResult(
                name=spec.name,
                status="failed",
                duration_ms=duration_ms,
                failure_code=spec.failure_code,
                remediation=spec.remediation,
                detail=f"Unhandled exception: {type(exc).__name__}: {exc}",
                context={},
            )

    def _stage_index(self, spec: StageSpec) -> Dict[str, object]:
        """Run clean index build stage."""
        payload = self._run_cli_json(["index", str(self.repo_dir), "--json"], spec=spec)
        indexed_files = int(payload.get("indexed_files", 0))
        indexed_symbols = int(payload.get("indexed_symbols", 0))
        failed_files = int(payload.get("failed", 0))
        if indexed_files <= 0 or indexed_symbols <= 0 or failed_files > 0:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Index stage did not produce a clean indexed result.",
                context={
                    "indexed_files": indexed_files,
                    "indexed_symbols": indexed_symbols,
                    "failed": failed_files,
                    "payload": payload,
                },
            )
        return {
            "indexed_files": indexed_files,
            "indexed_symbols": indexed_symbols,
            "failed": failed_files,
        }

    def _stage_watch_incremental(self, spec: StageSpec) -> Dict[str, object]:
        """Run watch mode and verify incremental update is observed."""
        self._run_cli_json(
            ["watch", "init", str(self.repo_dir), "--config", str(self.watch_config), "--json"],
            spec=spec,
        )
        start_payload = self._run_cli_json(
            ["watch", "start", "--config", str(self.watch_config), "--daemon", "--json"],
            spec=spec,
        )
        if start_payload.get("started") is not True:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Watch daemon did not report started=true.",
                context={"payload": start_payload},
            )
        self._watch_started = True
        try:
            running_payload: Optional[Dict[str, object]] = None
            for _ in range(200):
                payload = self._read_watch_state()
                if payload.get("running") is True:
                    running_payload = payload
                    break
                time.sleep(0.1)
            if running_payload is None:
                status_payload = self._run_cli_json(
                    ["watch", "status", "--config", str(self.watch_config), "--json"],
                    spec=spec,
                    timeout_seconds=30.0,
                )
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="Watch daemon never reached running=true status.",
                    context={"status_payload": status_payload},
                )

            self._write_watch_target_update(revision=2)

            processed_payload: Optional[Dict[str, object]] = None
            last_payload: Dict[str, object] = dict(running_payload)
            for attempt in range(300):
                payload = self._read_watch_state()
                if payload:
                    last_payload = payload

                indexed_files = int(last_payload.get("indexed_files", 0))
                last_batch = last_payload.get("last_batch", {})
                batch_indexed = 0
                if isinstance(last_batch, dict):
                    batch_indexed = int(last_batch.get("indexed_files", 0))
                if indexed_files > 0 or batch_indexed > 0:
                    processed_payload = last_payload
                    break

                error_count = int(last_payload.get("error_count", 0))
                batch_failed = int(last_batch.get("failed", 0)) if isinstance(last_batch, dict) else 0
                if error_count > 0 or batch_failed > 0:
                    raise StageFailure(
                        code=spec.failure_code,
                        remediation=spec.remediation,
                        detail="Watch stage observed processing errors before incremental update completed.",
                        context={"payload": last_payload},
                    )

                if attempt in {40, 120, 220}:
                    self._write_watch_target_update(revision=2 + attempt)
                time.sleep(0.1)
            if processed_payload is None:
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="Watch stage did not process the file change in time.",
                    context={
                        "watch_state_file": str(self.runtime_dir / "watch_state.json"),
                        "last_payload": last_payload,
                    },
                )
            return {
                "pid": int(start_payload.get("pid", 0)),
                "indexed_files": int(processed_payload.get("indexed_files", 0)),
                "status": str(processed_payload.get("status", "")),
            }
        finally:
            if self._watch_started:
                self._run_cli_json(
                    ["watch", "stop", "--config", str(self.watch_config), "--json"],
                    spec=spec,
                )
                self._watch_started = False

    def _stage_resume_status(self, spec: StageSpec) -> Dict[str, object]:
        """Validate resume contract after watch/index flow."""
        payload = self._run_cli_json(["status", "--json"], spec=spec)
        if payload.get("resume_decision") != "resume_ok":
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="status --json did not report resume_decision=resume_ok.",
                context={"payload": payload},
            )
        if payload.get("needs_reindex") is True:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="status --json reported needs_reindex=true after smoke workflow.",
                context={"payload": payload},
            )
        return {
            "resume_decision": payload.get("resume_decision"),
            "needs_reindex": payload.get("needs_reindex"),
            "resume_reason_codes": payload.get("resume_reason_codes", []),
        }

    def _stage_search(self, spec: StageSpec) -> Dict[str, object]:
        """Run retrieval stage and validate result schema/coverage."""
        payload = self._run_cli_json(
            ["search", self._updated_phrase, "--json", "--top-k", "5"],
            spec=spec,
        )
        results = payload.get("hits")
        if not isinstance(results, list) or not results:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Search returned no results for smoke query.",
                context={"payload": payload},
            )
        expected_file = Path(self._watch_target).as_posix()
        if not any(
            str(item.get("path", "")).replace("\\", "/") == expected_file
            for item in results
            if isinstance(item, dict)
        ):
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Search results do not include the smoke fixture file.",
                context={"expected_file": expected_file, "payload": payload},
            )
        summary = payload.get("summary", {})
        if isinstance(summary, dict) and summary.get("needs_reindex") is True:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Search metadata reported needs_reindex=true.",
                context={"summary": summary},
            )
        return {
            "query": self._updated_phrase,
            "total_results": len(results),
            "top_file": str(results[0].get("path", "")) if isinstance(results[0], dict) else "",
        }

    def _stage_inspect(self, spec: StageSpec) -> Dict[str, object]:
        """Run inspect stage and verify grouped summary payload fields."""
        payload = self._run_cli_json(["inspect", str(self.repo_dir), "--json"], spec=spec)
        warning_summary = payload.get("warning_summary")
        if not isinstance(warning_summary, dict):
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Inspect payload missing warning_summary object.",
                context={"payload": payload},
            )
        required_summary_keys = {
            "total_warnings",
            "by_warning_type",
            "by_path_class",
            "reports_by_path_class",
            "top_files",
        }
        if not required_summary_keys.issubset(set(warning_summary.keys())):
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Inspect warning_summary missing required keys.",
                context={
                    "required_keys": sorted(required_summary_keys),
                    "actual_keys": sorted(warning_summary.keys()),
                },
            )
        return {
            "inspect_payload_schema_version": payload.get("inspect_payload_schema_version"),
            "warnings_total": int(payload.get("total", 0)),
            "reports_total": int(payload.get("reports_total", 0)),
        }

    def run(self) -> Dict[str, object]:
        """Execute smoke workflow and build machine-readable result payload."""
        start = time.perf_counter()
        setup_failure: Optional[StageFailure] = None
        try:
            self._setup()
        except StageFailure as failure:
            setup_failure = failure
        try:
            if setup_failure is None:
                stage_results, failed = _execute_stage_plan(STAGE_SPECS, self._run_stage)
            else:
                first = STAGE_SPECS[0]
                first_failure = StageResult(
                    name=first.name,
                    status="failed",
                    duration_ms=0,
                    failure_code=setup_failure.code,
                    remediation=setup_failure.remediation,
                    detail=setup_failure.detail,
                    context=setup_failure.context or {},
                )
                stage_results, failed = _execute_stage_plan(
                    STAGE_SPECS,
                    lambda spec: first_failure if spec.name == first.name else StageResult(
                        name=spec.name,
                        status="not_run",
                        duration_ms=0,
                        failure_code="blocked_by_prior_stage_failure",
                        remediation="Fix the previous failed stage and rerun the smoke harness.",
                        detail=f"Blocked by {first.name} ({first_failure.failure_code})",
                        context={"blocked_by_stage": first.name},
                    ),
                )
        finally:
            self._cleanup()
        total_duration_ms = int((time.perf_counter() - start) * 1000)
        passed_count = sum(1 for result in stage_results if result.status == "passed")
        failed_count = sum(1 for result in stage_results if result.status == "failed")
        not_run_count = sum(1 for result in stage_results if result.status == "not_run")
        payload: Dict[str, object] = {
            "ok": failed is None,
            "duration_ms": total_duration_ms,
            "summary": {
                "total_stages": len(STAGE_SPECS),
                "passed": passed_count,
                "failed": failed_count,
                "not_run": not_run_count,
            },
            "stage_order": [spec.name for spec in STAGE_SPECS],
            "stages": [result.as_dict() for result in stage_results],
        }
        if self._repo_dir is not None:
            payload["repo"] = str(self._repo_dir)
        if self._cache_dir is not None:
            payload["cache_dir"] = str(self._cache_dir)
        if failed is not None:
            payload["failure"] = {
                "stage": failed.name,
                "code": failed.failure_code,
                "remediation": failed.remediation,
                "detail": failed.detail,
            }
        return payload


def _render_markdown(payload: Dict[str, object]) -> str:
    """Render smoke payload as compact markdown."""
    summary = payload.get("summary", {})
    lines = [
        "# Smoke Harness",
        "",
        f"- ok: {payload.get('ok')}",
        f"- duration_ms: {payload.get('duration_ms')}",
        f"- stages: total={summary.get('total_stages')} passed={summary.get('passed')} failed={summary.get('failed')} not_run={summary.get('not_run')}",
        "",
        "## Stage Results",
    ]
    for item in payload.get("stages", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        status = item.get("status")
        code = item.get("failure_code")
        detail = item.get("detail")
        lines.append(f"- {name}: {status}")
        if code:
            lines.append(f"  - failure_code: {code}")
        if detail:
            lines.append(f"  - detail: {detail}")
    return "\n".join(lines)


def main() -> int:
    """Run end-to-end smoke harness for index/watch/resume/search/inspect flow."""
    parser = argparse.ArgumentParser(
        description="Run full Gloggur smoke workflow with stage-specific diagnostics."
    )
    parser.add_argument("--repo", type=Path, default=None, help="Use an existing repo path instead of fixture generation.")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", type=Path, default=None, help="Write output to a file path.")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep temporary smoke artifacts on disk for debugging.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout for each CLI stage command.",
    )
    args = parser.parse_args()

    harness = SmokeHarness(
        repo=args.repo,
        keep_artifacts=args.keep_artifacts,
        timeout_seconds=args.timeout_seconds,
    )
    payload = harness.run()
    if args.format == "json":
        output = json.dumps(payload, indent=2)
    else:
        output = _render_markdown(payload)

    if args.output:
        args.output.write_text(output + "\n", encoding="utf8")
    else:
        print(output)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
