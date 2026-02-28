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


@dataclass(frozen=True)
class StageSpec:
    """Definition for one artifact smoke stage."""

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
    """Result for an artifact smoke stage."""

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
        name="index_source",
        failure_code="artifact_smoke_index_failed",
        remediation="Run `gloggur index <repo> --json` and fix index/runtime errors before retrying.",
    ),
    StageSpec(
        name="publish_artifact",
        failure_code="artifact_smoke_publish_failed",
        remediation="Run `gloggur artifact publish --json --destination <path>` and inspect the structured failure payload.",
    ),
    StageSpec(
        name="validate_artifact",
        failure_code="artifact_smoke_validate_failed",
        remediation="Run `gloggur artifact validate --json --artifact <path>` and repair or republish the artifact.",
    ),
    StageSpec(
        name="restore_artifact",
        failure_code="artifact_smoke_restore_failed",
        remediation="Run `gloggur artifact restore --json --artifact <path> --destination <cache-dir>` and inspect restore diagnostics.",
    ),
    StageSpec(
        name="restored_status",
        failure_code="artifact_smoke_status_failed",
        remediation="Run `gloggur status --json` against the restored cache directory and inspect resume metadata.",
    ),
    StageSpec(
        name="restored_search",
        failure_code="artifact_smoke_search_failed",
        remediation="Run `gloggur search <query> --json` against the restored cache directory and verify retrieval results.",
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
                    remediation="Fix the previous failed stage and rerun the artifact smoke harness.",
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


class ArtifactSmokeHarness:
    """Run publish -> validate -> restore artifact smoke workflow."""

    def __init__(self, repo: Optional[Path], keep_artifacts: bool, timeout_seconds: float) -> None:
        self._external_repo = repo
        self._keep_artifacts = keep_artifacts
        self._timeout_seconds = timeout_seconds
        self._workspace_dir: Optional[Path] = None
        self._repo_dir: Optional[Path] = None
        self._source_cache_dir: Optional[Path] = None
        self._restored_cache_dir: Optional[Path] = None
        self._artifacts_dir: Optional[Path] = None
        self._artifact_path: Optional[Path] = None
        self._search_query = "artifact smoke retrieval phrase"

    @property
    def repo_dir(self) -> Path:
        if self._repo_dir is None:
            raise RuntimeError("artifact smoke repo not initialized")
        return self._repo_dir

    @property
    def source_cache_dir(self) -> Path:
        if self._source_cache_dir is None:
            raise RuntimeError("artifact smoke source cache not initialized")
        return self._source_cache_dir

    @property
    def restored_cache_dir(self) -> Path:
        if self._restored_cache_dir is None:
            raise RuntimeError("artifact smoke restored cache not initialized")
        return self._restored_cache_dir

    @property
    def artifacts_dir(self) -> Path:
        if self._artifacts_dir is None:
            raise RuntimeError("artifact smoke artifacts dir not initialized")
        return self._artifacts_dir

    @property
    def artifact_path(self) -> Path:
        if self._artifact_path is None:
            raise RuntimeError("artifact smoke artifact not published yet")
        return self._artifact_path

    def _setup(self) -> None:
        """Prepare fixture repository and isolated runtime paths."""
        if self._external_repo is not None:
            self._repo_dir = self._external_repo.resolve()
            if not self._repo_dir.exists():
                raise StageFailure(
                    code="artifact_smoke_index_failed",
                    remediation="Pass an existing --repo path or omit --repo to use fixture generation.",
                    detail=f"Provided repo path does not exist: {self._repo_dir}",
                    context={"repo": str(self._repo_dir)},
                )
            self._workspace_dir = Path(tempfile.mkdtemp(prefix="gloggur-artifact-smoke-runtime-"))
        else:
            self._workspace_dir = Path(tempfile.mkdtemp(prefix="gloggur-artifact-smoke-"))
            self._repo_dir = self._workspace_dir / "repo"
            self._repo_dir.mkdir(parents=True, exist_ok=True)
            (self._repo_dir / "smoke_target.py").write_text(
                "def artifact_smoke_target() -> str:\n"
                f'    """{self._search_query}"""\n'
                '    return "artifact-smoke"\n',
                encoding="utf8",
            )
        self._source_cache_dir = self._workspace_dir / "source-cache"
        self._source_cache_dir.mkdir(parents=True, exist_ok=True)
        (self._source_cache_dir / ".local_embedding_fallback").touch(exist_ok=True)
        self._restored_cache_dir = self._workspace_dir / "restored-cache"
        self._artifacts_dir = self._workspace_dir / "artifacts"
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup(self) -> None:
        """Remove temporary artifacts unless debugging is requested."""
        if self._workspace_dir is None:
            return
        if self._keep_artifacts:
            return
        shutil.rmtree(self._workspace_dir, ignore_errors=True)

    def _env(self, cache_dir: Path) -> Dict[str, str]:
        """Build deterministic environment for artifact smoke commands."""
        env = dict(os.environ)
        env["GLOGGUR_CACHE_DIR"] = str(cache_dir)
        env["GLOGGUR_LOCAL_FALLBACK"] = "1"
        return env

    def _run_cli_json(
        self,
        args: List[str],
        *,
        cache_dir: Path,
        spec: StageSpec,
    ) -> Dict[str, object]:
        """Run CLI command and return JSON payload or deterministic stage failure."""
        cmd = [sys.executable, "-m", "gloggur.cli.main", *args]
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.repo_dir),
                env=self._env(cache_dir),
                timeout=self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail=(
                    f"Command timed out after {self._timeout_seconds:.1f}s: {' '.join(cmd)}"
                ),
                context={"args": args, "timeout_seconds": self._timeout_seconds},
            ) from exc

        payload = _parse_json_payload(completed.stdout)
        if completed.returncode != 0:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail=f"Command exited with code {completed.returncode}: {' '.join(cmd)}",
                context={
                    "args": args,
                    "returncode": completed.returncode,
                    "stdout": _truncate(completed.stdout),
                    "stderr": _truncate(completed.stderr),
                    "payload": payload or {},
                },
            )
        if payload is None:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail=f"Command did not emit a JSON object: {' '.join(cmd)}",
                context={
                    "args": args,
                    "stdout": _truncate(completed.stdout),
                    "stderr": _truncate(completed.stderr),
                },
            )
        return payload

    def _run_stage(self, spec: StageSpec) -> StageResult:
        """Execute one stage and return deterministic result payload."""
        started = time.perf_counter()
        try:
            context = self._execute_stage(spec)
            duration_ms = int((time.perf_counter() - started) * 1000)
            return StageResult(
                name=spec.name,
                status="passed",
                duration_ms=duration_ms,
                context=context,
            )
        except StageFailure as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            return StageResult(
                name=spec.name,
                status="failed",
                duration_ms=duration_ms,
                failure_code=exc.code,
                remediation=exc.remediation,
                detail=exc.detail,
                context=exc.context,
            )

    def _execute_stage(self, spec: StageSpec) -> Dict[str, object]:
        """Dispatch one stage implementation."""
        if spec.name == "index_source":
            payload = self._run_cli_json(
                ["index", str(self.repo_dir), "--json"],
                cache_dir=self.source_cache_dir,
                spec=spec,
            )
            total_symbols = payload.get("indexed_symbols")
            if not isinstance(total_symbols, int) or total_symbols <= 0:
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="Index did not report a positive indexed_symbols count.",
                    context={"payload": payload},
                )
            return {
                "indexed_symbols": total_symbols,
                "indexed_files": payload.get("indexed_files"),
            }

        if spec.name == "publish_artifact":
            payload = self._run_cli_json(
                [
                    "artifact",
                    "publish",
                    "--json",
                    "--destination",
                    str(self.artifacts_dir),
                ],
                cache_dir=self.source_cache_dir,
                spec=spec,
            )
            artifact_path = payload.get("artifact_path")
            if not isinstance(artifact_path, str) or not Path(artifact_path).exists():
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="artifact publish did not produce a readable artifact_path.",
                    context={"payload": payload},
                )
            self._artifact_path = Path(artifact_path)
            return {
                "artifact_path": artifact_path,
                "archive_sha256": payload.get("archive_sha256"),
                "archive_bytes": payload.get("archive_bytes"),
            }

        if spec.name == "validate_artifact":
            payload = self._run_cli_json(
                [
                    "artifact",
                    "validate",
                    "--json",
                    "--artifact",
                    str(self.artifact_path),
                ],
                cache_dir=self.source_cache_dir,
                spec=spec,
            )
            if payload.get("valid") is not True:
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="artifact validate did not report valid=true.",
                    context={"payload": payload},
                )
            return {
                "artifact_path": str(self.artifact_path),
                "archive_sha256": payload.get("archive_sha256"),
                "manifest_sha256": payload.get("manifest_sha256"),
            }

        if spec.name == "restore_artifact":
            payload = self._run_cli_json(
                [
                    "artifact",
                    "restore",
                    "--json",
                    "--artifact",
                    str(self.artifact_path),
                    "--destination",
                    str(self.restored_cache_dir),
                ],
                cache_dir=self.source_cache_dir,
                spec=spec,
            )
            if payload.get("restored") is not True:
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="artifact restore did not report restored=true.",
                    context={"payload": payload},
                )
            if not (self.restored_cache_dir / "index.db").exists():
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="Restored cache directory is missing index.db.",
                    context={"payload": payload},
                )
            return {
                "destination_cache_dir": payload.get("destination_cache_dir"),
                "restored_files": payload.get("restored_files"),
                "restored_bytes": payload.get("restored_bytes"),
            }

        if spec.name == "restored_status":
            payload = self._run_cli_json(
                ["status", "--json"],
                cache_dir=self.restored_cache_dir,
                spec=spec,
            )
            total_symbols = payload.get("total_symbols")
            if not isinstance(total_symbols, int) or total_symbols <= 0:
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="Restored cache status did not report a positive total_symbols count.",
                    context={"payload": payload},
                )
            if payload.get("resume_decision") != "resume_ok":
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="Restored cache status did not report resume_decision=resume_ok.",
                    context={"payload": payload},
                )
            return {
                "total_symbols": total_symbols,
                "resume_decision": payload.get("resume_decision"),
            }

        if spec.name == "restored_search":
            payload = self._run_cli_json(
                ["search", self._search_query, "--json", "--top-k", "3"],
                cache_dir=self.restored_cache_dir,
                spec=spec,
            )
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="Restored cache search did not emit metadata payload.",
                    context={"payload": payload},
                )
            total_results = metadata.get("total_results")
            if not isinstance(total_results, int) or total_results <= 0:
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail="Restored cache search returned no results for smoke query.",
                    context={"payload": payload},
                )
            return {
                "query": self._search_query,
                "total_results": total_results,
            }

        raise RuntimeError(f"Unknown stage: {spec.name}")

    def run(self) -> Dict[str, object]:
        """Execute the artifact smoke harness and return machine-readable payload."""
        try:
            self._setup()
        except StageFailure as exc:
            failed_stage = StageResult(
                name=STAGE_SPECS[0].name,
                status="failed",
                duration_ms=0,
                failure_code=exc.code,
                remediation=exc.remediation,
                detail=exc.detail,
                context=exc.context,
            )
            blocked_results = [failed_stage]
            for spec in STAGE_SPECS[1:]:
                blocked_results.append(
                    StageResult(
                        name=spec.name,
                        status="not_run",
                        duration_ms=0,
                        failure_code="blocked_by_prior_stage_failure",
                        remediation="Fix the previous failed stage and rerun the artifact smoke harness.",
                        detail=f"Blocked by {failed_stage.name} ({failed_stage.failure_code})",
                        context={"blocked_by_stage": failed_stage.name},
                    )
                )
            return {
                "ok": False,
                "stage_order": [spec.name for spec in STAGE_SPECS],
                "stages": [result.as_dict() for result in blocked_results],
                "summary": {
                    "total_stages": len(blocked_results),
                    "passed": 0,
                    "failed": 1,
                    "not_run": len(blocked_results) - 1,
                },
                "failure": {
                    "stage": failed_stage.name,
                    "code": failed_stage.failure_code,
                    "remediation": failed_stage.remediation,
                    "detail": failed_stage.detail,
                },
            }
        try:
            results, failed_result = _execute_stage_plan(STAGE_SPECS, self._run_stage)
            payload: Dict[str, object] = {
                "ok": failed_result is None,
                "stage_order": [spec.name for spec in STAGE_SPECS],
                "stages": [result.as_dict() for result in results],
                "summary": {
                    "total_stages": len(results),
                    "passed": sum(1 for item in results if item.status == "passed"),
                    "failed": sum(1 for item in results if item.status == "failed"),
                    "not_run": sum(1 for item in results if item.status == "not_run"),
                },
            }
            if failed_result is not None:
                payload["failure"] = {
                    "stage": failed_result.name,
                    "code": failed_result.failure_code,
                    "remediation": failed_result.remediation,
                    "detail": failed_result.detail,
                }
            return payload
        finally:
            self._cleanup()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args for artifact smoke harness."""
    parser = argparse.ArgumentParser(
        description="Run artifact publish/restore smoke workflow with stage-specific diagnostics."
    )
    parser.add_argument("--repo", type=Path, default=None, help="Optional existing repository path.")
    parser.add_argument("--format", choices=("json",), default="json")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        default=False,
        help="Keep temporary artifact smoke workspace on disk for debugging.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Per-command timeout for artifact smoke stages.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for artifact smoke harness."""
    args = _parse_args(argv)
    harness = ArtifactSmokeHarness(
        repo=args.repo,
        keep_artifacts=args.keep_artifacts,
        timeout_seconds=args.timeout_seconds,
    )
    payload = harness.run()
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
