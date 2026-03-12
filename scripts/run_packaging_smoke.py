from __future__ import annotations

import argparse
import hashlib
import json
import os
import site
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
    """Definition for one packaging smoke stage."""

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
    """Result for a packaging smoke stage."""

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


BASE_STAGE_SPECS: List[StageSpec] = [
    StageSpec(
        name="build_artifacts",
        failure_code="packaging_build_failed",
        remediation=(
            "Run `python -m build --wheel --sdist` in repo root and fix packaging/build errors before retrying."
        ),
    ),
    StageSpec(
        name="install_from_sdist",
        failure_code="packaging_install_failed",
        remediation=(
            "Run `pip install <sdist>` in an isolated venv and resolve dependency/install failures."
        ),
    ),
    StageSpec(
        name="upgrade_to_wheel",
        failure_code="packaging_upgrade_failed",
        remediation=(
            "Run `pip install --upgrade --force-reinstall <wheel>` and fix upgrade conflicts."
        ),
    ),
    StageSpec(
        name="cli_help",
        failure_code="packaging_help_failed",
        remediation=(
            "Run `gloggur --help` in the isolated install environment and fix entrypoint/import issues."
        ),
    ),
    StageSpec(
        name="cli_status",
        failure_code="packaging_status_failed",
        remediation=(
            "Run `gloggur status --json` in the isolated install environment and inspect structured errors."
        ),
    ),
]


def _stage_specs(*, skip_install_smoke: bool) -> List[StageSpec]:
    """Return enabled stage specs based on runtime options."""
    if skip_install_smoke:
        return [BASE_STAGE_SPECS[0]]
    return list(BASE_STAGE_SPECS)


def _truncate(value: str, limit: int = 400) -> str:
    """Truncate potentially large output values for stable payload shape."""
    if len(value) <= limit:
        return value
    return value[:limit] + "...(truncated)"


def _sha256_file(path: Path) -> str:
    """Return SHA256 digest for a file path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json_payload(raw: str) -> Optional[Dict[str, object]]:
    """Parse JSON payload from command output, including prefixed logs."""
    if not raw:
        return None
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


def _normalize_path(path: str) -> str:
    """Normalize a path for prefix comparisons across subprocess payloads."""
    return str(Path(path).resolve())


def _execute_stage_plan(
    specs: List[StageSpec],
    runner: Callable[[StageSpec], StageResult],
) -> Tuple[List[StageResult], Optional[StageResult]]:
    """Execute stages in order and block downstream stages after first failure."""
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
                    remediation="Fix the previous failed stage and rerun the packaging smoke harness.",
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


class PackagingSmokeHarness:
    """Run packaging validation in deterministic stages."""

    def __init__(
        self,
        *,
        repo: Path,
        keep_artifacts: bool,
        timeout_seconds: float,
        skip_install_smoke: bool,
    ) -> None:
        self._repo = repo.resolve()
        self._keep_artifacts = keep_artifacts
        self._timeout_seconds = timeout_seconds
        self._skip_install_smoke = skip_install_smoke
        self._workspace_dir: Optional[Path] = None
        self._dist_dir: Optional[Path] = None
        self._venv_dir: Optional[Path] = None
        self._runtime_dir: Optional[Path] = None
        self._wheel_path: Optional[Path] = None
        self._sdist_path: Optional[Path] = None

    @property
    def dist_dir(self) -> Path:
        if self._dist_dir is None:
            raise RuntimeError("dist_dir not initialized")
        return self._dist_dir

    @property
    def venv_dir(self) -> Path:
        if self._venv_dir is None:
            raise RuntimeError("venv_dir not initialized")
        return self._venv_dir

    @property
    def runtime_dir(self) -> Path:
        if self._runtime_dir is None:
            raise RuntimeError("runtime_dir not initialized")
        return self._runtime_dir

    @property
    def wheel_path(self) -> Path:
        if self._wheel_path is None:
            raise RuntimeError("wheel_path not initialized")
        return self._wheel_path

    @property
    def sdist_path(self) -> Path:
        if self._sdist_path is None:
            raise RuntimeError("sdist_path not initialized")
        return self._sdist_path

    def _setup(self) -> None:
        """Prepare workspace and validate repo preconditions."""
        if not self._repo.exists():
            raise StageFailure(
                code="packaging_build_failed",
                remediation="Pass an existing --repo path with pyproject.toml.",
                detail=f"Provided repo path does not exist: {self._repo}",
                context={"repo": str(self._repo)},
            )
        pyproject = self._repo / "pyproject.toml"
        if not pyproject.exists():
            raise StageFailure(
                code="packaging_build_failed",
                remediation="Ensure pyproject.toml exists before running packaging smoke.",
                detail=f"Missing pyproject.toml in repo: {self._repo}",
                context={"repo": str(self._repo)},
            )

        self._workspace_dir = Path(tempfile.mkdtemp(prefix="gloggur-packaging-smoke-"))
        self._dist_dir = self._workspace_dir / "dist"
        self._venv_dir = self._workspace_dir / "venv"
        self._runtime_dir = self._workspace_dir / "runtime"
        self._dist_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup(self) -> None:
        """Cleanup temporary workspace unless keep_artifacts is requested."""
        if self._workspace_dir is None:
            return
        if self._keep_artifacts:
            return
        shutil.rmtree(self._workspace_dir, ignore_errors=True)

    def _venv_bin(self, name: str) -> Path:
        """Resolve venv binary path in a cross-platform way."""
        scripts_dir = "Scripts" if os.name == "nt" else "bin"
        suffix = ".exe" if os.name == "nt" else ""
        return self.venv_dir / scripts_dir / f"{name}{suffix}"

    def _venv_site_packages(self) -> Path:
        """Resolve the smoke venv site-packages directory."""
        if os.name == "nt":
            return self.venv_dir / "Lib" / "site-packages"
        return (
            self.venv_dir
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )

    def _outer_site_packages(self) -> List[str]:
        """Return current interpreter site-packages paths for dependency fallback."""
        return [_normalize_path(path) for path in site.getsitepackages()]

    def _packaging_env(self, *, include_venv_site_packages: bool) -> Dict[str, str]:
        """Build environment with deterministic dependency fallback ordering."""
        pythonpath_entries: List[str] = []
        if include_venv_site_packages:
            pythonpath_entries.append(_normalize_path(str(self._venv_site_packages())))
        pythonpath_entries.extend(self._outer_site_packages())
        return {
            "PYTHONPATH": os.pathsep.join(pythonpath_entries),
            "GLOGGUR_EMBEDDING_PROVIDER": "test",
        }

    def _run_command(
        self,
        cmd: List[str],
        *,
        spec: StageSpec,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Path] = None,
        timeout_seconds: Optional[float] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run subprocess command and map failures to deterministic stage failures."""
        timeout = self._timeout_seconds if timeout_seconds is None else timeout_seconds
        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd or self._repo),
                env=run_env,
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
        return completed

    def _run_stage(self, spec: StageSpec) -> StageResult:
        """Run one stage and return deterministic pass/fail result."""
        start = time.perf_counter()
        try:
            if spec.name == "build_artifacts":
                context = self._stage_build_artifacts(spec)
            elif spec.name == "install_from_sdist":
                context = self._stage_install_from_sdist(spec)
            elif spec.name == "upgrade_to_wheel":
                context = self._stage_upgrade_to_wheel(spec)
            elif spec.name == "cli_help":
                context = self._stage_cli_help(spec)
            elif spec.name == "cli_status":
                context = self._stage_cli_status(spec)
            else:
                raise StageFailure(
                    code=spec.failure_code,
                    remediation=spec.remediation,
                    detail=f"Unknown stage requested: {spec.name}",
                )
            duration_ms = int((time.perf_counter() - start) * 1000)
            return StageResult(
                name=spec.name, status="passed", duration_ms=duration_ms, context=context
            )
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
        except Exception as exc:  # pragma: no cover - defensive fallback
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

    def _stage_build_artifacts(self, spec: StageSpec) -> Dict[str, object]:
        """Build wheel+sdist and validate resulting artifact metadata."""
        self._run_command(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--sdist",
                "--no-isolation",
                "--outdir",
                str(self.dist_dir),
            ],
            spec=spec,
        )

        wheel_candidates = sorted(self.dist_dir.glob("*.whl"))
        sdist_candidates = sorted(self.dist_dir.glob("*.tar.gz")) + sorted(
            self.dist_dir.glob("*.zip")
        )
        if not wheel_candidates or not sdist_candidates:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Build did not produce both wheel and sdist artifacts.",
                context={
                    "dist_dir": str(self.dist_dir),
                    "wheels": [item.name for item in wheel_candidates],
                    "sdists": [item.name for item in sdist_candidates],
                },
            )

        self._wheel_path = wheel_candidates[-1]
        self._sdist_path = sdist_candidates[-1]
        return {
            "dist_dir": str(self.dist_dir),
            "wheel": {
                "path": str(self.wheel_path),
                "sha256": _sha256_file(self.wheel_path),
                "bytes": self.wheel_path.stat().st_size,
            },
            "sdist": {
                "path": str(self.sdist_path),
                "sha256": _sha256_file(self.sdist_path),
                "bytes": self.sdist_path.stat().st_size,
            },
        }

    def _stage_install_from_sdist(self, spec: StageSpec) -> Dict[str, object]:
        """Create isolated venv and install from sdist."""
        self._run_command(
            [sys.executable, "-m", "venv", str(self.venv_dir)],
            spec=spec,
        )
        pip_bin = self._venv_bin("pip")
        self._run_command(
            [
                str(pip_bin),
                "install",
                "--force-reinstall",
                "--no-deps",
                "--no-build-isolation",
                str(self.sdist_path),
            ],
            spec=spec,
            env=self._packaging_env(include_venv_site_packages=False),
        )
        install_context = self._inspect_installed_package(spec)
        return {
            "venv": str(self.venv_dir),
            "installed_from": str(self.sdist_path),
            **install_context,
        }

    def _stage_upgrade_to_wheel(self, spec: StageSpec) -> Dict[str, object]:
        """Upgrade isolated environment to built wheel artifact."""
        pip_bin = self._venv_bin("pip")
        self._run_command(
            [
                str(pip_bin),
                "install",
                "--upgrade",
                "--force-reinstall",
                "--no-deps",
                str(self.wheel_path),
            ],
            spec=spec,
            env=self._packaging_env(include_venv_site_packages=True),
        )
        install_context = self._inspect_installed_package(spec)
        return {
            "upgraded_to": str(self.wheel_path),
            **install_context,
        }

    def _inspect_installed_package(self, spec: StageSpec) -> Dict[str, object]:
        """Verify the installed package resolves from the smoke venv, not the repo checkout."""
        python_bin = self._venv_bin("python")
        completed = self._run_command(
            [
                str(python_bin),
                "-c",
                (
                    "import importlib.metadata, json, pathlib, gloggur; "
                    "print(json.dumps({"
                    "'module_path': str(pathlib.Path(gloggur.__file__).resolve()), "
                    "'version': importlib.metadata.version('gloggur')"
                    "}))"
                ),
            ],
            spec=spec,
            env=self._packaging_env(include_venv_site_packages=True),
            cwd=self.runtime_dir,
        )
        payload = _parse_json_payload(completed.stdout)
        if payload is None:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Installed-package inspection did not produce a parseable JSON payload.",
                context={
                    "stdout": _truncate(completed.stdout.strip()),
                    "stderr": _truncate(completed.stderr.strip()),
                },
            )

        module_path = payload.get("module_path")
        if not isinstance(module_path, str) or not module_path:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Installed-package inspection payload is missing module_path.",
                context={"payload": payload},
            )

        normalized_module_path = _normalize_path(module_path)
        normalized_venv = _normalize_path(str(self.venv_dir))
        normalized_repo = _normalize_path(str(self._repo))
        if not normalized_module_path.startswith(normalized_venv + os.sep):
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Installed gloggur module did not resolve from the smoke venv.",
                context={
                    "module_path": normalized_module_path,
                    "venv": normalized_venv,
                },
            )
        if normalized_module_path.startswith(normalized_repo + os.sep):
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="Installed gloggur module resolved from the repo checkout instead of the smoke venv.",
                context={
                    "module_path": normalized_module_path,
                    "repo": normalized_repo,
                },
            )
        return {
            "installed_module_path": normalized_module_path,
            "installed_version": payload.get("version"),
        }

    def _stage_cli_help(self, spec: StageSpec) -> Dict[str, object]:
        """Verify installed console script resolves and renders help."""
        gloggur_bin = self._venv_bin("gloggur")
        completed = self._run_command(
            [str(gloggur_bin), "--help"],
            spec=spec,
            env=self._packaging_env(include_venv_site_packages=True),
            cwd=self.runtime_dir,
        )
        return {
            "command": str(gloggur_bin),
            "help_excerpt": _truncate(completed.stdout.strip(), limit=200),
        }

    def _stage_cli_status(self, spec: StageSpec) -> Dict[str, object]:
        """Verify installed CLI executes status JSON in isolated environment."""
        cache_dir = self.runtime_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        gloggur_bin = self._venv_bin("gloggur")
        completed = self._run_command(
            [str(gloggur_bin), "status", "--json"],
            spec=spec,
            env={
                "GLOGGUR_CACHE_DIR": str(cache_dir),
                **self._packaging_env(include_venv_site_packages=True),
            },
            cwd=self.runtime_dir,
        )
        payload = _parse_json_payload(completed.stdout)
        if payload is None:
            payload = _parse_json_payload(completed.stderr)
        if payload is None:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="status --json did not produce a parseable JSON payload.",
                context={
                    "stdout": _truncate(completed.stdout.strip()),
                    "stderr": _truncate(completed.stderr.strip()),
                },
            )
        if "resume_decision" not in payload:
            raise StageFailure(
                code=spec.failure_code,
                remediation=spec.remediation,
                detail="status --json payload is missing resume_decision.",
                context={"payload": payload},
            )
        return {
            "resume_decision": payload.get("resume_decision"),
            "needs_reindex": payload.get("needs_reindex"),
            "cache_dir": str(cache_dir),
        }

    def run(self) -> Dict[str, object]:
        """Run packaging smoke and return machine-readable payload."""
        start = time.perf_counter()
        setup_failure: Optional[StageFailure] = None
        try:
            self._setup()
        except StageFailure as failure:
            setup_failure = failure

        specs = _stage_specs(skip_install_smoke=self._skip_install_smoke)
        try:
            if setup_failure is None:
                stage_results, failed = _execute_stage_plan(specs, self._run_stage)
            else:
                first = specs[0]
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
                    specs,
                    lambda spec: (
                        first_failure
                        if spec.name == first.name
                        else StageResult(
                            name=spec.name,
                            status="not_run",
                            duration_ms=0,
                            failure_code="blocked_by_prior_stage_failure",
                            remediation="Fix the previous failed stage and rerun the packaging smoke harness.",
                            detail=f"Blocked by {first.name} ({first_failure.failure_code})",
                            context={"blocked_by_stage": first.name},
                        )
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
                "total_stages": len(specs),
                "passed": passed_count,
                "failed": failed_count,
                "not_run": not_run_count,
            },
            "stage_order": [spec.name for spec in specs],
            "stages": [result.as_dict() for result in stage_results],
            "repo": str(self._repo),
            "skip_install_smoke": self._skip_install_smoke,
        }
        if failed is not None:
            payload["failure"] = {
                "stage": failed.name,
                "code": failed.failure_code,
                "remediation": failed.remediation,
                "detail": failed.detail,
            }
        return payload


def _render_markdown(payload: Dict[str, object]) -> str:
    """Render a compact markdown view for manual inspection."""
    summary = payload.get("summary", {})
    lines = [
        "# Packaging Smoke",
        "",
        f"- ok: {payload.get('ok')}",
        f"- duration_ms: {payload.get('duration_ms')}",
        (
            "- stages: "
            f"total={summary.get('total_stages')} "
            f"passed={summary.get('passed')} "
            f"failed={summary.get('failed')} "
            f"not_run={summary.get('not_run')}"
        ),
        f"- repo: {payload.get('repo')}",
        f"- skip_install_smoke: {payload.get('skip_install_smoke')}",
        "",
        "## Stage Results",
    ]
    for stage in payload.get("stages", []):
        if not isinstance(stage, dict):
            continue
        lines.append(
            f"- `{stage.get('name')}`: {stage.get('status')} " f"({stage.get('duration_ms')}ms)"
        )
        if stage.get("failure_code"):
            lines.append(f"  - code: `{stage.get('failure_code')}`")
        if stage.get("detail"):
            lines.append(f"  - detail: {stage.get('detail')}")
    failure = payload.get("failure")
    if isinstance(failure, dict):
        lines.extend(
            [
                "",
                "## Failure",
                f"- stage: `{failure.get('stage')}`",
                f"- code: `{failure.get('code')}`",
                f"- detail: {failure.get('detail')}",
                f"- remediation: {failure.get('remediation')}",
            ]
        )
    return "\n".join(lines)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args for packaging smoke harness."""
    parser = argparse.ArgumentParser(description="Run packaging/distribution smoke harness")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path (default: current gloggur repo root)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep temporary build/install artifacts for debugging.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-command timeout in seconds.",
    )
    parser.add_argument(
        "--skip-install-smoke",
        action="store_true",
        help="Run build-only validation and skip install/upgrade/CLI stages.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for packaging smoke harness."""
    args = _parse_args(argv)
    harness = PackagingSmokeHarness(
        repo=args.repo,
        keep_artifacts=args.keep_artifacts,
        timeout_seconds=args.timeout_seconds,
        skip_install_smoke=args.skip_install_smoke,
    )
    payload = harness.run()
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_render_markdown(payload))
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
