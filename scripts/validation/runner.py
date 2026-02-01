from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CommandResult:
    command: List[str]
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    json_data: Optional[object] = None
    timed_out: bool = False


class MissingDependencyError(RuntimeError):
    def __init__(self, module: str, message: str) -> None:
        super().__init__(message)
        self.module = module


class CommandRunner:
    def __init__(
        self,
        base_cmd: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        default_timeout: float = 120.0,
    ) -> None:
        self.base_cmd = base_cmd or [sys.executable, "-m", "gloggur.cli.main"]
        self.cwd = cwd
        self.env = env
        self.default_timeout = default_timeout

    def run_command(self, cmd: List[str], capture_json: bool = True) -> CommandResult:
        full_cmd = self.base_cmd + cmd
        start = time.perf_counter()
        merged_env = os.environ.copy()
        if self.env:
            merged_env.update(self.env)
        try:
            completed = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                env=merged_env,
                timeout=self.default_timeout,
                check=False,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            json_data = None
            if capture_json:
                json_data = self._parse_json(completed.stdout)
            return CommandResult(
                command=full_cmd,
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_code=completed.returncode,
                duration_ms=duration_ms,
                json_data=json_data,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return CommandResult(
                command=full_cmd,
                stdout=stdout if isinstance(stdout, str) else stdout.decode("utf8", "ignore"),
                stderr=stderr if isinstance(stderr, str) else stderr.decode("utf8", "ignore"),
                exit_code=-1,
                duration_ms=duration_ms,
                json_data=None,
                timed_out=True,
            )

    def run_index(self, path: str, **kwargs: object) -> Dict[str, object]:
        cmd = ["index", path, "--json"]
        config_path = kwargs.get("config_path")
        embedding_provider = kwargs.get("embedding_provider")
        if config_path:
            cmd += ["--config", str(config_path)]
        if embedding_provider:
            cmd += ["--embedding-provider", str(embedding_provider)]
        result = self.run_command(cmd, capture_json=True)
        return self._require_json(result)

    def run_search(self, query: str, **kwargs: object) -> Dict[str, object]:
        cmd = ["search", query, "--json"]
        config_path = kwargs.get("config_path")
        kind = kwargs.get("kind")
        file_path = kwargs.get("file_path")
        top_k = kwargs.get("top_k")
        stream = kwargs.get("stream")
        if config_path:
            cmd += ["--config", str(config_path)]
        if kind:
            cmd += ["--kind", str(kind)]
        if file_path:
            cmd += ["--file", str(file_path)]
        if top_k is not None:
            cmd += ["--top-k", str(top_k)]
        if stream:
            cmd += ["--stream"]
        capture_json = not bool(stream)
        result = self.run_command(cmd, capture_json=capture_json)
        if stream:
            return self._parse_stream_results(result)
        return self._require_json(result)

    def run_validate(self, path: str, **kwargs: object) -> Dict[str, object]:
        cmd = ["validate", path, "--json"]
        config_path = kwargs.get("config_path")
        if config_path:
            cmd += ["--config", str(config_path)]
        result = self.run_command(cmd, capture_json=True)
        return self._require_json(result)

    def run_status(self, **kwargs: object) -> Dict[str, object]:
        cmd = ["status", "--json"]
        config_path = kwargs.get("config_path")
        if config_path:
            cmd += ["--config", str(config_path)]
        result = self.run_command(cmd, capture_json=True)
        return self._require_json(result)

    def run_clear_cache(self, **kwargs: object) -> Dict[str, object]:
        cmd = ["clear-cache", "--json"]
        config_path = kwargs.get("config_path")
        if config_path:
            cmd += ["--config", str(config_path)]
        result = self.run_command(cmd, capture_json=True)
        return self._require_json(result)

    @staticmethod
    def _missing_dependency_message(result: CommandResult) -> Optional[tuple[str, str]]:
        combined = "\n".join([result.stderr or "", result.stdout or ""]).strip()
        if not combined:
            return None
        match = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", combined)
        if not match:
            return None
        module = match.group(1)
        install_hint = "pip install -e ."
        if module in {"pytest", "black", "mypy", "ruff"}:
            install_hint = "pip install -e '.[dev]'"
        message = (
            f"Missing dependency '{module}'. Install project requirements (e.g. {install_hint}) "
            "and re-run the validation."
        )
        return module, message

    @staticmethod
    def _parse_json(stdout: str) -> Optional[object]:
        if stdout == "" or stdout.isspace():
            return None
        data = stdout.strip()
        if not data:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    @staticmethod
    def _format_command(result: CommandResult) -> str:
        return " ".join(result.command)

    @staticmethod
    def _parse_stream_results(result: CommandResult) -> Dict[str, object]:
        if result.exit_code != 0:
            missing = CommandRunner._missing_dependency_message(result)
            if missing:
                module, message = missing
                raise MissingDependencyError(module, message)
            parsed_count = 0
            parsing_errors: List[Dict[str, object]] = []
            for line_number, raw_line in enumerate(result.stdout.splitlines(), start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    json.loads(stripped)
                    parsed_count += 1
                except json.JSONDecodeError as exc:
                    parsing_errors.append(
                        {
                            "line_number": line_number,
                            "line_content": CommandRunner._truncate(stripped, 100),
                            "error": str(exc),
                        }
                    )
            parsing_error_lines = []
            for error in parsing_errors:
                parsing_error_lines.append(
                    f"  - Line {error['line_number']}: {error['line_content']} ({error['error']})"
                )
            parsing_error_details = "\n".join(parsing_error_lines)
            error_message = (
                "Command failed while parsing stream results\n"
                f"Command: {CommandRunner._format_command(result)}\n"
                f"Exit Code: {result.exit_code}\n"
                f"Stdout: {CommandRunner._truncate(result.stdout.strip(), 500)}\n"
                f"Stderr: {CommandRunner._truncate(result.stderr.strip(), 500)}\n"
                f"Parsed JSON lines before failure: {parsed_count}\n"
                f"Parsing errors: {len(parsing_errors)}"
            )
            if parsing_error_details:
                error_message = f"{error_message}\nParsing error details:\n{parsing_error_details}"
            error = RuntimeError(error_message)
            error.parsing_errors = parsing_errors
            raise error
        results: List[object] = []
        parsing_errors: List[Dict[str, object]] = []
        total_lines = 0
        for line_number, raw_line in enumerate(result.stdout.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            total_lines += 1
            try:
                results.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                parsing_errors.append(
                    {
                        "line_number": line_number,
                        "line_content": CommandRunner._truncate(stripped, 100),
                        "error": str(exc),
                    }
                )
        response: Dict[str, object] = {"stream": True, "results": results, "parsing_errors": parsing_errors}
        if total_lines > 0 and parsing_errors:
            failure_rate = len(parsing_errors) / total_lines
            if failure_rate > 0.10:
                response["warnings"] = [
                    f"High parsing error rate: {len(parsing_errors)} of {total_lines} lines failed to parse."
                ]
        return response

    @staticmethod
    def _require_json(result: CommandResult) -> Dict[str, object]:
        if result.exit_code != 0:
            missing = CommandRunner._missing_dependency_message(result)
            if missing:
                module, message = missing
                raise MissingDependencyError(module, message)
            raise RuntimeError(
                "Command failed while expecting JSON output\n"
                f"Command: {CommandRunner._format_command(result)}\n"
                f"Exit Code: {result.exit_code}\n"
                f"Duration: {result.duration_ms:.2f} ms\n"
                f"Stdout: {CommandRunner._truncate(result.stdout.strip(), 500)}\n"
                f"Stderr: {CommandRunner._truncate(result.stderr.strip(), 500)}"
            )
        if result.json_data is None or not isinstance(result.json_data, dict):
            stdout = result.stdout
            stripped = stdout.strip()
            command = CommandRunner._format_command(result)
            if not stripped or stdout.isspace():
                raise ValueError(
                    "Expected JSON object but command produced no output\n"
                    f"Command: {command}\n"
                    "Hint: Command may have produced empty output.\n"
                    f"Stdout: {CommandRunner._truncate(stripped, 200)}"
                )
            if result.json_data is None:
                raise ValueError(
                    "Expected JSON object but command produced invalid JSON output\n"
                    f"Command: {command}\n"
                    "Hint: Command may have produced non-JSON output.\n"
                    f"Stdout: {CommandRunner._truncate(stripped, 200)}"
                )
            raise ValueError(
                "Expected JSON object but received a different JSON type\n"
                f"Command: {command}\n"
                f"Stdout: {CommandRunner._truncate(stripped, 200)}"
            )
        return result.json_data
