from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass
import shutil
import tempfile
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

from scripts.validation.logging_utils import log_event
from scripts.validation.reporter import Reporter, TestResult

logger = logging.getLogger(__name__)

@dataclass
class RetryConfig:
    max_attempts: int = 1  # 1 means no retry
    initial_backoff_ms: float = 1000.0
    max_backoff_ms: float = 30000.0
    backoff_multiplier: float = 2.0
    retryable_exceptions: tuple = (ConnectionError, TimeoutError)
    retryable_exit_codes: Optional[Union[Set[int], Callable[[int], bool]]] = None
    retry_on_nonzero_exit: bool = False


@dataclass
class CommandResult:
    command: List[str]
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    json_data: Optional[object] = None
    timed_out: bool = False
    timeout_used: Optional[float] = None
    retry_attempts: int = 0
    partial_output: bool = False


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
        retry_config: Optional[RetryConfig] = None,
        debug: bool = False,
    ) -> None:
        self.base_cmd = base_cmd or [sys.executable, "-m", "gloggur.cli.main"]
        self.cwd = cwd
        self.env = env
        self.default_timeout = default_timeout
        self.retry_config = retry_config or RetryConfig()
        self.debug = debug

    def with_env(self, extra_env: Dict[str, str], *, cwd: Optional[str] = None) -> "CommandRunner":
        merged_env = dict(self.env or {})
        merged_env.update(extra_env)
        return CommandRunner(
            base_cmd=list(self.base_cmd),
            cwd=cwd or self.cwd,
            env=merged_env,
            default_timeout=self.default_timeout,
            retry_config=self.retry_config,
            debug=self.debug,
        )

    @contextmanager
    def isolated_env(
        self,
        *,
        cache_dir: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        prefix: str = "gloggur-cache-",
        cwd: Optional[str] = None,
    ) -> Iterator["CommandRunner"]:
        owns_cache = cache_dir is None
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix=prefix)
        merged_env = dict(self.env or {})
        merged_env["GLOGGUR_CACHE_DIR"] = cache_dir
        if extra_env:
            merged_env.update(extra_env)
        runner = CommandRunner(
            base_cmd=list(self.base_cmd),
            cwd=cwd or self.cwd,
            env=merged_env,
            default_timeout=self.default_timeout,
            retry_config=self.retry_config,
            debug=self.debug,
        )
        try:
            yield runner
        finally:
            if owns_cache:
                shutil.rmtree(cache_dir, ignore_errors=True)

    def run_command(
        self,
        cmd: List[str],
        capture_json: bool = True,
        timeout: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        full_cmd = self.base_cmd + cmd
        merged_env = os.environ.copy()
        if self.env:
            merged_env.update(self.env)
        if extra_env:
            merged_env.update(extra_env)
        timeout_value = self.default_timeout if timeout is None else timeout
        if self._debug_enabled():
            log_event(
                logger,
                logging.DEBUG,
                "command.start",
                command=full_cmd,
                env=merged_env,
                timeout_s=timeout_value,
            )
        return self._execute_with_retry(full_cmd, merged_env, timeout_value, capture_json)

    def run_command_streaming(
        self,
        cmd: List[str],
        line_callback: Callable[[str], None],
        capture_json: bool = False,
        timeout: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        full_cmd = self.base_cmd + cmd
        merged_env = os.environ.copy()
        if self.env:
            merged_env.update(self.env)
        if extra_env:
            merged_env.update(extra_env)
        timeout_value = self.default_timeout if timeout is None else timeout
        if self._debug_enabled():
            log_event(
                logger,
                logging.DEBUG,
                "command.stream.start",
                command=full_cmd,
                env=merged_env,
                timeout_s=timeout_value,
            )
        start = time.perf_counter()
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.cwd,
            env=merged_env,
        )
        stdout, stderr, timed_out = self._stream_output(process, timeout_value, line_callback)
        duration_ms = (time.perf_counter() - start) * 1000
        json_data = None
        if capture_json and stdout:
            json_data = self._parse_json(stdout)
        if self._debug_enabled():
            log_event(
                logger,
                logging.DEBUG,
                "command.stream.finish",
                command=full_cmd,
                duration_ms=duration_ms,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
            )
        return CommandResult(
            command=full_cmd,
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode if process.returncode is not None else -1,
            duration_ms=duration_ms,
            json_data=json_data,
            timed_out=timed_out,
            timeout_used=timeout_value,
            retry_attempts=0,
            partial_output=timed_out,
        )

    def _stream_output(
        self,
        process: subprocess.Popen[str],
        timeout: float,
        line_callback: Callable[[str], None],
    ) -> Tuple[str, str, bool]:
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        lock = threading.Lock()

        def read_stdout() -> None:
            if process.stdout is None:
                return
            for line in iter(process.stdout.readline, ""):
                with lock:
                    stdout_lines.append(line)
                line_callback(line)

        def read_stderr() -> None:
            if process.stderr is None:
                return
            for line in iter(process.stderr.readline, ""):
                with lock:
                    stderr_lines.append(line)

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        if timed_out:
            stderr = f"WARNING: Command timed out after {timeout}s; partial output captured.\n{stderr}"
        return stdout, stderr, timed_out

    def _execute_with_retry(
        self,
        full_cmd: List[str],
        merged_env: Dict[str, str],
        timeout: float,
        capture_json: bool,
    ) -> CommandResult:
        def should_retry_exit_code(exit_code: int) -> bool:
            predicate = self.retry_config.retryable_exit_codes
            if predicate is None:
                return self.retry_config.retry_on_nonzero_exit and exit_code != 0
            if isinstance(predicate, set):
                return exit_code in predicate
            return bool(predicate(exit_code))

        def should_retry_exception(exc: BaseException) -> bool:
            return isinstance(exc, self.retry_config.retryable_exceptions)

        def sleep_backoff(attempt_index: int) -> None:
            backoff_delay = min(
                self.retry_config.initial_backoff_ms * (self.retry_config.backoff_multiplier**attempt_index),
                self.retry_config.max_backoff_ms,
            )
            time.sleep(backoff_delay / 1000)

        last_exc: Optional[BaseException] = None
        max_attempts = max(1, self.retry_config.max_attempts)
        for attempt in range(max_attempts):
            start = time.perf_counter()
            try:
                completed = subprocess.run(
                    full_cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.cwd,
                    env=merged_env,
                    timeout=timeout,
                    check=False,
                )
                duration_ms = (time.perf_counter() - start) * 1000
                if self._debug_enabled():
                    log_event(
                        logger,
                        logging.DEBUG,
                        "command.finish",
                        command=full_cmd,
                        duration_ms=duration_ms,
                        exit_code=completed.returncode,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                    )
                is_success = completed.returncode == 0
                should_retry = (not is_success) and should_retry_exit_code(completed.returncode)
                if should_retry and attempt + 1 < max_attempts:
                    logger.warning("Command failed (exit %s), retrying attempt %d", completed.returncode, attempt + 2)
                    sleep_backoff(attempt)
                    continue
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
                    timeout_used=timeout,
                    retry_attempts=attempt,
                    partial_output=False,
                )
            except subprocess.TimeoutExpired as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                stderr_text = stderr if isinstance(stderr, str) else stderr.decode("utf8", "ignore")
                warning = f"WARNING: Command timed out after {timeout}s; partial output captured."
                stderr_text = f"{warning}\n{stderr_text}".strip()
                if self._debug_enabled():
                    log_event(
                        logger,
                        logging.DEBUG,
                        "command.timeout",
                        command=full_cmd,
                        duration_ms=duration_ms,
                        timeout_s=timeout,
                        stdout=stdout if isinstance(stdout, str) else stdout.decode("utf8", "ignore"),
                        stderr=stderr_text,
                    )
                logger.error("Command timed out after %.2fs", timeout)
                should_retry = should_retry_exception(exc)
                if should_retry and attempt + 1 < max_attempts:
                    sleep_backoff(attempt)
                    continue
                return CommandResult(
                    command=full_cmd,
                    stdout=stdout if isinstance(stdout, str) else stdout.decode("utf8", "ignore"),
                    stderr=stderr_text,
                    exit_code=-1,
                    duration_ms=duration_ms,
                    json_data=None,
                    timed_out=True,
                    timeout_used=timeout,
                    retry_attempts=attempt,
                    partial_output=True,
                )
            except self.retry_config.retryable_exceptions as exc:
                last_exc = exc
                logger.warning("Retryable exception: %s", exc)
                if attempt + 1 >= max_attempts:
                    raise
                sleep_backoff(attempt)
        if last_exc:
            raise last_exc
        raise RuntimeError("Command execution failed without returning a result.")

    def run_index(self, path: str, timeout: Optional[float] = None, **kwargs: object) -> Dict[str, object]:
        cmd = ["index", path, "--json"]
        config_path = kwargs.get("config_path")
        embedding_provider = kwargs.get("embedding_provider")
        if config_path:
            cmd += ["--config", str(config_path)]
        if embedding_provider:
            cmd += ["--embedding-provider", str(embedding_provider)]
        result = self.run_command(cmd, capture_json=True, timeout=timeout)
        return self._require_json(result)

    def run_search(self, query: str, timeout: Optional[float] = None, **kwargs: object) -> Dict[str, object]:
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
        result = self.run_command(cmd, capture_json=capture_json, timeout=timeout)
        if stream:
            return self._parse_stream_results(result)
        return self._require_json(result)

    def run_validate(self, path: str, timeout: Optional[float] = None, **kwargs: object) -> Dict[str, object]:
        cmd = ["validate", path, "--json"]
        config_path = kwargs.get("config_path")
        if config_path:
            cmd += ["--config", str(config_path)]
        result = self.run_command(cmd, capture_json=True, timeout=timeout)
        return self._require_json(result)

    def run_status(self, timeout: Optional[float] = None, **kwargs: object) -> Dict[str, object]:
        cmd = ["status", "--json"]
        config_path = kwargs.get("config_path")
        if config_path:
            cmd += ["--config", str(config_path)]
        result = self.run_command(cmd, capture_json=True, timeout=timeout)
        return self._require_json(result)

    def run_clear_cache(self, timeout: Optional[float] = None, **kwargs: object) -> Dict[str, object]:
        cmd = ["clear-cache", "--json"]
        config_path = kwargs.get("config_path")
        if config_path:
            cmd += ["--config", str(config_path)]
        result = self.run_command(cmd, capture_json=True, timeout=timeout)
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
            timeout_info = ""
            if result.timed_out:
                timeout_info = (
                    f"Timeout: Command timed out after {result.timeout_used}s. Partial output may be available.\n"
                )
            retry_info = ""
            if result.retry_attempts > 0:
                retry_info = f"Retry Attempts: {result.retry_attempts}\n"
            partial_warning = ""
            if result.partial_output:
                partial_warning = "Warning: Output may be incomplete due to timeout\n"
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
                f"{timeout_info}"
                f"{retry_info}"
                f"{partial_warning}"
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
            timeout_message = ""
            if result.timed_out:
                timeout_message = (
                    f"Command timed out after {result.timeout_used}s\n"
                    "Hint: Partial output may be available.\n"
                )
            retry_message = ""
            if result.retry_attempts > 0:
                retry_message = f"Retry Attempts: {result.retry_attempts}\n"
            partial_warning = ""
            if result.partial_output:
                partial_warning = "Warning: Output may be incomplete due to timeout\n"
            raise RuntimeError(
                "Command failed while expecting JSON output\n"
                f"Command: {CommandRunner._format_command(result)}\n"
                f"Exit Code: {result.exit_code}\n"
                f"Duration: {result.duration_ms:.2f} ms\n"
                f"{timeout_message}"
                f"{retry_message}"
                f"{partial_warning}"
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

    def _debug_enabled(self) -> bool:
        return self.debug or bool(os.getenv("GLOGGUR_DEBUG_LOGS"))


@dataclass(frozen=True)
class TestTask:
    name: str
    run: Callable[[], TestResult]
    section: str = "General"


@dataclass(frozen=True)
class TestOutcome:
    name: str
    result: TestResult
    section: str


class TestOrchestrator:
    def __init__(self, reporter: Reporter, max_workers: Optional[int] = None) -> None:
        self.reporter = reporter
        self.max_workers = max_workers
        self._logger = logging.getLogger(__name__)

    def run(self, tasks: Iterable[TestTask]) -> List[TestOutcome]:
        outcomes: List[TestOutcome] = []
        if self.max_workers == 1:
            for task in tasks:
                outcome = self._run_task(task)
                self.reporter.add_test_result_to_section(outcome.section, outcome.name, outcome.result)
                outcomes.append(outcome)
                self._logger.info("Task completed: %s (section=%s)", outcome.name, outcome.section)
            return outcomes
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map: Dict[object, TestTask] = {}
            for task in tasks:
                ctx = copy_context()
                future = executor.submit(ctx.run, self._run_task, task)
                future_map[future] = task
            for future in as_completed(future_map):
                outcome = future.result()
                self.reporter.add_test_result_to_section(outcome.section, outcome.name, outcome.result)
                outcomes.append(outcome)
                self._logger.info("Task completed: %s (section=%s)", outcome.name, outcome.section)
        return outcomes

    @staticmethod
    def _run_task(task: TestTask) -> TestOutcome:
        try:
            result = task.run()
        except Exception as exc:
            result = TestResult(passed=False, message=f"Unhandled exception: {exc}")
        return TestOutcome(name=task.name, result=result, section=task.section)
