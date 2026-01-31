from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


class TestFixtures:
    def __init__(self, cache_dir: str = ".gloggur-cache") -> None:
        self.cache_dir = cache_dir
        self._temp_dirs: List[Path] = []

    def create_temp_repo(self, files: Dict[str, str]) -> Path:
        repo_dir = Path(tempfile.mkdtemp(prefix="gloggur-test-"))
        for relative_path, content in files.items():
            target = repo_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf8")
        self._temp_dirs.append(repo_dir)
        return repo_dir

    @staticmethod
    def create_sample_python_file() -> str:
        return (
            "\n".join(
                [
                    "\"\"\"Sample module for validation tests.\"\"\"",
                    "",
                    "class Greeter:",
                    "    \"\"\"Simple greeting class.\"\"\"",
                    "    def __init__(self, name: str) -> None:",
                    "        self.name = name",
                    "",
                    "    def greet(self) -> str:",
                    "        return f'Hello, {self.name}!'",
                    "",
                    "def add(a: int, b: int) -> int:",
                    "    \"\"\"Add two integers.\"\"\"",
                    "    return a + b",
                ]
            )
            + "\n"
        )

    def cleanup_cache(self) -> None:
        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def backup_cache(self) -> Path:
        backup_dir = Path(tempfile.mkdtemp(prefix="gloggur-cache-backup-"))
        cache_path = Path(self.cache_dir)
        if cache_path.exists():
            shutil.rmtree(backup_dir)
            shutil.copytree(cache_path, backup_dir)
        return backup_dir

    def restore_cache(self, backup_path: Path) -> None:
        cache_path = Path(self.cache_dir)
        if cache_path.exists():
            shutil.rmtree(cache_path)
        shutil.copytree(backup_path, cache_path)

    def cleanup_temp_repos(self) -> None:
        for path in self._temp_dirs:
            if path.exists():
                shutil.rmtree(path)
        self._temp_dirs.clear()
