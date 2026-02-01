from __future__ import annotations

import ast
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from tree_sitter_language_pack import get_parser

_LANGUAGE_BY_EXTENSION: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
}

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FixtureFile:
    content: str
    language: Optional[str] = None
    validate: bool = True


@dataclass(frozen=True)
class FixtureTemplate:
    name: str
    description: str
    files: Dict[str, FixtureFile]
    tags: Tuple[str, ...] = ()


class FixtureRegistry:
    def __init__(self) -> None:
        self._templates: Dict[str, FixtureTemplate] = {}

    def register(self, template: FixtureTemplate, overwrite: bool = False) -> None:
        if not overwrite and template.name in self._templates:
            raise ValueError(f"Fixture template already registered: {template.name}")
        self._templates[template.name] = template

    def get(self, name: str) -> FixtureTemplate:
        if name not in self._templates:
            available = ", ".join(sorted(self._templates))
            raise KeyError(f"Unknown fixture template: {name}. Available: {available}")
        return self._templates[name]

    def list(self, tags: Optional[Iterable[str]] = None) -> List[FixtureTemplate]:
        if not tags:
            return list(self._templates.values())
        tag_set = set(tags)
        return [template for template in self._templates.values() if tag_set.issubset(template.tags)]

    def compose(
        self,
        name: str,
        fixtures: Sequence[Union[str, FixtureTemplate]],
        description: str,
        tags: Optional[Iterable[str]] = None,
        register: bool = False,
    ) -> FixtureTemplate:
        files: Dict[str, FixtureFile] = {}
        for fixture in fixtures:
            template = self.get(fixture) if isinstance(fixture, str) else fixture
            for path, file in template.files.items():
                if path in files:
                    raise ValueError(f"Fixture composition conflict at {path}")
                files[path] = file
        composed = FixtureTemplate(
            name=name,
            description=description,
            files=files,
            tags=tuple(tags or ()),
        )
        if register:
            self.register(composed)
        return composed


FIXTURE_REGISTRY = FixtureRegistry()


class TestFixtures:
    __test__ = False
    registry = FIXTURE_REGISTRY

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        if cache_dir is None:
            cache_dir = str(Path(tempfile.mkdtemp(prefix="gloggur-cache-")))
            self._owns_cache_dir = True
        else:
            self._owns_cache_dir = False
        self._cache_dir = cache_dir
        self._temp_dirs: List[Path] = []
        self._backup_dirs: List[Path] = []
        self._cache_dirs: List[Path] = []
        self._cache_lock = threading.Lock()
        self._cache_dir_local = threading.local()
        self._cache_dir_local.value = cache_dir

    @property
    def cache_dir(self) -> str:
        return self._get_cache_dir()

    @cache_dir.setter
    def cache_dir(self, value: str) -> None:
        with self._cache_lock:
            self._cache_dir = value
            self._owns_cache_dir = False
        self._cache_dir_local.value = value

    def _get_cache_dir(self) -> str:
        local_cache_dir = getattr(self._cache_dir_local, "value", None)
        if local_cache_dir:
            return local_cache_dir
        with self._cache_lock:
            return self._cache_dir

    def __enter__(self) -> "TestFixtures":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup_temp_repos()
        self.cleanup_backup_dirs()
        if self._owns_cache_dir:
            self._cleanup_owned_cache_dir()
        self.cleanup_cache_dirs()

    def create_temp_repo(
        self,
        files: Dict[str, Union[str, FixtureFile]],
        validate: bool = True,
        cache_dir: Optional[str] = None,
    ) -> Path:
        if cache_dir is None:
            self.new_cache_dir()
        else:
            self._cache_dir_local.value = str(cache_dir)
        normalized_files = self._normalize_files(files)
        if validate:
            self._validate_fixture_files(normalized_files)
        repo_dir = Path(tempfile.mkdtemp(prefix="gloggur-test-"))
        for relative_path, fixture_file in normalized_files.items():
            target = repo_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fixture_file.content, encoding="utf8")
        self._temp_dirs.append(repo_dir)
        logger.info("Created temp repo %s with %d files", repo_dir, len(normalized_files))
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

    @staticmethod
    def create_sample_javascript_file() -> str:
        return (
            "\n".join(
                [
                    "/** Sample module for validation tests. */",
                    "",
                    "export class Greeter {",
                    "  constructor(name) {",
                    "    this.name = name;",
                    "  }",
                    "",
                    "  greet() {",
                    "    return `Hello, ${this.name}!`;",
                    "  }",
                    "}",
                    "",
                    "export function add(a, b) {",
                    "  return a + b;",
                    "}",
                ]
            )
            + "\n"
        )

    @staticmethod
    def create_sample_typescript_file() -> str:
        return (
            "\n".join(
                [
                    "/** Sample module for validation tests. */",
                    "",
                    "export interface User {",
                    "  id: number;",
                    "  name: string;",
                    "}",
                    "",
                    "export class Greeter {",
                    "  constructor(private name: string) {}",
                    "",
                    "  greet(): string {",
                    "    return `Hello, ${this.name}!`;",
                    "  }",
                    "}",
                    "",
                    "export function add(a: number, b: number): number {",
                    "  return a + b;",
                    "}",
                ]
            )
            + "\n"
        )

    @staticmethod
    def create_sample_tsx_file() -> str:
        return (
            "\n".join(
                [
                    "import React from \"react\";",
                    "",
                    "type Props = {",
                    "  title: string;",
                    "};",
                    "",
                    "export const App: React.FC<Props> = ({ title }) => {",
                    "  return <div>{title}</div>;",
                    "};",
                ]
            )
            + "\n"
        )

    @staticmethod
    def create_sample_rust_file() -> str:
        return (
            "\n".join(
                [
                    "pub struct Greeter {",
                    "    name: String,",
                    "}",
                    "",
                    "impl Greeter {",
                    "    pub fn new(name: &str) -> Self {",
                    "        Self {",
                    "            name: name.to_string(),",
                    "        }",
                    "    }",
                    "",
                    "    pub fn greet(&self) -> String {",
                    "        format!(\"Hello, {}!\", self.name)",
                    "    }",
                    "}",
                    "",
                    "pub fn add(a: i32, b: i32) -> i32 {",
                    "    a + b",
                    "}",
                ]
            )
            + "\n"
        )

    @staticmethod
    def create_sample_go_file() -> str:
        return (
            "\n".join(
                [
                    "package main",
                    "",
                    "import \"fmt\"",
                    "",
                    "func add(a int, b int) int {",
                    "    return a + b",
                    "}",
                    "",
                    "func main() {",
                    "    fmt.Println(add(1, 2))",
                    "}",
                ]
            )
            + "\n"
        )

    @staticmethod
    def create_sample_java_file() -> str:
        return (
            "\n".join(
                [
                    "public class Main {",
                    "    public static int add(int a, int b) {",
                    "        return a + b;",
                    "    }",
                    "",
                    "    public static void main(String[] args) {",
                    "        System.out.println(add(1, 2));",
                    "    }",
                    "}",
                ]
            )
            + "\n"
        )

    def create_temp_repo_from_fixture(
        self,
        name: str,
        validate: bool = True,
        cache_dir: Optional[str] = None,
    ) -> Path:
        template = self.registry.get(name)
        return self.create_temp_repo(template.files, validate=validate, cache_dir=cache_dir)

    def create_temp_repo_from_fixtures(
        self,
        fixtures: Sequence[Union[str, FixtureTemplate]],
        validate: bool = True,
        name: str = "composed",
        description: str = "Composed fixture",
        cache_dir: Optional[str] = None,
    ) -> Path:
        template = self.registry.compose(
            name=name,
            fixtures=fixtures,
            description=description,
        )
        return self.create_temp_repo(template.files, validate=validate, cache_dir=cache_dir)

    @classmethod
    def list_fixture_templates(cls, tags: Optional[Iterable[str]] = None) -> List[FixtureTemplate]:
        return cls.registry.list(tags=tags)

    @classmethod
    def get_fixture_template(cls, name: str) -> FixtureTemplate:
        return cls.registry.get(name)

    def cleanup_cache(self) -> None:
        cache_dir = self._get_cache_dir()
        if cache_dir and os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
            logger.debug("Removed cache dir %s", cache_dir)

    def _cleanup_owned_cache_dir(self) -> None:
        with self._cache_lock:
            cache_dir = self._cache_dir
        if cache_dir and os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
            logger.debug("Removed cache dir %s", cache_dir)

    def create_cache_dir(self, prefix: str = "gloggur-cache-") -> Path:
        with self._cache_lock:
            cache_dir = Path(tempfile.mkdtemp(prefix=prefix))
            self._cache_dirs.append(cache_dir)
            logger.debug("Created cache dir %s", cache_dir)
            return cache_dir

    def new_cache_dir(self, prefix: str = "gloggur-cache-") -> Path:
        cache_dir = self.create_cache_dir(prefix=prefix)
        self._cache_dir_local.value = str(cache_dir)
        return cache_dir

    def cleanup_cache_dirs(self) -> None:
        with self._cache_lock:
            cache_dirs = list(self._cache_dirs)
            self._cache_dirs.clear()
        for path in cache_dirs:
            if path.exists():
                shutil.rmtree(path)
                logger.debug("Removed cache dir %s", path)

    def backup_cache(self) -> Optional[Path]:
        cache_path = Path(self._get_cache_dir())
        if not cache_path.exists():
            return None
        backup_dir = Path(tempfile.mkdtemp(prefix="gloggur-cache-backup-"))
        # copytree requires destination to not exist
        shutil.rmtree(backup_dir)
        shutil.copytree(cache_path, backup_dir)
        self._backup_dirs.append(backup_dir)
        return backup_dir

    def restore_cache(self, backup_path: Path) -> None:
        cache_path = Path(self._get_cache_dir())
        if cache_path.exists():
            shutil.rmtree(cache_path)
        shutil.copytree(backup_path, cache_path)

    def cleanup_temp_repos(self) -> None:
        for path in self._temp_dirs:
            if path.exists():
                shutil.rmtree(path)
                logger.debug("Removed temp repo %s", path)
        self._temp_dirs.clear()

    def cleanup_backup_dirs(self) -> None:
        for path in self._backup_dirs:
            if path.exists():
                shutil.rmtree(path)
                logger.debug("Removed cache backup dir %s", path)
        self._backup_dirs.clear()

    @staticmethod
    def _normalize_files(files: Dict[str, Union[str, FixtureFile]]) -> Dict[str, FixtureFile]:
        normalized: Dict[str, FixtureFile] = {}
        for path, content in files.items():
            if isinstance(content, FixtureFile):
                normalized[path] = content
            else:
                normalized[path] = FixtureFile(content=content)
        return normalized

    def _validate_fixture_files(self, files: Dict[str, FixtureFile]) -> None:
        errors: List[str] = []
        for path, fixture_file in files.items():
            if not fixture_file.validate:
                continue
            language = fixture_file.language or self._infer_language(path)
            if not language:
                continue
            try:
                self._validate_source(path, fixture_file.content, language)
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            logger.error("Fixture validation failed for %d files", len(errors))
            raise ValueError("Fixture validation failed:\n" + "\n".join(errors))

    @staticmethod
    def _infer_language(path: str) -> Optional[str]:
        return _LANGUAGE_BY_EXTENSION.get(Path(path).suffix.lower())

    @staticmethod
    def _validate_source(path: str, source: str, language: str) -> None:
        if language == "python":
            try:
                ast.parse(source, filename=path)
            except SyntaxError as exc:
                raise ValueError(f"{path}: Python syntax error: {exc.msg}") from exc
            return
        parser = get_parser(language)
        tree = parser.parse(bytes(source, "utf8"))
        if TestFixtures._tree_has_error(tree.root_node):
            raise ValueError(f"{path}: {language} syntax error")

    @staticmethod
    def _tree_has_error(node) -> bool:
        if getattr(node, "is_error", False) or getattr(node, "is_missing", False):
            return True
        if node.type == "ERROR":
            return True
        for child in node.children:
            if TestFixtures._tree_has_error(child):
                return True
        return False


def _register_default_fixtures(registry: FixtureRegistry) -> None:
    python_basic = FixtureTemplate(
        name="python_basic",
        description="Single-file Python fixture with class and function.",
        files={"sample.py": FixtureFile(TestFixtures.create_sample_python_file(), language="python")},
        tags=("python", "single-file"),
    )
    python_multi = FixtureTemplate(
        name="python_multi_file",
        description="Multi-file Python package fixture.",
        files={
            "src/app.py": FixtureFile(
                "\n".join(
                    [
                        "from src.utils.math import add",
                        "from src.models.user import User",
                        "",
                        "def build_message(user: User) -> str:",
                        "    return f\"Hello, {user.name}!\"",
                        "",
                        "def total(a: int, b: int) -> int:",
                        "    return add(a, b)",
                        "",
                    ]
                )
                + "\n",
                language="python",
            ),
            "src/utils/math.py": FixtureFile(
                "\n".join(
                    [
                        "def add(a: int, b: int) -> int:",
                        "    return a + b",
                        "",
                        "def multiply(a: int, b: int) -> int:",
                        "    return a * b",
                        "",
                    ]
                )
                + "\n",
                language="python",
            ),
            "src/models/user.py": FixtureFile(
                "\n".join(
                    [
                        "from dataclasses import dataclass",
                        "",
                        "@dataclass",
                        "class User:",
                        "    id: int",
                        "    name: str",
                        "",
                    ]
                )
                + "\n",
                language="python",
            ),
            "README.md": FixtureFile("# Sample Python package\n", validate=False),
        },
        tags=("python", "multi-file"),
    )
    javascript_basic = FixtureTemplate(
        name="javascript_basic",
        description="Single-file JavaScript fixture with class and function.",
        files={"src/index.js": FixtureFile(TestFixtures.create_sample_javascript_file(), language="javascript")},
        tags=("javascript", "single-file"),
    )
    typescript_basic = FixtureTemplate(
        name="typescript_basic",
        description="Single-file TypeScript fixture with interface, class, and function.",
        files={"src/index.ts": FixtureFile(TestFixtures.create_sample_typescript_file(), language="typescript")},
        tags=("typescript", "single-file"),
    )
    tsx_basic = FixtureTemplate(
        name="tsx_basic",
        description="TSX fixture with React component.",
        files={"src/App.tsx": FixtureFile(TestFixtures.create_sample_tsx_file(), language="tsx")},
        tags=("tsx", "single-file"),
    )
    rust_basic = FixtureTemplate(
        name="rust_basic",
        description="Single-file Rust fixture with struct and functions.",
        files={"src/lib.rs": FixtureFile(TestFixtures.create_sample_rust_file(), language="rust")},
        tags=("rust", "single-file"),
    )
    go_basic = FixtureTemplate(
        name="go_basic",
        description="Single-file Go fixture with basic functions.",
        files={"main.go": FixtureFile(TestFixtures.create_sample_go_file(), language="go")},
        tags=("go", "single-file"),
    )
    java_basic = FixtureTemplate(
        name="java_basic",
        description="Single-file Java fixture with main method.",
        files={"src/Main.java": FixtureFile(TestFixtures.create_sample_java_file(), language="java")},
        tags=("java", "single-file"),
    )
    edge_empty = FixtureTemplate(
        name="edge_empty_repo",
        description="Empty repository fixture.",
        files={},
        tags=("edge", "empty"),
    )
    edge_unsupported = FixtureTemplate(
        name="edge_unsupported_files",
        description="Repo with unsupported file extensions only.",
        files={
            "README.txt": FixtureFile("just text\n", validate=False),
            "notes.md": FixtureFile("# Notes\n", validate=False),
        },
        tags=("edge", "unsupported"),
    )
    edge_nested = FixtureTemplate(
        name="edge_nested_paths",
        description="Deeply nested paths with valid Python content.",
        files={
            "src/nested/deep/path/feature/handler.py": FixtureFile(
                "\n".join(
                    [
                        "def handle(value: int) -> int:",
                        "    return value * 2",
                        "",
                    ]
                )
                + "\n",
                language="python",
            ),
        },
        tags=("edge", "python"),
    )

    for template in [
        python_basic,
        python_multi,
        javascript_basic,
        typescript_basic,
        tsx_basic,
        rust_basic,
        go_basic,
        java_basic,
        edge_empty,
        edge_unsupported,
        edge_nested,
    ]:
        registry.register(template)

    registry.register(
        registry.compose(
            name="multi_language_repo",
            fixtures=["python_basic", "javascript_basic", "typescript_basic"],
            description="Mixed-language repo with Python, JavaScript, and TypeScript.",
            tags=("multi-language", "mixed"),
        )
    )


_register_default_fixtures(FIXTURE_REGISTRY)
