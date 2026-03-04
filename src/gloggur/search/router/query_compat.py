from __future__ import annotations

import shlex
from dataclasses import dataclass

_COMPAT_COMMANDS = {"rg", "grep"}
_NOOP_FLAGS = {
    "-R",
    "-r",
    "-n",
    "--line-number",
    "-H",
    "--with-filename",
    "--no-heading",
}
_VALUE_FLAGS = {
    "-m",
    "--max-count",
    "-A",
    "--after-context",
    "-B",
    "--before-context",
    "-C",
    "--context",
    "--max-columns",
    "--max-filesize",
    "-t",
    "--type",
    "-T",
    "--type-not",
    "-j",
    "--threads",
    "--encoding",
    "--engine",
    "--sort",
    "--sortr",
}
_INLINE_VALUE_PREFIXES = ("-m", "-A", "-B", "-C", "-j", "-t", "-T")
_INLINE_EQ_VALUE_PREFIXES = (
    "--max-count=",
    "--after-context=",
    "--before-context=",
    "--context=",
    "--max-columns=",
    "--max-filesize=",
    "--type=",
    "--type-not=",
    "--threads=",
    "--encoding=",
    "--engine=",
    "--sort=",
    "--sortr=",
)


@dataclass(frozen=True)
class ParsedQueryCompat:
    source: str
    command: str | None = None
    pattern: str | None = None
    pattern_quoted: bool = False
    path_filters: tuple[str, ...] = ()
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    case_mode: str | None = None
    word_match: bool = False
    fixed_string: bool = False
    unknown_flags: tuple[str, ...] = ()
    fallback_used: bool = False
    parse_error: str | None = None

    def to_debug_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "command": self.command,
            "pattern": self.pattern,
            "pattern_quoted": self.pattern_quoted,
            "path_filters": list(self.path_filters),
            "include_globs": list(self.include_globs),
            "exclude_globs": list(self.exclude_globs),
            "case_mode": self.case_mode,
            "word_match": self.word_match,
            "fixed_string": self.fixed_string,
            "unknown_flags": list(self.unknown_flags),
            "fallback_used": self.fallback_used,
            "parse_error": self.parse_error,
        }


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _is_quoted_token(raw_token: str) -> bool:
    if len(raw_token) < 2:
        return False
    if raw_token[0] == raw_token[-1] and raw_token[0] in {"'", '"'}:
        return True
    return False


def parse_query_compat(query: str) -> ParsedQueryCompat:
    raw = query.strip()
    if not raw:
        return ParsedQueryCompat(source="plain")
    lowered = raw.split(maxsplit=1)[0].strip().lower()
    if lowered not in _COMPAT_COMMANDS:
        return ParsedQueryCompat(source="plain")

    try:
        argv = shlex.split(raw)
        argv_raw = shlex.split(raw, posix=False)
    except ValueError as exc:
        return ParsedQueryCompat(
            source="grep_compat",
            command=lowered,
            fallback_used=True,
            parse_error=f"{type(exc).__name__}: {exc}",
        )
    if not argv:
        return ParsedQueryCompat(source="plain")
    if len(argv_raw) != len(argv):
        argv_raw = list(argv)

    command = argv[0].lower()
    if command not in _COMPAT_COMMANDS:
        return ParsedQueryCompat(source="plain")

    include_globs: list[str] = []
    exclude_globs: list[str] = []
    path_filters: list[str] = []
    unknown_flags: list[str] = []
    positional: list[tuple[str, bool]] = []
    case_mode: str | None = None
    word_match = False
    fixed_string = False

    index = 1
    while index < len(argv):
        token = argv[index]
        raw_token = argv_raw[index]
        if token in {"-i", "--ignore-case"}:
            case_mode = "ignore"
            index += 1
            continue
        if token in {"-S", "--smart-case"}:
            case_mode = "smart"
            index += 1
            continue
        if token == "-w":
            word_match = True
            index += 1
            continue
        if token == "-F":
            fixed_string = True
            index += 1
            continue
        if token in _NOOP_FLAGS:
            index += 1
            continue
        if token in _VALUE_FLAGS:
            if index + 1 >= len(argv):
                return ParsedQueryCompat(
                    source="grep_compat",
                    command=command,
                    fallback_used=True,
                    parse_error=f"{token} missing required value",
                )
            index += 2
            continue
        if token.startswith(_INLINE_EQ_VALUE_PREFIXES):
            index += 1
            continue
        if token.startswith(_INLINE_VALUE_PREFIXES) and len(token) > 2:
            index += 1
            continue
        if token in {"-g", "--glob"}:
            if index + 1 >= len(argv):
                return ParsedQueryCompat(
                    source="grep_compat",
                    command=command,
                    fallback_used=True,
                    parse_error=f"{token} missing required value",
                )
            value = argv[index + 1].strip()
            if value.startswith("!"):
                normalized = value[1:].strip()
                if normalized:
                    exclude_globs.append(normalized)
            elif value:
                include_globs.append(value)
            index += 2
            continue
        if token.startswith("--glob="):
            value = token[len("--glob=") :].strip()
            if value.startswith("!"):
                normalized = value[1:].strip()
                if normalized:
                    exclude_globs.append(normalized)
            elif value:
                include_globs.append(value)
            index += 1
            continue
        if token.startswith("-g") and len(token) > 2:
            value = token[2:].strip()
            if value.startswith("!"):
                normalized = value[1:].strip()
                if normalized:
                    exclude_globs.append(normalized)
            elif value:
                include_globs.append(value)
            index += 1
            continue
        if token.startswith("-"):
            unknown_flags.append(token)
            index += 1
            continue
        positional.append((token, _is_quoted_token(raw_token)))
        index += 1

    if not positional:
        return ParsedQueryCompat(
            source="grep_compat",
            command=command,
            fallback_used=True,
            parse_error="no search pattern found in grep-compatible query",
            unknown_flags=_unique(unknown_flags),
        )

    pattern, pattern_quoted = positional[0]
    if len(positional) > 1:
        path_filters.extend(token for token, _ in positional[1:])
    return ParsedQueryCompat(
        source="grep_compat",
        command=command,
        pattern=pattern,
        pattern_quoted=pattern_quoted,
        path_filters=_unique(path_filters),
        include_globs=_unique(include_globs),
        exclude_globs=_unique(exclude_globs),
        case_mode=case_mode,
        word_match=word_match,
        fixed_string=fixed_string,
        unknown_flags=_unique(unknown_flags),
        fallback_used=False,
    )
