from __future__ import annotations

from gloggur.search.router.query_compat import parse_query_compat


def test_query_compat_parses_rg_with_flags_and_glob() -> None:
    parsed = parse_query_compat('rg -S -g "*.py" AuthToken')

    assert parsed.source == "grep_compat"
    assert parsed.command == "rg"
    assert parsed.pattern == "AuthToken"
    assert parsed.pattern_quoted is False
    assert parsed.case_mode == "smart"
    assert parsed.include_globs == ("*.py",)
    assert parsed.path_filters == ()
    assert parsed.fallback_used is False


def test_query_compat_parses_grep_with_path_filter() -> None:
    parsed = parse_query_compat("grep -R foo_bar src/")

    assert parsed.source == "grep_compat"
    assert parsed.command == "grep"
    assert parsed.pattern == "foo_bar"
    assert parsed.pattern_quoted is False
    assert parsed.path_filters == ("src/",)
    assert parsed.fallback_used is False


def test_query_compat_fallbacks_to_plain_on_parse_failure() -> None:
    parsed = parse_query_compat("rg -g")

    assert parsed.source == "grep_compat"
    assert parsed.fallback_used is True
    assert parsed.parse_error is not None


def test_query_compat_collects_unknown_flags() -> None:
    parsed = parse_query_compat("rg --no-such-flag Foo")

    assert parsed.source == "grep_compat"
    assert parsed.pattern == "Foo"
    assert parsed.unknown_flags == ("--no-such-flag",)


def test_query_compat_skips_values_for_common_rg_flags() -> None:
    parsed = parse_query_compat("rg --max-count 10 Foo")

    assert parsed.source == "grep_compat"
    assert parsed.pattern == "Foo"
    assert parsed.fallback_used is False


def test_query_compat_glob_exclude_forms_are_equivalent() -> None:
    parsed_eq = parse_query_compat("rg --glob=!*.py Foo")
    parsed_short = parse_query_compat("rg -g!*.py Foo")
    parsed_split = parse_query_compat("rg --glob !*.py Foo")

    assert parsed_eq.exclude_globs == ("*.py",)
    assert parsed_short.exclude_globs == ("*.py",)
    assert parsed_split.exclude_globs == ("*.py",)


def test_query_compat_case_mode_last_flag_wins() -> None:
    first_ignore = parse_query_compat("rg -i -S Foo")
    first_smart = parse_query_compat("rg -S -i Foo")

    assert first_ignore.case_mode == "smart"
    assert first_smart.case_mode == "ignore"


def test_query_compat_tracks_quoted_pattern_tokens() -> None:
    quoted = parse_query_compat('rg "id"')
    unquoted = parse_query_compat("rg id")

    assert quoted.pattern == "id"
    assert quoted.pattern_quoted is True
    assert unquoted.pattern == "id"
    assert unquoted.pattern_quoted is False


def test_query_compat_leaves_plain_queries_untouched() -> None:
    parsed = parse_query_compat("where is AuthToken validated")

    assert parsed.source == "plain"
    assert parsed.pattern is None
