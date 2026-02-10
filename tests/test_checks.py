from scripts.verification.checks import Checks, Optional, Range


def test_optional_missing_field_ok() -> None:
    """Optional fields may be omitted without error."""
    schema = {"required": int, "maybe": Optional(str)}
    data = {"required": 1}
    result = Checks.check_json_structure(data, schema)
    assert result.ok is True


def test_optional_present_none_ok() -> None:
    """Optional fields accept None values."""
    schema = {"required": int, "maybe": Checks.optional(str)}
    data = {"required": 1, "maybe": None}
    result = Checks.check_json_structure(data, schema)
    assert result.ok is True


def test_optional_wrong_type_reports_optional() -> None:
    """Optional fields report type errors with optional label."""
    schema = {"required": int, "maybe": Checks.optional(str)}
    data = {"required": 1, "maybe": 42}
    result = Checks.check_json_structure(data, schema)
    assert result.ok is False
    assert result.details is not None
    assert any("optional" in error["expected"] for error in result.details["errors"])


def test_range_bounds_check() -> None:
    """Range checker enforces min/max bounds."""
    schema = {"score": Range(min_value=0.0, max_value=1.0)}
    good = Checks.check_json_structure({"score": 0.5}, schema)
    low = Checks.check_json_structure({"score": -0.1}, schema)
    high = Checks.check_json_structure({"score": 1.5}, schema)
    assert good.ok is True
    assert low.ok is False
    assert high.ok is False


def test_range_type_mismatch() -> None:
    """Range checker reports type mismatches."""
    schema = {"score": Range(min_value=0.0, max_value=1.0)}
    result = Checks.check_json_structure({"score": "nope"}, schema)
    assert result.ok is False
    assert result.details is not None
    assert any("range" in error["expected"] for error in result.details["errors"])


def test_none_for_dict_and_list_reports_specific_message() -> None:
    """None for dict/list fields reports specific errors."""
    dict_schema = {"meta": {"count": int}}
    list_schema = {"items": [int]}
    dict_result = Checks.check_json_structure({"meta": None}, dict_schema)
    list_result = Checks.check_json_structure({"items": None}, list_schema)
    assert dict_result.ok is False
    assert list_result.ok is False
    assert dict_result.details is not None
    assert list_result.details is not None
    assert any(
        error["message"] == "expected dict, got None (NoneType)" for error in dict_result.details["errors"]
    )
    assert any(
        error["message"] == "expected list, got None (NoneType)" for error in list_result.details["errors"]
    )


def test_tuple_union_type_checking() -> None:
    """Tuple types act as unions in schema checks."""
    schema = {"value": (int, float)}
    good = Checks.check_json_structure({"value": 1.5}, schema)
    bad = Checks.check_json_structure({"value": "nope"}, schema)
    assert good.ok is True
    assert bad.ok is False
    assert bad.details is not None
    assert any(error["expected"] == "int or float" for error in bad.details["errors"])
