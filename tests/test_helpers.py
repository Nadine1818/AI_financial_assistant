"""
Tests for app.utils.helpers

Run with:
    pytest tests/test_helpers.py -v
"""
import pytest
from app.utils.helpers import (
    clean_text,
    truncate_text,
    format_currency,
    format_percentage,
    format_large_number,
    safe_parse_json,
    extract_field,
    estimate_tokens,
    mask_sensitive,
)


class TestCleanText:
    def test_strips_edges(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_raises_on_non_string(self):
        with pytest.raises(TypeError):
            clean_text(123)


class TestTruncateText:
    def test_short_text_unchanged(self):
        assert truncate_text("hi", 10) == "hi"

    def test_truncates_long_text(self):
        result = truncate_text("hello world", 8)
        assert len(result) <= 8
        assert result.endswith("...")

    def test_raises_on_zero_max(self):
        with pytest.raises(ValueError):
            truncate_text("hello", 0)


class TestFormatCurrency:
    def test_gbp(self):
        assert format_currency(1234.5) == "£1,234.50"

    def test_usd(self):
        assert format_currency(500.0, "USD") == "$500.00"

    def test_negative(self):
        assert format_currency(-500.0) == "-£500.00"


class TestFormatPercentage:
    def test_ratio(self):
        assert format_percentage(0.045) == "4.50%"

    def test_already_percentage(self):
        assert format_percentage(4.5) == "4.50%"

    def test_negative(self):
        assert "-" in format_percentage(-0.12)


class TestFormatLargeNumber:
    def test_millions(self):
        assert "M" in format_large_number(1_500_000)

    def test_billions(self):
        assert "B" in format_large_number(2_000_000_000)

    def test_thousands(self):
        assert "K" in format_large_number(45_000)

    def test_small_number(self):
        assert format_large_number(999) == "999"


class TestSafeParseJson:
    def test_clean_json(self):
        assert safe_parse_json('{"key": "value"}') == {"key": "value"}

    def test_markdown_wrapped(self):
        text = '```json\n{"key": "value"}\n```'
        assert safe_parse_json(text) == {"key": "value"}

    def test_returns_none_on_failure(self):
        assert safe_parse_json("not json at all!!!") is None

    def test_empty_string(self):
        assert safe_parse_json("") is None


class TestExtractField:
    def test_nested(self):
        data = {"a": {"b": {"c": 42}}}
        assert extract_field(data, "a", "b", "c") == 42

    def test_missing_key(self):
        data = {"a": 1}
        assert extract_field(data, "a", "b") is None

    def test_default(self):
        assert extract_field({}, "missing", default="x") == "x"


class TestMaskSensitive:
    def test_masks_api_key(self):
        result = mask_sensitive("sk-abc123XYZ789")
        assert result.startswith("sk-a")
        assert "*" in result

    def test_short_value(self):
        result = mask_sensitive("ab")
        assert result == "**"