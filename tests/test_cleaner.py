"""
Tests for app.ingestion.cleaner

Run with:
    pytest tests/test_cleaner.py -v
"""
import pytest 
from app.ingestion.cleaner import (
    normalize,
    remove_boilerplate_lines,
    remove_short_lines,
    collapse_blank_lines,
    normalize_financial_symbols,
    clean_document,
)


class TestNormalize:
    def test_strips_whitespace(self):
        assert normalize("  hello  ") == "hello"

    def test_collapses_tabs(self):
        assert normalize("hello\t\tworld") == "hello world"

    def test_empty_string(self):
        assert normalize("") == ""


class TestRemoveBoilerplateLines:
    def test_removes_page_numbers(self):
        lines = ["Page 1 of 10", "Real content here with enough chars"]
        result = remove_boilerplate_lines(lines)
        assert "Page 1 of 10" not in result

    def test_removes_confidential(self):
        lines = ["Confidential", "Actual financial data line here"]
        result = remove_boilerplate_lines(lines)
        assert "Confidential" not in result

    def test_keeps_real_content(self):
        lines = ["Total revenue: £1,234,567 for Q3 2024"]
        result = remove_boilerplate_lines(lines)
        assert result == lines


class TestRemoveShortLines:
    def test_removes_short(self):
        lines = ["Hi", "This is a long enough line to keep"]
        result = remove_short_lines(lines)
        assert "Hi" not in result

    def test_keeps_long(self):
        line = "This line is definitely long enough to survive"
        result = remove_short_lines([line])
        assert line in result


class TestNormalizeFinancialSymbols:
    def test_gbp_to_symbol(self):
        assert "£" in normalize_financial_symbols("GBP 1,000")

    def test_accounting_negative(self):
        result = normalize_financial_symbols("(1,234.56)")
        assert "-1,234.56" in result

    def test_usd_to_symbol(self):
        assert "$" in normalize_financial_symbols("USD 500")


class TestCleanDocument:
    def test_empty_returns_empty(self):
        assert clean_document("") == ""

    def test_whitespace_only_returns_empty(self):
        assert clean_document("   \n\n  ") == ""

    def test_reduces_length(self):
        noisy = "Page 1 of 5\n" * 10 + "Real content that matters here today"
        cleaned = clean_document(noisy)
        assert len(cleaned) < len(noisy)

    def test_preserves_financial_content(self):
        text = "Total balance: GBP 2,400.00 as of January 2024"
        result = clean_document(text)
        assert "2,400.00" in result