# General-purpose utility functions used across the financial AI assistant.
"""
Design principles:
- Every function does ONE thing (Single Responsibility)
- All functions are pure (no side effects) where possible
- Financial data is treated with extra care: no silent failures
- Logging is light here — the logger.py module owns that responsibility
"""
import json
import os
import re
import time
import unicodedata
from contextlib import contextmanager
from typing import Any, Generator

def clean_text(text: str) -> str:
    """
    Clean and normalize text data for consistent processing.
        Steps:
      1. Normalize unicode (e.g. fancy quotes → standard quotes)
      2. Remove non-printable / control characters
      3. Collapse multiple whitespace into single spaces
      4. Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
 
    # Step 1: Normalize unicode to NFC (canonical composition)
    # e.g. "café" stored as "cafe\u0301" → "café" as a single char
    text = unicodedata.normalize("NFC", text)
 
    # Step 2: Remove control characters (tabs become spaces, newlines kept)
    # \x00-\x08, \x0b-\x0c, \x0e-\x1f are "invisible" control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
 
    # Step 3: Collapse multiple whitespace (spaces, tabs) into one space
    # But preserve newlines — they carry structural meaning in financial docs
    text = re.sub(r"[ \t]+", " ", text)
 
    # Step 4: Strip edges
    return text.strip()

def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum character count.
 
    Used when: building prompts with context windows in mind, or displaying
    snippets in logs without flooding them.
 
    Args:
        text:      Input string
        max_chars: Maximum allowed character count (inclusive of suffix)
        suffix:    Appended when truncation happens (default "...")
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
 
    if len(text) <= max_chars:
        return text
 
    # Leave room for the suffix inside the max_chars budget
    cut = max_chars - len(suffix)
    if cut <= 0:
        # Edge case: suffix is longer than the budget — just truncate hard
        return text[:max_chars]
 
    return text[:cut] + suffix

def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a simple but robust regex.
 
    This handles the most common financial text patterns (abbreviations like
    "Inc.", "Ltd.", decimal numbers like "3.5%") reasonably well.
 
    Returns: List of non-empty sentence strings.
    """
    # Split on . ! ? followed by whitespace + uppercase, or end of string
    # Negative lookbehind prevents splitting on "Inc." "Ltd." "U.S." etc. 
    sentence_endings = re.compile(
        # dont split if . belongs to any of these 
        r"(?<!\b(?:Mr|Mrs|Dr|Inc|Ltd|Corp|vs|etc|approx|est|fig|no))"
        r"(?<![0-9])"       # don't split on decimal points
        r"[.!?]" # . / ! / ? are sentence endings 
        r"(?=\s+[A-Z]|$)"
    )
    parts = sentence_endings.split(text)
    return [s.strip() for s in parts if s.strip()]

# This is a very rough estimate, but it helps us gauge the size of text inputs/outputs -> 4 chars/token
def estimate_tokens(text: str, model: str = "gpt-4") -> int:   
    if not text:
        return 0
    # ~4 chars per token is a safe conservative estimate (slightly over-counts)
    return max(1, len(text) // 4)

def fits_in_context(text: str, max_tokens: int, model: str = "gpt-4") -> bool:
    """
    Quick guard: will this text fit in the model's context window?
 
    Use this before sending chunks to the LLM to catch oversized inputs early.
    """
    return estimate_tokens(text, model) <= max_tokens

# financial formatting , used by response_generator.py & verifier.py before reaching user
def format_currency(
    amount: float,
    currency: str = "GBP",
    decimal_places: int = 2,
) -> str:
    """
    Format a float as a human-readable currency string.
 
    Examples:
        format_currency(1234567.5)          → "£1,234,567.50"
        format_currency(1234567.5, "USD")   → "$1,234,567.50"
        format_currency(-500.0, "EUR")      → "-€500.00"
    """
    symbols = {
        "GBP": "£",
        "USD": "$",
        "EUR": "€",
        "JPY": "¥",
    }
    symbol = symbols.get(currency.upper(), currency.upper() + " ")
 
    # Handle negatives cleanly: show -£500 not £-500
    negative = amount < 0
    abs_amount = abs(amount)
 
    formatted = f"{abs_amount:,.{decimal_places}f}"
    result = f"{symbol}{formatted}"
 
    return f"-{result}" if negative else result

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a decimal ratio or percentage value as a percentage string.
 
    Auto-detects whether input is a ratio (0.045) or already a percentage (4.5).
 
    Examples:
        format_percentage(0.045)   → "4.50%"
        format_percentage(4.5)     → "4.50%"   (detected as already %)
        format_percentage(-0.12)   → "-12.00%"
    """
    # If the absolute value is <= 1.0, treat as ratio and multiply by 100
    if abs(value) <= 1.0:
        percentage = value * 100
    else:
        percentage = value
 
    return f"{percentage:.{decimal_places}f}%"

def format_large_number(value: float) -> str:
    """
    Abbreviate large numbers for readable display in summaries.
 
    Examples:
        format_large_number(1_500_000)     → "£1.5M"  (if you add currency separately)
        format_large_number(2_300_000_000) → "2.3B"
        format_large_number(45_000)        → "45.0K"
        format_large_number(999)           → "999"
    """
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
 
    if abs_val >= 1_000_000_000:
        return f"{sign}{abs_val / 1_000_000_000:.1f}B"
    elif abs_val >= 1_000_000:
        return f"{sign}{abs_val / 1_000_000:.1f}M"
    elif abs_val >= 1_000:
        return f"{sign}{abs_val / 1_000:.1f}K"
    else:
        return f"{sign}{abs_val:.0f}"
    
# SAFE JSON PARSING 
# prevent pipeline from crashing due to malformed JSON in LLM responses or config files
def safe_parse_json(text: str) -> dict | list | None:
    """
    Attempt to parse JSON from text, including when it's wrapped in markdown.
 
    LLMs frequently return JSON like:
        ```json
        {"key": "value"}
        ```
    or with extra explanation text before/after the JSON block.
 
    Returns:
        Parsed Python object (dict or list), or None if parsing fails entirely.
 
    Note: We return None (not raise) so callers can apply fallback logic.
    This is intentional — in financial systems, a parse failure should trigger
    a fallback or re-prompt, not a 500 error.
    """
    if not text or not text.strip():
        return None
 
    # Strategy 1: Direct parse (clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
 
    # Strategy 2: Extract from markdown code block
    # Matches ```json ... ``` or ``` ... ```
    code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
 
    # Strategy 3: Find first { ... } or [ ... ] in the text
    json_like = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if json_like:
        try:
            return json.loads(json_like.group(1))
        except json.JSONDecodeError:
            pass
 
    # All strategies failed
    return None
 
def extract_field(data: dict, *keys: str, default: Any = None) -> Any:
    """
    Safely extract a nested field from a dict using a chain of keys.
 
    Much cleaner than chained .get() calls across your codebase.
 
    Example:
        data = {"user": {"profile": {"balance": 5000}}}
        extract_field(data, "user", "profile", "balance")  → 5000
        extract_field(data, "user", "missing", "key")      → None
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current

# timing & performance utilities for latency tracking in response_generator.py and other critical paths
@contextmanager
def timer(label: str = "operation") -> Generator[dict, None, None]:
    """
    Context manager that measures elapsed time of a code block.
 
    Usage:
        with timer("LLM call") as t:
            response = llm.invoke(prompt)
        print(f"Took {t['elapsed_ms']}ms")
    """
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed_ms"] = round(elapsed * 1000, 2)
        result["elapsed_s"] = round(elapsed, 4)
        result["label"] = label

def retry(func, retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry a callable on exception with exponential backoff.
 
    Used for: LLM API calls (rate limits), vector DB queries, external APIs.
 
    Args:
        func:    A zero-argument callable (use lambda or functools.partial)
        retries: Number of retry attempts (not counting the first try)
        delay:   Initial wait in seconds between retries
        backoff: Multiplier applied to delay after each failure
 
    Example:
        result = retry(lambda: openai_client.chat(...), retries=3, delay=1.0)
 
    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc = None
    wait = delay
 
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(wait)
                wait *= backoff
 
    raise last_exc  # type: ignore[misc]

# ENVIRONMENT & CONFIG VALIDATION
# Fail loudly at startup, not silently at runtime inside a financial query.
def require_env_vars(*var_names: str) -> dict[str, str]:
    missing = []
    found = {}
 
    for name in var_names:
        value = os.environ.get(name)
        if value is None:
            missing.append(name)
        else:
            found[name] = value
 
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Check your .env file and ensure it's loaded (e.g., via python-dotenv)."
        )
 
    return found

def mask_sensitive(value: str, visible_chars: int = 4) -> str:
    """
    Mask a sensitive string (API key, token) for safe logging.
 
    Example:
        mask_sensitive("sk-abc123XYZ789")  → "sk-a**********"
        mask_sensitive("Bearer token123")  → "Bear**********"
    """
    if len(value) <= visible_chars:
        return "*" * len(value)
    return value[:visible_chars] + "*" * (len(value) - visible_chars)