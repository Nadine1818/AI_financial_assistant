# Cleans raw extracted text before it enters the chunking + embedding pipeline.

"""
Responsibilities:
    1. Normalize whitespace and unicode (delegate to helpers.clean_text)
    2. Remove financial document boilerplate (page numbers, headers, footers)
    3. Deduplicate repeated lines (common in scraped reports)
    4. Filter out lines that carry zero semantic value
    5. Optionally strip tables (when plain text extraction mangled them)
 
Design principles:
    - Every cleaning step is a separate, testable function
    - The main entry point `clean_document` composes them in order
    - Steps are opt-in via flags so callers control aggressiveness
    - Logs every meaningful transformation at DEBUG level for auditability

"""
import re
from app.utils.helpers import clean_text, truncate_text
from app.utils.logger import get_logger
 
logger = get_logger(__name__)
# __name__ here is "app.ingestion.cleaner"

# CONSTANTS
# Lines shorter than this are often just noise (e.g. page numbers, "Table of Contents", etc.)
MIN_LINE_LENGTH = 10

# Regex patterns that match common financial document boilerplate.
# Each is compiled once at module load (fast) rather than inside loops (slow).
BOILERPLATE_PATTERNS: list[re.Pattern] = [
    # Page numbering: "Page 3 of 12", "- 3 -", "3 | 12"
    re.compile(r"^\s*[-–]?\s*page\s+\d+\s*(of\s+\d+)?\s*[-–]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*[|/]\s*\d+\s*$"),
 
    # Standalone page numbers: a line that is just a number
    re.compile(r"^\s*\d{1,4}\s*$"),
 
    # Common footer/header stamps
    re.compile(r"^\s*confidential\s*$", re.IGNORECASE),
    re.compile(r"^\s*for\s+internal\s+use\s+only\s*$", re.IGNORECASE),
    re.compile(r"^\s*draft\s*$", re.IGNORECASE),
    re.compile(r"^\s*proprietary\s+and\s+confidential\s*$", re.IGNORECASE),
 
    # Copyright lines: "© 2024 Acme Corp", "Copyright 2024"
    re.compile(r"^\s*(©|copyright|\(c\))\s*\d{4}", re.IGNORECASE),
 
    # "Continued on next page" / "Continued from previous page"
    re.compile(r"^\s*continued\s+(on|from)\s+", re.IGNORECASE),
 
    # Horizontal rules left over from PDF extraction: "----", "====", "____"
    re.compile(r"^\s*[-=_]{3,}\s*$"),
]

# Patterns that signal a line is a mangled table fragment.
# e.g. "| 1,234.56 | | 2,345.67 |" — these look like data but are unreadable.
TABLE_FRAGMENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"(\|\s*){2,}"),          # Multiple pipe chars → table row
    re.compile(r"(\t\s*){3,}"),           # Many tabs → TSV table row
    re.compile(r"^\s*[\d,.\s]{20,}\s*$"), # Long run of numbers/commas only
]

# UNICODE & WHITESPACE NORMALIZATION
# Delegated to helpers.clean_text, which applies Unicode NFKC normalization and collapses all
def normalize(text: str) -> str:
    """
    This is always the first step — everything else assumes clean unicode.
    Delegates to helpers.clean_text so there's a single source of truth.
    """
    return clean_text(text)

# LINE-LEVEL CLEANING
# Financial documents are best cleaned line by line.
# A "bad" line shouldn't corrupt the surrounding good lines.
def remove_boilerplate_lines(lines: list[str]) -> list[str]:
    """
    Remove lines that match known boilerplate patterns.
 
    Each line is tested against every pattern in BOILERPLATE_PATTERNS.
    If any pattern matches, the line is dropped and logged at DEBUG.
 
    Args:
        lines: List of strings, one per line.
 
    Returns:
        Filtered list with boilerplate lines removed.
    """
    cleaned = []
    removed_count = 0
 
    for line in lines:
        is_boilerplate = any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS)
        if is_boilerplate:
            removed_count += 1
            logger.debug("Removed boilerplate line: %r", truncate_text(line, 60))
        else:
            cleaned.append(line)
 
    if removed_count:
        logger.debug("Removed %d boilerplate line(s)", removed_count)
 
    return cleaned

def remove_short_lines(lines: list[str], min_length: int = MIN_LINE_LENGTH) -> list[str]:
    """
    Drop lines that are too short to carry semantic meaning.
    """
    before = len(lines)
    cleaned = [line for line in lines if len(line.strip()) >= min_length]
    removed = before - len(cleaned)
 
    if removed:
        logger.debug("Removed %d short line(s) (< %d chars)", removed, min_length)
 
    return cleaned

def remove_short_lines(lines: list[str], min_length: int = MIN_LINE_LENGTH) -> list[str]:
    """
    Drop lines that are too short to carry semantic meaning.
 
    A line of 3–9 chars is almost never a complete financial statement.
    Common culprits: stray punctuation, single words, page artifacts.
 
    Args:
        lines:      List of line strings.
        min_length: Minimum character count to keep a line.
    """
    before = len(lines)
    cleaned = [line for line in lines if len(line.strip()) >= min_length]
    removed = before - len(cleaned)
 
    if removed:
        logger.debug("Removed %d short line(s) (< %d chars)", removed, min_length)
 
    return cleaned

def remove_table_fragments(lines: list[str]) -> list[str]:
    """
    Remove lines that are clearly mangled table fragments from PDF extraction.
    handle the actual table parsing in loader.py if we want to preserve the data, but for now we just want to remove the noise.
    Args:
        lines: List of line strings.
    """
    cleaned = []
    removed_count = 0
 
    for line in lines:
        is_fragment = any(p.search(line) for p in TABLE_FRAGMENT_PATTERNS)
        if is_fragment:
            removed_count += 1
            logger.debug("Removed table fragment: %r", truncate_text(line, 60))
        else:
            cleaned.append(line)
 
    if removed_count:
        logger.debug("Removed %d table fragment line(s)", removed_count)
 
    return cleaned

# DOCUMENT-LEVEL CLEANING
# after line-level cleaning, we can rejoin the lines and do any final document-wide cleanup if needed.
def collapse_blank_lines(text: str, max_consecutive: int = 2) -> str:
    """
    Reduce runs of blank lines to at most `max_consecutive`.
 
    After removing boilerplate and short lines, we often end up with
    large gaps of empty lines. These waste tokens without adding meaning.
 
    Args:
        text:             Full document string (post line-cleaning).
        max_consecutive:  Max blank lines allowed in a row (default 2).
    """
    # Match 3+ consecutive newlines (i.e. 2+ blank lines) and collapse them
    pattern = re.compile(r"\n{" + str(max_consecutive + 1) + r",}")
    collapsed = pattern.sub("\n" * max_consecutive, text)
 
    if collapsed != text:
        logger.debug("Collapsed excessive blank lines (max %d consecutive)", max_consecutive)
 
    return collapsed

def normalize_financial_symbols(text: str) -> str:
    """
    Standardize financial symbols and notation for consistent embedding.
 
    Why this matters: "GBP 1,234" and "£1,234" and "1234 GBP" all mean the
    same thing, but will get different embeddings. Normalizing to a consistent
    form improves retrieval accuracy for financial queries.
 
    Transformations:
        "GBP "  → "£"
        "USD "  → "$"
        "EUR "  → "€"
        "1,234,567" → kept as-is (commas in numbers are fine)
        "(1,234)"   → "-1,234"  (accounting negative notation)
    """
    # Currency code to symbol
    text = re.sub(r"\bGBP\s*", "£", text)
    text = re.sub(r"\bUSD\s*", "$", text)
    text = re.sub(r"\bEUR\s*", "€", text)
 
    # Accounting negatives: (1,234.56) → -1,234.56
    # This notation is standard in UK financial statements
    text = re.sub(r"\((\d[\d,\.]*)\)", r"-\1", text)
 
    logger.debug("Normalized financial symbols")
    return text

def deduplicate_lines(lines: list[str]) -> list[str]:
    """
    Remove consecutive duplicate lines.
 
    Scraped financial reports often repeat section headers or table headers
    on every page. This removes runs of identical adjacent lines.
 
    Note: Only removes CONSECUTIVE duplicates, not all duplicates.
    Reason: The same sentence appearing in two different sections may be
    legitimately repeated (e.g. a disclaimer at the start and end).
 
    Example:
        ["Revenue", "Revenue", "£1,234", "£1,234", "£1,234", "Costs"]
        → ["Revenue", "£1,234", "Costs"]
    """
    if not lines:
        return []
 
    deduped = [lines[0]]
    removed_count = 0
 
    for line in lines[1:]:
        if line.strip() != deduped[-1].strip():
            deduped.append(line)
        else:
            removed_count += 1
            logger.debug("Removed duplicate line: %r", truncate_text(line, 60))
 
    if removed_count:
        logger.debug("Removed %d consecutive duplicate line(s)", removed_count)
 
    return deduped 

# MAIN ENTRY POINT
# composes all cleaning steps in correct order, the fn used by loader.py & chunker.py
def clean_document(
    text: str,
    remove_boilerplate: bool = True,
    remove_short: bool = True,
    deduplicate: bool = True,
    # we only set it to true if we KNOW tha tables are mangled
    strip_table_fragments: bool = False,  # opt-in: aggressive, may lose data
    normalize_symbols: bool = True,
) -> str:
    
    if not text or not text.strip():
        logger.warning("clean_document received empty or whitespace-only text")
        return ""
 
    original_length = len(text)
    logger.debug("Starting document cleaning | input length: %d chars", original_length)
 
    # ── Step 1: Unicode + whitespace normalization ──────────────────────────
    # Always first — everything downstream assumes clean unicode
    text = normalize(text)
 
    # ── Step 2: Financial symbol normalization ───────────────────────────────
    if normalize_symbols:
        text = normalize_financial_symbols(text)
 
    # ── Step 3: Line-level cleaning ──────────────────────────────────────────
    # Split into lines, apply filters, rejoin
    lines = text.splitlines()
    logger.debug("Split into %d lines for line-level cleaning", len(lines))
 
    if remove_boilerplate:
        lines = remove_boilerplate_lines(lines)
 
    if remove_short:
        lines = remove_short_lines(lines)
 
    if deduplicate:
        lines = deduplicate_lines(lines)
 
    if strip_table_fragments:
        lines = remove_table_fragments(lines)
 
    # Rejoin lines back into a single string
    text = "\n".join(lines)
 
    # ── Step 4: Document-level cleanup ──────────────────────────────────────
    text = collapse_blank_lines(text)
 
    # ── Final: strip edges ───────────────────────────────────────────────────
    text = text.strip()
 
    final_length = len(text)
    reduction_pct = ((original_length - final_length) / original_length * 100) if original_length else 0
 
    logger.info(
        "Document cleaned | %d → %d chars (%.1f%% reduction)",
        original_length,
        final_length,
        reduction_pct,
    )
 
    return text

# filter out empty documents after cleaning, and log how many were skipped
def clean_documents(texts: list[str], **kwargs) -> list[str]:
    cleaned = []
    skipped = 0
 
    for i, text in enumerate(texts):
        result = clean_document(text, **kwargs)
        if result:
            cleaned.append(result)
        else:
            skipped += 1
            logger.warning("Document %d was empty after cleaning — skipped", i)
 
    logger.info(
        "Batch cleaning complete | %d documents cleaned, %d skipped",
        len(cleaned),
        skipped,
    )
 
    return cleaned