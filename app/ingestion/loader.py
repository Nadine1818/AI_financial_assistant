# Loads raw files from disk and returns them as uniform Document objects.
"""
Responsibilities:
    1. Accept a file path (or directory of files)
    2. Detect the file type and dispatch to the correct loader
    3. Extract raw text + structured metadata from each file
    4. Return a Document object with a consistent shape
    5. Never clean, chunk, or embed — that is the job of downstream modules
Design principles:
    - Strategy Pattern: one loader class per file type, identical interface
    - Fail loudly on unsupported types (no silent data loss)
    - Metadata is always populated — auditability requires knowing the source
    - CSV and JSON are converted to plain text so cleaner.py stays format-agnostic
    - Logging at every meaningful step for observability
 
Supported formats:
    .pdf   → pdfplumber (preserves layout better than PyPDF2 for financial docs)
    .csv   → csv stdlib (no pandas dependency at this layer)
    .json  → json stdlib
    .txt   → plain read
    .md    → plain read (treated same as .txt)
"""
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable
 
import pdfplumber
 
from app.config.settings import settings
from app.utils.logger import get_logger
 
logger = get_logger(__name__)
# __name__ here is "app.ingestion.loader"

# DOCUMENT DATACLASS
# Why a dataclass and not a plain dict?
#   - Type-safe: downstream code knows exactly what fields exist
#   - IDE autocomplete works (document.content, document.metadata)
#   - Immutability is easy to add later (frozen=True)
#   - Easier to extend (add fields without breaking callers)
@dataclass
class Document:
    """
    A single unit of loaded content ready for cleaning and chunking.
 
    Attributes:
        content:   The raw extracted text. This is what cleaner.py receives.
        metadata:  A dict describing the source. Always populated.
                   Standard keys:
                       "source"      → absolute file path (str)
                       "filename"    → just the file name (str)
                       "file_type"   → extension without dot, e.g. "pdf" (str)
                       "file_size_bytes" → size of the file on disk (int)
                   Loader-specific keys (added by each loader):
                       PDF  → "page_count", "page_number" (one Doc per page)
                       CSV  → "row_count", "column_names"
                       JSON → "top_level_keys"
    """
    content: str
    metadata: dict = field(default_factory=dict)
 
    def __post_init__(self):
        # Guard: content must always be a string (even if empty)
        if not isinstance(self.content, str):
            raise TypeError(
                f"Document.content must be str, got {type(self.content).__name__}"
            )
 
    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"Document(source={self.metadata.get('source', 'unknown')!r}, "
            f"chars={len(self.content)}, preview={preview!r})"
        )

# LOADER PROTOCOL
# Defines the interface that all loaders must implement.
@runtime_checkable
class DocumentLoader(Protocol):
    def load(self, path: Path) -> list[Document]:
        """Load a file and return one or more Documents."""
        ...
# BASE LOADER 
# Shared logic that every loader inherits.
# Keeps individual loaders focused on extraction only.
class BaseLoader:
    """
    Base class with shared utilities for all file loaders.
 
    Subclasses must implement _extract(path) → list[Document].
    The public .load() method wraps _extract() with:
        - File existence check
        - File size logging
        - Error handling with a clear message
    """
 
    # Which extensions this loader handles (overridden by subclasses)
    supported_extensions: tuple[str, ...] = ()
 
    def load(self, path: Path) -> list[Document]:
        """
        Public entry point. Validates the file then calls _extract().
 
        Args:
            path: Resolved Path object pointing to a file on disk.
 
        Returns:
            List of Document objects (may be >1 for multi-page PDFs).
 
        Raises:
            FileNotFoundError: If the path doesn't exist.
            ValueError:        If the file is empty.
            RuntimeError:      If extraction fails for any reason.
        """
        path = Path(path).resolve()
 
        # ── Guard 1: file must exist ─────────────────────────────────────────
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
 
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
 
        # ── Guard 2: file must not be empty ──────────────────────────────────
        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError(f"File is empty (0 bytes): {path}")
 
        logger.info(
            "Loading file | path=%s | size=%d bytes | loader=%s",
            path.name,
            file_size,
            self.__class__.__name__,
        )
 
        # ── Extract ──────────────────────────────────────────────────────────
        try:
            documents = self._extract(path)
        except Exception as exc:
            # Wrap any extraction error with context about which file failed.
            # This makes debugging much faster in production.
            raise RuntimeError(
                f"[{self.__class__.__name__}] Failed to load '{path.name}': {exc}"
            ) from exc
 
        logger.info(
            "Loaded %d document(s) from '%s'",
            len(documents),
            path.name,
        )
 
        return documents
 
    def _extract(self, path: Path) -> list[Document]:
        """Override in subclasses to implement extraction logic."""
        raise NotImplementedError
 
    def _base_metadata(self, path: Path) -> dict:
        """
        Build the standard metadata dict that all loaders include.
 
        Every Document gets at minimum:
            source, filename, file_type, file_size_bytes
 
        Loader-specific metadata is merged on top in each _extract().
        """
        return {
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix.lstrip(".").lower(),  # "pdf", "csv", etc.
            "file_size_bytes": path.stat().st_size,
        }
# PDF LOADER 
# Why one Document per page?
#   - Chunker.py gets more granular units to work with
#   - Retrieval can cite exact page numbers (auditability)
#   - Avoids huge single-document blobs that exceed context windows
class PDFLoader(BaseLoader):
    """
    Loads PDF files using pdfplumber.
 
    Returns one Document per page so downstream chunking is page-aware.
    Pages that yield no extractable text are skipped with a warning.
    """
 
    supported_extensions = (".pdf",)
 
    def _extract(self, path: Path) -> list[Document]:
        documents = []
        base_meta = self._base_metadata(path)
 
        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            base_meta["page_count"] = total_pages
 
            logger.debug("PDF has %d page(s): %s", total_pages, path.name)
 
            for page_num, page in enumerate(pdf.pages, start=1):
                # extract_text() returns None if the page has no text layer
                # (e.g. a scanned image — we skip those for now)
                raw_text = page.extract_text()
 
                if not raw_text or not raw_text.strip():
                    logger.warning(
                        "Page %d/%d of '%s' yielded no text — skipping "
                        "(possibly a scanned image or graphic-only page)",
                        page_num,
                        total_pages,
                        path.name,
                    )
                    continue
 
                # Each page gets its own Document with page-specific metadata
                page_meta = {
                    **base_meta,
                    "page_number": page_num,
                    "total_pages": total_pages,
                }
 
                documents.append(Document(content=raw_text, metadata=page_meta))
                logger.debug(
                    "Extracted page %d/%d | %d chars",
                    page_num,
                    total_pages,
                    len(raw_text),
                )
 
        if not documents:
            logger.warning(
                "PDF '%s' yielded 0 documents after extraction "
                "(all pages were blank or image-only)",
                path.name,
            )
 
        return documents
 
# CSV LOADER
# convert each row into a readable sentence
# one document for the whole file (not one per row) to preserve context
class CSVLoader(BaseLoader):
    """
    Loads CSV files and converts rows to readable text for embedding.
 
    Each row becomes a "key: value | key: value" sentence.
    The full file becomes a single Document (chunker handles splitting).
    """
 
    supported_extensions = (".csv",)
 
    def _extract(self, path: Path) -> list[Document]:
        base_meta = self._base_metadata(path)
        rows_text: list[str] = []
        column_names: list[str] = []
 
        with open(path, newline="", encoding="utf-8-sig") as f:
            # utf-8-sig handles BOM characters that Excel adds to CSV exports
            reader = csv.DictReader(f)
 
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header row: {path.name}")
 
            column_names = list(reader.fieldnames)
            logger.debug(
                "CSV columns: %s", ", ".join(column_names)
            )
 
            for i, row in enumerate(reader):
                # Convert each row to "Column: Value | Column: Value" format
                # Skip columns with no value to keep sentences concise
                row_parts = [
                    f"{col}: {val.strip()}"
                    for col, val in row.items()
                    if val and val.strip()
                ]
                if row_parts:
                    rows_text.append(" | ".join(row_parts))
 
        row_count = len(rows_text)
        logger.debug("CSV parsed: %d data rows from '%s'", row_count, path.name)
 
        # Join all rows into one block of text, one row per line
        content = "\n".join(rows_text)
 
        meta = {
            **base_meta,
            "row_count": row_count,
            "column_names": column_names,
        }
 
        return [Document(content=content, metadata=meta)]
    
# JSON LOADER
# Similar to CSV, we convert JSON objects into readable text.
class JSONLoader(BaseLoader):
    """
    Loads JSON files and serializes them to indented text.
 
    Handles both dict (single object) and list (array of objects).
    For lists, each item is serialized separately and joined with a separator.
    """
 
    supported_extensions = (".json",)
 
    def _extract(self, path: Path) -> list[Document]:
        base_meta = self._base_metadata(path)
 
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in '{path.name}': {e}"
                ) from e
 
        # ── Serialize back to readable text ──────────────────────────────────
        if isinstance(data, list):
            # Array of objects → serialize each element with a separator
            # This is common in Open Banking responses: [{"txn": ...}, {"txn": ...}]
            parts = [json.dumps(item, indent=2, ensure_ascii=False) for item in data]
            content = "\n---\n".join(parts)
            top_level_keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
            logger.debug(
                "JSON is a list of %d item(s) from '%s'", len(data), path.name
            )
        elif isinstance(data, dict):
            content = json.dumps(data, indent=2, ensure_ascii=False)
            top_level_keys = list(data.keys())
            logger.debug(
                "JSON is a dict with keys: %s from '%s'",
                ", ".join(top_level_keys),
                path.name,
            )
        else:
            # Scalar value (rare but valid JSON): just stringify it
            content = str(data)
            top_level_keys = []
 
        meta = {
            **base_meta,
            "top_level_keys": top_level_keys,
        }
 
        return [Document(content=content, metadata=meta)]

# TXT/MD LOADER
# .txt  → financial notes, internal memos, plain summaries
# .md   → internal knowledge base docs, README-style reference material
# Both are treated identically — Markdown syntax is left as-is.
# The LLM understands Markdown natively, so stripping it would lose structure.
class TextLoader(BaseLoader):
    """
    Loads .txt and .md files as a single Document.
 
    Tries UTF-8 first, falls back to latin-1 for legacy financial exports
    (some bank export tools still produce latin-1 encoded files).
    """
 
    supported_extensions = (".txt", ".md")
 
    def _extract(self, path: Path) -> list[Document]:
        base_meta = self._base_metadata(path)
 
        # ── Encoding fallback ─────────────────────────────────────────────────
        # Try UTF-8 first (modern default). If that fails, fall back to
        # latin-1, which is a superset of ASCII and never raises a decode error.
        # This is a pragmatic choice for financial exports from legacy systems.
        try:
            content = path.read_text(encoding="utf-8")
            logger.debug("Read '%s' as UTF-8", path.name)
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")
            logger.warning(
                "UTF-8 decode failed for '%s' — fell back to latin-1. "
                "Consider re-encoding the file.",
                path.name,
            )
 
        meta = {
            **base_meta,
            "line_count": content.count("\n") + 1,
        }
 
        return [Document(content=content, metadata=meta)]
 
# LOADER FACTORY
# The Factory Pattern: given a file path, return the correct loader instance.
# Registry maps file extension → loader class
# To support a new format: add one line here, write the loader class above.
LOADER_REGISTRY: dict[str, type[BaseLoader]] = {
    ".pdf":  PDFLoader,
    ".csv":  CSVLoader,
    ".json": JSONLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
}
def get_loader(path: Path) -> BaseLoader:
    """
    Return the appropriate loader instance for the given file extension.
 
    Args:
        path: File path (the extension is all that matters here).
 
    Returns:
        An instantiated loader ready to call .load(path) on.
 
    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = path.suffix.lower()
 
    loader_class = LOADER_REGISTRY.get(ext)
 
    if loader_class is None:
        supported = ", ".join(sorted(LOADER_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported file type '{ext}' for file '{path.name}'. "
            f"Supported extensions: {supported}"
        )
 
    logger.debug("Dispatching '%s' → %s", path.name, loader_class.__name__)
    return loader_class()
 
# PUBLIC API (main entry point for external code)
def load_file(path: str | Path) -> list[Document]:
    """
    Load a single file from disk and return its Document(s).
 
    This is the main entry point for loading a single file.
    Used by: main.py, notebooks, and direct calls in tests.
 
    Args:
        path: Path to the file (str or Path object).
 
    Returns:
        List of Document objects (>1 for multi-page PDFs).
 
    Example:
        docs = load_file("data/raw/bank_statement.pdf")
        for doc in docs:
            print(doc.metadata["page_number"], doc.content[:100])
    """
    path = Path(path).resolve()
    loader = get_loader(path)
    return loader.load(path)

def load_directory(
    directory: str | Path,
    recursive: bool = False,
) -> list[Document]:
    """
    Load all supported files from a directory.
 
    Skips unsupported file types with a warning rather than crashing.
    This is intentional: a data directory may contain .DS_Store, images,
    or other files that should be silently ignored.
 
    Args:
        directory: Path to a directory on disk.
        recursive: If True, walk subdirectories too (default: False).
                   Set True for nested knowledge bases.
 
    Returns:
        Flat list of all Document objects from all loaded files.
 
    Example:
        docs = load_directory("data/raw/", recursive=True)
        print(f"Loaded {len(docs)} document chunks from {directory}")
    """
    directory = Path(directory).resolve()
 
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
 
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
 
    # Collect all file paths
    if recursive:
        # rglob("*") walks all subdirectories
        all_paths = [p for p in directory.rglob("*") if p.is_file()]
    else:
        all_paths = [p for p in directory.iterdir() if p.is_file()]
 
    # Sort for deterministic ordering (important for reproducible pipelines)
    all_paths.sort()
 
    logger.info(
        "Scanning directory '%s' | %d file(s) found | recursive=%s",
        directory.name,
        len(all_paths),
        recursive,
    )
 
    all_documents: list[Document] = []
    skipped: list[str] = []
 
    for file_path in all_paths:
        # Skip hidden files (e.g. .DS_Store, .gitkeep)
        if file_path.name.startswith("."):
            logger.debug("Skipping hidden file: %s", file_path.name)
            continue
 
        ext = file_path.suffix.lower()
        if ext not in LOADER_REGISTRY:
            skipped.append(file_path.name)
            logger.debug(
                "Skipping unsupported file type '%s': %s", ext, file_path.name
            )
            continue
 
        try:
            docs = load_file(file_path)
            all_documents.extend(docs)
        except Exception as exc:
            # Log the error but keep processing other files.
            # One bad file shouldn't stop the entire ingestion job.
            logger.error(
                "Failed to load '%s': %s — skipping this file",
                file_path.name,
                exc,
            )
 
    logger.info(
        "Directory load complete | %d document(s) loaded | %d file(s) skipped",
        len(all_documents),
        len(skipped),
    )
 
    if skipped:
        logger.debug("Skipped files: %s", ", ".join(skipped))
 
    return all_documents
 