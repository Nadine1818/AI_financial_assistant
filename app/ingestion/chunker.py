# Splits cleaned Document content into overlapping chunks ready for embedding.
"""
Responsibilities:
    1. Accept Document objects from loader.py (after cleaner.py has run)
    2. Split their content into fixed-size overlapping chunks
    3. Preserve and enrich metadata on every chunk (source, chunk index, etc.)
    4. Return new Document objects — one per chunk

Chunking strategy — Recursive Character Splitting:
    We try to split on natural boundaries in order of preference:
        1. Double newline  (\n\n)  → paragraph boundary (best)
        2. Single newline  (\n)     → line boundary
        3. Period + space  (". ")    → sentence boundary
        4. Single space    (" ")     → word boundary (last resort)
        5. Hard cut                  → character boundary (absolute fallback)

Design principles:
    - Input:  list[Document] from cleaner.py
    - Output: list[Document], one per chunk, with enriched metadata
    - Chunk size and overlap come from settings.py (single source of truth)
    - Every chunk knows its origin: source file, chunk index, char offset
    - Short documents that fit in one chunk are returned as-is (no splitting)
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
 
from app.config.settings import settings
from app.utils.logger import get_logger
 
logger = get_logger(__name__)
# Log lines → app.ingestion.chunker

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,           # from settings.py (default 500)
    chunk_overlap=settings.CHUNK_OVERLAP,     # from settings.py (default 100)
    separators=["\n\n", "\n", ". ", " ", ""], 
    length_function=len,                      # character count (not token count)
    is_separator_regex=False,                 # separators are plain strings
)
 
logger.debug(
    "Chunker initialised | chunk_size=%d | chunk_overlap=%d",
    settings.CHUNK_SIZE,
    settings.CHUNK_OVERLAP,
)

def chunk_text(text: str, metadata: dict = None) -> list[Document]:
    """
    Split a single cleaned text string into overlapping Document chunks.
 
    Each returned Document has:
        page_content  → the chunk text
        metadata      → source metadata + chunk_index + char_offset
 
    Why attach metadata to every chunk?
    - Traceability: Each chunk knows its source document and position, which is crucial for debugging and retrieval.
    - Enrichment: We can add more info later (e.g. chunk_total) without changing the overall structure.
 
    Args:
        text:     A cleaned document string from cleaner.py
        metadata: Optional dict of source info (e.g. {"source": "report.pdf"})
                  This is merged into every chunk's metadata.
 
    Returns:
        List of LangChain Document objects, one per chunk.
        If text is empty, returns an empty list.
 
    """
    if not text or not text.strip():
        logger.warning("chunk_text received empty text — returning []")
        return []
 
    base_metadata = metadata or {}
 
    # LangChain splits the text and wraps each chunk in a Document
    chunks: list[Document] = _splitter.create_documents(
        texts=[text],
        metadatas=[base_metadata],  # applied to all chunks from this text
    )
 
    # Enrich each chunk with its position information
    # LangChain doesn't add chunk_index by default so we add it here
    char_offset = 0
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)
        chunk.metadata["char_offset"] = char_offset # position of this chunk's start in the original text
        char_offset += len(chunk.page_content)
        
 
    logger.info(
        "Chunked document | %d chars → %d chunk(s) | source: %s",
        len(text),
        len(chunks),
        base_metadata.get("source", "unknown"),
    )
 
    return chunks
 
 
def chunk_documents(documents: list[dict]) -> list[Document]:
    """
    Chunk a batch of documents.
 
    Expects a list of dicts with:
        {
            "text":     str,   ← cleaned text from cleaner.py
            "metadata": dict,  ← source info (filename, url, type, etc.)
        }
 
    This is the function embedder.py will call — it takes the full
    batch output from cleaner.py and returns all chunks in one flat list.
 
    Args:
        documents: List of {"text": ..., "metadata": ...} dicts.
 
    Returns:
        Flat list of all Document chunks across all documents.
 
        docs = [
            {"text": clean_text_1, "metadata": {"source": "report.pdf"}},
            {"text": clean_text_2, "metadata": {"source": "transactions.csv"}},
        ]
        all_chunks = chunk_documents(docs)
    """
    if not documents:
        logger.warning("chunk_documents received empty list")
        return []
 
    all_chunks: list[Document] = []
 
    for i, doc in enumerate(documents):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
 
        if not text:
            logger.warning("Document %d has no text — skipping", i)
            continue
 
        chunks = chunk_text(text, metadata=metadata)
        all_chunks.extend(chunks)
 
    logger.info(
        "Batch chunking complete | %d document(s) → %d total chunk(s)",
        len(documents),
        len(all_chunks),
    )
 
    return all_chunks
 