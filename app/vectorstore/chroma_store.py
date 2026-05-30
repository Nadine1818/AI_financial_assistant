"""
app/vectorstore/chroma_store.py

Persists embedded Document chunks to ChromaDB and exposes search.

Position in the pipeline:
    embedder.py → [chroma_store.py] ← retriever.py


ChromaDB is a local vector database. It stores your embedded chunks on
disk so you don't re-embed documents every time your app restarts.

What this file does:
    1. get_vectorstore()   → connect to (or create) the ChromaDB collection
    2. add_documents()     → persist embedded chunks to ChromaDB
    3. similarity_search() → find top-k chunks closest to a query
    4. delete_collection() → wipe and reset (useful during development)

Design:
    - One singleton vectorstore instance, built lazily on first call
    - All config (path, collection name) comes from settings.py
    - embedder._embedder is passed in so Chroma can embed queries internally
"""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config.settings import settings
from app.ingestion.embedder import _embedder
from app.utils.helpers import timer
from app.utils.logger import get_logger

logger = get_logger(__name__)
# Log lines → app.vectorstore.chroma_store

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """
    Return the ChromaDB vectorstore, creating it on first call.

    Lazy singleton pattern:
        We don't connect to ChromaDB at import time because:
        1. The data directory may not exist yet on first run
        2. Tests may want to inject a different path
        3. Import-time side effects are hard to reason about

    On first call:
        - Creates settings.CHROMA_PATH directory if it doesn't exist
        - Opens the existing collection if it exists on disk
        - Creates a new empty collection if it doesn't

    On subsequent calls:
        - Returns the already-open connection (fast, no disk I/O)

    Returns:
        LangChain Chroma instance connected to local ChromaDB.
    """
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    # Ensure the persistence directory exists
    # parents=True → create intermediate dirs (e.g. data/processed/)
    # exist_ok=True → don't error if it already exists
    settings.CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Connecting to ChromaDB | path: %s | collection: %s",
        settings.CHROMA_PATH,
        settings.CHROMA_COLLECTION_NAME,
    )

    with timer("chroma connect") as t:
        _vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=_embedder,   # used to embed queries at search time
            persist_directory=str(settings.CHROMA_PATH),
        )

    logger.info("ChromaDB ready in %sms", t["elapsed_ms"])
    return _vectorstore

def add_documents(chunks: list[Document]) -> None:
    """
    Persist a list of embedded Document chunks to ChromaDB.

    What happens internally:
        1. LangChain extracts page_content from each Document
        2. Extracts metadata (source, chunk_index, char_offset, etc.)
        3. Uses the stored embeddings (already in metadata from embedder.py)
           OR re-embeds if not present (slightly slower)
        4. Writes everything to the ChromaDB collection on disk

    Deduplication:
        ChromaDB uses document IDs to avoid storing duplicates.
        We generate a deterministic ID from source + chunk_index so
        re-ingesting the same file doesn't create duplicate entries.
        Format: "annual_report.pdf::chunk_3"

    Args:
        chunks: list[Document] from embedder.embed_documents()
                Each chunk must have page_content and metadata.
    """
    if not chunks:
        logger.warning("add_documents called with empty list — nothing stored")
        return

    vs = get_vectorstore()

    # Build deterministic IDs for deduplication
    # If source is missing, fall back to chunk index alone
    ids = [
        f"{chunk.metadata.get('source', 'unknown')}::chunk_{chunk.metadata.get('chunk_index', i)}"
        for i, chunk in enumerate(chunks)
    ]

    logger.info("Adding %d chunk(s) to ChromaDB...", len(chunks))

    with timer("chroma add") as t:
        vs.add_documents(documents=chunks, ids=ids)

    logger.info(
        "Stored %d chunk(s) in ChromaDB in %sms",
        len(chunks),
        t["elapsed_ms"],
    )


def similarity_search(
    query: str,
    top_k: int | None = None,
    filter: dict | None = None,
) -> list[Document]:
    """
    Find the top-k Document chunks most semantically similar to a query.

    This is called by retriever.py during the RAG pipeline:
        user question → embed → similarity_search → top chunks → LLM

    How it works:
        1. Embeds the query string using _embedder (same model as ingestion)
        2. Computes cosine similarity between query vector and all stored vectors
        3. Returns the top_k closest matches as Document objects
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    k = top_k or settings.RETRIEVAL_TOP_K
    vs = get_vectorstore()

    logger.debug(
        "Similarity search | query: %r | top_k: %d | filter: %s",
        query[:60],
        k,
        filter,
    )

    with timer("chroma search") as t:
        results = vs.similarity_search(
            query=query,
            k=k,
            filter=filter,  # None = no filter = search entire collection
        )

    logger.info(
        "Retrieved %d chunk(s) in %sms for query: %r",
        len(results),
        t["elapsed_ms"],
        query[:60],
    )

    return results


def similarity_search_with_scores(
    query: str,
    top_k: int | None = None,
    filter: dict | None = None,
) -> list[tuple[Document, float]]:
    """
    Same as similarity_search but also returns relevance scores.

    Returns list of (Document, score) tuples where score is cosine distance:
        0.0 → identical meaning (perfect match)
        1.0 → completely unrelated

    Lower score = more relevant. Counterintuitive but standard for distance.

    Why use this over similarity_search?
        When you want to apply a relevance threshold — e.g. only use chunks
        with score < 0.3 (highly relevant), discard the rest.
        This is important for financial accuracy: better to say "I don't know"
        than to hallucinate from a weakly-relevant chunk.

    Used by: verifier.py to filter low-confidence retrievals.
    """
    k = top_k or settings.RETRIEVAL_TOP_K
    vs = get_vectorstore()

    logger.debug(
        "Similarity search with scores | query: %r | top_k: %d",
        query[:60],
        k,
    )

    with timer("chroma search scored") as t:
        results = vs.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    # Log score distribution — useful for tuning retrieval quality
    if results:
        scores = [score for _, score in results]
        logger.debug(
            "Score range: %.4f (best) → %.4f (worst)",
            min(scores),
            max(scores),
        )

    logger.info(
        "Retrieved %d chunk(s) with scores in %sms",
        len(results),
        t["elapsed_ms"],
    )

    return results


def get_collection_stats() -> dict:
    """
    Return basic stats about the current ChromaDB collection.

    Useful for:
    - Sanity checking after ingestion ("did my documents actually get stored?")
    - Monitoring collection size over time
    - Debugging empty retrieval results

    Returns:
        {
            "collection_name": "financial_docs",
            "document_count": 142,
            "persist_path": "/path/to/chroma_db"
        }
    """
    vs = get_vectorstore()
    count = vs._collection.count()

    stats = {
        "collection_name": settings.CHROMA_COLLECTION_NAME,
        "document_count": count,
        "persist_path": str(settings.CHROMA_PATH),
    }

    logger.info("Collection stats: %s", stats)
    return stats


def delete_collection() -> None:
    """
    Wipe the entire ChromaDB collection and reset the singleton.

    USE WITH CAUTION — this deletes all stored vectors permanently.

    Used when:
    - During development when you want to re-ingest from scratch
    - After changing the embedding model (old vectors are incompatible
      with new model dimensions — you MUST delete and re-embed)
    - In tests to start with a clean slate

    After calling this, the next call to get_vectorstore() will create
    a fresh empty collection.
    """
    global _vectorstore

    vs = get_vectorstore()
    vs.delete_collection()
    _vectorstore = None  # reset singleton so next call creates fresh

    logger.warning(
        "ChromaDB collection '%s' deleted — all vectors wiped",
        settings.CHROMA_COLLECTION_NAME,
    )