"""
app/retrieval/bm25_index.py

Builds and caches a BM25 (keyword) search index over the same chunks
stored in ChromaDB, for use in hybrid retrieval (retriever.py's
retrieve_hybrid()).

Why BM25 alongside dense/semantic search?
    Embeddings are strong at matching MEANING but weak at matching EXACT
    terms — account numbers, specific dates, ticker symbols, precise
    dollar figures. BM25 is a classic keyword-frequency ranking
    algorithm (the same family that powers traditional search engines)
    and is strong at exactly what embeddings are weak at. Combining both
    catches queries either method alone would miss.

Position in the pipeline:
    chroma_store.py → [bm25_index.py] → retriever.py (retrieve_hybrid)

Design:
    - Lazy singleton, same pattern as chroma_store.get_vectorstore() and
      embedder.get_embedder() — nothing loads until get_bm25_retriever()
      is actually called, and it's built once, then reused.
    - reset_bm25_index() lets callers force a rebuild after new documents
      are ingested. The index is a snapshot at build time — it does NOT
      automatically pick up documents added to ChromaDB afterward.
"""

from langchain_community.retrievers import BM25Retriever

from app.vectorstore.chroma_store import get_all_documents
from app.config.settings import settings
from app.utils.helpers import timer
from app.utils.logger import get_logger

logger = get_logger(__name__)
# __name__ here is "app.retrieval.bm25_index"


_bm25_retriever: BM25Retriever | None = None


def get_bm25_retriever() -> BM25Retriever:
    """
    Return the BM25 retriever, building it on first call only.

    Returns:
        A LangChain BM25Retriever indexed over every chunk currently in
        ChromaDB at the time of the first call.

    Raises:
        ValueError: If the ChromaDB collection is empty — there's
            nothing to build a keyword index over. BM25Retriever itself
            errors confusingly on an empty corpus, so this catches it
            early with a clearer message pointing at the actual cause.
    """
    global _bm25_retriever

    if _bm25_retriever is not None:
        return _bm25_retriever

    docs = get_all_documents()

    if not docs:
        raise ValueError(
            "Cannot build BM25 index — the ChromaDB collection is empty. "
            "Run ingestion first (main.ingest())."
        )

    logger.info("Building BM25 index over %d chunk(s)...", len(docs))

    with timer("bm25 index build") as t:
        _bm25_retriever = BM25Retriever.from_documents(docs)
        _bm25_retriever.k = settings.RETRIEVAL_TOP_K

    logger.info("BM25 index built in %sms", t["elapsed_ms"])
    return _bm25_retriever


def reset_bm25_index() -> None:
    """
    Clear the cached BM25 index, forcing a rebuild on next use.

    Call this after re-ingesting documents (e.g. main.py's 'reingest'
    command) — otherwise the BM25 index keeps searching a stale snapshot
    from before the new documents were added. ChromaDB itself doesn't
    have this problem since similarity_search always queries live.
    """
    global _bm25_retriever
    _bm25_retriever = None
    logger.info("BM25 index cleared — will rebuild on next use.")