"""
retrieval/retriever.py — Semantic retrieval for the Financial AI Assistant.

Position in the pipeline:
    chroma_store.py → [retriever.py] → response_generator.py

The retriever is the bridge between the vector store and the LLM.
Its job: given a user's question, find the most relevant document chunks
and return them in a form the generator can use.

This file exposes two public functions:
    retrieve(query)         → list[Document]  (standard RAG path)
    retrieve_with_scores(query) → list[tuple[Document, float]]  (scored path)
"""

from langchain_core.documents import Document

from app.vectorstore.chroma_store import (
    similarity_search,
    similarity_search_with_scores,
)
from app.config.settings import settings
from app.utils.logger import get_logger
from app.utils.helpers import truncate, estimate_tokens

logger = get_logger(__name__)

# ── Relevance threshold ────────────────────────────────────────────────────────
# Cosine distance scores from ChromaDB: 0.0 = identical, 1.0 = unrelated.
# Chunks with a score ABOVE this threshold are considered too weakly related
# to be useful — including them would inject noise into the LLM's context.
#
# 0.45 is a reasonable starting point for financial prose.
RELEVANCE_THRESHOLD = 0.45


def retrieve(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> list[Document]:
    """
    Retrieve the most relevant document chunks for a user query.

    This is the standard path used by response_generator.py.
    It returns clean Document objects ready to be formatted into a prompt.

    What happens here:
        1. Call similarity_search_with_scores so we can apply a threshold
        2. Filter out chunks whose score is above RELEVANCE_THRESHOLD
        3. Log a warning if nothing passes the threshold (signals a gap
           in your document coverage — important to know in production)
        4. Return just the Document objects (scores discarded)

    Args:
        query:         The user's question, exactly as typed.
        top_k:         How many chunks to retrieve. Defaults to
                       settings.RETRIEVAL_TOP_K (set in .env).
        source_filter: Optional filename to restrict search to a single
                       document. e.g. "apple_10k_2023.pdf"
                       Passed as a ChromaDB metadata filter.

    Returns:
        list[Document] — ordered by relevance (most relevant first).
        May be empty if no chunks pass the relevance threshold.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty.")

    k = top_k or settings.RETRIEVAL_TOP_K

    # Build optional metadata filter for ChromaDB
    # ChromaDB filter syntax: {"key": "value"} for exact match
    chroma_filter = {"source": source_filter} if source_filter else None

    logger.info(
        "Retrieving chunks | query: %r | top_k: %d | source: %s",
        truncate(query, 80),
        k,
        source_filter or "all",
    )

    # Use scored search so we can apply the relevance threshold
    scored_results = similarity_search_with_scores(
        query=query,
        top_k=k,
        filter=chroma_filter,
    )

    # Threshold filtering 
    # Keep only chunks where the distance score is below our threshold.
    # Remember: lower score = more similar = more relevant.
    before = len(scored_results)
    filtered = [
        (doc, score)
        for doc, score in scored_results
        if score <= RELEVANCE_THRESHOLD
    ]
    after = len(filtered)

    if before > 0 and after == 0:
        # Nothing passed the threshold — this is important signal.
        # It means either:
        #   a) The question is outside your document coverage
        #   b) The query phrasing doesn't match how docs are written
        #   c) Your threshold is too strict
        logger.warning(
            "All %d retrieved chunks failed relevance threshold (%.2f). "
            "Query may be out of scope: %r",
            before,
            RELEVANCE_THRESHOLD,
            truncate(query, 80),
        )
        return []

    if after < before:
        logger.debug(
            "Filtered %d → %d chunks after applying threshold %.2f",
            before,
            after,
            RELEVANCE_THRESHOLD,
        )

    # Extract just the Document objects, already ordered by relevance
    docs = [doc for doc, _ in filtered]

    _log_retrieved_chunks(docs)
    return docs


def retrieve_with_scores(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> list[tuple[Document, float]]:
    """
    Same as retrieve() but returns (Document, score) tuples.

    Useي when the caller needs to know HOW relevant each chunk is,
    not just which chunks were found. Used by:
        - verifier.py: to flag low-confidence answers
        - notebooks: for retrieval quality analysis and threshold tuning

    Returns:
        list of (Document, float) tuples, ordered best → worst score.
        Threshold filtering still applies — weak matches are excluded.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty.")

    k = top_k or settings.RETRIEVAL_TOP_K
    chroma_filter = {"source": source_filter} if source_filter else None

    scored_results = similarity_search_with_scores(
        query=query,
        top_k=k,
        filter=chroma_filter,
    )

    # Apply the same threshold as retrieve()
    filtered = [
        (doc, score)
        for doc, score in scored_results
        if score <= RELEVANCE_THRESHOLD
    ]

    logger.info(
        "retrieve_with_scores: %d/%d chunks passed threshold",
        len(filtered),
        len(scored_results),
    )

    return filtered


def format_context(docs: list[Document]) -> str:
    """
    Format a list of retrieved chunks into a single context string
    ready to be injected into an LLM prompt.

    Why formatting belongs in the retriever (not the generator):
        The retriever knows the structure of the chunks — their metadata,
        their source, their order. Formatting here keeps the generator
        clean and focused on prompt logic.

    Output format (one block per chunk):
        [Source: filename.pdf | Chunk 3]

    The source label is important for financial contexts:
        - Lets the LLM cite where information came from
        - Helps verifier.py check if claims are grounded in a real source

    Args:
        docs: list[Document] from retrieve()

    Returns:
        A single formatted string to insert into the prompt as {context}.
        Returns empty string if docs is empty.
    """
    if not docs:
        logger.warning("format_context called with empty document list")
        return ""

    blocks = []
    total_tokens = 0

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        content = doc.page_content.strip()

        block = f"[Source: {source} | Chunk {chunk_index}]\n{content}"
        blocks.append(block)

        chunk_tokens = estimate_tokens(content)
        total_tokens += chunk_tokens
        logger.debug(
            "  Chunk %s from %s → ~%d tokens",
            chunk_index,
            source,
            chunk_tokens,
        )

    logger.info(
        "Context built: %d chunks | ~%d total tokens",
        len(blocks),
        total_tokens,
    )

    # Join blocks with a blank line separator for readability
    return "\n\n".join(blocks)


# Internal helpers 

def _log_retrieved_chunks(docs: list[Document]) -> None:
    """
    Log a summary of retrieved chunks.
    Kept separate to keep retrieve() readable.
    """
    logger.info("Retrieved %d chunk(s):", len(docs))
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        preview = truncate(doc.page_content.replace("\n", " "), max_chars=80)
        logger.debug("  [%d] %s::chunk_%s → %r", i + 1, source, chunk_idx, preview)