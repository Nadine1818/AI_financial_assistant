"""
app/ingestion/embedder.py

Converts Document chunks into vector embeddings and stores them.

Position in the pipeline:
    loader.py → cleaner.py → chunker.py → [embedder.py] → vectorstore

Embedding model: sentence-transformers/all-MiniLM-L6-v2
    - Free, runs locally (no API key, no cost, no data leaving your machine)
    - 384-dimensional vectors (small = fast)
    - Trained on 1B+ sentence pairs — excellent general semantic understanding
    - Widely used in production RAG systems

"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.config.settings import settings
from app.utils.helpers import timer, estimate_tokens
from app.utils.logger import get_logger

logger = get_logger(__name__)
# Log lines → app.ingestion.embedder


# ---------------------------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------------------------

# Override the default model from settings with our free local model.
# settings.EMBEDDING_MODEL defaults to "text-embedding-3-small" (OpenAI).
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Model kwargs passed to the underlying sentence-transformers library.
# device="cpu" → run on CPU (no GPU required for this small model)
_MODEL_KWARGS = {"device": "cpu"}

# Encoding kwargs control the embedding computation itself.
# normalize_embeddings=True → L2-normalizes vectors to unit length.
# Why normalize? Cosine similarity (used by ChromaDB) is equivalent to
# dot product on normalized vectors — faster and more stable retrieval.
_ENCODE_KWARGS = {"normalize_embeddings": True}


# ---------------------------------------------------------------------------
# EMBEDDER SINGLETON
# Built once at module load, reused for every embed call.
# Loading a transformer model takes ~1-2 seconds and ~90MB RAM.
# We do it once here so the first embed() call isn't slow.
# ---------------------------------------------------------------------------

def _build_embedder() -> HuggingFaceEmbeddings:
    """
    Load the embedding model and return a LangChain embedder instance.
    """
    logger.info("Loading embedding model: %s", _EMBEDDING_MODEL)

    with timer("embedding model load") as t:
        embedder = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL,
            model_kwargs=_MODEL_KWARGS,
            encode_kwargs=_ENCODE_KWARGS,
        )

    logger.info("Embedding model loaded in %sms", t["elapsed_ms"])
    return embedder


# Module-level singleton — loaded once, reused everywhere
_embedder = _build_embedder()

def embed_documents(chunks: list[Document]) -> list[Document]:
    """
    Embed a list of Document chunks and attach vectors to their metadata.

    Why attach embeddings to metadata instead of returning raw vectors?
        Keeping embeddings inside the Document object means downstream code
        (chroma_store.py) receives one clean list of Documents and doesn't
        need to zip/align a separate list of vectors. Less room for error.

    What happens internally (LangChain handles this):
        1. Extract page_content from each Document
        2. Batch the texts (sends multiple at once, much faster than one-by-one)
        3. Run each text through the transformer model
        4. Return a list of vectors (one per document)

    Args:
        chunks: list[Document] from chunker.py — each has page_content + metadata

    Returns:
        The same list of Documents with "embedding" added to each metadata dict.
        Shape: chunks[i].metadata["embedding"] = [0.12, -0.45, ...] (384 floats)

    """
    if not chunks:
        logger.warning("embed_documents received empty list — returning []")
        return []

    logger.info("Embedding %d chunk(s)...", len(chunks))

    # Extract raw text from each Document for the embedding model
    texts = [chunk.page_content for chunk in chunks]

    # Estimate token cost for awareness (free model, but good habit to track)
    total_tokens = sum(estimate_tokens(t) for t in texts)
    logger.debug("Estimated token count for batch: ~%d tokens", total_tokens)

    with timer("embed batch") as t:
        # embed_documents returns list[list[float]] — one vector per text
        vectors: list[list[float]] = _embedder.embed_documents(texts)

    logger.info(
        "Embedded %d chunk(s) in %sms | vector dimensions: %d",
        len(chunks),
        t["elapsed_ms"],
        len(vectors[0]) if vectors else 0,
    )

    # Attach each vector to its Document's metadata
    for chunk, vector in zip(chunks, vectors):
        chunk.metadata["embedding"] = vector

    return chunks


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string for retrieval.

    This is used at query time (not ingestion time):
        user asks a question → embed_query() → search vector store → retrieve chunks

    Why a separate function from embed_documents?
        Some embedding models use different prompts for queries vs documents
        to improve retrieval quality (asymmetric embedding).
        LangChain's .embed_query() handles this automatically — for models
        that don't distinguish, it's identical to embed_documents.

    Args:
        query: The user's question as a plain string.

    Returns:
        A single embedding vector: list[float] with 384 dimensions.

    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    logger.debug("Embedding query: %r", query[:80])  # log first 80 chars only

    with timer("embed query") as t:
        vector = _embedder.embed_query(query)

    logger.debug(
        "Query embedded in %sms | dimensions: %d",
        t["elapsed_ms"],
        len(vector),
    )

    return vector


def get_embedding_dimensions() -> int:
    """
    Return the vector dimensions produced by the current embedding model.

    Useful for:
    - Validating vector store configuration (ChromaDB needs to know dimensions)
    - Logging/debugging
    - Switching models and catching dimension mismatches early

    all-MiniLM-L6-v2 → 384 dimensions
    """
    # Embed a dummy string to get the vector shape
    # We cache this so it only runs once (the result is stored in the singleton)
    if not hasattr(_embedder, "_cached_dimensions"):
        dummy_vector = _embedder.embed_query("dimension check")
        _embedder._cached_dimensions = len(dummy_vector)
        logger.debug("Embedding dimensions: %d", _embedder._cached_dimensions)

    return _embedder._cached_dimensions