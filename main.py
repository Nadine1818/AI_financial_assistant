"""
main.py — Entry point for the Financial AI Assistant.

Run with:
    python main.py

What this file does:
    1. Ingest documents from data/raw/ into ChromaDB  (one-time or on demand)
    2. Start an interactive conversation loop
    3. For every question: retrieve → generate → verify
    4. Print the safe answer + sources to the terminal

This file contains NO business logic.
Every step delegates to the module that owns it:
    Ingestion  → loader, cleaner, chunker, embedder
    Retrieval  → retriever
    Generation → response_generator
    Validation → verifier
"""

import sys

from app.config.settings import settings
from app.utils.logger import get_logger
from app.utils.helpers import require_env_vars

# Ingestion pipeline
from app.ingestion.loader import load_directory
from app.ingestion.cleaner import clean_document
from app.ingestion.chunker import chunk_documents
from app.vectorstore.chroma_store import add_documents, collection_exists

# RAG + validation
from app.generation.response_generator import generate_with_history
from app.validation.verifier import verify

logger = get_logger(__name__)


# STARTUP CHECKS 

def _check_env() -> None:
    """
    Fail loudly at startup if required environment variables are missing.

    Better to crash here with a clear message than to crash silently
    inside an LLM call three steps into the pipeline.
    """
    require_env_vars("OPENAI_API_KEY")
    logger.info("Environment checks passed.")


# INGESTION PIPELINE

def ingest(force: bool = False) -> None:
    """
    Load, clean, chunk, and embed all documents in data/raw/.

    Skipped automatically if the ChromaDB collection already exists,
    unless force=True is passed (useful when you add new documents).

    Args:
        force: Re-ingest even if the collection already exists.
    """
    if collection_exists() and not force:
        logger.info(
            "ChromaDB collection '%s' already exists — skipping ingestion. "
            "Pass force=True to re-ingest.",
            settings.CHROMA_COLLECTION_NAME,
        )
        return

    logger.info("Starting ingestion from: %s", settings.RAW_DATA_DIR)

    # Step 1: Load raw files
    raw_docs = load_directory(settings.RAW_DATA_DIR, recursive=False)

    if not raw_docs:
        logger.warning(
            "No documents found in %s — add .pdf, .csv, .txt, or .json files "
            "to data/raw/ before running.",
            settings.RAW_DATA_DIR,
        )
        return

    logger.info("Loaded %d raw document(s).", len(raw_docs))

    # Step 2: Clean 
    # clean_document() operates on strings, so we clean each doc's content
    # and put it back — metadata is preserved unchanged.
    for doc in raw_docs:
        doc.content = clean_document(doc.content)

    # Filter out any documents that became empty after cleaning
    raw_docs = [d for d in raw_docs if d.content.strip()]
    logger.info("%d document(s) remain after cleaning.", len(raw_docs))

    # Step 3: Chunk 
    # chunk_documents() expects list[dict] with "text" and "metadata" keys
    # (matching the interface your chunker.py defines)
    doc_dicts = [
        {"text": doc.content, "metadata": doc.metadata}
        for doc in raw_docs
    ]
    chunks = chunk_documents(doc_dicts)
    logger.info("Produced %d chunk(s) across all documents.", len(chunks))

    # Step 4: Embed + store in ChromaDB
    add_documents(chunks)
    logger.info("Ingestion complete. ChromaDB is ready.")


# CONVERSATION LOOP 

def chat() -> None:
    """
    Run an interactive terminal conversation loop.

    Maintains chat_history so multi-turn follow-up questions work
    via generate_with_history() and CONDENSE_PROMPT.

    Commands:
        'quit' or 'exit'  → end the session
        'reset'           → clear conversation history
        'reingest'        → re-run ingestion (pick up new documents)
    """
    print("\n" + "=" * 60)
    print("  Financial AI Assistant")
    print(f"  Model : {settings.LLM_MODEL}")
    print(f"  Docs  : {settings.RAW_DATA_DIR}")
    print("=" * 60)
    print("  Commands: 'quit' | 'reset' | 'reingest'")
    print("=" * 60 + "\n")

    # chat_history holds (human_message, ai_message) tuples — oldest first.
    # Passed to generate_with_history() on every turn.
    chat_history: list[tuple[str, str]] = []

    while True:
        # Get user input 
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C or Ctrl+D → exit gracefully
            print("\nGoodbye.")
            break

        if not question:
            continue

        # Handle commands before invoking the LLM
        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if question.lower() == "reset":
            chat_history.clear()
            print("── Conversation history cleared. ──\n")
            continue

        if question.lower() == "reingest":
            print("── Re-ingesting documents... ──")
            ingest(force=True)
            print("── Done. ──\n")
            continue

        # Generate response with RAG + history
        try:
            gen_result = generate_with_history(
                question=question,
                chat_history=chat_history,
            )
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            print(f"\nAssistant: Sorry, something went wrong: {exc}\n")
            continue

        # Verify 
        try:
            ver_result = verify(gen_result)
        except Exception as exc:
            logger.error("Verification failed: %s", exc)
            # Verification failure is non-fatal — show the unverified answer
            # with a warning rather than crashing the session.
            ver_result = None

        # Print answer + sources + verdict
        if ver_result:
            answer  = ver_result.safe_answer
            sources = ver_result.sources
            verdict = ver_result.verdict
        else:
            answer  = gen_result.answer
            sources = gen_result.sources
            verdict = "UNVERIFIED"

        print(f"\nAssistant: {answer}")

        if sources:
            print(f"\n  Sources : {', '.join(sources)}")

        print(f"  Verdict : {verdict}\n")

        # Update history 
        # Store the original question (not the condensed one) so the history
        # reads naturally when shown back to the user or the condense LLM.
        chat_history.append((question, answer))


# ENTRY POINT 

def main() -> None:
    """
    Application entry point. Called when you run `python main.py`.

    Order:
        1. Check environment variables
        2. Ingest documents (skipped if ChromaDB already populated)
        3. Start the conversation loop
    """
    _check_env()
    ingest()
    chat()


if __name__ == "__main__":
    main()