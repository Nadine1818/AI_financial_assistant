# Orchestrates the full RAG turn — the brain of the financial AI assistant.
"""
Position in the pipeline:
    retriever.py + prompts.py + llm.py → [response_generator.py] → verifier.py

Single responsibility:
    Wire retrieval → prompt building → LLM invocation into one clean function.
    This module owns the REQUEST-RESPONSE CYCLE. It does not own:
        - How chunks are retrieved          (retriever.py)
        - What the prompts say              (prompts.py)
        - How the LLM is called            (llm.py)
        - Whether the answer is correct     (verifier.py)

What lives here:
    generate()          → standard single-turn RAG response
    generate_with_history() → multi-turn: condense follow-up → RAG

Flow for generate():
    1. Retrieve relevant chunks for the query         (retriever.retrieve)
    2. If nothing retrieved → use NO_CONTEXT_PROMPT   (safe fallback)
    3. Format chunks into a context string            (retriever.format_context)
    4. Build messages from RAG_PROMPT                 (prompts.get_prompt)
    5. Invoke the LLM                                 (llm.invoke)
    6. Return a GenerationResult (answer + metadata)

Flow for generate_with_history():
    1. Condense the follow-up + history → standalone question (CONDENSE_PROMPT)
    2. Run the condensed question through generate()
    (All the RAG logic is reused — no duplication)

Why return a GenerationResult dataclass instead of a plain string?
    verifier.py needs the context alongside the answer to fact-check it.
    Callers (main.py, API layer) need the source list for citations.
    A plain string loses all of that. The dataclass keeps everything together
    without coupling the generator to the verifier's internals.
"""

from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.retrieval.retriever import retrieve, retrieve_hybrid, format_context
from app.generation.prompts import get_prompt
from app.generation.llm import invoke
from app.config.settings import settings
from app.utils.helpers import timer, truncate_text
from app.utils.logger import get_logger

logger = get_logger(__name__)
# __name__ here is "app.generation.response_generator"


# GENERATION RESULT 
# The contract between response_generator.py and its callers.
#
# Why a dataclass instead of a plain dict?
#   - Type-safe: callers know exactly what fields exist
#   - verifier.py receives `result.context` and `result.answer` cleanly
#   - main.py can display `result.sources` for citations without parsing
#   - Easy to extend (add fields) without breaking existing callers

@dataclass
class GenerationResult:
    """
    The output of a single RAG turn.

    Attributes:
        answer:        The LLM's response string, ready to show the user.
        context:       The formatted context string that was injected into
                       the prompt. verifier.py uses this to fact-check.
        sources:       List of source filenames cited in the context.
                       Extracted from chunk metadata for easy display.
        question:      The question that was actually sent to the LLM.
                       For multi-turn, this is the condensed question.
        retrieved_chunks: How many chunks were retrieved (0 = no context found).
        used_fallback: True if no context was found and NO_CONTEXT_PROMPT
                       was used instead of RAG_PROMPT.
    """
    answer: str
    context: str
    sources: list[str]
    question: str
    retrieved_chunks: int
    used_fallback: bool = False
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.answer[:80].replace("\n", " ")
        return (
            f"GenerationResult("
            f"chunks={self.retrieved_chunks}, "
            f"fallback={self.used_fallback}, "
            f"sources={self.sources}, "
            f"answer={preview!r})"
        )


# HELPERS 

def _extract_sources(docs: list) -> list[str]:
    """
    Pull unique source filenames from a list of retrieved Document objects.

    Uses dict.fromkeys() to deduplicate while preserving insertion order
    (important: most-relevant source stays first).

    Args:
        docs: list[Document] from retriever.retrieve()

    Returns:
        Deduplicated list of source filenames, e.g. ["statement.pdf", "q3.pdf"]
    """
    sources = [
        doc.metadata.get("source", "unknown")
        for doc in docs
    ]
    # dict.fromkeys preserves order and removes duplicates
    return list(dict.fromkeys(sources))


# GENERATE (SINGLE TURN)

def generate(
    question: str,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> GenerationResult:
    """
    Run a full RAG turn for a single question and return the result.

    This is the main function. Everything else in this module supports it.

    Steps:
        1. Retrieve relevant chunks from ChromaDB
        2. If zero chunks pass the relevance threshold → fallback path
        3. Format chunks into a context string
        4. Build prompt messages (RAG_PROMPT or NO_CONTEXT_PROMPT)
        5. Invoke the LLM
        6. Return a GenerationResult with everything verifier.py needs

    Args:
        question:      The user's question, exactly as typed.
        top_k:         Number of chunks to retrieve. Defaults to
                       settings.RETRIEVAL_TOP_K.
        source_filter: Restrict retrieval to a single source file.
                       e.g. "bank_statement_jan.pdf"

    Returns:
        GenerationResult with answer, context, sources, and metadata.

    Raises:
        ValueError: If question is empty.
        RuntimeError: If LLM invocation fails after all retries.

    Example:
        result = generate("What was my total spend in January?")
        print(result.answer)
        print(result.sources)
    """
    if not question or not question.strip():
        raise ValueError("generate() received an empty question.")

    logger.info(
        "Starting RAG generation | question: %r",
        truncate_text(question, 100),
    )

    # Step 1: Retrieve relevant chunks for the question
    # Uses hybrid (dense + BM25) retrieval if settings.USE_HYBRID_RETRIEVAL
    # is True, otherwise the original dense-only path. Defaults to dense-
    # only (False) so this is an explicit opt-in, not a silent behavior
    # change — see app/retrieval/retriever.py for what each does
    # differently (retrieve_hybrid() does NOT apply RELEVANCE_THRESHOLD;
    # see that file's docstring for why).
    with timer("retrieval") as t_retrieve:
        if settings.USE_HYBRID_RETRIEVAL:
            docs = retrieve_hybrid(
                query=question,
                top_k=top_k,
                source_filter=source_filter,
            )
        else:
            docs = retrieve(
                query=question,
                top_k=top_k,
                source_filter=source_filter,
            )

    logger.info(
        "Retrieval complete | %d chunk(s) retrieved | elapsed=%sms",
        len(docs),
        t_retrieve["elapsed_ms"],
    )

    # Step 2: No-context fallback 
    # If the retriever found nothing above the relevance threshold,
    # we must NOT pass an empty context to RAG_PROMPT — the LLM will
    # hallucinate an answer from general knowledge.
    # Instead, NO_CONTEXT_PROMPT tells the LLM to honestly say it couldn't
    # find anything and suggest what document the user might upload.
    if not docs:
        logger.warning(
            "No relevant chunks found for question: %r — using fallback prompt",
            truncate_text(question, 100),
        )

        fallback_prompt   = get_prompt("no_context")
        fallback_messages = fallback_prompt.format_messages(question=question)

        with timer("llm_fallback") as t_llm:
            answer = invoke(fallback_messages)

        logger.info(
            "Fallback response generated | elapsed=%sms",
            t_llm["elapsed_ms"],
        )

        return GenerationResult(
            answer=answer,
            context="",
            sources=[],
            question=question,
            retrieved_chunks=0,
            used_fallback=True,
            metadata={
                "retrieval_ms": t_retrieve["elapsed_ms"],
                "llm_ms": t_llm["elapsed_ms"],
            },
        )

    # Step 3: Format context string from retrieved chunks
    # format_context() joins chunks into labelled blocks:
    #   [Source: statement.pdf | Chunk 2]
    #   <chunk text>
    context = format_context(docs)
    sources = _extract_sources(docs)

    logger.debug(
        "Context built | %d chars | sources: %s",
        len(context),
        sources,
    )

    # Step 4: Build prompt messages for RAG
    # get_prompt("rag") returns the RAG_PROMPT ChatPromptTemplate.
    # .format_messages() validates that {context} and {question} are both
    # provided — raises KeyError immediately if either is missing.
    rag_prompt = get_prompt("rag")
    messages   = rag_prompt.format_messages(
        context=context,
        question=question,
    )

    logger.debug(
        "Prompt built | %d message(s) | total chars ≈ %d",
        len(messages),
        sum(len(m.content) for m in messages),
    )

    # Step 5: Invoke LLM
    with timer("llm_invoke") as t_llm:
        answer = invoke(messages)

    logger.info(
        "Generation complete | elapsed=%sms | sources: %s",
        t_llm["elapsed_ms"],
        sources,
    )

    # Step 6: Return result
    return GenerationResult(
        answer=answer,
        context=context,
        sources=sources,
        question=question,
        retrieved_chunks=len(docs),
        used_fallback=False,
        metadata={
            "retrieval_ms": t_retrieve["elapsed_ms"],
            "llm_ms": t_llm["elapsed_ms"],
            "model": settings.LLM_MODEL,
        },
    )


# GENERATE WITH HISTORY (MULTI-TURN) 

def generate_with_history(
    question: str,
    chat_history: list[tuple[str, str]],
    top_k: int | None = None,
    source_filter: str | None = None,
) -> GenerationResult:
    """
    Multi-turn RAG: condense a follow-up question then run generate().

    After condensing, the question is passed straight to generate() —
    all RAG logic is reused, no duplication.

    Args:
        question:     The user's latest follow-up question.
        chat_history: List of (human_message, ai_message) string tuples,
                      oldest first. Each tuple = one prior exchange.
                      e.g. [("What was my balance?", "Your balance was £2,400.")]
        top_k:        Passed through to generate().
        source_filter: Passed through to generate().

    Returns:
        GenerationResult — same shape as generate(), with the condensed
        question stored in result.question for transparency.

    Example:
        history = [
            ("What was my balance in January?", "Your balance was £2,400."),
        ]
        result = generate_with_history("What about February?", history)
        print(result.question)  # "What was my account balance in February?"
        print(result.answer)
    """
    if not question or not question.strip():
        raise ValueError("generate_with_history() received an empty question.")

    logger.info(
        "Multi-turn generation | follow-up: %r | history turns: %d",
        truncate_text(question, 80),
        len(chat_history),
    )

    # Step 1: Build LangChain message history 
    # CONDENSE_PROMPT uses MessagesPlaceholder(variable_name="chat_history").
    # LangChain expects a list of BaseMessage objects, not raw strings.
    # We convert the (human, ai) tuples here.
    lc_history: list[BaseMessage] = []
    for human_msg, ai_msg in chat_history:
        lc_history.append(HumanMessage(content=human_msg))
        lc_history.append(AIMessage(content=ai_msg))

    # Step 2: Condense the follow-up into a standalone question 
    condense_prompt   = get_prompt("condense")
    condense_messages = condense_prompt.format_messages(
        chat_history=lc_history,
        question=question,
    )

    logger.debug("Condensing follow-up question with %d history messages", len(lc_history))

    with timer("condense") as t_condense:
        condensed_question = invoke(condense_messages).strip()

    logger.info(
        "Question condensed | '%s' → '%s' | elapsed=%sms",
        truncate_text(question, 60),
        truncate_text(condensed_question, 60),
        t_condense["elapsed_ms"],
    )

    # Step 3: Run standard RAG on the condensed question
    # generate() handles everything from retrieval through LLM invocation.
    result = generate(
        question=condensed_question,
        top_k=top_k,
        source_filter=source_filter,
    )

    # Record the condensing latency in metadata for observability
    result.metadata["condense_ms"] = t_condense["elapsed_ms"]
    result.metadata["original_question"] = question

    return result