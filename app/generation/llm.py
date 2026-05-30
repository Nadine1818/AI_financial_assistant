# Owns the LLM client — initialisation, configuration, and invocation.
"""
Position in the pipeline:
    prompts.py → [llm.py] → response_generator.py

Single responsibility:
    This file does ONE thing: take a list of formatted messages and return
    the LLM's response string. Nothing about prompts, retrieval, or validation
    lives here — that is the job of the modules above and below it.

Why isolate the LLM client in its own module?
    - Swapping models (gpt-4o-mini → gpt-4o → Claude) means changing ONE file
    - Retry logic, timeout config, and cost tracking live in one place
    - response_generator.py stays clean — it just calls invoke()
    - Easy to mock in tests: patch app.generation.llm.invoke

What lives here:
    _llm          → the singleton ChatOpenAI instance (built once at import)
    invoke()      → sends messages, returns a string, handles retries
    invoke_json() → same but parses the response as JSON (for guardrail prompt)
    get_llm()     → exposes the raw client when LangChain chains need it

LangChain model used: ChatOpenAI
    - Maps directly to the OpenAI chat completions API
    - Accepts list[BaseMessage] (SystemMessage, HumanMessage, AIMessage)
    - Returns an AIMessage whose .content is the response string
    - Integrates natively with ChatPromptTemplate.format_messages()
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration

from app.config.settings import settings
from app.utils.helpers import retry, safe_parse_json, timer
from app.utils.logger import get_logger

logger = get_logger(__name__)
# __name__ here is "app.generation.llm"


# LLM CLIENT (SINGLETON)
# Built once at module import. Every call to invoke() reuses this instance.
#
# Why a singleton?
#   ChatOpenAI holds an HTTP connection pool internally. Recreating it on
#   every request wastes time and connections. One instance shared across
#   the app is the standard pattern.
#
# Settings mapped from settings.py (single source of truth):
#   model          → LLM_MODEL        (default: "gpt-4o-mini")
#   temperature    → LLM_TEMPERATURE  (default: 0.0 — deterministic)
#   max_tokens     → LLM_MAX_TOKENS   (default: 1024)
#
# temperature=0.0 is critical for a financial assistant.
# At temperature 0, the model always picks the highest-probability token.
# This makes outputs reproducible and eliminates creative "hallucination"
# that higher temperatures encourage.

_llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    max_tokens=settings.LLM_MAX_TOKENS,
    api_key=settings.OPENAI_API_KEY,
)

logger.debug(
    "LLM client initialised | model=%s | temperature=%.1f | max_tokens=%d",
    settings.LLM_MODEL,
    settings.LLM_TEMPERATURE,
    settings.LLM_MAX_TOKENS,
)


# INVOKE 

def invoke(messages: list[BaseMessage], retries: int = 3) -> str:
    """
    Send a list of messages to the LLM and return the response as a string.

    This is the main function response_generator.py calls on every turn.

    How it works:
        1. Wrap the _llm.invoke() call in helpers.retry() for resilience
           against transient OpenAI API errors (rate limits, timeouts)
        2. Wrap in helpers.timer() to log latency on every call
        3. Extract the string content from the AIMessage response
        4. Log token usage if OpenAI returns it (cost visibility)

    Args:
        messages: A list of LangChain BaseMessage objects.
                  Produced by ChatPromptTemplate.format_messages().
                  Typically: [SystemMessage, HumanMessage]

        retries:  How many times to retry on failure (default 3).
                  Uses exponential backoff via helpers.retry().

    Returns:
        The LLM's response as a plain string.

    Raises:
        RuntimeError: If all retries are exhausted (wraps the original error).

    Example:
        from app.generation.prompts import get_prompt

        prompt   = get_prompt("rag")
        messages = prompt.format_messages(context=ctx, question=q)
        answer   = invoke(messages)
    """
    if not messages:
        raise ValueError("invoke() received an empty messages list.")

    logger.info(
        "Invoking LLM | model=%s | messages=%d",
        settings.LLM_MODEL,
        len(messages),
    )

    # Call with retry + timing 
    # helpers.retry() takes a zero-argument callable and retries it on any
    # exception with exponential backoff (1s → 2s → 4s by default).
    # helpers.timer() is a context manager that measures elapsed time.
    with timer("LLM invoke") as t:
        try:
            response = retry(
                func=lambda: _llm.invoke(messages),
                retries=retries,
                delay=1.0,
                backoff=2.0,
            )
        except Exception as exc:
            # All retries exhausted — wrap with context for easier debugging
            raise RuntimeError(
                f"LLM invoke failed after {retries} retries "
                f"(model={settings.LLM_MODEL}): {exc}"
            ) from exc

    # ── Log latency ───────────────────────────────────────────────────────────
    logger.info(
        "LLM response received | elapsed=%sms | model=%s",
        t["elapsed_ms"],
        settings.LLM_MODEL,
    )

    # ── Extract response string ───────────────────────────────────────────────
    # _llm.invoke() returns an AIMessage object.
    # AIMessage.content is the string we want.
    content = response.content

    if not content or not content.strip():
        logger.warning("LLM returned an empty response — check your prompt.")

    # ── Log token usage (if available) ───────────────────────────────────────
    # OpenAI returns token counts in response_metadata. Logging this gives
    # you visibility into cost per request — important at MVP scale.
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    if usage:
        logger.info(
            "Token usage | prompt=%d | completion=%d | total=%d",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        )

    logger.debug("LLM response preview: %r", content[:120].replace("\n", " "))

    return content


# ── INVOKE JSON ───────────────────────────────────────────────────────────────

def invoke_json(messages: list[BaseMessage], retries: int = 3) -> dict | list | None:
    """
    Same as invoke() but parses the response as JSON.

    Used by verifier.py when calling GUARDRAIL_PROMPT, which asks the LLM
    to respond with a structured JSON verdict:
        {"verdict": "PASS", "explanation": "..."}

    Why not just call invoke() and parse manually in verifier.py?
        Keeping JSON parsing here means verifier.py stays focused on
        validation logic. Also, safe_parse_json() handles markdown-wrapped
        JSON (```json ... ```) which LLMs frequently produce — centralising
        that handling avoids bugs when verifier.py forgets to strip fences.

    Args:
        messages: Formatted messages, typically from GUARDRAIL_PROMPT.
        retries:  Retry attempts passed through to invoke().

    Returns:
        Parsed Python object (dict or list), or None if parsing fails.
        Returns None rather than raising so verifier.py can apply
        fallback logic — a parse failure is not a hard crash.

    Example:
        messages = get_prompt("guardrail").format_messages(
            context=ctx, question=q, answer=draft
        )
        result = invoke_json(messages)
        # result → {"verdict": "PASS", "explanation": "All claims supported."}
    """
    raw = invoke(messages, retries=retries)

    parsed = safe_parse_json(raw)

    if parsed is None:
        logger.warning(
            "invoke_json: failed to parse LLM response as JSON. "
            "Raw response: %r",
            raw[:200],
        )
    else:
        logger.debug("invoke_json: parsed successfully → %s", type(parsed).__name__)

    return parsed


# ── GET LLM ───────────────────────────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    """
    Return the raw ChatOpenAI singleton.

    Use this when you need to pass the LLM directly into a LangChain
    chain or graph — e.g. LangGraph nodes, LangChain Expression Language
    (LCEL) chains, or ConversationalRetrievalChain.

    invoke() is preferred for direct calls — it adds retry and timing.
    get_llm() is for framework integrations that manage invocation themselves.

    Example (LCEL chain):
        from app.generation.llm import get_llm
        from app.generation.prompts import get_prompt

        chain = get_prompt("rag") | get_llm()
        response = chain.invoke({"context": ctx, "question": q})
    """
    return _llm