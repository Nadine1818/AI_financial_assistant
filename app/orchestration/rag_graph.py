"""
app/orchestration/rag_graph.py

LangGraph orchestration for the corrective-RAG loop.

Position in the pipeline:
    response_generator.py + verifier.py → [rag_graph.py] → main.py (or callers)

Why this file exists:
    response_generator.generate() and verifier.verify() form a straight
    line: retrieve → generate → verify → done. If verification returns
    FAIL, the old behavior was to show a hardcoded refusal
    (verifier.FAIL_RESPONSE) and stop there.

    This module adds ONE new capability on top of those same functions,
    without modifying either of them: if verification fails, rewrite the
    query and try again — up to MAX_RETRIES times — before finally
    falling back to the safe refusal. This is the "Corrective RAG" pattern.

Design principle:
    This file does NOT duplicate retrieval, generation, or verification
    logic. It only WIRES the existing generate() and verify() functions
    into a graph and adds the retry/rewrite behavior around them. If you
    change how generation or verification work, you change those files —
    this file's job is only the control flow between them.

Usage:
    from app.orchestration.rag_graph import run

    result = run("What was my total spend in January?")
    print(result.safe_answer)
    print(result.metadata["graph_attempts"])   # how many tries it took
"""

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.generation.response_generator import generate, GenerationResult
from app.validation.verifier import verify, VerificationResult
from app.generation.prompts import get_prompt
from app.generation.llm import invoke
from app.utils.helpers import truncate_text
from app.utils.logger import get_logger

logger = get_logger(__name__)
# __name__ here is "app.orchestration.rag_graph"


# CONSTANTS

# Total attempts = 1 initial generation + up to MAX_RETRIES rewrite-and-retry
# cycles. Capped low on purpose: each retry is a full retrieve+generate+
# verify cycle (3 LLM calls), so this bounds both latency and cost per
# question. 2 retries (3 attempts total) is a reasonable starting point —
# tune via testing, not guesswork.
MAX_RETRIES = 2


# GRAPH STATE

class RAGState(TypedDict):
    """
    The state object passed between every node in the graph.

    LangGraph nodes are plain functions: state in, partial state out.
    Each field here is either read or written by at least one node.

    Attributes:
        original_question: The user's question, exactly as asked. Never
            mutated — the rewrite node always rewrites FROM this, not from
            the previous rewrite, to avoid drifting away from what the
            user actually asked over multiple retries.
        current_query: What actually gets sent to retrieval this attempt.
            Starts equal to original_question; replaced by the rewrite
            node on each retry.
        attempt: 0-indexed count of how many generate+verify cycles have
            run. Used both for logging and to enforce MAX_RETRIES.
        gen_result: The most recent GenerationResult from generate().
        ver_result: The most recent VerificationResult from verify().
    """
    original_question: str
    current_query: str
    attempt: int
    gen_result: Optional[GenerationResult]
    ver_result: Optional[VerificationResult]


# NODES
# Each node is a plain function: (state) -> partial state update.
# LangGraph merges the returned dict into the existing state.

def _generate_node(state: RAGState) -> dict:
    """
    Run a standard RAG turn on state["current_query"].

    Delegates entirely to response_generator.generate() — no retrieval
    or prompting logic lives here.
    """
    logger.info(
        "Graph: generate | attempt=%d | query=%r",
        state["attempt"],
        truncate_text(state["current_query"], 80),
    )

    result = generate(question=state["current_query"])

    return {"gen_result": result}


def _verify_node(state: RAGState) -> dict:
    """
    Fact-check the most recent generation. Delegates entirely to
    verifier.verify() — no guardrail logic lives here.
    """
    result = verify(state["gen_result"])

    logger.info(
        "Graph: verify | verdict=%s | attempt=%d",
        result.verdict,
        state["attempt"],
    )

    return {"ver_result": result}


def _rewrite_node(state: RAGState) -> dict:
    """
    Rewrite the ORIGINAL question into a new search query, informed by
    why the previous attempt failed verification.

    Always rewrites from original_question, not current_query — so a bad
    rewrite on attempt 1 doesn't compound into a worse one on attempt 2.

    Also increments `attempt`, since a rewrite is what starts a new cycle.
    """
    rewrite_prompt = get_prompt("rewrite")
    messages = rewrite_prompt.format_messages(
        question=state["original_question"],
        failed_answer=state["gen_result"].answer,
        explanation=state["ver_result"].explanation,
    )

    new_query = invoke(messages).strip()

    logger.info(
        "Graph: rewrite | attempt %d → %d | %r → %r",
        state["attempt"],
        state["attempt"] + 1,
        truncate_text(state["current_query"], 60),
        truncate_text(new_query, 60),
    )

    return {
        "current_query": new_query,
        "attempt": state["attempt"] + 1,
    }


# CONDITIONAL EDGE

def _should_retry(state: RAGState) -> str:
    """
    Decide what happens after verification: retry, or stop.

    Only FAIL triggers a retry. PASS and UNCERTAIN both end the graph —
    UNCERTAIN already gets a caveat appended by verifier.py, which is a
    reasonable place to stop; only a clear FAIL is worth spending another
    full retrieve+generate+verify cycle on.

    Returns:
        "rewrite" → route to the rewrite node, then loop back to generate.
        "end"     → stop here; the current ver_result is final.
    """
    verdict = state["ver_result"].verdict

    if verdict == "FAIL" and state["attempt"] < MAX_RETRIES:
        return "rewrite"

    if verdict == "FAIL":
        logger.warning(
            "Graph: FAIL verdict persisted after %d attempt(s) — "
            "giving up, returning safe refusal.",
            state["attempt"] + 1,
        )

    return "end"


# GRAPH ASSEMBLY

def _build_graph():
    """
    Wire the nodes and edges together.

        generate → verify → (conditional) → rewrite → generate  [loop]
                                          └→ END
    """
    graph = StateGraph(RAGState)

    graph.add_node("generate", _generate_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("rewrite", _rewrite_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        _should_retry,
        {"rewrite": "rewrite", "end": END},
    )
    graph.add_edge("rewrite", "generate")

    return graph.compile()


# Compiled once at import time — cheap (no model loading happens here,
# unlike embedder.py's model). Reused across every call to run().
_compiled_graph = _build_graph()


# PUBLIC ENTRY POINT

def run(question: str) -> VerificationResult:
    """
    Answer a question with automatic corrective retries on FAIL verdicts.

    Same call shape as generate() + verify() chained together, but adds
    the rewrite-and-retry loop transparently. Callers (main.py, an API
    layer) don't need to know retries happened at all — they just get a
    VerificationResult back, same as calling verify(generate(question)).

    Args:
        question: The user's question, exactly as typed.

    Returns:
        VerificationResult — same shape verifier.verify() always returns.
        result.metadata["graph_attempts"] tells you how many generate+
        verify cycles it took (1 = succeeded on the first try).

    Raises:
        ValueError: If question is empty.

    Example:
        result = run("What was my total spend in January?")
        print(result.safe_answer)
        print(result.verdict)
        print(result.metadata["graph_attempts"])
    """
    if not question or not question.strip():
        raise ValueError("run() received an empty question.")

    logger.info("Graph: starting | question=%r", truncate_text(question, 100))

    initial_state: RAGState = {
        "original_question": question,
        "current_query": question,
        "attempt": 0,
        "gen_result": None,
        "ver_result": None,
    }

    final_state = _compiled_graph.invoke(initial_state)

    ver_result = final_state["ver_result"]
    ver_result.metadata["graph_attempts"] = final_state["attempt"] + 1

    logger.info(
        "Graph: done | verdict=%s | total attempts=%d",
        ver_result.verdict,
        ver_result.metadata["graph_attempts"],
    )

    return ver_result


# MULTI-TURN ENTRY POINT

def _condense_question(question: str, chat_history: list[tuple[str, str]]) -> str:
    """
    Rewrite a follow-up question into a standalone one, using prior turns.

    Mirrors the condense step inside response_generator.generate_with_history()
    exactly (same CONDENSE_PROMPT, same tuple → BaseMessage conversion) —
    kept here rather than imported from there so the graph's multi-turn
    entry point doesn't depend on generate_with_history()'s internals,
    which run() intentionally bypasses (see run_with_history() below).

    Args:
        question: The user's latest follow-up, exactly as typed.
        chat_history: List of (human_message, ai_message) tuples, oldest
            first. Empty list = first turn, no condensing needed.

    Returns:
        A standalone question. If chat_history is empty, returns the
        question unchanged (no LLM call needed).
    """
    if not chat_history:
        return question

    lc_history: list[BaseMessage] = []
    for human_msg, ai_msg in chat_history:
        lc_history.append(HumanMessage(content=human_msg))
        lc_history.append(AIMessage(content=ai_msg))

    condense_prompt = get_prompt("condense")
    messages = condense_prompt.format_messages(
        chat_history=lc_history,
        question=question,
    )

    condensed = invoke(messages).strip()

    logger.info(
        "Graph: condensed follow-up | %r → %r",
        truncate_text(question, 60),
        truncate_text(condensed, 60),
    )

    return condensed


def run_with_history(
    question: str,
    chat_history: list[tuple[str, str]],
) -> VerificationResult:
    """
    Multi-turn version of run(): condense a follow-up question against
    prior turns, then run the corrective RAG graph on the standalone
    question.

    This is the function main.py's chat loop should call — it's the
    multi-turn equivalent of response_generator.generate_with_history(),
    but with the corrective retry loop included.

    Args:
        question: The user's latest follow-up question, exactly as typed.
        chat_history: List of (human_message, ai_message) tuples, oldest
            first. Pass [] for the first turn of a conversation.

    Returns:
        VerificationResult, same as run(). result.metadata also includes
        "original_question" — the pre-condense follow-up, for transparency
        (mirrors generate_with_history()'s existing behavior).

    Example:
        history = [("What was my balance in January?", "£2,400.")]
        result = run_with_history("What about February?", history)
        print(result.safe_answer)
    """
    if not question or not question.strip():
        raise ValueError("run_with_history() received an empty question.")

    condensed_question = _condense_question(question, chat_history)

    result = run(condensed_question)
    result.metadata["original_question"] = question

    return result