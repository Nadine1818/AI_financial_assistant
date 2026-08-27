# Single source of truth for every prompt sent to the LLM.
"""
Position in the pipeline:
    retriever.py → [prompts.py] → response_generator.py

Design principles:
    - NO prompt string lives anywhere else in the codebase
    - All templates use LangChain's ChatPromptTemplate (not f-strings)
    - Variables are declared explicitly → missing one raises an error immediately
    - Each template has one job and is independently testable
    - Financial assistant constraints are enforced at the prompt level first,
      then again in verifier.py (defence in depth)

Why ChatPromptTemplate over f-strings?
    f-strings are silent about missing variables — you get a broken prompt
    at runtime with no warning. ChatPromptTemplate validates variables at
    build time and maps cleanly to the OpenAI messages API
    (system / human / assistant roles).

Templates in this file:
    RAG_PROMPT          → main QA template: context + question → answer
    GUARDRAIL_PROMPT    → self-check: does the answer contradict the context?
    CONDENSE_PROMPT     → multi-turn: rephrase follow-up question as standalone
    NO_CONTEXT_PROMPT   → fallback when retrieval returns nothing
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from app.utils.logger import get_logger

logger = get_logger(__name__)
# __name__ here is "app.generation.prompts"


# SYSTEM PERSONA 
# Extracted as a constant so it can be reused across multiple templates
# without copy-pasting. Changing the persona here updates every template at once.
#
# Key constraints baked into the persona (not just guidelines):
#   1. Only answer from provided context  → prevents hallucination
#   2. Cite sources by filename           → auditability requirement
#   3. Refuse if answer not in context    → financial safety
#   4. Never guess at numbers             → critical for financial accuracy
#   5. Flag uncertainty explicitly        → compliance requirement

SYSTEM_PERSONA = """\
You are a precise and trustworthy financial AI assistant.

Your role is to answer questions about the user's financial documents \
(bank statements, reports, transaction records, etc.).

Rules you must always follow:
1. ONLY use information present in the provided context. \
   Never use prior knowledge to fill in gaps.
2. Cite the source of your answer using the filename from the context block, \
   e.g. "According to bank_statement.pdf..."
3. If the context does not contain enough information to answer the question, \
   respond exactly with: "I don't have enough information in the provided \
   documents to answer this question."
4. Never invent, estimate, or extrapolate numerical values. \
   If a number is not explicitly stated in the context, say so.
5. If the answer involves currency, always include the currency symbol \
   (£, $, €) exactly as it appears in the source document.
6. Keep answers concise and factual. Avoid filler phrases like \
   "Great question!" or "Certainly!".
"""


# RAG PROMPT 
# The main template. Used on every standard question-answering turn.
#
# Variables:
#   {context}  → formatted string from retriever.format_context()
#                Each block is labelled "[Source: filename | Chunk N]"
#   {question} → the user's question, exactly as typed
#
# Why is context injected into the human message, not the system message?
#   The system message defines behaviour (static). The human message carries
#   the per-request data (dynamic). Mixing them makes caching harder and
#   makes the template less reusable across turn types.

RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PERSONA),
    (
        "human",
        """\
Here is the relevant context retrieved from your financial documents:

{context}

---

Based solely on the context above, please answer the following question:

{question}""",
    ),
])

logger.debug("RAG_PROMPT registered | variables: %s", RAG_PROMPT.input_variables)


# GUARDRAIL PROMPT 
# Asks the LLM to verify its own answer against the retrieved context.
# This is the "LLM-as-judge" pattern — cheap self-check before returning
# a response to the user.
#
# Used by verifier.py after response_generator.py produces a draft answer.
#
# Variables:
#   {context}  → same context block used to generate the answer
#   {question} → the original user question
#   {answer}   → the draft answer produced by RAG_PROMPT
#
# Expected output: a JSON object with two keys:
#   "verdict":     "PASS" | "FAIL" | "UNCERTAIN"
#   "explanation": one sentence explaining the verdict
#
# Why JSON output?
#   verifier.py can parse it reliably with helpers.safe_parse_json().
#   Free-text verdicts are ambiguous and hard to act on programmatically.

GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """\
You are a financial fact-checker. Your only job is to verify whether \
a given answer is fully supported by the provided context.

You must respond with ONLY a JSON object — no preamble, no explanation outside the JSON.

JSON format:
{{
  "verdict": "PASS" | "FAIL" | "UNCERTAIN",
  "explanation": "<one sentence>"
}}

Verdict definitions:
  PASS      → Every factual claim in the answer is directly supported by the context.
  FAIL      → The answer contains at least one claim not found in the context, \
               or contradicts the context.
  UNCERTAIN → The context is ambiguous and the answer may or may not be correct.
""",
    ),
    (
        "human",
        """\
Context:
{context}

Question:
{question}

Answer to verify:
{answer}

Respond with the JSON verdict only.""",
    ),
])

logger.debug("GUARDRAIL_PROMPT registered | variables: %s", GUARDRAIL_PROMPT.input_variables)


# CONDENSE PROMPT 
# Handles multi-turn conversations. When a user asks a follow-up question
# like "What about the previous month?", the retriever can't search for that
# directly — "previous month" has no meaning without the conversation history.
#
# This prompt rewrites the follow-up into a self-contained question that
# can be passed directly to the retriever.
#
# Variables:
#   {chat_history} → MessagesPlaceholder — LangChain injects the list of
#                    prior HumanMessage / AIMessage objects here automatically
#   {question}     → the user's latest follow-up question
#
# Expected output: a single rephrased question string. Nothing else.
#
#   history:  "What was my balance in January?"  → "Your balance was £2,400."
#   question: "What about February?"
#   output:   "What was my account balance in February?"

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """\
You are a query rewriter for a financial document retrieval system.

Given a conversation history and a follow-up question, rewrite the \
follow-up into a single, self-contained question that can be understood \
without the conversation history.

Rules:
- Output ONLY the rewritten question. No preamble, no explanation.
- Preserve all financial specifics (dates, amounts, account names).
- If the follow-up is already self-contained, return it unchanged.
""",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    (
        "human",
        "Follow-up question: {question}\n\nRewritten question:",
    ),
])

logger.debug("CONDENSE_PROMPT registered | variables: %s", CONDENSE_PROMPT.input_variables)


# NO-CONTEXT FALLBACK PROMPT 
# Used when the retriever returns zero chunks above the relevance threshold.
# Instead of sending an empty context block to the LLM (which causes
# hallucination), we use this prompt to generate a safe, honest refusal.
#
# Variables:
#   {question} → the user's question (so we can reference it in the response)
#
# Why not just return a hardcoded string?
#   A hardcoded string can't reference the specific question or suggest
#   what document the user might need to upload. The LLM can do that.

NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PERSONA),
    (
        "human",
        """\
The retrieval system found no relevant document chunks for the following question:

{question}

Respond by:
1. Telling the user clearly that you couldn't find relevant information \
   in the uploaded documents.
2. Suggesting what type of document they might need to upload \
   (e.g. "a bank statement", "a profit & loss report") based on the question.
3. Keeping the response under 3 sentences.

Do not attempt to answer the question from general knowledge.""",
    ),
])

logger.debug("NO_CONTEXT_PROMPT registered | variables: %s", NO_CONTEXT_PROMPT.input_variables)


# REWRITE PROMPT
# Used by the corrective-RAG loop (app/orchestration/rag_graph.py) when
# verifier.py returns a FAIL verdict — the answer contained claims not
# supported by the retrieved context. Instead of giving up immediately,
# we ask the LLM to rewrite the question into a query more likely to
# retrieve the right documents, then retry generation.
#
# Variables:
#   {question}       → the ORIGINAL user question (not a prior rewrite —
#                       always rewrite from the original to avoid drifting
#                       further from what the user actually asked)
#   {failed_answer}   → the answer that failed verification
#   {explanation}     → the guardrail LLM's explanation of why it failed
#
# Expected output: a single rewritten search query. Nothing else.

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """\
You are a query rewriter for a financial document retrieval system.

A previous attempt to answer a question failed fact-checking — the answer \
contained claims not supported by the retrieved documents. Your job is to \
rewrite the ORIGINAL question into a new search query more likely to \
retrieve the documents actually needed to answer it correctly.

Rules:
- Output ONLY the rewritten query. No preamble, no explanation.
- Preserve all financial specifics from the original question (dates, \
  amounts, account names, document types).
- Try different phrasing, more specific terms, or a narrower focus than \
  the original — the goal is better retrieval, not a different question.
""",
    ),
    (
        "human",
        """\
Original question:
{question}

Previous answer (failed verification):
{failed_answer}

Why it failed:
{explanation}

Rewritten search query:""",
    ),
])

logger.debug("REWRITE_PROMPT registered | variables: %s", REWRITE_PROMPT.input_variables)


# PROMPT REGISTRY
# A single dict that maps prompt names to their templates.
# response_generator.py imports from here by name — no direct imports
# of individual templates needed outside this file.
#
# This also makes it easy to iterate over all prompts in tests:
#   for name, prompt in PROMPT_REGISTRY.items():
#       assert prompt.input_variables  # every prompt declares its variables

PROMPT_REGISTRY: dict[str, ChatPromptTemplate] = {
    "rag":         RAG_PROMPT,
    "guardrail":   GUARDRAIL_PROMPT,
    "condense":    CONDENSE_PROMPT,
    "no_context":  NO_CONTEXT_PROMPT,
    "rewrite":     REWRITE_PROMPT,
}


def get_prompt(name: str) -> ChatPromptTemplate:
    """
    Retrieve a prompt template by name from the registry.

    Args:
        name: One of "rag", "guardrail", "condense", "no_context"

    Returns:
        The corresponding ChatPromptTemplate.

    Raises:
        KeyError: If the name is not in the registry — fail loudly,
                  because a missing prompt is a code bug, not a user error.

    Example:
        prompt = get_prompt("rag")
        messages = prompt.format_messages(context=ctx, question=q)
    """
    if name not in PROMPT_REGISTRY:
        available = ", ".join(f'"{k}"' for k in PROMPT_REGISTRY)
        raise KeyError(
            f"Unknown prompt name: '{name}'. "
            f"Available prompts: {available}"
        )

    logger.debug("Fetched prompt: '%s'", name)
    return PROMPT_REGISTRY[name]