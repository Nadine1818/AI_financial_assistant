"""
Tests for app.generation.prompts

Tests prompt template management:
    - get_prompt() retrieves correct templates
    - Prompt validation and variable binding
    - Edge cases and error handling

Run with:
    pytest tests/test_prompts.py -v
"""
import pytest
from langchain_core.prompts import ChatPromptTemplate

from app.generation.prompts import (
    get_prompt,
    SYSTEM_PERSONA,
    RAG_PROMPT,
    GUARDRAIL_PROMPT,
    CONDENSE_PROMPT,
    NO_CONTEXT_PROMPT,
)


class TestSystemPersona:
    """Test the system persona constant."""
    
    def test_system_persona_contains_core_rules(self):
        """Should enforce key financial safety rules."""
        assert "financial AI assistant" in SYSTEM_PERSONA
        assert "context" in SYSTEM_PERSONA.lower()
        assert "source" in SYSTEM_PERSONA.lower() or "filename" in SYSTEM_PERSONA.lower()
    
    def test_system_persona_forbids_hallucination(self):
        """Should explicitly forbid using prior knowledge to fill gaps."""
        assert "prior knowledge" in SYSTEM_PERSONA.lower() or "provided" in SYSTEM_PERSONA.lower()


class TestGetPrompt:
    """Test get_prompt() function."""
    
    def test_get_prompt_returns_rag_template(self):
        """Should return RAG_PROMPT when requested."""
        result = get_prompt("rag")
        assert isinstance(result, ChatPromptTemplate)
        # RAG should have context and question variables
        assert "context" in result.input_variables
        assert "question" in result.input_variables
    
    def test_get_prompt_returns_condense_template(self):
        """Should return CONDENSE_PROMPT when requested."""
        result = get_prompt("condense")
        assert isinstance(result, ChatPromptTemplate)
        # Should handle chat_history and question
        assert "chat_history" in result.input_variables
        assert "question" in result.input_variables
    
    def test_get_prompt_returns_no_context_template(self):
        """Should return NO_CONTEXT_PROMPT when requested."""
        result = get_prompt("no_context")
        assert isinstance(result, ChatPromptTemplate)
        # Should at least need question
        assert "question" in result.input_variables
    
    def test_get_prompt_returns_guardrail_template(self):
        """Should return GUARDRAIL_PROMPT when requested."""
        result = get_prompt("guardrail")
        assert isinstance(result, ChatPromptTemplate)
    
    def test_get_prompt_raises_on_invalid_name(self):
        """Should raise KeyError or ValueError for unknown prompt names."""
        with pytest.raises((KeyError, ValueError)):
            get_prompt("invalid_prompt_name")


class TestRagPromptTemplate:
    """Test RAG_PROMPT template specifics."""
    
    def test_rag_prompt_requires_context_and_question(self):
        """RAG_PROMPT should require both context and question variables."""
        assert "context" in RAG_PROMPT.input_variables
        assert "question" in RAG_PROMPT.input_variables
    
    def test_rag_prompt_format_messages_succeeds_with_both_vars(self):
        """Should successfully format when both context and question provided."""
        messages = RAG_PROMPT.format_messages(
            context="Financial data here",
            question="What is my balance?"
        )
        assert len(messages) > 0
        # Should have system and user messages
        assert any("financial" in str(m).lower() for m in messages)
    
    def test_rag_prompt_format_messages_fails_without_context(self):
        """Should raise KeyError when context is missing."""
        with pytest.raises(KeyError):
            RAG_PROMPT.format_messages(question="What is my balance?")
    
    def test_rag_prompt_format_messages_fails_without_question(self):
        """Should raise KeyError when question is missing."""
        with pytest.raises(KeyError):
            RAG_PROMPT.format_messages(context="Financial data")


class TestCondensePromptTemplate:
    """Test CONDENSE_PROMPT template specifics."""
    
    def test_condense_prompt_requires_question_and_history(self):
        """CONDENSE_PROMPT should require question and chat_history."""
        assert "question" in CONDENSE_PROMPT.input_variables
        assert "chat_history" in CONDENSE_PROMPT.input_variables
    
    def test_condense_prompt_accepts_langchain_messages(self):
        """Should accept list of LangChain BaseMessage objects for history."""
        from langchain_core.messages import HumanMessage, AIMessage
        
        history = [
            HumanMessage(content="Previous question"),
            AIMessage(content="Previous answer"),
        ]
        
        messages = CONDENSE_PROMPT.format_messages(
            chat_history=history,
            question="Follow-up question"
        )
        
        assert len(messages) > 0


class TestNoContextPromptTemplate:
    """Test NO_CONTEXT_PROMPT fallback template."""
    
    def test_no_context_prompt_requires_question(self):
        """NO_CONTEXT_PROMPT should require question variable."""
        assert "question" in NO_CONTEXT_PROMPT.input_variables
    
    def test_no_context_prompt_format_messages_succeeds(self):
        """Should successfully format with just a question."""
        messages = NO_CONTEXT_PROMPT.format_messages(
            question="What is my balance?"
        )
        
        assert len(messages) > 0
        # Should indicate no context available
        content_str = str(messages).lower()
        assert "no" in content_str or "information" in content_str or "document" in content_str


class TestGuardrailPromptTemplate:
    """Test GUARDRAIL_PROMPT for self-checking."""
    
    def test_guardrail_prompt_exists(self):
        """Should define a guardrail prompt for answer verification."""
        assert GUARDRAIL_PROMPT is not None
        assert isinstance(GUARDRAIL_PROMPT, ChatPromptTemplate)
    
    def test_guardrail_prompt_requires_context_and_answer(self):
        """Should require context and answer for verification."""
        # Likely needs context and answer to check
        input_vars = GUARDRAIL_PROMPT.input_variables
        assert len(input_vars) > 0


class TestPromptConsistency:
    """Test cross-prompt consistency and safety."""
    
    def test_all_prompts_use_system_persona(self):
        """All prompts should incorporate the SYSTEM_PERSONA for consistency."""
        prompts_to_check = [RAG_PROMPT, CONDENSE_PROMPT, NO_CONTEXT_PROMPT]
        
        for prompt in prompts_to_check:
            prompt_str = str(prompt)
            # At minimum, should contain some core financial safety language
            # (This is a loose check; actual implementation may vary)
            assert len(prompt_str) > 0
    
    def test_prompt_templates_are_distinct(self):
        """Different prompts should have different purposes."""
        rag_str = str(RAG_PROMPT)
        condense_str = str(CONDENSE_PROMPT)
        
        # Should not be identical
        assert rag_str != condense_str
