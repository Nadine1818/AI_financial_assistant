"""
Tests for app.generation.response_generator

Tests the RAG orchestration layer:
    - generate() single-turn logic
    - generate_with_history() multi-turn logic
    - GenerationResult dataclass

Run with:
    pytest tests/test_response_generator.py -v
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from app.generation.response_generator import (
    generate,
    generate_with_history,
    GenerationResult,
    _extract_sources,
)


class TestExtractSources:
    """Test source extraction from retrieved documents."""
    
    def test_extracts_unique_sources_preserving_order(self):
        """Should deduplicate and preserve insertion order."""
        docs = [
            Document(page_content="content1", metadata={"source": "file1.pdf"}),
            Document(page_content="content2", metadata={"source": "file2.pdf"}),
            Document(page_content="content3", metadata={"source": "file1.pdf"}),
        ]
        result = _extract_sources(docs)
        assert result == ["file1.pdf", "file2.pdf"]
    
    def test_handles_missing_source_metadata(self):
        """Should use 'unknown' for docs without source metadata."""
        docs = [
            Document(page_content="content", metadata={}),
        ]
        result = _extract_sources(docs)
        assert result == ["unknown"]
    
    def test_empty_list_returns_empty(self):
        """Should return empty list for empty input."""
        result = _extract_sources([])
        assert result == []


class TestGenerationResult:
    """Test GenerationResult dataclass."""
    
    def test_creates_with_required_fields(self):
        """Should construct with all required fields."""
        result = GenerationResult(
            answer="Test answer",
            context="Test context",
            sources=["doc.pdf"],
            question="Test question",
            retrieved_chunks=1,
        )
        assert result.answer == "Test answer"
        assert result.retrieved_chunks == 1
    
    def test_defaults_used_fallback_to_false(self):
        """Should default used_fallback to False."""
        result = GenerationResult(
            answer="", context="", sources=[], question="", retrieved_chunks=0
        )
        assert result.used_fallback is False
    
    def test_metadata_defaults_to_empty_dict(self):
        """Should default metadata to empty dict."""
        result = GenerationResult(
            answer="", context="", sources=[], question="", retrieved_chunks=0
        )
        assert result.metadata == {}
    
    def test_repr_shows_key_info(self):
        """__repr__ should show chunks, fallback, sources, and answer preview."""
        result = GenerationResult(
            answer="This is a test answer that is very long and should be truncated",
            context="ctx",
            sources=["file.pdf"],
            question="q",
            retrieved_chunks=5,
            used_fallback=False,
        )
        repr_str = repr(result)
        assert "chunks=5" in repr_str
        assert "fallback=False" in repr_str
        assert "['file.pdf']" in repr_str
        assert "This is a test answer" in repr_str


class TestGenerate:
    """Test single-turn RAG generation."""
    
    @patch("app.generation.response_generator.retrieve")
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    def test_generate_with_retrieved_context(self, mock_get_prompt, mock_invoke, mock_retrieve):
        """Should generate answer when context is found."""
        # Setup mocks
        mock_retrieve.return_value = [
            Document(
                page_content="Financial data here",
                metadata={"source": "statement.pdf"}
            )
        ]
        mock_invoke.return_value = "Based on the context, the answer is..."
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = [
            MagicMock(content="System prompt"),
            MagicMock(content="User question"),
        ]
        mock_get_prompt.return_value = mock_prompt
        
        # Execute
        result = generate("What is my balance?")
        
        # Assert
        assert isinstance(result, GenerationResult)
        assert result.answer == "Based on the context, the answer is..."
        assert result.retrieved_chunks == 1
        assert result.used_fallback is False
        assert result.sources == ["statement.pdf"]
        assert "What is my balance?" in result.question
    
    @patch("app.generation.response_generator.retrieve")
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    def test_generate_fallback_when_no_context(self, mock_get_prompt, mock_invoke, mock_retrieve):
        """Should use NO_CONTEXT_PROMPT when retrieval is empty."""
        # Setup
        mock_retrieve.return_value = []
        mock_invoke.return_value = "I don't have information about that."
        mock_fallback_prompt = MagicMock()
        mock_fallback_prompt.format_messages.return_value = [MagicMock(content="Fallback")]
        
        def get_prompt_side_effect(prompt_type):
            if prompt_type == "no_context":
                return mock_fallback_prompt
            return MagicMock()
        
        mock_get_prompt.side_effect = get_prompt_side_effect
        
        # Execute
        result = generate("Unknown question?")
        
        # Assert
        assert result.used_fallback is True
        assert result.retrieved_chunks == 0
        assert result.sources == []
        assert result.context == ""
    
    def test_generate_raises_on_empty_question(self):
        """Should raise ValueError for empty questions."""
        with pytest.raises(ValueError, match="empty question"):
            generate("")
        
        with pytest.raises(ValueError, match="empty question"):
            generate("   ")
    
    @patch("app.generation.response_generator.retrieve")
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    def test_generate_respects_top_k_parameter(self, mock_get_prompt, mock_invoke, mock_retrieve):
        """Should pass top_k to retriever."""
        mock_retrieve.return_value = []
        mock_invoke.return_value = "test"
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = []
        mock_get_prompt.return_value = mock_prompt
        
        generate("query", top_k=20)
        
        mock_retrieve.assert_called_once()
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs["top_k"] == 20
    
    @patch("app.generation.response_generator.retrieve")
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    def test_generate_respects_source_filter(self, mock_get_prompt, mock_invoke, mock_retrieve):
        """Should pass source_filter to retriever."""
        mock_retrieve.return_value = []
        mock_invoke.return_value = "test"
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = []
        mock_get_prompt.return_value = mock_prompt
        
        generate("query", source_filter="specific.pdf")
        
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs["source_filter"] == "specific.pdf"


class TestGenerateWithHistory:
    """Test multi-turn RAG with question condensing."""
    
    @patch("app.generation.response_generator.generate")
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    def test_condenses_follow_up_question(self, mock_get_prompt, mock_invoke, mock_generate):
        """Should condense follow-up question with chat history."""
        # Setup
        history = [
            ("What is my balance?", "Your balance is £1,000."),
        ]
        mock_condense_prompt = MagicMock()
        mock_condense_prompt.format_messages.return_value = [MagicMock()]
        mock_invoke.return_value = "What was my balance in January?"
        
        def get_prompt_side_effect(prompt_type):
            if prompt_type == "condense":
                return mock_condense_prompt
            return MagicMock()
        
        mock_get_prompt.side_effect = get_prompt_side_effect
        mock_generate.return_value = GenerationResult(
            answer="Your balance in January was £900.",
            context="",
            sources=[],
            question="What was my balance in January?",
            retrieved_chunks=1,
        )
        
        # Execute
        result = generate_with_history("What about January?", history)
        
        # Assert
        assert result.question == "What was my balance in January?"
        assert "original_question" in result.metadata
        assert result.metadata["original_question"] == "What about January?"
    
    @patch("app.generation.response_generator.generate")
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    def test_builds_langchain_message_history(self, mock_get_prompt, mock_invoke, mock_generate):
        """Should convert tuple history to LangChain BaseMessage objects."""
        history = [
            ("Q1", "A1"),
            ("Q2", "A2"),
        ]
        mock_condense_prompt = MagicMock()
        mock_condense_prompt.format_messages.return_value = [MagicMock()]
        mock_invoke.return_value = "Condensed question"
        mock_generate.return_value = GenerationResult(
            answer="", context="", sources=[], question="", retrieved_chunks=0
        )
        
        def get_prompt_side_effect(prompt_type):
            if prompt_type == "condense":
                return mock_condense_prompt
            return MagicMock()
        
        mock_get_prompt.side_effect = get_prompt_side_effect
        
        generate_with_history("Follow-up", history)
        
        # Verify format_messages was called with chat_history parameter
        call_kwargs = mock_condense_prompt.format_messages.call_args[1]
        assert "chat_history" in call_kwargs
        lc_history = call_kwargs["chat_history"]
        
        # Should have 4 messages: 2 exchanges × (human + AI)
        assert len(lc_history) == 4
        assert isinstance(lc_history[0], HumanMessage)
        assert isinstance(lc_history[1], AIMessage)
    
    def test_raises_on_empty_follow_up_question(self):
        """Should raise ValueError for empty follow-up questions."""
        with pytest.raises(ValueError, match="empty question"):
            generate_with_history("", [])
