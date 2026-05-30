"""
Integration tests for the full RAG pipeline

Tests end-to-end RAG flows with integrated components:
    - Single-turn generation with retrieval + LLM
    - Multi-turn generation with condensing
    - Error handling across components
    - Performance/latency checks

Run with:
    pytest tests/test_integration_rag.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from app.generation.response_generator import generate, generate_with_history


class TestRagIntegrationSingleTurn:
    """Integration tests for single-turn RAG."""
    
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    @patch("app.generation.response_generator.retrieve")
    def test_full_rag_flow_with_context(self, mock_retrieve, mock_get_prompt, mock_invoke):
        """Test complete single-turn RAG flow: retrieve → format → prompt → LLM."""
        # Setup: retriever finds relevant chunks
        docs = [
            Document(
                page_content="Q3 2024 revenue was £5,234,000",
                metadata={"source": "financial_report_q3.pdf"}
            ),
            Document(
                page_content="Operating expenses were £1,234,000",
                metadata={"source": "financial_report_q3.pdf"}
            ),
        ]
        mock_retrieve.return_value = docs
        
        # Setup: prompt formatting
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = [
            MagicMock(content="System: You are a financial assistant"),
            MagicMock(content="User: What was Q3 revenue?"),
        ]
        mock_get_prompt.return_value = mock_prompt
        
        # Setup: LLM response
        mock_invoke.return_value = "According to financial_report_q3.pdf, Q3 2024 revenue was £5,234,000."
        
        # Execute
        result = generate("What was our Q3 2024 revenue?")
        
        # Verify entire flow
        assert result.answer == "According to financial_report_q3.pdf, Q3 2024 revenue was £5,234,000."
        assert result.retrieved_chunks == 2
        assert "financial_report_q3.pdf" in result.sources
        assert not result.used_fallback
    
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    @patch("app.generation.response_generator.retrieve")
    def test_rag_fallback_when_no_relevant_chunks(self, mock_retrieve, mock_get_prompt, mock_invoke):
        """Test RAG fallback path when retriever finds nothing."""
        mock_retrieve.return_value = []
        
        mock_fallback_prompt = MagicMock()
        mock_fallback_prompt.format_messages.return_value = [MagicMock(content="Fallback")]
        
        def prompt_selector(prompt_type):
            if prompt_type == "no_context":
                return mock_fallback_prompt
            return MagicMock()
        
        mock_get_prompt.side_effect = prompt_selector
        mock_invoke.return_value = "I don't have that information in the documents."
        
        # Execute
        result = generate("Very obscure question?")
        
        # Verify fallback path taken
        assert result.used_fallback is True
        assert result.retrieved_chunks == 0
        assert result.context == ""
    
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    @patch("app.generation.response_generator.retrieve")
    def test_rag_respects_source_filtering(self, mock_retrieve, mock_get_prompt, mock_invoke):
        """Test that source filtering is applied through entire pipeline."""
        docs = [
            Document(
                page_content="Filtered content",
                metadata={"source": "bank_statement.pdf"}
            ),
        ]
        mock_retrieve.return_value = docs
        
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = [MagicMock()]
        mock_get_prompt.return_value = mock_prompt
        mock_invoke.return_value = "Answer from bank_statement.pdf"
        
        # Execute with source filter
        result = generate("Balance?", source_filter="bank_statement.pdf")
        
        # Verify source filter was passed to retriever
        mock_retrieve.assert_called_once()
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs["source_filter"] == "bank_statement.pdf"


class TestRagIntegrationMultiTurn:
    """Integration tests for multi-turn RAG with history."""
    
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    @patch("app.generation.response_generator.retrieve")
    def test_multi_turn_with_condensing(self, mock_retrieve, mock_get_prompt, mock_invoke):
        """Test multi-turn with question condensing from history."""
        # Setup: condense prompt returns a standalone question
        history = [
            ("What was my balance in January?", "Your balance was £1,000."),
        ]
        
        mock_condense_prompt = MagicMock()
        mock_condense_prompt.format_messages.return_value = [MagicMock()]
        
        mock_rag_prompt = MagicMock()
        mock_rag_prompt.format_messages.return_value = [MagicMock()]
        
        def prompt_selector(prompt_type):
            if prompt_type == "condense":
                return mock_condense_prompt
            return mock_rag_prompt
        
        mock_get_prompt.side_effect = prompt_selector
        
        # invoke() called twice: once for condensing, once for RAG
        mock_invoke.side_effect = [
            "What was my account balance in February?",  # Condensed question
            "Your February balance was £1,100.",         # RAG response
        ]
        
        mock_retrieve.return_value = [
            Document(page_content="February balance: £1,100", metadata={"source": "statement.pdf"})
        ]
        
        # Execute
        result = generate_with_history("What about February?", history)
        
        # Verify condensing happened
        assert "February" in result.question or "balance" in result.question.lower()
        assert result.metadata["original_question"] == "What about February?"
    
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    @patch("app.generation.response_generator.retrieve")
    def test_multi_turn_preserves_full_history(self, mock_retrieve, mock_get_prompt, mock_invoke):
        """Test that multi-turn uses complete chat history for condensing."""
        history = [
            ("Q1", "A1"),
            ("Q2", "A2"),
            ("Q3", "A3"),
        ]
        
        mock_condense_prompt = MagicMock()
        mock_condense_prompt.format_messages.return_value = [MagicMock()]
        
        mock_rag_prompt = MagicMock()
        mock_rag_prompt.format_messages.return_value = [MagicMock()]
        
        def prompt_selector(prompt_type):
            if prompt_type == "condense":
                return mock_condense_prompt
            return mock_rag_prompt
        
        mock_get_prompt.side_effect = prompt_selector
        mock_invoke.side_effect = ["Condensed", "Final answer"]
        mock_retrieve.return_value = []
        
        # Execute
        generate_with_history("Follow-up", history)
        
        # Verify entire history was used
        condense_call_kwargs = mock_condense_prompt.format_messages.call_args[1]
        history_list = condense_call_kwargs["chat_history"]
        
        # Should have 6 messages: 3 exchanges × (human + AI)
        assert len(history_list) == 6


class TestRagErrorRecovery:
    """Integration tests for error handling across RAG components."""
    
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    @patch("app.generation.response_generator.retrieve")
    def test_rag_handles_retriever_error(self, mock_retrieve, mock_get_prompt, mock_invoke):
        """Test RAG degrades gracefully if retriever fails."""
        mock_retrieve.side_effect = Exception("Vector DB connection failed")
        
        # Should raise or handle the error
        with pytest.raises(Exception):
            generate("What is my balance?")
    
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    @patch("app.generation.response_generator.retrieve")
    def test_rag_handles_llm_error_with_fallback(self, mock_retrieve, mock_get_prompt, mock_invoke):
        """Test that LLM errors are propagated (verifier can handle)."""
        mock_retrieve.return_value = [
            Document(page_content="Some content", metadata={"source": "doc.pdf"})
        ]
        
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = [MagicMock()]
        mock_get_prompt.return_value = mock_prompt
        
        mock_invoke.side_effect = Exception("LLM API timeout")
        
        with pytest.raises(Exception, match="timeout"):
            generate("Question")


class TestRagMetadataTracking:
    """Integration tests for metadata collection during RAG."""
    
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    @patch("app.generation.response_generator.retrieve")
    def test_rag_tracks_latencies(self, mock_retrieve, mock_get_prompt, mock_invoke):
        """Test that RAG collects timing metadata."""
        mock_retrieve.return_value = [
            Document(page_content="Content", metadata={"source": "doc.pdf"})
        ]
        
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = [MagicMock()]
        mock_get_prompt.return_value = mock_prompt
        mock_invoke.return_value = "Answer"
        
        result = generate("Question")
        
        # Verify metadata contains timing info
        assert "retrieval_ms" in result.metadata
        assert "llm_ms" in result.metadata
        assert result.metadata["retrieval_ms"] >= 0
        assert result.metadata["llm_ms"] >= 0
    
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    @patch("app.generation.response_generator.retrieve")
    def test_multi_turn_tracks_condense_latency(self, mock_retrieve, mock_get_prompt, mock_invoke):
        """Test that multi-turn tracks condensing time separately."""
        mock_condense_prompt = MagicMock()
        mock_condense_prompt.format_messages.return_value = [MagicMock()]
        
        mock_rag_prompt = MagicMock()
        mock_rag_prompt.format_messages.return_value = [MagicMock()]
        
        def prompt_selector(prompt_type):
            if prompt_type == "condense":
                return mock_condense_prompt
            return mock_rag_prompt
        
        mock_get_prompt.side_effect = prompt_selector
        mock_invoke.side_effect = ["Condensed", "Answer"]
        mock_retrieve.return_value = []
        
        result = generate_with_history("Q", [])
        
        # Verify condense latency tracked
        assert "condense_ms" in result.metadata
        assert result.metadata["condense_ms"] >= 0
