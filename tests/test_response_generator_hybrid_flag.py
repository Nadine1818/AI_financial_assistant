"""
Tests for the USE_HYBRID_RETRIEVAL flag in response_generator.generate().

Kept separate from test_response_generator.py so the existing file (and
its 15 tests covering the default dense-only path) stays untouched —
this file only tests the NEW branch added on top of it.

Run with:
    python -m pytest tests/test_response_generator_hybrid_flag.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.generation.response_generator import generate


def _mock_prompt():
    mock_prompt = MagicMock()
    mock_prompt.format_messages.return_value = [MagicMock()]
    return mock_prompt


class TestHybridFlagOff:
    @patch("app.generation.response_generator.settings")
    @patch("app.generation.response_generator.retrieve_hybrid")
    @patch("app.generation.response_generator.retrieve")
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    def test_uses_dense_retrieve_when_flag_false(
        self, mock_get_prompt, mock_invoke, mock_retrieve, mock_retrieve_hybrid, mock_settings
    ):
        mock_settings.USE_HYBRID_RETRIEVAL = False
        mock_settings.LLM_MODEL = "gpt-4o-mini"
        mock_get_prompt.return_value = _mock_prompt()
        mock_invoke.return_value = "an answer"
        mock_retrieve.return_value = [
            Document(page_content="text", metadata={"source": "f.pdf", "chunk_index": 0}),
        ]

        generate("What was my balance?")

        mock_retrieve.assert_called_once()
        mock_retrieve_hybrid.assert_not_called()


class TestHybridFlagOn:
    @patch("app.generation.response_generator.settings")
    @patch("app.generation.response_generator.retrieve_hybrid")
    @patch("app.generation.response_generator.retrieve")
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    def test_uses_hybrid_retrieve_when_flag_true(
        self, mock_get_prompt, mock_invoke, mock_retrieve, mock_retrieve_hybrid, mock_settings
    ):
        mock_settings.USE_HYBRID_RETRIEVAL = True
        mock_settings.LLM_MODEL = "gpt-4o-mini"
        mock_get_prompt.return_value = _mock_prompt()
        mock_invoke.return_value = "an answer"
        mock_retrieve_hybrid.return_value = [
            Document(page_content="text", metadata={"source": "f.pdf", "chunk_index": 0}),
        ]

        generate("What was my balance?")

        mock_retrieve_hybrid.assert_called_once()
        mock_retrieve.assert_not_called()

    @patch("app.generation.response_generator.settings")
    @patch("app.generation.response_generator.retrieve_hybrid")
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    def test_hybrid_path_passes_through_top_k_and_source_filter(
        self, mock_get_prompt, mock_invoke, mock_retrieve_hybrid, mock_settings
    ):
        mock_settings.USE_HYBRID_RETRIEVAL = True
        mock_settings.LLM_MODEL = "gpt-4o-mini"
        mock_get_prompt.return_value = _mock_prompt()
        mock_invoke.return_value = "an answer"
        mock_retrieve_hybrid.return_value = [
            Document(page_content="text", metadata={"source": "f.pdf", "chunk_index": 0}),
        ]

        generate("What was my balance?", top_k=3, source_filter="statement.pdf")

        call_kwargs = mock_retrieve_hybrid.call_args.kwargs
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["source_filter"] == "statement.pdf"

    @patch("app.generation.response_generator.settings")
    @patch("app.generation.response_generator.retrieve_hybrid")
    @patch("app.generation.response_generator.invoke")
    @patch("app.generation.response_generator.get_prompt")
    def test_hybrid_empty_results_triggers_fallback(
        self, mock_get_prompt, mock_invoke, mock_retrieve_hybrid, mock_settings
    ):
        """Same fallback-to-NO_CONTEXT_PROMPT behavior should apply
        whether docs came back empty from retrieve() or retrieve_hybrid()."""
        mock_settings.USE_HYBRID_RETRIEVAL = True
        mock_settings.LLM_MODEL = "gpt-4o-mini"
        mock_get_prompt.return_value = _mock_prompt()
        mock_invoke.return_value = "I don't have enough information..."
        mock_retrieve_hybrid.return_value = []

        result = generate("What was my balance?")

        assert result.used_fallback is True
        assert result.retrieved_chunks == 0