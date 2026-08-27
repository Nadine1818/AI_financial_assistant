"""
Tests for app.retrieval.bm25_index

Run with:
    python -m pytest tests/test_bm25_index.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

import app.retrieval.bm25_index as bm25_index


@pytest.fixture(autouse=True)
def reset_singleton():
    """Every test starts with a clean (unbuilt) BM25 singleton."""
    bm25_index._bm25_retriever = None
    yield
    bm25_index._bm25_retriever = None


class TestGetBm25Retriever:
    @patch("app.retrieval.bm25_index.get_all_documents")
    def test_builds_index_from_documents(self, mock_get_all):
        mock_get_all.return_value = [
            Document(page_content="Revenue was £10,000 in January.", metadata={"source": "a.pdf", "chunk_index": 0}),
            Document(page_content="Expenses totalled £4,500 in January.", metadata={"source": "a.pdf", "chunk_index": 1}),
        ]

        retriever = bm25_index.get_bm25_retriever()

        assert retriever is not None
        mock_get_all.assert_called_once()

    @patch("app.retrieval.bm25_index.get_all_documents")
    def test_raises_on_empty_collection(self, mock_get_all):
        mock_get_all.return_value = []

        with pytest.raises(ValueError, match="empty"):
            bm25_index.get_bm25_retriever()

    @patch("app.retrieval.bm25_index.get_all_documents")
    def test_singleton_reused_across_calls(self, mock_get_all):
        mock_get_all.return_value = [
            Document(page_content="some text", metadata={"source": "a.pdf", "chunk_index": 0}),
        ]

        first = bm25_index.get_bm25_retriever()
        second = bm25_index.get_bm25_retriever()

        assert first is second
        mock_get_all.assert_called_once()  # only built once, not twice


class TestResetBm25Index:
    @patch("app.retrieval.bm25_index.get_all_documents")
    def test_reset_forces_rebuild(self, mock_get_all):
        mock_get_all.return_value = [
            Document(page_content="some text", metadata={"source": "a.pdf", "chunk_index": 0}),
        ]

        first = bm25_index.get_bm25_retriever()
        bm25_index.reset_bm25_index()
        second = bm25_index.get_bm25_retriever()

        assert first is not second
        assert mock_get_all.call_count == 2