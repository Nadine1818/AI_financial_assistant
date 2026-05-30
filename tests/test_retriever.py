"""
Tests for app.retrieval.retriever

Tests semantic retrieval:
    - retrieve() with threshold filtering
    - retrieve_with_scores() for score-based filtering
    - Error handling and edge cases

Run with:
    pytest tests/test_retriever.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.retrieval.retriever import (
    retrieve,
    retrieve_with_scores,
    RELEVANCE_THRESHOLD,
)


class TestRetrieve:
    """Test standard retrieve() function."""
    
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_retrieve_returns_documents_above_threshold(self, mock_search):
        """Should return only documents below relevance threshold."""
        # Scores: 0.0 = identical, 1.0 = unrelated
        # RELEVANCE_THRESHOLD = 0.45
        # Keep scores < 0.45, discard >= 0.45
        mock_search.return_value = [
            (Document(page_content="Good match", metadata={"source": "f1.pdf"}), 0.2),
            (Document(page_content="Poor match", metadata={"source": "f2.pdf"}), 0.6),
        ]
        
        result = retrieve("test query")
        
        # Should only return the good match
        assert len(result) == 1
        assert result[0].page_content == "Good match"
    
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_retrieve_returns_empty_when_no_matches(self, mock_search):
        """Should return empty list when nothing passes threshold."""
        mock_search.return_value = [
            (Document(page_content="Bad match", metadata={"source": "f.pdf"}), 0.8),
        ]
        
        result = retrieve("obscure question")
        
        assert result == []
    
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_retrieve_passes_top_k_to_search(self, mock_search):
        """Should pass top_k parameter to similarity_search_with_scores."""
        mock_search.return_value = []
        
        retrieve("query", top_k=50)
        
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["top_k"] == 50
    
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_retrieve_uses_default_top_k_from_settings(self, mock_search):
        """Should use settings.RETRIEVAL_TOP_K when top_k is None."""
        mock_search.return_value = []
        
        retrieve("query", top_k=None)
        
        # Should be called with some positive k value from settings
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["top_k"] > 0
    
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_retrieve_applies_source_filter(self, mock_search):
        """Should pass source filter as ChromaDB metadata filter."""
        mock_search.return_value = []
        
        retrieve("query", source_filter="specific.pdf")
        
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs.get("filter") == {"source": "specific.pdf"}
    
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_retrieve_no_filter_when_source_none(self, mock_search):
        """Should not apply filter when source_filter is None."""
        mock_search.return_value = []
        
        retrieve("query", source_filter=None)
        
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs.get("filter") is None
    
    def test_retrieve_raises_on_empty_query(self):
        """Should raise ValueError for empty query strings."""
        with pytest.raises(ValueError, match="empty"):
            retrieve("")
        
        with pytest.raises(ValueError, match="empty"):
            retrieve("   ")
    
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_retrieve_preserves_order_by_relevance(self, mock_search):
        """Should return documents in relevance order (best first)."""
        mock_search.return_value = [
            (Document(page_content="Best", metadata={"source": "f1.pdf"}), 0.1),
            (Document(page_content="Good", metadata={"source": "f2.pdf"}), 0.3),
            (Document(page_content="Acceptable", metadata={"source": "f3.pdf"}), 0.4),
        ]
        
        result = retrieve("query")
        
        assert len(result) == 3
        assert result[0].page_content == "Best"
        assert result[1].page_content == "Good"
        assert result[2].page_content == "Acceptable"


class TestRetrieveWithScores:
    """Test retrieve_with_scores() function."""
    
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_retrieve_with_scores_returns_tuples(self, mock_search):
        """Should return (Document, score) tuples."""
        doc = Document(page_content="content", metadata={"source": "f.pdf"})
        mock_search.return_value = [(doc, 0.25)]
        
        result = retrieve_with_scores("query")
        
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert result[0][0] == doc
        assert result[0][1] == 0.25
    
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_retrieve_with_scores_filters_by_threshold(self, mock_search):
        """Should only return scores below RELEVANCE_THRESHOLD."""
        mock_search.return_value = [
            (Document(page_content="Good", metadata={"source": "f1.pdf"}), 0.2),
            (Document(page_content="Bad", metadata={"source": "f2.pdf"}), 0.7),
        ]
        
        result = retrieve_with_scores("query")
        
        assert len(result) == 1
        assert result[0][1] == 0.2
    
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_retrieve_with_scores_returns_empty_on_no_matches(self, mock_search):
        """Should return empty list when nothing passes threshold."""
        mock_search.return_value = []
        
        result = retrieve_with_scores("query")
        
        assert result == []


class TestFormatContext:
    """Test format_context() helper function."""
    
    def test_format_context_creates_labeled_blocks(self):
        """Should format documents as [Source: X | Chunk N] followed by content."""
        from app.retrieval.retriever import format_context
        
        docs = [
            Document(page_content="First chunk", metadata={"source": "doc1.pdf", "chunk_index": 1}),
            Document(page_content="Second chunk", metadata={"source": "doc2.pdf", "chunk_index": 2}),
        ]
        
        result = format_context(docs)
        
        assert "[Source: doc1.pdf | Chunk 1]" in result
        assert "First chunk" in result
        assert "[Source: doc2.pdf | Chunk 2]" in result
        assert "Second chunk" in result
    
    def test_format_context_empty_list(self):
        """Should return empty string for empty document list."""
        from app.retrieval.retriever import format_context
        
        result = format_context([])
        
        assert result == ""
    
    def test_format_context_preserves_chunk_numbers(self):
        """Should number chunks sequentially starting from 1."""
        from app.retrieval.retriever import format_context
        
        docs = [
            Document(page_content="A", metadata={"source": "f.pdf", "chunk_index": 1}),
            Document(page_content="B", metadata={"source": "f.pdf", "chunk_index": 2}),
            Document(page_content="C", metadata={"source": "f.pdf", "chunk_index": 3}),
        ]
        
        result = format_context(docs)
        
        assert "Chunk 1" in result
        assert "Chunk 2" in result
        assert "Chunk 3" in result
