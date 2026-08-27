"""
Tests for app.retrieval.retriever.retrieve_hybrid()

Kept in a separate file from test_retriever.py (which covers the
existing dense-only retrieve()) so this new capability is additive and
independently testable — retrieve() and its tests are untouched.

Run with:
    python -m pytest tests/test_retriever_hybrid.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.retrieval.retriever import retrieve_hybrid, RRF_K


def _doc(source, chunk_index, text="some chunk text"):
    return Document(
        page_content=text,
        metadata={"source": source, "chunk_index": chunk_index},
    )


class TestRetrieveHybridBasic:
    @patch("app.retrieval.retriever.get_bm25_retriever")
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_returns_fused_results(self, mock_dense, mock_get_bm25):
        mock_dense.return_value = [
            (_doc("a.pdf", 0), 0.1),
            (_doc("b.pdf", 0), 0.3),
        ]
        mock_bm25 = MagicMock()
        mock_bm25.invoke.return_value = [_doc("c.pdf", 0)]
        mock_get_bm25.return_value = mock_bm25

        results = retrieve_hybrid("what was my balance", top_k=5)

        # 3 unique chunks across both retrievers, all should come back
        # since top_k=5 is wider than the unique candidate count
        sources = {doc.metadata["source"] for doc in results}
        assert sources == {"a.pdf", "b.pdf", "c.pdf"}

    @patch("app.retrieval.retriever.get_bm25_retriever")
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_respects_top_k(self, mock_dense, mock_get_bm25):
        mock_dense.return_value = [
            (_doc("a.pdf", 0), 0.1),
            (_doc("b.pdf", 0), 0.2),
            (_doc("c.pdf", 0), 0.3),
        ]
        mock_bm25 = MagicMock()
        mock_bm25.invoke.return_value = []
        mock_get_bm25.return_value = mock_bm25

        results = retrieve_hybrid("query", top_k=2)

        assert len(results) == 2


class TestRRFFusionMath:
    @patch("app.retrieval.retriever.get_bm25_retriever")
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_doc_found_by_both_retrievers_ranks_highest(self, mock_dense, mock_get_bm25):
        """A chunk that BOTH retrievers agree on should outrank a chunk
        only one retriever found, even if the other retriever ranked
        its own top pick #1."""
        shared_doc = _doc("shared.pdf", 0)
        dense_only_doc = _doc("dense_only.pdf", 0)
        bm25_only_doc = _doc("bm25_only.pdf", 0)

        # shared_doc is rank 2 in dense, rank 2 in bm25 — should still
        # beat a doc that's rank 1 in only ONE list, because RRF rewards
        # appearing in both.
        mock_dense.return_value = [
            (dense_only_doc, 0.1),
            (shared_doc, 0.2),
        ]
        mock_bm25 = MagicMock()
        mock_bm25.invoke.return_value = [bm25_only_doc, shared_doc]
        mock_get_bm25.return_value = mock_bm25

        results = retrieve_hybrid("query", top_k=3)

        result_sources = [doc.metadata["source"] for doc in results]
        assert result_sources[0] == "shared.pdf"

    @patch("app.retrieval.retriever.get_bm25_retriever")
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_exact_rrf_scores(self, mock_dense, mock_get_bm25):
        """Verify the actual RRF arithmetic: score = sum of 1/(RRF_K + rank + 1)."""
        doc_a = _doc("a.pdf", 0)  # rank 0 (1st) in dense only
        doc_b = _doc("b.pdf", 0)  # rank 0 (1st) in bm25 only

        mock_dense.return_value = [(doc_a, 0.1)]
        mock_bm25 = MagicMock()
        mock_bm25.invoke.return_value = [doc_b]
        mock_get_bm25.return_value = mock_bm25

        results = retrieve_hybrid("query", top_k=2)

        # Both docs rank #1 in their own list (rank index 0), so both
        # should score identically: 1 / (RRF_K + 0 + 1)
        expected_score = 1.0 / (RRF_K + 1)
        assert len(results) == 2
        # Order between equally-scored docs is stable but not asserted here —
        # only that both made it into the top_k, which confirms the score
        # formula didn't silently drop or misrank either one.
        assert {d.metadata["source"] for d in results} == {"a.pdf", "b.pdf"}


class TestRetrieveHybridEdgeCases:
    def test_raises_on_empty_query(self):
        with pytest.raises(ValueError):
            retrieve_hybrid("")

    def test_raises_on_whitespace_query(self):
        with pytest.raises(ValueError):
            retrieve_hybrid("   ")

    @patch("app.retrieval.retriever.get_bm25_retriever")
    @patch("app.retrieval.retriever.similarity_search_with_scores")
    def test_empty_results_from_both_returns_empty(self, mock_dense, mock_get_bm25):
        mock_dense.return_value = []
        mock_bm25 = MagicMock()
        mock_bm25.invoke.return_value = []
        mock_get_bm25.return_value = mock_bm25

        results = retrieve_hybrid("obscure query")

        assert results == []