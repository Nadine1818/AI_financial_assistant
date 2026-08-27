"""
Tests for app.orchestration.rag_graph

Tests the corrective-RAG control flow in isolation:
    - PASS on first try → no retry, 1 attempt
    - FAIL then PASS → exactly one retry, 2 attempts
    - FAIL persisting → stops at MAX_RETRIES, returns the final FAIL

generate(), verify(), and invoke() are mocked here because their own
correctness is already covered by test_response_generator.py,
test_llm.py, and the (network-dependent) verifier tests. This file only
tests that rag_graph.py wires them together and loops correctly.

Run with:
    python -m pytest tests/test_rag_graph.py -v
"""
import pytest
from unittest.mock import patch, MagicMock

from app.orchestration.rag_graph import run, MAX_RETRIES
from app.generation.response_generator import GenerationResult
from app.validation.verifier import VerificationResult


def _gen_result(answer="draft answer"):
    return GenerationResult(
        answer=answer,
        context="[Source: f.pdf | Chunk 0]\nsome context",
        sources=["f.pdf"],
        question="rewritten or original question",
        retrieved_chunks=1,
        used_fallback=False,
        metadata={},
    )


def _ver_result(verdict, answer="draft answer", explanation="because"):
    return VerificationResult(
        verdict=verdict,
        explanation=explanation,
        safe_answer=answer,
        sources=["f.pdf"],
        original_answer=answer,
        metadata={},
    )


class TestPassOnFirstTry:
    @patch("app.orchestration.rag_graph.verify")
    @patch("app.orchestration.rag_graph.generate")
    def test_no_retry_when_pass(self, mock_generate, mock_verify):
        mock_generate.return_value = _gen_result()
        mock_verify.return_value = _ver_result("PASS")

        result = run("What was my balance in January?")

        assert result.verdict == "PASS"
        assert result.metadata["graph_attempts"] == 1
        mock_generate.assert_called_once()
        mock_verify.assert_called_once()


class TestUncertainDoesNotRetry:
    @patch("app.orchestration.rag_graph.verify")
    @patch("app.orchestration.rag_graph.generate")
    def test_no_retry_when_uncertain(self, mock_generate, mock_verify):
        mock_generate.return_value = _gen_result()
        mock_verify.return_value = _ver_result("UNCERTAIN")

        result = run("What was my balance in January?")

        assert result.verdict == "UNCERTAIN"
        assert result.metadata["graph_attempts"] == 1
        mock_generate.assert_called_once()


class TestFailThenPass:
    @patch("app.orchestration.rag_graph.invoke")
    @patch("app.orchestration.rag_graph.verify")
    @patch("app.orchestration.rag_graph.generate")
    def test_retries_once_then_succeeds(self, mock_generate, mock_verify, mock_invoke):
        mock_generate.side_effect = [_gen_result("bad answer"), _gen_result("good answer")]
        mock_verify.side_effect = [_ver_result("FAIL", "bad answer"), _ver_result("PASS", "good answer")]
        mock_invoke.return_value = "rewritten query"

        result = run("What was my balance in January?")

        assert result.verdict == "PASS"
        assert result.metadata["graph_attempts"] == 2
        assert mock_generate.call_count == 2
        assert mock_verify.call_count == 2
        mock_invoke.assert_called_once()  # rewrite node called exactly once


class TestFailPersists:
    @patch("app.orchestration.rag_graph.invoke")
    @patch("app.orchestration.rag_graph.verify")
    @patch("app.orchestration.rag_graph.generate")
    def test_stops_at_max_retries(self, mock_generate, mock_verify, mock_invoke):
        mock_generate.return_value = _gen_result("bad answer")
        mock_verify.return_value = _ver_result("FAIL", "bad answer")
        mock_invoke.return_value = "rewritten query"

        result = run("What was my balance in January?")

        assert result.verdict == "FAIL"
        # 1 initial attempt + MAX_RETRIES retries = MAX_RETRIES + 1 total
        assert result.metadata["graph_attempts"] == MAX_RETRIES + 1
        assert mock_generate.call_count == MAX_RETRIES + 1
        assert mock_invoke.call_count == MAX_RETRIES  # one rewrite per retry


class TestRewriteUsesOriginalQuestion:
    @patch("app.orchestration.rag_graph.invoke")
    @patch("app.orchestration.rag_graph.verify")
    @patch("app.orchestration.rag_graph.generate")
    def test_rewrite_always_based_on_original_question(self, mock_generate, mock_verify, mock_invoke):
        """The rewrite prompt should always reference the ORIGINAL question,
        not a previous rewrite — even on the second retry."""
        mock_generate.return_value = _gen_result("bad answer")
        mock_verify.return_value = _ver_result("FAIL", "bad answer")
        mock_invoke.return_value = "rewritten query"

        run("original question text")

        # Every call to invoke() (the rewrite LLM call) should have been
        # built from a prompt containing the original question, not the
        # previous rewrite output.
        for call in mock_invoke.call_args_list:
            messages = call.args[0]
            combined = " ".join(m.content for m in messages)
            assert "original question text" in combined


class TestEmptyQuestion:
    def test_raises_on_empty_question(self):
        with pytest.raises(ValueError):
            run("")

    def test_raises_on_whitespace_question(self):
        with pytest.raises(ValueError):
            run("   ")


class TestRunWithHistory:
    @patch("app.orchestration.rag_graph.invoke")
    @patch("app.orchestration.rag_graph.verify")
    @patch("app.orchestration.rag_graph.generate")
    def test_first_turn_skips_condensing(self, mock_generate, mock_verify, mock_invoke):
        """No history yet — should NOT call invoke() to condense."""
        from app.orchestration.rag_graph import run_with_history

        mock_generate.return_value = _gen_result()
        mock_verify.return_value = _ver_result("PASS")

        result = run_with_history("What was my balance in January?", [])

        assert result.verdict == "PASS"
        mock_invoke.assert_not_called()  # no history → no condense LLM call

    @patch("app.orchestration.rag_graph.invoke")
    @patch("app.orchestration.rag_graph.verify")
    @patch("app.orchestration.rag_graph.generate")
    def test_follow_up_condenses_before_running_graph(self, mock_generate, mock_verify, mock_invoke):
        from app.orchestration.rag_graph import run_with_history

        mock_invoke.return_value = "What was my balance in February?"
        mock_generate.return_value = _gen_result()
        mock_verify.return_value = _ver_result("PASS")

        history = [("What was my balance in January?", "£2,400.")]
        result = run_with_history("What about February?", history)

        assert result.verdict == "PASS"
        assert result.metadata["original_question"] == "What about February?"
        mock_invoke.assert_called_once()  # condense step ran
        # generate() should receive the CONDENSED question, not the raw follow-up
        called_question = mock_generate.call_args.kwargs.get("question") or mock_generate.call_args.args[0]
        assert called_question == "What was my balance in February?"

    def test_raises_on_empty_question(self):
        from app.orchestration.rag_graph import run_with_history
        with pytest.raises(ValueError):
            run_with_history("", [])