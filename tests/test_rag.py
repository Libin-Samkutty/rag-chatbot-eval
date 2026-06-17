"""
tests/test_rag.py — Unit tests for the RAG pipeline components.

These tests do NOT make real API calls. The chunker is pure Python, so it
can be tested directly. The retriever tests use mocked OpenAI responses.
"""

import pytest

from rag.chunker import chunk_text


# --- Chunker tests ---

class TestChunker:

    def test_short_text_produces_one_chunk(self):
        """Text shorter than CHUNK_SIZE should produce exactly one chunk."""
        text = "This is a short article about machine learning."
        chunks = chunk_text(text, source="test.txt")
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["source"] == "test.txt"
        assert chunks[0]["chunk_index"] == 0

    def test_long_text_produces_multiple_chunks(self):
        """Text much longer than CHUNK_SIZE should produce multiple chunks."""
        # Generate a long text (~1000 tokens)
        sentence = "The transformer model uses self-attention mechanisms. "
        long_text = sentence * 100  # ~1000 tokens, well above CHUNK_SIZE=512
        chunks = chunk_text(long_text, source="long.txt")
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        """Adjacent chunks should share some content due to the overlap window."""
        sentence = "Word " * 200  # ~200 tokens
        text = sentence * 5       # ~1000 tokens
        chunks = chunk_text(text, source="overlap_test.txt")
        assert len(chunks) >= 2

        # The end of chunk 0 should appear at the start of chunk 1
        # (due to the 50-token overlap)
        end_of_first = chunks[0]["text"][-100:]  # last 100 chars of chunk 0
        start_of_second = chunks[1]["text"][:100]  # first 100 chars of chunk 1

        # There should be some overlap — at least some shared words
        words_first = set(end_of_first.split())
        words_second = set(start_of_second.split())
        shared = words_first & words_second
        assert len(shared) > 0, "Expected overlap between adjacent chunks"

    def test_chunk_metadata(self):
        """Each chunk should carry correct source and index metadata."""
        text = "Test content. " * 100
        chunks = chunk_text(text, source="meta_test.txt")
        for i, chunk in enumerate(chunks):
            assert chunk["source"] == "meta_test.txt"
            assert chunk["chunk_index"] == i

    def test_empty_text_produces_no_chunks(self):
        """Empty string should produce no chunks (or one empty chunk — both acceptable)."""
        chunks = chunk_text("", source="empty.txt")
        # Either no chunks or one empty chunk
        assert len(chunks) <= 1

    def test_chunk_text_is_string(self):
        """All chunk texts should be strings, not bytes."""
        text = "Machine learning is a subset of artificial intelligence. " * 20
        chunks = chunk_text(text, source="type_test.txt")
        for chunk in chunks:
            assert isinstance(chunk["text"], str)

    def test_chunk_size_constant(self):
        """CHUNK_SIZE should be 512 tokens."""
        from rag.chunker import CHUNK_SIZE
        assert CHUNK_SIZE == 512

    def test_domain_tag_ww1(self):
        """Files prefixed ww1_ should receive domain_tag == 'ww1'."""
        chunks = chunk_text("World War I began in 1914.", source="ww1_world_war_i.txt")
        assert chunks[0]["domain_tag"] == "ww1"

    def test_domain_tag_historical_figures(self):
        """Files prefixed figures_ should receive domain_tag == 'historical_figures'."""
        chunks = chunk_text(
            "Napoleon Bonaparte was a French military leader.",
            source="figures_napoleon_bonaparte.txt",
        )
        assert chunks[0]["domain_tag"] == "historical_figures"

    def test_domain_tag_general_fallback(self):
        """Files with no recognised prefix should receive domain_tag == 'general'."""
        chunks = chunk_text("Some generic content.", source="some_file.txt")
        assert chunks[0]["domain_tag"] == "general"


# --- Eval unit tests (no API calls) ---

class TestEvalModels:

    def test_checklist_item_construction(self):
        """ChecklistItem should accept valid tier values."""
        from eval.models import ChecklistItem
        item = ChecklistItem(
            key="faith_no_hallucination",
            question="Does the answer avoid hallucinations?",
            result=True,
            tier=1,
        )
        assert item.key == "faith_no_hallucination"
        assert item.result is True
        assert item.tier == 1

    def test_dimension_result_construction(self):
        """DimensionResult should store all fields correctly."""
        from eval.models import ChecklistItem, DimensionResult
        item = ChecklistItem(
            key="faith_no_hallucination",
            question="No hallucinations?",
            result=True,
            tier=1,
        )
        dim = DimensionResult(
            name="faithfulness",
            passed=True,
            items=[item],
            score=None,
            reason="All checks passed.",
            tier1_failed=[],
            tier2_pass_rate=1.0,
        )
        assert dim.passed is True
        assert dim.name == "faithfulness"
        assert len(dim.items) == 1

    def test_eval_result_overall_passed(self):
        """overall_passed should be False if any dimension fails."""
        from eval.models import DimensionResult, EvalResult

        def _dim(name: str, passed: bool) -> DimensionResult:
            return DimensionResult(
                name=name,
                passed=passed,
                items=[],
                score=None,
                reason="ok" if passed else "failed",
                tier1_failed=[] if passed else ["some_key"],
                tier2_pass_rate=1.0 if passed else 0.0,
            )

        result = EvalResult(
            faithfulness=_dim("faithfulness", True),
            answer_relevancy=_dim("answer_relevancy", False),  # one failure
            completeness=_dim("completeness", True),
            context_precision=_dim("context_precision", True),
            context_recall=_dim("context_recall", True),
            coherence=_dim("coherence", True),
            historical_balance=_dim("historical_balance", True),
            toxicity=_dim("toxicity", True),
            latency_ms=500.0,
            overall_passed=False,
        )
        assert result.overall_passed is False

    def test_eval_result_all_pass(self):
        """overall_passed should be True when all dimensions pass."""
        from eval.models import DimensionResult, EvalResult

        def _dim(name: str) -> DimensionResult:
            return DimensionResult(
                name=name,
                passed=True,
                items=[],
                score=None,
                reason="passed",
                tier1_failed=[],
                tier2_pass_rate=1.0,
            )

        result = EvalResult(
            faithfulness=_dim("faithfulness"),
            answer_relevancy=_dim("answer_relevancy"),
            completeness=_dim("completeness"),
            context_precision=_dim("context_precision"),
            context_recall=_dim("context_recall"),
            coherence=_dim("coherence"),
            historical_balance=_dim("historical_balance"),
            toxicity=_dim("toxicity"),
            latency_ms=200.0,
            overall_passed=True,
        )
        assert result.overall_passed is True


# --- Checklist evaluation logic tests (no API calls) ---

class TestChecklistEvaluators:

    def test_faithfulness_tier1_failure(self):
        """Faithfulness dimension must fail when a Tier 1 item is False."""
        from eval.models import ChecklistItem
        from eval.checklists.faithfulness import evaluate_faithfulness
        items = [
            ChecklistItem(key="faith_no_hallucination", question="?", result=False, tier=1),
            ChecklistItem(key="faith_no_contradiction", question="?", result=True, tier=1),
            ChecklistItem(key="faith_temporal_accuracy", question="?", result=True, tier=2),
            ChecklistItem(key="faith_numeric_fidelity", question="?", result=True, tier=2),
            ChecklistItem(key="faith_proper_naming", question="?", result=True, tier=2),
        ]
        result = evaluate_faithfulness(items)
        assert result.passed is False
        assert "faith_no_hallucination" in result.tier1_failed

    def test_faithfulness_passes_all_ok(self):
        """Faithfulness dimension passes when all items are True."""
        from eval.models import ChecklistItem
        from eval.checklists.faithfulness import evaluate_faithfulness
        items = [
            ChecklistItem(key="faith_no_hallucination", question="?", result=True, tier=1),
            ChecklistItem(key="faith_no_contradiction", question="?", result=True, tier=1),
            ChecklistItem(key="faith_temporal_accuracy", question="?", result=True, tier=2),
            ChecklistItem(key="faith_numeric_fidelity", question="?", result=True, tier=2),
            ChecklistItem(key="faith_proper_naming", question="?", result=True, tier=2),
        ]
        result = evaluate_faithfulness(items)
        assert result.passed is True
        assert result.tier1_failed == []
        assert result.tier2_pass_rate == 1.0

    def test_faithfulness_tier2_threshold_fail(self):
        """Faithfulness fails when fewer than 2 Tier 2 items pass."""
        from eval.models import ChecklistItem
        from eval.checklists.faithfulness import evaluate_faithfulness
        items = [
            ChecklistItem(key="faith_no_hallucination", question="?", result=True, tier=1),
            ChecklistItem(key="faith_no_contradiction", question="?", result=True, tier=1),
            ChecklistItem(key="faith_temporal_accuracy", question="?", result=False, tier=2),
            ChecklistItem(key="faith_numeric_fidelity", question="?", result=False, tier=2),
            ChecklistItem(key="faith_proper_naming", question="?", result=True, tier=2),
        ]
        result = evaluate_faithfulness(items)
        # Only 1 of 3 Tier 2 items passed — below threshold of 2
        assert result.passed is False

    def test_completeness_geval_passes_above_threshold(self):
        """Completeness G-Eval passes when score >= 0.7."""
        import asyncio
        from unittest.mock import patch
        from eval.deepeval_eval import score_completeness

        with patch("eval.deepeval_eval._run_geval_sync", return_value=0.8):
            result = asyncio.run(score_completeness("What caused WW1?", ["context chunk"], "The answer."))

        assert result.passed is True
        assert result.score == pytest.approx(0.8)
        assert result.name == "completeness"

    def test_completeness_geval_fails_below_threshold(self):
        """Completeness G-Eval fails when score < 0.7."""
        import asyncio
        from unittest.mock import patch
        from eval.deepeval_eval import score_completeness

        with patch("eval.deepeval_eval._run_geval_sync", return_value=0.5):
            result = asyncio.run(score_completeness("What caused WW1?", ["context chunk"], "The answer."))

        assert result.passed is False
        assert result.score == pytest.approx(0.5)

    def test_context_precision_threshold(self):
        """Context precision passes when >=75% of chunks are relevant."""
        from eval.models import ChecklistItem
        from eval.checklists.context_precision import evaluate_context_precision
        # 3 of 4 chunks relevant = 75% = exactly threshold → passes
        items = [
            ChecklistItem(key="precision_chunk_relevant", question="?", result=True, tier=2),
            ChecklistItem(key="precision_chunk_relevant", question="?", result=True, tier=2),
            ChecklistItem(key="precision_chunk_relevant", question="?", result=True, tier=2),
            ChecklistItem(key="precision_chunk_relevant", question="?", result=False, tier=2),
        ]
        result = evaluate_context_precision(items)
        assert result.passed is True  # 0.75 >= 0.75

    def test_context_recall_below_threshold(self):
        """Context recall fails when fewer than 80% of claims are covered."""
        from eval.models import ChecklistItem
        from eval.checklists.context_recall import evaluate_context_recall
        # 3 of 5 = 60% < 80%
        items = [
            ChecklistItem(key="recall_claim_covered", question="?", result=True, tier=2),
            ChecklistItem(key="recall_claim_covered", question="?", result=True, tier=2),
            ChecklistItem(key="recall_claim_covered", question="?", result=True, tier=2),
            ChecklistItem(key="recall_claim_covered", question="?", result=False, tier=2),
            ChecklistItem(key="recall_claim_covered", question="?", result=False, tier=2),
        ]
        result = evaluate_context_recall(items)
        assert result.passed is False
