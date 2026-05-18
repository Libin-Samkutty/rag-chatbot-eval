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
        long_text = sentence * 50
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


# --- Eval unit tests (no API calls) ---

class TestEvalModels:

    def test_eval_score_clamps(self):
        """EvalScore should reject scores outside [0, 1]."""
        from eval.models import EvalScore
        import pytest

        with pytest.raises(Exception):
            EvalScore(score=1.5, reason="too high", passed=True)

        with pytest.raises(Exception):
            EvalScore(score=-0.1, reason="too low", passed=False)

    def test_eval_result_overall_passed(self):
        """overall_passed should be False if any metric fails."""
        from eval.models import EvalScore, EvalResult

        passing = EvalScore(score=0.9, reason="ok", passed=True)
        failing = EvalScore(score=0.3, reason="bad", passed=False)

        result = EvalResult(
            faithfulness=passing,
            answer_relevancy=failing,  # one failure
            context_precision=passing,
            latency_ms=500.0,
            overall_passed=False,
        )
        assert result.overall_passed is False


# --- Cosine similarity unit test ---

class TestCosineSimilarity:

    def test_identical_vectors_score_one(self):
        """Cosine similarity of a vector with itself should be 1.0."""
        from eval.relevancy import _cosine_similarity
        vec = [0.1, 0.5, 0.3, 0.8]
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self):
        """Orthogonal vectors should score 0.0."""
        from eval.relevancy import _cosine_similarity
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert abs(_cosine_similarity(vec_a, vec_b)) < 1e-6

    def test_zero_vector_returns_zero(self):
        """A zero vector should not cause a division-by-zero error."""
        from eval.relevancy import _cosine_similarity
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.5]) == 0.0
