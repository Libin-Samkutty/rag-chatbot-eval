"""
tests/evals/test_lexical_eval.py — Lexical regression tests for golden dataset.

Runs with ZERO API calls — no OPENAI_API_KEY, no --run-evals flag required.
BERTScore uses roberta-large (~500 MB, cached in ~/.cache/huggingface/ after
the first run; subsequent runs use the local cache).

    pytest tests/evals/test_lexical_eval.py -v
    pytest tests/ -v -k "not eval_suite"   # lexical tests included automatically

Design invariant: adversarial entries (question_type == "adversarial") are
excluded from lexical scoring. Their reference_answer is a meta-refusal
statement, making BLEU/ROUGE/BERTScore comparisons meaningless.
"""

import json
from pathlib import Path

from eval.lexical_eval import compute_lexical_scores

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"


def _load_non_adversarial() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text())
    return [q for q in data if q.get("question_type") != "adversarial"]


def test_rouge_l_perfect_on_identical_strings() -> None:
    """Identical strings must produce ROUGE-L = 1.0 and passed=True."""
    result = compute_lexical_scores("x", "x")
    assert result.rouge_l == 1.0
    assert result.passed is True


def test_lexical_result_model_shape() -> None:
    """LexicalResult must expose all four attributes with the correct types."""
    result = compute_lexical_scores(
        "World War I began in 1914.",
        "World War I started in 1914.",
    )
    assert hasattr(result, "bleu")
    assert hasattr(result, "rouge_l")
    assert hasattr(result, "bertscore_f1")
    assert hasattr(result, "passed")
    assert isinstance(result.bleu, float)
    assert isinstance(result.rouge_l, float)
    assert isinstance(result.bertscore_f1, float)
    assert isinstance(result.passed, bool)


def test_bleu_logged_not_gated() -> None:
    """
    BLEU is logged for observability but is excluded from the pass gate.

    passed must equal (rouge_l >= 0.25 AND bertscore_f1 >= 0.85) regardless
    of the BLEU value. This test enforces that logic using two strings that
    are semantically similar but lexically distinct (different n-grams).
    """
    answer = "World War I ended in November 1918 when an armistice was signed."
    reference = "The conflict concluded in the autumn of 1918 via a ceasefire agreement."
    result = compute_lexical_scores(answer, reference)

    assert hasattr(result, "bleu")
    assert isinstance(result.bleu, float)
    # Verify the gate formula — bleu must not appear in this expression
    assert result.passed == (result.rouge_l >= 0.25 and result.bertscore_f1 >= 0.85)


def test_adversarial_entries_excluded() -> None:
    """
    Design decision: adversarial entries must not be passed to compute_lexical_scores().

    Their reference_answer is a short meta-refusal statement ("This question is
    outside the scope of the World History knowledge base."), not a factual answer.
    Comparing a real model answer against a refusal string would produce meaningless
    similarity scores — high BLEU/ROUGE/BERTScore would not indicate quality, and
    low scores would be false negatives.

    This test confirms the golden dataset has adversarial entries and that each one
    carries a recognisably short meta-refusal reference_answer (< 200 characters),
    not a factual answer that would support meaningful lexical comparison.
    """
    data = json.loads(GOLDEN_PATH.read_text())
    adversarial = [q for q in data if q.get("question_type") == "adversarial"]

    assert adversarial, (
        "Golden dataset must contain at least one adversarial entry — "
        "check golden_dataset.json has question_type == 'adversarial' entries."
    )

    for entry in adversarial:
        ref = entry.get("reference_answer", "")
        assert len(ref) < 200, (
            f"Adversarial entry {entry['id']} has a long reference_answer "
            f"({len(ref)} chars) — expected a short meta-refusal statement, "
            "not a factual answer. Do not pass adversarial entries to "
            "compute_lexical_scores()."
        )


def test_bertscore_average_above_threshold() -> None:
    """
    Self-similarity check: each non-adversarial reference_answer compared to
    itself must yield mean BERTScore F1 >= 0.85.

    This validates that:
      1. The roberta-large model loaded and ran correctly.
      2. compute_lexical_scores() returns sensible bertscore_f1 values.

    Note: the first run downloads roberta-large (~500 MB) to ~/.cache/huggingface/.
    Subsequent runs use the local cache and complete in seconds.
    """
    entries = _load_non_adversarial()
    assert entries, "No non-adversarial entries found in golden_dataset.json"

    scores = []
    for entry in entries:
        ref = entry["reference_answer"]
        result = compute_lexical_scores(ref, ref)
        scores.append(result.bertscore_f1)

    mean_score = sum(scores) / len(scores)
    assert mean_score >= 0.85, (
        f"Mean BERTScore F1 on self-similarity check is {mean_score:.4f} "
        f"(over {len(scores)} non-adversarial entries) — expected >= 0.85. "
        "Check that roberta-large loaded correctly and compute_lexical_scores() "
        "returns valid bertscore_f1 values."
    )
