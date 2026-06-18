"""
eval/lexical_eval.py — BLEU, ROUGE-L, and BERTScore lexical metrics.

API-free complementary layer for golden dataset regression testing.
This module is NEVER imported from eval/runner.py — it lives entirely
in the test path (tests/evals/test_lexical_eval.py).

Exclusion rule: adversarial entries (question_type == "adversarial") must
not be passed here. Their reference_answer is a meta-refusal statement
("This question is outside the scope of..."), making similarity scores
meaningless.
"""

import logging

import nltk

from eval.models import LexicalResult

logger = logging.getLogger(__name__)

# Ensure NLTK punkt tokenizer data is present (needed by some evaluate backends).
# quiet=True suppresses output; a missing download is non-fatal.
nltk.download("punkt_tab", quiet=True)

# Module-level metric caches — avoid reloading evaluate metrics on every call.
_bleu_metric = None
_rouge_metric = None


def compute_lexical_scores(answer: str, reference: str) -> LexicalResult:
    """
    Compute BLEU-1, ROUGE-L F1, and BERTScore F1 for answer vs. reference.

    Synchronous — HuggingFace evaluate and bert_score are not async-capable.
    On any exception, returns a zero-score LexicalResult(passed=False) and
    logs a warning so callers can continue without crashing.

    Gate logic (see LexicalResult docstring):
      passed = rouge_l >= 0.25 AND bertscore_f1 >= 0.85
      bleu is logged for observability and excluded from the pass gate.
    """
    global _bleu_metric, _rouge_metric

    try:
        import bert_score as bs
        import evaluate

        if _bleu_metric is None:
            _bleu_metric = evaluate.load("bleu")
        if _rouge_metric is None:
            _rouge_metric = evaluate.load("rouge")

        # BLEU-1 unigram precision.
        # references must be a list-of-lists for the evaluate BLEU metric.
        bleu_result = _bleu_metric.compute(
            predictions=[answer],
            references=[[reference]],
            max_order=1,
        )
        bleu = float(bleu_result.get("bleu", 0.0))

        # ROUGE-L F1.
        rouge_result = _rouge_metric.compute(
            predictions=[answer],
            references=[reference],
        )
        rouge_l = float(rouge_result.get("rougeL", 0.0))

        # BERTScore F1 using roberta-large (~500 MB, cached in ~/.cache/huggingface/).
        # verbose=False suppresses the per-call progress bar.
        _, _, f1 = bs.score(
            [answer],
            [reference],
            model_type="roberta-large",
            verbose=False,
        )
        bertscore_f1 = float(f1[0].item())

        passed = rouge_l >= 0.25 and bertscore_f1 >= 0.85

        return LexicalResult(
            bleu=bleu,
            rouge_l=rouge_l,
            bertscore_f1=bertscore_f1,
            passed=passed,
        )

    except Exception as e:
        logger.warning("lexical_eval failed (non-fatal): %s", e)
        return LexicalResult(bleu=0.0, rouge_l=0.0, bertscore_f1=0.0, passed=False)
