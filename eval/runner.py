"""
eval/runner.py — Orchestrates all eight evaluation dimensions for one chat turn.

All eight dimensions run concurrently via asyncio.gather, so the total
eval time is approximately the slowest individual dimension rather than
the sum of all eight.

Dimension map:
  Layer 1 — Retrieval:
    context_precision   → RAGAS ContextPrecision (eval/ragas_eval.py)
    context_recall      → RAGAS ContextRecall    (eval/ragas_eval.py)

  Layer 2 — Generation:
    faithfulness        → RAGAS-inspired GPT-4o judge (eval/ragas_eval.py)
    answer_relevancy    → RAGAS-inspired GPT-4o judge (eval/ragas_eval.py)
    completeness        → DeepEval G-Eval              (eval/deepeval_eval.py)
    coherence           → DeepEval G-Eval              (eval/deepeval_eval.py)
    historical_balance  → DeepEval G-Eval              (eval/deepeval_eval.py)
    toxicity            → DeepEval G-Eval              (eval/deepeval_eval.py)

Adding a new dimension:
  1. Implement async score_<name>() in eval/ragas_eval.py or eval/deepeval_eval.py
  2. Import it here and add it to asyncio.gather
  3. Add the result to the EvalResult constructor
  4. Add the field to EvalResult in eval/models.py
"""

import asyncio
import logging
import os
import time

from openai import AsyncOpenAI

from eval.deepeval_eval import (
    score_coherence,
    score_completeness,
    score_historical_balance,
    score_toxicity,
)
from eval.models import EvalResult
from eval.ragas_eval import (
    score_answer_relevancy,
    score_context_precision,
    score_context_recall,
    score_faithfulness,
)

logger = logging.getLogger(__name__)


async def run_evals(
    question: str,
    context: list[str],
    answer: str,
    reference_answer: str | None,
    start_time: float,
    openai_client: AsyncOpenAI,
    domain_tag: str | None = None,
) -> EvalResult:
    """
    Run all eight evaluation dimensions concurrently and return EvalResult.

    Args:
        question:         The user's original question.
        context:          The list of retrieved chunk texts passed to the LLM.
        answer:           The model's generated answer.
        reference_answer: Gold reference answer (from golden dataset), or None
                          for live chat mode. context_recall falls back to a
                          reference-free heuristic when None.
        start_time:       time.perf_counter() value captured before the RAG
                          pipeline started — used for end-to-end latency.
        openai_client:    A shared AsyncOpenAI client instance.

    Returns:
        EvalResult with all eight DimensionResult fields populated.
    """
    (
        faithfulness_result,
        relevancy_result,
        completeness_result,
        precision_result,
        recall_result,
        coherence_result,
        balance_result,
        toxicity_result,
    ) = await asyncio.gather(
        score_faithfulness(context, answer, openai_client),
        score_answer_relevancy(question, answer, openai_client),
        score_completeness(question, context, answer),
        score_context_precision(context, question, answer, openai_client),
        score_context_recall(context, answer, reference_answer, openai_client),
        score_coherence(question, answer),
        score_historical_balance(question, answer),
        score_toxicity(answer),
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    overall_passed = all([
        faithfulness_result.passed,
        relevancy_result.passed,
        completeness_result.passed,
        precision_result.passed,
        recall_result.passed,
        coherence_result.passed,
        balance_result.passed,
        toxicity_result.passed,
    ])

    result = EvalResult(
        faithfulness=faithfulness_result,
        answer_relevancy=relevancy_result,
        completeness=completeness_result,
        context_precision=precision_result,
        context_recall=recall_result,
        coherence=coherence_result,
        historical_balance=balance_result,
        toxicity=toxicity_result,
        latency_ms=round(latency_ms, 1),
        overall_passed=overall_passed,
    )
    _log_to_mlflow(result, domain_tag, question)
    return result


def _log_to_mlflow(
    result: EvalResult,
    domain_tag: str | None,
    question: str,
) -> None:
    """Log eval metrics and tags to MLflow if MLFLOW_TRACKING_URI is configured.

    Skips silently when the env var is absent so environments without mlflow
    installed continue to work. Any logging failure is non-fatal.
    """
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow  # late import — not required when tracking URI is unset

        dimensions = [
            "faithfulness",
            "answer_relevancy",
            "completeness",
            "context_precision",
            "context_recall",
            "coherence",
            "historical_balance",
            "toxicity",
        ]
        geval_scores = {
            "completeness_score": result.completeness.score,
            "coherence_score": result.coherence.score,
            "historical_balance_score": result.historical_balance.score,
            "toxicity_score": result.toxicity.score,
        }

        with mlflow.start_run():
            for dim in dimensions:
                mlflow.log_metric(
                    f"{dim}_passed",
                    int(getattr(result, dim).passed),
                )
            for name, value in geval_scores.items():
                if value is not None:
                    mlflow.log_metric(name, value)
            mlflow.log_metric("latency_ms", result.latency_ms)
            mlflow.log_metric("overall_passed", int(result.overall_passed))

            mlflow.set_tag("domain_tag", domain_tag or "all")
            mlflow.set_tag("overall_passed", str(result.overall_passed).lower())
            mlflow.set_tag("question", question[:100])
    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)
