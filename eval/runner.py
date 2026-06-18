"""
eval/runner.py — Orchestrates all eight evaluation dimensions for one chat turn.

Cost optimisation: previously ran 8 concurrent GPT-4o calls (one per dimension).
Now runs 2 concurrent calls — one for all four RAGAS dimensions, one for all
four G-Eval dimensions. Total API cost reduced ~4x per chat turn.

Dimension map:
  RAGAS batch (eval/ragas_eval.py → score_ragas_all):
    faithfulness        → RAGAS-inspired GPT-4o judge
    answer_relevancy    → RAGAS-inspired GPT-4o judge
    context_precision   → RAGAS ContextPrecision
    context_recall      → RAGAS ContextRecall

  G-Eval batch (eval/deepeval_eval.py → score_geval_all):
    completeness        → G-Eval criteria (direct GPT-4o)
    coherence           → G-Eval criteria (direct GPT-4o)
    historical_balance  → G-Eval criteria (direct GPT-4o)
    toxicity            → G-Eval criteria (direct GPT-4o)

Adding a new dimension:
  1. Implement async score_<name>() in eval/ragas_eval.py or eval/deepeval_eval.py
  2. Import it here and add it to asyncio.gather
  3. Add the result to the EvalResult constructor
  4. Add the field to EvalResult in eval/models.py
"""

import asyncio
import time

from openai import AsyncOpenAI

# Cost optimisation: individual dimension imports replaced by combined batch functions.
# Originals preserved below for reference.
from eval.deepeval_eval import score_geval_all
# from eval.deepeval_eval import (
#     score_coherence,
#     score_completeness,
#     score_historical_balance,
#     score_toxicity,
# )
from eval.models import EvalResult
from eval.ragas_eval import score_ragas_all
# from eval.ragas_eval import (
#     score_answer_relevancy,
#     score_context_precision,
#     score_context_recall,
#     score_faithfulness,
# )


async def run_evals(
    question: str,
    context: list[str],
    answer: str,
    reference_answer: str | None,
    start_time: float,
    openai_client: AsyncOpenAI,
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
    # Cost optimisation: 8 individual GPT-4o calls → 2 concurrent batch calls.
    # RAGAS and G-Eval batches still run concurrently against each other.
    (
        faithfulness_result,
        relevancy_result,
        precision_result,
        recall_result,
    ), (
        completeness_result,
        coherence_result,
        balance_result,
        toxicity_result,
    ) = await asyncio.gather(
        score_ragas_all(question, context, answer, reference_answer, openai_client),
        score_geval_all(question, context, answer, openai_client),
    )

    # --- Pre-consolidation: 8 individual concurrent calls (preserved for reference) ---
    # (
    #     faithfulness_result,
    #     relevancy_result,
    #     completeness_result,
    #     precision_result,
    #     recall_result,
    #     coherence_result,
    #     balance_result,
    #     toxicity_result,
    # ) = await asyncio.gather(
    #     score_faithfulness(context, answer, openai_client),
    #     score_answer_relevancy(question, answer, openai_client),
    #     score_completeness(question, context, answer),
    #     score_context_precision(context, question, answer, openai_client),
    #     score_context_recall(context, answer, reference_answer, openai_client),
    #     score_coherence(question, answer),
    #     score_historical_balance(question, answer),
    #     score_toxicity(answer),
    # )

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

    return EvalResult(
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
