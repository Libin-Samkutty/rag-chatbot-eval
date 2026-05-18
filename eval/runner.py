"""
eval/runner.py — Orchestrates all evaluation metrics for one chat turn.

All three scored metrics run concurrently via asyncio.gather, so the total
eval time is roughly the slowest individual metric (the LLM-as-judge
faithfulness call) rather than the sum of all three.

Adding a new metric:
  1. Create eval/your_metric.py with an async score_* function
  2. Import it below
  3. Add it to the asyncio.gather call
  4. Add the result to the EvalResult constructor
  5. Add the field to EvalResult in eval/models.py
"""

import asyncio
import time

from openai import AsyncOpenAI

from config import settings
from eval.context_precision import score_context_precision
from eval.faithfulness import score_faithfulness
from eval.models import EvalResult
from eval.relevancy import score_answer_relevancy


async def run_evals(
    question: str,
    context: list[str],
    answer: str,
    start_time: float,
    client: AsyncOpenAI,
) -> EvalResult:
    """
    Run all four evaluation metrics and return a combined EvalResult.

    Args:
        question:   The user's original question.
        context:    The list of retrieved chunk texts.
        answer:     The model's generated answer.
        start_time: time.perf_counter() value from before the RAG pipeline
                    started — used to compute end-to-end latency.
        client:     A shared AsyncOpenAI client instance.

    Returns:
        EvalResult containing scores for all metrics.
    """
    # Run the three scored metrics concurrently.
    # asyncio.gather launches all coroutines at once and waits for all to finish.
    # This is more efficient than awaiting them one by one.
    faithfulness_score, relevancy_score, precision_score = await asyncio.gather(
        score_faithfulness(
            context=context,
            answer=answer,
            client=client,
            model=settings.eval_model,
        ),
        score_answer_relevancy(
            question=question,
            answer=answer,
            client=client,
            embedding_model=settings.embedding_model,
        ),
        score_context_precision(
            context=context,
            answer=answer,
            client=client,
            embedding_model=settings.embedding_model,
        ),
    )

    # Measure total wall-clock latency from when the request arrived.
    latency_ms = (time.perf_counter() - start_time) * 1000

    # overall_passed is True only if all three scored metrics pass.
    # Latency is informational — it does not contribute to overall_passed.
    overall_passed = all([
        faithfulness_score.passed,
        relevancy_score.passed,
        precision_score.passed,
    ])

    return EvalResult(
        faithfulness=faithfulness_score,
        answer_relevancy=relevancy_score,
        context_precision=precision_score,
        latency_ms=round(latency_ms, 1),
        overall_passed=overall_passed,
    )
