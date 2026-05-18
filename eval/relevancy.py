"""
eval/relevancy.py — Answer relevancy evaluation metric.

Question: Does the answer actually address the question that was asked?

Method: Cosine similarity between embeddings.
  - Embed the original question.
  - Embed the model's answer.
  - Compute cosine similarity between the two vectors.
  - A high similarity means the answer is semantically close to the question,
    i.e. it is on-topic.

Why cosine similarity here?
  No LLM call is needed — embedding similarity is fast, cheap, and works well
  for topical relevance. It won't catch subtle off-topic answers, but it
  reliably catches cases where the model answers a different question entirely.

Pass threshold: 0.75
"""

import logging
import math

from openai import AsyncOpenAI

from eval.models import EvalScore

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.75


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Cosine similarity = dot(A, B) / (|A| * |B|)
    Returns a value between -1 and 1 (in practice 0–1 for embeddings).
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


async def _embed(text: str, client: AsyncOpenAI, model: str) -> list[float]:
    """Return the embedding vector for a piece of text."""
    response = await client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


async def score_answer_relevancy(
    question: str,
    answer: str,
    client: AsyncOpenAI,
    embedding_model: str = "text-embedding-3-small",
) -> EvalScore:
    """
    Evaluate how relevant the answer is to the question.

    Args:
        question:        The user's original question.
        answer:          The model's generated answer.
        client:          An async OpenAI client instance.
        embedding_model: The OpenAI embedding model to use.

    Returns:
        EvalScore with score, reason, and passed flag.
    """
    try:
        # Embed both texts concurrently using asyncio — see runner.py for how
        # this is orchestrated with asyncio.gather at a higher level.
        question_vec = await _embed(question, client, embedding_model)
        answer_vec = await _embed(answer, client, embedding_model)

        score = _cosine_similarity(question_vec, answer_vec)
        # Clamp to [0, 1] — cosine similarity can technically be negative
        score = max(0.0, min(1.0, score))

        if score >= PASS_THRESHOLD:
            reason = "Answer is semantically aligned with the question."
        elif score >= 0.5:
            reason = "Answer is partially relevant but may not fully address the question."
        else:
            reason = "Answer appears off-topic relative to the question."

    except Exception as e:
        logger.warning("Answer relevancy eval failed: %s", e)
        score = 0.0
        reason = f"Eval error: {e}"

    rounded_score = round(score, 4)
    return EvalScore(
        score=rounded_score,
        reason=reason,
        passed=rounded_score >= PASS_THRESHOLD,
    )
