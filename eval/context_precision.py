"""
eval/context_precision.py — Context precision evaluation metric.

Question: Were the retrieved chunks actually useful for generating the answer?

Method: Per-chunk cosine similarity against the answer.
  - Embed each retrieved chunk.
  - Embed the answer.
  - Compute cosine similarity between each chunk and the answer.
  - Score = mean similarity across all chunks.

Interpretation:
  - High score: the retrieved chunks are semantically close to the answer,
    meaning retrieval pulled relevant documents.
  - Low score: retrieval pulled unrelated chunks; the model likely answered
    from its parametric (training) memory rather than the knowledge base.

Why this matters:
  A low context precision score is a signal to investigate your chunking
  strategy, embedding model, or the queries being sent to ChromaDB.

Pass threshold: 0.6

Note: This is a simplified version of the RAGAS context precision metric.
The full RAGAS version uses an LLM judge per chunk; we use embedding
similarity to keep costs down.
"""

import logging
import math

from openai import AsyncOpenAI

from eval.models import EvalScore

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.6


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors — same implementation as relevancy.py."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def score_context_precision(
    context: list[str],
    answer: str,
    client: AsyncOpenAI,
    embedding_model: str = "text-embedding-3-small",
) -> EvalScore:
    """
    Evaluate whether the retrieved chunks were relevant to the answer.

    Args:
        context:         List of retrieved chunk texts.
        answer:          The model's generated answer.
        client:          An async OpenAI client instance.
        embedding_model: The OpenAI embedding model to use.

    Returns:
        EvalScore with score, reason, and passed flag.
    """
    if not context:
        return EvalScore(score=0.0, reason="No context was retrieved.", passed=False)

    try:
        # Embed the answer once
        answer_response = await client.embeddings.create(
            input=answer, model=embedding_model
        )
        answer_vec = answer_response.data[0].embedding

        # Embed all chunks in a single batch request (more efficient than
        # one request per chunk)
        chunks_response = await client.embeddings.create(
            input=context, model=embedding_model
        )
        chunk_vecs = [item.embedding for item in chunks_response.data]

        # Compute per-chunk similarity and take the mean
        similarities = [
            max(0.0, _cosine_similarity(chunk_vec, answer_vec))
            for chunk_vec in chunk_vecs
        ]
        score = sum(similarities) / len(similarities)
        score = min(1.0, score)  # Clamp

        if score >= PASS_THRESHOLD:
            reason = f"Retrieved chunks were relevant (mean similarity {score:.2f})."
        else:
            reason = (
                f"Retrieved chunks had low overlap with the answer "
                f"(mean similarity {score:.2f}). Check retrieval quality."
            )

    except Exception as e:
        logger.warning("Context precision eval failed: %s", e)
        score = 0.0
        reason = f"Eval error: {e}"

    rounded_score = round(score, 4)
    return EvalScore(
        score=rounded_score,
        reason=reason,
        passed=rounded_score >= PASS_THRESHOLD,
    )
