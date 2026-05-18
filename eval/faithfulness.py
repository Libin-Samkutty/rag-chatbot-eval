"""
eval/faithfulness.py — Faithfulness evaluation metric.

Question: Does the answer contain ONLY information supported by the context?

Method: LLM-as-judge.
  - We give gpt-4o-mini the retrieved context and the answer.
  - We ask it to verify whether each factual claim in the answer is grounded
    in the context.
  - It returns a JSON object with a score (0–1) and a reason.

Why LLM-as-judge here?
  Faithfulness requires semantic understanding that rule-based methods cannot
  reliably provide. A model reading both context and answer can identify
  subtle hallucinations that cosine similarity would miss.

Pass threshold: 0.7
"""

import json
import logging

from openai import AsyncOpenAI

from eval.models import EvalScore

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.7

# The judge prompt is intentionally short and direct.
# Longer prompts with many examples tend to cause the model to be lenient.
FAITHFULNESS_PROMPT = """You are an evaluation judge assessing whether an AI answer is faithful to its source context.

CONTEXT (the only information the answer should use):
{context}

ANSWER (the AI's response to evaluate):
{answer}

Your task: Decide whether every factual claim in the ANSWER is directly supported by the CONTEXT.
- Score 1.0 means every claim is grounded in the context.
- Score 0.0 means the answer contains significant information not in the context.
- Score 0.5 means the answer is partially grounded.

Respond with ONLY valid JSON, no preamble:
{{"score": <float 0.0-1.0>, "reason": "<one sentence explaining your score>"}}"""


async def score_faithfulness(
    context: list[str],
    answer: str,
    client: AsyncOpenAI,
    model: str = "gpt-4o-mini",
) -> EvalScore:
    """
    Evaluate how faithful the answer is to the retrieved context.

    Args:
        context:  List of retrieved chunk texts used to generate the answer.
        answer:   The model's generated answer.
        client:   An async OpenAI client instance.
        model:    The OpenAI model to use as the judge.

    Returns:
        EvalScore with score, reason, and passed flag.
    """
    # Join the chunks into a single context block for the prompt
    context_text = "\n\n---\n\n".join(context)

    prompt = FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # response_format forces the model to return valid JSON,
            # eliminating the need for fragile regex parsing.
            response_format={"type": "json_object"},
            temperature=0,  # Deterministic for eval consistency
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        score = float(data["score"])
        reason = str(data["reason"])

    except Exception as e:
        # If the judge call fails, return a neutral failing score rather than
        # crashing the whole request — eval failures should be visible but
        # should not block the user from seeing the chat answer.
        logger.warning("Faithfulness eval failed: %s", e)
        score = 0.0
        reason = f"Eval error: {e}"

    return EvalScore(
        score=score,
        reason=reason,
        passed=score >= PASS_THRESHOLD,
    )
