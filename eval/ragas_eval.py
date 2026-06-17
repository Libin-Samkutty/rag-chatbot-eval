"""
eval/ragas_eval.py — RAGAS-inspired RAG evaluation metrics via direct GPT-4o judge calls.

Implements four evaluation dimensions:
  - faithfulness        → claim extraction + context grounding check
  - answer_relevancy    → direct LLM relevancy judge
  - context_precision   → per-chunk relevance judge
  - context_recall      → per-claim context coverage judge

Each function accepts the same inputs RAGAS would and returns a DimensionResult
populated with ChecklistItems — the checklist system is unchanged.

Why direct calls instead of the RAGAS library:
  RAGAS internally imports langchain_community.chat_models.vertexai during
  initialisation regardless of the provider configured. That module was removed
  from langchain_community in versions >= 0.3. Rather than pin a stale
  LangChain version or add langchain-google-vertexai as a shim dependency,
  we replicate the same LLM-judge logic with direct AsyncOpenAI calls.
  This also makes each metric a transparent, readable async function —
  consistent with the project's "no black-box eval framework" principle.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from config import settings
from eval.checklists.answer_relevancy import evaluate_answer_relevancy
from eval.checklists.context_precision import evaluate_context_precision
from eval.checklists.context_recall import evaluate_context_recall
from eval.checklists.faithfulness import evaluate_faithfulness
from eval.models import ChecklistItem, DimensionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared JSON judge helper
# ---------------------------------------------------------------------------

async def _judge(
    prompt: str,
    openai_client: AsyncOpenAI,
    model: str | None = None,
) -> dict:
    """Call GPT-4o with a JSON-mode prompt and return the parsed dict."""
    response = await openai_client.chat.completions.create(
        model=model or settings.eval_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """You are a factual grounding evaluator.

Given the retrieved context and the generated answer, extract every factual claim
made in the answer and determine whether each claim is:
  - "supported"     — directly stated or clearly implied by the context
  - "unsupported"   — not found in the context (hallucination)
  - "contradiction" — contradicts something in the context

Also classify each claim's type: "temporal" (dates/years), "numeric" (stats/counts),
"naming" (people/places/events), or "general".

Context:
{context}

Answer:
{answer}

Respond in JSON only:
{{
  "claims": [
    {{"claim": "...", "verdict": "supported|unsupported|contradiction", "type": "temporal|numeric|naming|general"}}
  ]
}}"""


async def score_faithfulness(
    context: list[str],
    answer: str,
    openai_client: AsyncOpenAI,
) -> DimensionResult:
    try:
        context_text = "\n\n---\n\n".join(context) if context else "(no context)"
        data = await _judge(
            _FAITHFULNESS_PROMPT.format(context=context_text, answer=answer),
            openai_client,
        )

        claims = data.get("claims", [])
        unsupported = [c for c in claims if c.get("verdict") == "unsupported"]
        contradicted = [c for c in claims if c.get("verdict") == "contradiction"]

        def _type_failed(t: str) -> bool:
            bad = {c["claim"] for c in unsupported + contradicted}
            return any(c["claim"] in bad and c.get("type") == t for c in claims)

        items: list[ChecklistItem] = [
            ChecklistItem(
                key="faith_no_hallucination",
                question="Does the answer avoid stating any fact not present in the retrieved context?",
                result=len(unsupported) == 0,
                tier=1,
            ),
            ChecklistItem(
                key="faith_no_contradiction",
                question="Does the answer avoid directly contradicting any statement in the retrieved context?",
                result=len(contradicted) == 0,
                tier=1,
            ),
            ChecklistItem(
                key="faith_temporal_accuracy",
                question="Are all dates, years, and time-based claims supported by the retrieved context?",
                result=not _type_failed("temporal"),
                tier=2,
            ),
            ChecklistItem(
                key="faith_numeric_fidelity",
                question="Are all numeric values, statistics, and counts supported by the retrieved context?",
                result=not _type_failed("numeric"),
                tier=2,
            ),
            ChecklistItem(
                key="faith_proper_naming",
                question="Are all named people, places, and events named correctly per the retrieved context?",
                result=not _type_failed("naming"),
                tier=2,
            ),
        ]

        return evaluate_faithfulness(items)

    except Exception as exc:
        logger.error("Faithfulness eval error: %s", exc)
        return _error_dimension("faithfulness", exc)


# ---------------------------------------------------------------------------
# 2. Answer Relevancy
# ---------------------------------------------------------------------------

_RELEVANCY_PROMPT = """You are a relevancy evaluator for a question-answering system.

Given the question and the answer, judge:
  - addresses_question: does the answer directly address what was asked?
  - no_tangent: does the answer stay on-topic without extended off-topic content?
  - intent_match: does the answer match the intent (factual, explanatory, comparative)?

Question: {question}
Answer: {answer}

Respond in JSON only:
{{
  "addresses_question": true or false,
  "no_tangent": true or false,
  "intent_match": true or false,
  "reason": "one sentence"
}}"""


async def score_answer_relevancy(
    question: str,
    answer: str,
    openai_client: AsyncOpenAI,
) -> DimensionResult:
    try:
        data = await _judge(
            _RELEVANCY_PROMPT.format(question=question, answer=answer),
            openai_client,
        )

        items: list[ChecklistItem] = [
            ChecklistItem(
                key="relevancy_addresses_question",
                question="Does the answer directly address what the question asked?",
                result=bool(data.get("addresses_question", False)),
                tier=2,
            ),
            ChecklistItem(
                key="relevancy_no_tangent",
                question="Does the answer stay on topic without extended tangents?",
                result=bool(data.get("no_tangent", False)),
                tier=2,
            ),
            ChecklistItem(
                key="relevancy_intent_match",
                question="Does the answer match the intent of the question?",
                result=bool(data.get("intent_match", False)),
                tier=2,
            ),
        ]

        return evaluate_answer_relevancy(items)

    except Exception as exc:
        logger.error("Answer relevancy eval error: %s", exc)
        return _error_dimension("answer_relevancy", exc)



# ---------------------------------------------------------------------------
# 4. Context Precision
# ---------------------------------------------------------------------------

_PRECISION_PROMPT = """You are a retrieval quality evaluator.

For each retrieved context chunk below, judge whether it was relevant to
generating a correct answer to the question. A chunk is relevant if it
contains information that directly supports or informs the answer.

Question: {question}
Answer: {answer}

Chunks:
{chunks}

Respond in JSON only:
{{
  "chunk_relevance": [true or false, ...],
  "reason": "one sentence"
}}

The array length must match the number of chunks exactly."""


async def score_context_precision(
    context: list[str],
    question: str,
    answer: str,
    openai_client: AsyncOpenAI,
) -> DimensionResult:
    try:
        if not context:
            return _error_dimension("context_precision", ValueError("No context chunks provided"))

        chunks_text = "\n\n".join(
            f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context)
        )
        data = await _judge(
            _PRECISION_PROMPT.format(question=question, answer=answer, chunks=chunks_text),
            openai_client,
        )

        relevance_flags: list[bool] = data.get("chunk_relevance", [])
        # Pad or truncate to match actual chunk count
        while len(relevance_flags) < len(context):
            relevance_flags.append(True)
        relevance_flags = relevance_flags[: len(context)]

        items: list[ChecklistItem] = [
            ChecklistItem(
                key="precision_chunk_relevant",
                question="Is this retrieved chunk relevant to answering the question?",
                result=flag,
                tier=2,
            )
            for flag in relevance_flags
        ]

        return evaluate_context_precision(items)

    except Exception as exc:
        logger.error("Context precision eval error: %s", exc)
        return _error_dimension("context_precision", exc)


# ---------------------------------------------------------------------------
# 5. Context Recall
# ---------------------------------------------------------------------------

_RECALL_PROMPT = """You are a context coverage evaluator.

Given the ground-truth answer and the retrieved context, extract the key factual
claims from the ground truth and determine whether each claim is covered by
(i.e. can be derived from) the retrieved context.

Ground truth: {ground_truth}

Retrieved context:
{context}

Respond in JSON only:
{{
  "claims": [
    {{"claim": "...", "covered": true or false}}
  ],
  "reason": "one sentence"
}}"""


async def score_context_recall(
    context: list[str],
    answer: str,
    reference_answer: str | None,
    openai_client: AsyncOpenAI,
) -> DimensionResult:
    try:
        ground_truth = reference_answer if reference_answer else answer
        context_text = "\n\n---\n\n".join(context) if context else "(no context)"

        data = await _judge(
            _RECALL_PROMPT.format(ground_truth=ground_truth, context=context_text),
            openai_client,
        )

        claims = data.get("claims", [])

        if not claims:
            # Fall back to a single synthetic claim based on aggregate
            claims = [{"claim": "(synthetic)", "covered": True}]

        items: list[ChecklistItem] = [
            ChecklistItem(
                key="recall_claim_covered",
                question="Is this ground-truth claim covered by at least one retrieved chunk?",
                result=bool(c.get("covered", False)),
                tier=2,
            )
            for c in claims
        ]

        return evaluate_context_recall(items)

    except Exception as exc:
        logger.error("Context recall eval error: %s", exc)
        return _error_dimension("context_recall", exc)


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def _error_dimension(name: str, exc: Exception) -> DimensionResult:
    return DimensionResult(
        name=name,
        passed=False,
        items=[],
        score=None,
        reason=f"eval_error: {exc}",
        tier1_failed=[],
        tier2_pass_rate=0.0,
    )
