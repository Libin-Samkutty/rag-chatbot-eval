"""
eval/ragas_eval.py — RAGAS-based evaluation metrics wrapper.

Implements five evaluation dimensions using RAGAS v0.4.x:
  - faithfulness        → RAGAS Faithfulness + claim classification
  - answer_relevancy    → RAGAS ResponseRelevancy
  - completeness        → Custom SingleTurnMetric
  - context_precision   → RAGAS ContextPrecision
  - context_recall      → RAGAS ContextRecall

Each public function returns a DimensionResult. All RAGAS calls are wrapped
in try/except — a single eval failure must not crash the entire request.

Judge model: GPT-4o (different family from the chatbot to avoid preference leakage).
Classification model: GPT-4o-mini (cheap secondary calls for claim categorisation).
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from openai import AsyncOpenAI, OpenAI

from config import settings
from eval.checklists.answer_relevancy import evaluate_answer_relevancy
from eval.checklists.completeness import evaluate_completeness
from eval.checklists.context_precision import evaluate_context_precision
from eval.checklists.context_recall import evaluate_context_recall
from eval.checklists.faithfulness import evaluate_faithfulness
from eval.models import ChecklistItem, DimensionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RAGAS judge configuration
# ---------------------------------------------------------------------------
# Lazy-initialised so we do not pay the import cost unless RAGAS is used.
_judge_llm: Any = None
_judge_embeddings: Any = None


def _get_ragas_llm() -> Any:
    """Return (and cache) the RAGAS judge LLM backed by GPT-4o."""
    global _judge_llm
    if _judge_llm is None:
        from ragas.llms import llm_factory

        _judge_llm = llm_factory(
            settings.eval_model,  # "gpt-4o"
            provider="openai",
            client=OpenAI(api_key=settings.openai_api_key),
        )
    return _judge_llm


def _get_ragas_embeddings() -> Any:
    """Return (and cache) the RAGAS embedding model backed by text-embedding-3-small."""
    global _judge_embeddings
    if _judge_embeddings is None:
        from ragas.embeddings import embedding_factory

        _judge_embeddings = embedding_factory(
            settings.embedding_model,  # "text-embedding-3-small"
            provider="openai",
            client=OpenAI(api_key=settings.openai_api_key),
        )
    return _judge_embeddings


# ---------------------------------------------------------------------------
# Claim categorisation helper (uses gpt-4o-mini for cost efficiency)
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = (
    "Classify this claim into one of: temporal, numeric, naming, general.\n"
    'Claim: "{claim}"\n'
    "Respond with one word only."
)


async def _classify_claim(claim: str, openai_client: AsyncOpenAI) -> str:
    """
    Return the claim category: 'temporal', 'numeric', 'naming', or 'general'.

    Uses gpt-4o-mini — this is a simple classification, not an eval judgement.
    """
    try:
        response = await openai_client.chat.completions.create(
            model=settings.eval_model_mini,  # "gpt-4o-mini"
            messages=[
                {"role": "user", "content": _CLASSIFY_PROMPT.format(claim=claim)}
            ],
            temperature=0,
            max_tokens=10,
        )
        category = response.choices[0].message.content.strip().lower()
        if category not in {"temporal", "numeric", "naming", "general"}:
            return "general"
        return category
    except Exception as exc:
        logger.warning("Claim classification failed for '%s': %s", claim[:80], exc)
        return "general"


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------

async def score_faithfulness(
    context: list[str],
    answer: str,
    openai_client: AsyncOpenAI,
) -> DimensionResult:
    """
    Score faithfulness using RAGAS Faithfulness.

    Extracts claim-level verdicts and maps them to the five-item checklist:
      Tier 1: faith_no_hallucination, faith_no_contradiction
      Tier 2: faith_temporal_accuracy, faith_numeric_fidelity, faith_proper_naming
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness

        metric = Faithfulness(llm=_get_ragas_llm())

        data = {
            "question": ["placeholder"],
            "answer": [answer],
            "contexts": [context],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[metric])

        ragas_score = result["faithfulness"]

        # Guard against NaN / None from RAGAS
        if ragas_score is None or (isinstance(ragas_score, float) and math.isnan(ragas_score)):
            raise ValueError("RAGAS returned NaN for faithfulness score")

        # ---------------------------------------------------------------
        # Extract per-claim verdicts if available; otherwise use the
        # aggregate score as a proxy to populate the checklist.
        # ---------------------------------------------------------------
        # Try to access intermediate claim-level data from RAGAS output.
        # RAGAS v0.4.x stores intermediate results in result.scores.
        unsupported_claims: list[str] = []
        contradicted_claims: list[str] = []
        supported_claims: list[str] = []

        try:
            # result.scores is a list of dicts, one per sample.
            scores_row = result.scores[0] if hasattr(result, "scores") else {}
            statements = scores_row.get("statements", [])
            verdicts = scores_row.get("verdicts", [])
            if statements and verdicts:
                for stmt, verdict in zip(statements, verdicts):
                    v = str(verdict).strip().lower()
                    if v in {"0", "false", "no", "unsupported"}:
                        unsupported_claims.append(stmt)
                    elif v in {"-1", "contradiction", "contradicts"}:
                        contradicted_claims.append(stmt)
                    else:
                        supported_claims.append(stmt)
        except Exception as extract_exc:
            logger.debug("Could not extract per-claim verdicts: %s", extract_exc)
            # Fall back: treat aggregate score as proxy.
            # score < 0.8 implies likely hallucination; < 0.5 implies contradiction.
            if ragas_score < 0.5:
                unsupported_claims = ["(proxy — aggregate score below 0.5)"]
                contradicted_claims = ["(proxy — aggregate score below 0.5)"]
            elif ragas_score < 0.8:
                unsupported_claims = ["(proxy — aggregate score below 0.8)"]

        # ---------------------------------------------------------------
        # Classify supported claims to populate Tier 2 items
        # ---------------------------------------------------------------
        all_claims = supported_claims + unsupported_claims + contradicted_claims
        if all_claims:
            categories = await asyncio.gather(
                *[_classify_claim(c, openai_client) for c in all_claims]
            )
        else:
            categories = []

        claim_categories: dict[str, str] = dict(zip(all_claims, categories))

        # Helper: did any claim of a given category fail (i.e. appear in
        # unsupported or contradicted)?
        bad_claims = set(unsupported_claims + contradicted_claims)

        def _category_has_failure(cat: str) -> bool:
            return any(
                claim in bad_claims
                for claim, c in claim_categories.items()
                if c == cat
            )

        def _category_exists(cat: str) -> bool:
            return any(c == cat for c in claim_categories.values())

        # ---------------------------------------------------------------
        # Build checklist items
        # ---------------------------------------------------------------
        items: list[ChecklistItem] = [
            ChecklistItem(
                key="faith_no_hallucination",
                question="Does the answer avoid stating any fact not present in the retrieved context?",
                result=len(unsupported_claims) == 0,
                tier=1,
            ),
            ChecklistItem(
                key="faith_no_contradiction",
                question="Does the answer avoid directly contradicting any statement in the retrieved context?",
                result=len(contradicted_claims) == 0,
                tier=1,
            ),
            ChecklistItem(
                key="faith_temporal_accuracy",
                question="Are all dates, years, and time-based claims in the answer supported by the retrieved context?",
                # Pass if no temporal claims failed (or no temporal claims exist).
                result=not _category_has_failure("temporal"),
                tier=2,
            ),
            ChecklistItem(
                key="faith_numeric_fidelity",
                question="Are all numeric values, statistics, and counts in the answer supported by the retrieved context?",
                result=not _category_has_failure("numeric"),
                tier=2,
            ),
            ChecklistItem(
                key="faith_proper_naming",
                question="Are all named people, places, and events in the answer named correctly and consistently with the retrieved context?",
                result=not _category_has_failure("naming"),
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

async def score_answer_relevancy(
    question: str,
    answer: str,
    openai_client: AsyncOpenAI,
) -> DimensionResult:
    """
    Score answer relevancy using RAGAS ResponseRelevancy.

    Maps the aggregate relevancy score onto three checklist items.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import ResponseRelevancy

        metric = ResponseRelevancy(
            llm=_get_ragas_llm(),
            embeddings=_get_ragas_embeddings(),
        )

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [[""]],  # ResponseRelevancy does not need context
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[metric])

        ragas_score = result["answer_relevancy"]

        if ragas_score is None or (isinstance(ragas_score, float) and math.isnan(ragas_score)):
            raise ValueError("RAGAS returned NaN for answer_relevancy score")

        # Map scalar score to three binary checklist items.
        # Thresholds are calibrated against industry benchmarks.
        addresses_q = ragas_score >= 0.75
        no_tangent = ragas_score >= 0.65
        intent_match = ragas_score >= 0.70

        items: list[ChecklistItem] = [
            ChecklistItem(
                key="relevancy_addresses_question",
                question="Does the answer directly address what the question asked rather than a related but different question?",
                result=addresses_q,
                tier=2,
            ),
            ChecklistItem(
                key="relevancy_no_tangent",
                question="Does the answer stay on topic without extended tangents unrelated to the question?",
                result=no_tangent,
                tier=2,
            ),
            ChecklistItem(
                key="relevancy_intent_match",
                question="Does the answer match the intent of the question (factual, explanatory, or comparative as appropriate)?",
                result=intent_match,
                tier=2,
            ),
        ]

        return evaluate_answer_relevancy(items)

    except Exception as exc:
        logger.error("Answer relevancy eval error: %s", exc)
        return _error_dimension("answer_relevancy", exc)


# ---------------------------------------------------------------------------
# 3. Completeness — custom RAGAS SingleTurnMetric
# ---------------------------------------------------------------------------

# Judge prompt for completeness.
_COMPLETENESS_PROMPT = """You are evaluating whether an AI answer is complete.

Question: {question}
Answer: {answer}

Respond in JSON only, no preamble:
{{
  "all_parts_addressed": true or false,
  "no_critical_omission": true or false,
  "reason": "one sentence"
}}

- all_parts_addressed: true if the answer addresses every distinct sub-part of the question.
- no_critical_omission: true if no critical fact is missing that would mislead the reader."""


async def score_completeness(
    question: str,
    answer: str,
    openai_client: AsyncOpenAI,
) -> DimensionResult:
    """
    Score completeness using a custom GPT-4o judge prompt.

    RAGAS does not provide a built-in completeness metric, so we use a
    direct LLM judge call that maps to the two-item checklist.
    """
    try:
        import json

        prompt = _COMPLETENESS_PROMPT.format(question=question, answer=answer)

        response = await openai_client.chat.completions.create(
            model=settings.eval_model,  # "gpt-4o"
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        all_parts = bool(data.get("all_parts_addressed", False))
        no_omission = bool(data.get("no_critical_omission", False))

        items: list[ChecklistItem] = [
            ChecklistItem(
                key="completeness_all_parts",
                question="Does the answer address all distinct sub-parts of the question?",
                result=all_parts,
                tier=2,
            ),
            ChecklistItem(
                key="completeness_no_omission",
                question="Does the answer avoid omitting any critical fact that, if missing, would mislead the reader?",
                result=no_omission,
                tier=2,
            ),
        ]

        return evaluate_completeness(items)

    except Exception as exc:
        logger.error("Completeness eval error: %s", exc)
        return _error_dimension("completeness", exc)


# ---------------------------------------------------------------------------
# 4. Context Precision
# ---------------------------------------------------------------------------

async def score_context_precision(
    context: list[str],
    answer: str,
    openai_client: AsyncOpenAI,
) -> DimensionResult:
    """
    Score context precision using RAGAS ContextPrecision.

    Creates one ChecklistItem per retrieved chunk based on whether each
    chunk contributed to the generated answer.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import ContextPrecision

        metric = ContextPrecision(llm=_get_ragas_llm())

        data = {
            "question": ["placeholder"],
            "answer": [answer],
            "contexts": [context],
            "ground_truth": [answer],  # Use answer as proxy when no reference
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[metric])

        ragas_score = result["context_precision"]

        if ragas_score is None or (isinstance(ragas_score, float) and math.isnan(ragas_score)):
            raise ValueError("RAGAS returned NaN for context_precision score")

        # Try to extract per-chunk relevance judgements.
        per_chunk_results: list[bool] = []
        try:
            scores_row = result.scores[0] if hasattr(result, "scores") else {}
            chunk_scores = scores_row.get("context_precision_scores", [])
            if chunk_scores:
                per_chunk_results = [bool(s) for s in chunk_scores]
        except Exception as extract_exc:
            logger.debug("Could not extract per-chunk scores: %s", extract_exc)

        # Fall back: distribute aggregate score across chunks proportionally.
        if not per_chunk_results:
            n = len(context)
            if n == 0:
                per_chunk_results = []
            else:
                # Mark the first floor(score * n) chunks as relevant.
                n_relevant = round(ragas_score * n)
                per_chunk_results = [True] * n_relevant + [False] * (n - n_relevant)

        items: list[ChecklistItem] = [
            ChecklistItem(
                key="precision_chunk_relevant",
                question="Is this retrieved chunk relevant to answering the question?",
                result=chunk_result,
                tier=2,
            )
            for chunk_result in per_chunk_results
        ]

        return evaluate_context_precision(items)

    except Exception as exc:
        logger.error("Context precision eval error: %s", exc)
        return _error_dimension("context_precision", exc)


# ---------------------------------------------------------------------------
# 5. Context Recall
# ---------------------------------------------------------------------------

async def score_context_recall(
    context: list[str],
    reference_answer: str | None,
    openai_client: AsyncOpenAI,
) -> DimensionResult:
    """
    Score context recall using RAGAS ContextRecall.

    When no reference_answer is provided (live chat mode), falls back to using
    the join of all context chunks as a self-reference — this measures whether
    the retrieved context is internally coherent rather than measuring coverage
    against a gold answer. Scores will be optimistically high in this mode;
    the metric is most meaningful in golden dataset eval mode.

    Creates one ChecklistItem per ground-truth claim.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import ContextRecall

        metric = ContextRecall(llm=_get_ragas_llm())

        # Use reference_answer if available; otherwise proxy with context text.
        ground_truth = reference_answer if reference_answer else "\n".join(context)

        data = {
            "question": ["placeholder"],
            "answer": [ground_truth],
            "contexts": [context],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[metric])

        ragas_score = result["context_recall"]

        if ragas_score is None or (isinstance(ragas_score, float) and math.isnan(ragas_score)):
            raise ValueError("RAGAS returned NaN for context_recall score")

        # Try to extract per-claim coverage from RAGAS intermediate output.
        per_claim_results: list[bool] = []
        try:
            scores_row = result.scores[0] if hasattr(result, "scores") else {}
            claim_scores = scores_row.get("context_recall_verdicts", [])
            if claim_scores:
                per_claim_results = [bool(s) for s in claim_scores]
        except Exception as extract_exc:
            logger.debug("Could not extract per-claim recall verdicts: %s", extract_exc)

        # Fall back: distribute aggregate score across synthetic claim slots.
        if not per_claim_results:
            # Use 5 slots as a reasonable proxy for a typical answer's claims.
            n_slots = 5
            n_covered = round(ragas_score * n_slots)
            per_claim_results = [True] * n_covered + [False] * (n_slots - n_covered)

        items: list[ChecklistItem] = [
            ChecklistItem(
                key="recall_claim_covered",
                question="Is this ground-truth claim covered by at least one of the retrieved chunks?",
                result=claim_result,
                tier=2,
            )
            for claim_result in per_claim_results
        ]

        return evaluate_context_recall(items)

    except Exception as exc:
        logger.error("Context recall eval error: %s", exc)
        return _error_dimension("context_recall", exc)


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def _error_dimension(name: str, exc: Exception) -> DimensionResult:
    """
    Return a failed DimensionResult when an eval call raises an exception.

    This ensures a single metric failure does not crash the entire request.
    The reason field encodes the exception so it is surfaced in the UI.
    """
    return DimensionResult(
        name=name,
        passed=False,
        items=[],
        score=None,
        reason=f"eval_error: {exc}",
        tier1_failed=[],
        tier2_pass_rate=0.0,
    )
