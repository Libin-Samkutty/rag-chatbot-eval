"""
eval/checklists/answer_relevancy.py — Answer Relevancy dimension checklist.

Two-tier logic:
  Tier 1: none — relevancy is a quality dimension, not a safety gate.
  Tier 2 (threshold gate): relevancy_addresses_question, relevancy_no_tangent,
    relevancy_intent_match — at least 2 out of 3 must pass.

Note: adversarial questions are EXPECTED to fail this dimension.
A correct refusal will score low on relevancy_addresses_question.

RAGAS ResponseRelevancy returns a semantic similarity score. ragas_eval.py
maps that score onto these three items before calling evaluate_answer_relevancy().
"""

from eval.models import ChecklistItem, DimensionResult

# No Tier 1 hard gate for answer relevancy.
TIER1_KEYS: set[str] = set()

# Minimum number of Tier 2 items that must pass.
TIER2_THRESHOLD: int = 2

CHECKLIST_TEMPLATES: list[ChecklistItem] = [
    ChecklistItem(
        key="relevancy_addresses_question",
        question="Does the answer directly address what the question asked rather than a related but different question?",
        result=False,
        tier=2,
    ),
    ChecklistItem(
        key="relevancy_no_tangent",
        question="Does the answer stay on topic without extended tangents unrelated to the question?",
        result=False,
        tier=2,
    ),
    ChecklistItem(
        key="relevancy_intent_match",
        question="Does the answer match the intent of the question (factual, explanatory, or comparative as appropriate)?",
        result=False,
        tier=2,
    ),
]


def _build_reason(tier2_pass_count: int, tier2_total: int) -> str:
    """Produce a single human-readable sentence explaining the outcome."""
    if tier2_pass_count >= TIER2_THRESHOLD:
        return (
            f"{tier2_pass_count}/{tier2_total} Tier 2 relevancy items passed "
            "— answer relevancy dimension passes."
        )
    return (
        f"Only {tier2_pass_count}/{tier2_total} Tier 2 items passed "
        f"(threshold is {TIER2_THRESHOLD})."
    )


def evaluate_answer_relevancy(items: list[ChecklistItem]) -> DimensionResult:
    """
    Apply two-tier logic to a fully-populated list of answer relevancy checklist items.

    Args:
        items: Three ChecklistItem instances with result filled in by the caller.

    Returns:
        DimensionResult for the answer_relevancy dimension.
    """
    tier1_items = [i for i in items if i.tier == 1]
    tier2_items = [i for i in items if i.tier == 2]

    # Tier 1 is empty for this dimension.
    tier1_failed: list[str] = [i.key for i in tier1_items if not i.result]
    tier2_pass_count = sum(1 for i in tier2_items if i.result)
    tier2_pass_rate = tier2_pass_count / len(tier2_items) if tier2_items else 1.0

    passed = tier2_pass_count >= TIER2_THRESHOLD

    reason = _build_reason(tier2_pass_count, len(tier2_items))

    return DimensionResult(
        name="answer_relevancy",
        passed=passed,
        items=items,
        score=None,
        reason=reason,
        tier1_failed=tier1_failed,
        tier2_pass_rate=tier2_pass_rate,
    )
