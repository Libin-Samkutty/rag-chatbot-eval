"""
eval/checklists/faithfulness.py — Faithfulness dimension checklist.

Two-tier logic:
  Tier 1 (hard gate): faith_no_hallucination, faith_no_contradiction
    — any False = dimension fails regardless of Tier 2.
  Tier 2 (threshold gate): faith_temporal_accuracy, faith_numeric_fidelity,
    faith_proper_naming — at least 2 out of 3 must pass.

RAGAS Faithfulness returns a claim-level score. ragas_eval.py maps those
verdicts onto these five items before calling evaluate_faithfulness().
"""

from eval.models import ChecklistItem, DimensionResult

# Keys that act as hard gates — any single failure kills the dimension.
TIER1_KEYS: set[str] = {"faith_no_hallucination", "faith_no_contradiction"}

# Minimum number of Tier 2 items that must pass.
TIER2_THRESHOLD: int = 2

# Templates — result field is a placeholder; ragas_eval.py fills it in.
CHECKLIST_TEMPLATES: list[ChecklistItem] = [
    ChecklistItem(
        key="faith_no_hallucination",
        question="Does the answer avoid stating any fact not present in the retrieved context?",
        result=False,
        tier=1,
    ),
    ChecklistItem(
        key="faith_no_contradiction",
        question="Does the answer avoid directly contradicting any statement in the retrieved context?",
        result=False,
        tier=1,
    ),
    ChecklistItem(
        key="faith_temporal_accuracy",
        question="Are all dates, years, and time-based claims in the answer supported by the retrieved context?",
        result=False,
        tier=2,
    ),
    ChecklistItem(
        key="faith_numeric_fidelity",
        question="Are all numeric values, statistics, and counts in the answer supported by the retrieved context?",
        result=False,
        tier=2,
    ),
    ChecklistItem(
        key="faith_proper_naming",
        question="Are all named people, places, and events in the answer named correctly and consistently with the retrieved context?",
        result=False,
        tier=2,
    ),
]


def _build_reason(
    tier1_failed: list[str],
    tier2_pass_count: int,
    tier2_total: int,
) -> str:
    """Produce a single human-readable sentence explaining the outcome."""
    if tier1_failed:
        keys_str = ", ".join(tier1_failed)
        return f"Tier 1 hard gate(s) failed: {keys_str}."
    if tier2_pass_count < TIER2_THRESHOLD:
        return (
            f"Only {tier2_pass_count}/{tier2_total} Tier 2 items passed "
            f"(threshold is {TIER2_THRESHOLD})."
        )
    return (
        f"All Tier 1 gates passed and {tier2_pass_count}/{tier2_total} "
        "Tier 2 items passed — faithfulness dimension passes."
    )


def evaluate_faithfulness(items: list[ChecklistItem]) -> DimensionResult:
    """
    Apply two-tier logic to a fully-populated list of faithfulness checklist items.

    Args:
        items: Five ChecklistItem instances with result filled in by the caller.

    Returns:
        DimensionResult for the faithfulness dimension.
    """
    tier1_items = [i for i in items if i.tier == 1]
    tier2_items = [i for i in items if i.tier == 2]

    tier1_failed = [i.key for i in tier1_items if not i.result]
    tier2_pass_count = sum(1 for i in tier2_items if i.result)
    tier2_pass_rate = tier2_pass_count / len(tier2_items) if tier2_items else 1.0

    passed = len(tier1_failed) == 0 and tier2_pass_count >= TIER2_THRESHOLD

    reason = _build_reason(tier1_failed, tier2_pass_count, len(tier2_items))

    return DimensionResult(
        name="faithfulness",
        passed=passed,
        items=items,
        score=None,
        reason=reason,
        tier1_failed=tier1_failed,
        tier2_pass_rate=tier2_pass_rate,
    )
