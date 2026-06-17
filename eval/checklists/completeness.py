"""
eval/checklists/completeness.py — Completeness dimension checklist.

Two-tier logic:
  Tier 1: none.
  Tier 2 (threshold gate): completeness_all_parts, completeness_no_omission
    — both items must pass (with only 2 items, a 50% threshold is meaningless).

RAGAS CompletenessMetric (custom SingleTurnMetric) returns two binary flags.
ragas_eval.py maps those flags onto these two items before calling
evaluate_completeness().
"""

from eval.models import ChecklistItem, DimensionResult

# No Tier 1 hard gate for completeness.
TIER1_KEYS: set[str] = set()

# Both Tier 2 items must pass (threshold = total items).
TIER2_THRESHOLD: int = 2

CHECKLIST_TEMPLATES: list[ChecklistItem] = [
    ChecklistItem(
        key="completeness_all_parts",
        question="Does the answer address all distinct sub-parts of the question?",
        result=False,
        tier=2,
    ),
    ChecklistItem(
        key="completeness_no_omission",
        question="Does the answer avoid omitting any critical fact that, if missing, would mislead the reader?",
        result=False,
        tier=2,
    ),
]


def _build_reason(tier2_pass_count: int, tier2_total: int) -> str:
    """Produce a single human-readable sentence explaining the outcome."""
    if tier2_pass_count >= TIER2_THRESHOLD:
        return (
            f"Both completeness items passed ({tier2_pass_count}/{tier2_total}) "
            "— completeness dimension passes."
        )
    failing = tier2_total - tier2_pass_count
    return (
        f"{failing} of {tier2_total} completeness item(s) failed — "
        "both must pass for this dimension."
    )


def evaluate_completeness(items: list[ChecklistItem]) -> DimensionResult:
    """
    Apply two-tier logic to a fully-populated list of completeness checklist items.

    Both Tier 2 items must pass (TIER2_THRESHOLD == len(items) == 2).

    Args:
        items: Two ChecklistItem instances with result filled in by the caller.

    Returns:
        DimensionResult for the completeness dimension.
    """
    tier1_items = [i for i in items if i.tier == 1]
    tier2_items = [i for i in items if i.tier == 2]

    tier1_failed: list[str] = [i.key for i in tier1_items if not i.result]
    tier2_pass_count = sum(1 for i in tier2_items if i.result)
    tier2_pass_rate = tier2_pass_count / len(tier2_items) if tier2_items else 1.0

    passed = tier2_pass_count >= TIER2_THRESHOLD

    reason = _build_reason(tier2_pass_count, len(tier2_items))

    return DimensionResult(
        name="completeness",
        passed=passed,
        items=items,
        score=None,
        reason=reason,
        tier1_failed=tier1_failed,
        tier2_pass_rate=tier2_pass_rate,
    )
