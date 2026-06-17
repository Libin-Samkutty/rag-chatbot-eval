"""
eval/checklists/context_recall.py — Context Recall dimension checklist.

Two-tier logic:
  Tier 1: none.
  Tier 2 (threshold gate): recall_claim_covered — evaluated per ground-truth claim.
    At least 80% of claims must be covered by the retrieved chunks.

RAGAS ContextRecall decomposes the reference answer into claims and checks
each against the retrieved context. ragas_eval.py creates one ChecklistItem
per claim before calling evaluate_context_recall().
"""

from eval.models import ChecklistItem, DimensionResult

# No Tier 1 hard gate for context recall.
TIER1_KEYS: set[str] = set()

# Proportion of Tier 2 items (claims) that must be covered.
TIER2_PASS_RATE_THRESHOLD: float = 0.80

# Template for a single claim — ragas_eval.py instantiates one per claim.
CHECKLIST_TEMPLATES: list[ChecklistItem] = [
    ChecklistItem(
        key="recall_claim_covered",
        question="Is this ground-truth claim covered by at least one of the retrieved chunks?",
        result=False,
        tier=2,
    ),
]


def _build_reason(tier2_pass_count: int, tier2_total: int, pass_rate: float) -> str:
    """Produce a single human-readable sentence explaining the outcome."""
    pct = round(pass_rate * 100, 1)
    threshold_pct = round(TIER2_PASS_RATE_THRESHOLD * 100, 1)
    if pass_rate >= TIER2_PASS_RATE_THRESHOLD:
        return (
            f"{tier2_pass_count}/{tier2_total} claims covered ({pct}%) "
            f"— context recall passes (threshold {threshold_pct}%)."
        )
    return (
        f"Only {tier2_pass_count}/{tier2_total} claims covered ({pct}%) "
        f"— below threshold of {threshold_pct}%."
    )


def evaluate_context_recall(items: list[ChecklistItem]) -> DimensionResult:
    """
    Apply per-claim threshold logic to a list of recall checklist items.

    One ChecklistItem per ground-truth claim is expected. Dimension passes if
    at least 80% of claims are marked as covered by the retrieved context.

    Args:
        items: One ChecklistItem per ground-truth claim, with result filled in.

    Returns:
        DimensionResult for the context_recall dimension.
    """
    tier1_items = [i for i in items if i.tier == 1]
    tier2_items = [i for i in items if i.tier == 2]

    tier1_failed: list[str] = [i.key for i in tier1_items if not i.result]
    tier2_pass_count = sum(1 for i in tier2_items if i.result)
    tier2_pass_rate = tier2_pass_count / len(tier2_items) if tier2_items else 1.0

    passed = tier2_pass_rate >= TIER2_PASS_RATE_THRESHOLD

    reason = _build_reason(tier2_pass_count, len(tier2_items), tier2_pass_rate)

    return DimensionResult(
        name="context_recall",
        passed=passed,
        items=items,
        score=None,
        reason=reason,
        tier1_failed=tier1_failed,
        tier2_pass_rate=tier2_pass_rate,
    )
