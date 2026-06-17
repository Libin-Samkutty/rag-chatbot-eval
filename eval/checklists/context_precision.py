"""
eval/checklists/context_precision.py — Context Precision dimension checklist.

Two-tier logic:
  Tier 1: none.
  Tier 2 (threshold gate): precision_chunk_relevant — evaluated per retrieved chunk.
    At least 75% of chunks must pass.

RAGAS ContextPrecision returns a per-chunk relevance score. ragas_eval.py
creates one ChecklistItem per chunk before calling evaluate_context_precision().
"""

from eval.models import ChecklistItem, DimensionResult

# No Tier 1 hard gate for context precision.
TIER1_KEYS: set[str] = set()

# Proportion of Tier 2 items (chunks) that must pass.
TIER2_PASS_RATE_THRESHOLD: float = 0.75

# Template for a single chunk — ragas_eval.py instantiates one per chunk.
CHECKLIST_TEMPLATES: list[ChecklistItem] = [
    ChecklistItem(
        key="precision_chunk_relevant",
        question="Is this retrieved chunk relevant to answering the question?",
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
            f"{tier2_pass_count}/{tier2_total} chunks relevant ({pct}%) "
            f"— context precision passes (threshold {threshold_pct}%)."
        )
    return (
        f"Only {tier2_pass_count}/{tier2_total} chunks relevant ({pct}%) "
        f"— below threshold of {threshold_pct}%."
    )


def evaluate_context_precision(items: list[ChecklistItem]) -> DimensionResult:
    """
    Apply per-chunk threshold logic to a list of precision checklist items.

    One ChecklistItem per retrieved chunk is expected. Dimension passes if
    at least 75% of chunks are marked relevant.

    Args:
        items: One ChecklistItem per retrieved chunk, with result filled in.

    Returns:
        DimensionResult for the context_precision dimension.
    """
    tier1_items = [i for i in items if i.tier == 1]
    tier2_items = [i for i in items if i.tier == 2]

    tier1_failed: list[str] = [i.key for i in tier1_items if not i.result]
    tier2_pass_count = sum(1 for i in tier2_items if i.result)
    tier2_pass_rate = tier2_pass_count / len(tier2_items) if tier2_items else 1.0

    passed = tier2_pass_rate >= TIER2_PASS_RATE_THRESHOLD

    reason = _build_reason(tier2_pass_count, len(tier2_items), tier2_pass_rate)

    return DimensionResult(
        name="context_precision",
        passed=passed,
        items=items,
        score=None,
        reason=reason,
        tier1_failed=tier1_failed,
        tier2_pass_rate=tier2_pass_rate,
    )
