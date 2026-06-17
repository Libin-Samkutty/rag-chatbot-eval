"""
eval/models.py — Pydantic models for the evaluation framework.

Three-layer model:
  ChecklistItem   — one binary Yes/No item with tier classification
  DimensionResult — result for a single eval dimension (faithfulness, etc.)
  EvalResult      — all 8 dimensions + latency + overall pass/fail
"""

from typing import Literal
from pydantic import BaseModel


class ChecklistItem(BaseModel):
    key: str               # e.g. "faith_no_hallucination"
    question: str          # the Yes/No question asked of the judge
    result: bool           # True = passed, False = failed
    tier: Literal[1, 2]    # 1 = hard gate, 2 = threshold gate


class DimensionResult(BaseModel):
    name: str                       # e.g. "faithfulness"
    passed: bool                    # final pass/fail for this dimension
    items: list[ChecklistItem]      # empty for G-Eval holistic dimensions
    score: float | None             # G-Eval score (0–1) for holistic dims; None for checklist dims
    reason: str                     # one-sentence explanation
    tier1_failed: list[str]         # keys of any Tier 1 items that failed
    tier2_pass_rate: float          # proportion of Tier 2 items that passed (0.0–1.0)


class EvalResult(BaseModel):
    faithfulness: DimensionResult
    answer_relevancy: DimensionResult
    completeness: DimensionResult
    context_precision: DimensionResult
    context_recall: DimensionResult
    coherence: DimensionResult
    historical_balance: DimensionResult
    toxicity: DimensionResult
    latency_ms: float
    overall_passed: bool   # True only if all 8 dimensions pass

    @classmethod
    def compute_overall(cls, dims: list[DimensionResult]) -> bool:
        return all(d.passed for d in dims)
