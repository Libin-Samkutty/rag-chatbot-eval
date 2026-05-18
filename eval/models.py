"""
eval/models.py — Pydantic models shared across all eval metrics.

EvalScore  : the result of a single metric (score, reason, passed)
EvalResult : the combined result of all four metrics for one chat turn
"""

from pydantic import BaseModel, Field


class EvalScore(BaseModel):
    """Result of a single evaluation metric."""

    score: float = Field(ge=0.0, le=1.0, description="Normalised 0–1 score")
    reason: str = Field(description="One-sentence human-readable explanation")
    passed: bool = Field(description="True if score meets the pass threshold")


class EvalResult(BaseModel):
    """Combined evaluation result for one chat turn."""

    faithfulness: EvalScore
    answer_relevancy: EvalScore
    context_precision: EvalScore
    latency_ms: float = Field(description="Total request latency in milliseconds")
    overall_passed: bool = Field(
        description="True only if all three scored metrics pass"
    )
