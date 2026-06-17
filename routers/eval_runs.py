"""
routers/eval_runs.py — GET /api/eval-runs

Returns paginated eval run records from the database, optionally filtered
by domain_tag. Used by the Eval Dashboard page in the Streamlit UI.
"""

from fastapi import APIRouter

from database import get_eval_runs

router = APIRouter()


@router.get("/eval-runs")
async def list_eval_runs(
    domain: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    Return paginated eval run records.

    Query params:
        domain: Filter by domain_tag (ww1, ww2, historical_figures, revolutions).
        limit:  Max records to return (server caps at 200).
        offset: Skip this many records for pagination.
    """
    runs = get_eval_runs(domain=domain, limit=limit, offset=offset)
    return {"runs": runs, "total": len(runs)}
