"""
routers/history.py — GET /api/history

Returns the last N eval runs from SQLite, ordered by most recent first.
Used by the History page in the Streamlit UI to display past eval scores.
"""

from fastapi import APIRouter

from database import get_history

router = APIRouter()


@router.get("/history")
async def history(limit: int = 50) -> list[dict]:
    """
    Return recent conversation history with all eval scores.

    Each record includes the new columns added in the 8-dimension overhaul:
    domain_tag, answer_relevancy_passed, completeness_passed,
    context_recall_passed, coherence_passed, historical_balance_passed,
    toxicity_passed, checklist_json.
    """
    # get_history() does SELECT * so all columns — old and new — are returned
    return get_history(limit=min(limit, 100))
