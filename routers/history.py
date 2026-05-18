"""
routers/history.py — GET /api/history

Returns the last 50 eval runs from SQLite, ordered by most recent first.
Used by the frontend to display past eval scores and detect trends.
"""

from fastapi import APIRouter

from database import get_history

router = APIRouter()


@router.get("/history")
async def history(limit: int = 50) -> list[dict]:
    """Return recent conversation history with eval scores."""
    return get_history(limit=min(limit, 100))  # Cap at 100 for safety
