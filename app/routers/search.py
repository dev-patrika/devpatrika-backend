from fastapi import APIRouter, Query, Depends
from sqlmodel import Session
from app.database import get_session

router = APIRouter(prefix="/search", tags=["Unified Search"])

@router.get("")
def unified_search(
    q: str = Query(..., min_length=1),
    session: Session = Depends(get_session)
):
    """
    Placeholder endpoint for cross-search queries (news stories + wiki terms).
    Fully implemented in v0.4.0.
    """
    return {
        "query": q,
        "results": {
            "news": [],
            "wiki": []
        },
        "message": "Unified database and vector search functionality will be activated in v0.4.0."
    }
