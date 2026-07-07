from fastapi import APIRouter, Query, Depends
from typing import List, Optional
from app.schemas.news_schema import NewsItemRead
from app.core.constants import TechCategory
from sqlmodel import Session
from app.database import get_session

router = APIRouter(prefix="/news", tags=["News"])

@router.get("", response_model=List[NewsItemRead])
def get_news(
    category: Optional[TechCategory] = None,
    q: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    session: Session = Depends(get_session)
):
    """
    Placeholder endpoint to retrieve daily tech news.
    Filters by category and search queries will be implemented in v0.2.0.
    """
    return []

@router.post("/ingest", response_model=dict)
def trigger_ingestion(session: Session = Depends(get_session)):
    """
    Placeholder endpoint to trigger feed ingestion manually.
    Implementation will be active in v0.2.0.
    """
    return {"status": "ingestion_triggered", "detail": "News ingestion tasks will run in the background (v0.2.0)."}
