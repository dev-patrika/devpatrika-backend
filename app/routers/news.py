from fastapi import APIRouter, Query, Depends, BackgroundTasks
from typing import List, Optional
from app.schemas.news_schema import NewsItemRead
from app.core.constants import TechCategory
from sqlmodel import Session, select
from app.database import get_session
from app.models.news import NewsItem
from app.services.ingestion.orchestrator import run_all_ingestions

router = APIRouter(prefix="/news", tags=["News"])

@router.get("", response_model=List[NewsItemRead])
def get_news(
    category: Optional[TechCategory] = None,
    q: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    session: Session = Depends(get_session)
):
    """
    Retrieve stored daily tech news articles.
    Supports filtering by category, search queries matching title or content,
    and sorting by publication date.
    """
    statement = select(NewsItem).order_by(NewsItem.published_at.desc())
    
    if category:
        statement = statement.where(NewsItem.category == category)
        
    if q:
        # SQLite LIKE is case-insensitive for standard characters
        statement = statement.where(
            NewsItem.title.like(f"%{q}%") | NewsItem.raw_content.like(f"%{q}%")
        )
        
    statement = statement.limit(limit)
    results = session.exec(statement).all()
    return results

@router.post("/ingest", response_model=dict)
def trigger_ingestion(
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """
    Trigger feed ingestion manually. Runs the ingestion loaders in a non-blocking background task.
    """
    def run_ingestion_in_background():
        from app.database import engine
        with Session(engine) as bg_session:
            run_all_ingestions(bg_session)

    background_tasks.add_task(run_ingestion_in_background)
    return {
        "status": "ingestion_triggered",
        "detail": "News ingestion tasks have been scheduled to run in the background."
    }

