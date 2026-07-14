from fastapi import APIRouter, Query, Depends, BackgroundTasks
from typing import List, Optional
from app.schemas.news_schema import NewsItemRead
from app.core.constants import TechCategory
from sqlmodel import Session, select
from app.database import get_session
from app.models.news import NewsItem
from app.agents.daily_brief import run_daily_brief_agent

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
        statement = statement.where(
            NewsItem.title.ilike(f"%{q}%") | NewsItem.raw_content.ilike(f"%{q}%")
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
    Trigger feed ingestion manually. Runs the ingestion loaders and LLM processing in a non-blocking background task.
    """
    def run_ingestion_in_background():
        from app.database import engine
        with Session(engine) as bg_session:
            run_daily_brief_agent(bg_session)

    background_tasks.add_task(run_ingestion_in_background)
    return {
        "status": "ingestion_triggered",
        "detail": "News ingestion and AI processing tasks have been scheduled to run in the background."
    }

@router.post("/process", response_model=dict)
def trigger_processing(
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """
    Trigger AI processing manually to summarize and categorize pending items in the database.
    """
    def run_processing_in_background():
        from app.database import engine
        from app.services.processing.pipeline import process_pending_items
        with Session(engine) as bg_session:
            process_pending_items(bg_session)

    background_tasks.add_task(run_processing_in_background)
    return {
        "status": "processing_triggered",
        "detail": "AI processing of pending news and GitHub repositories has been scheduled in the background."
    }

@router.get("/{news_id}/related", response_model=List[NewsItemRead])
def get_related_news_articles(news_id: int, session: Session = Depends(get_session)):
    """
    Retrieve semantically related news articles based on vector similarity.
    """
    from app.services.vectorstore.vector_service import get_related_articles
    related = get_related_articles(session, item_id=news_id, limit=3, threshold=0.3)
    return related

@router.get("/{news_id}", response_model=NewsItemRead)
def get_single_news(news_id: int, session: Session = Depends(get_session)):
    """
    Retrieve details of a single tech news article by its database ID.
    """
    from fastapi import HTTPException
    news_item = session.get(NewsItem, news_id)
    if not news_item:
        raise HTTPException(status_code=404, detail="News article not found")
    return news_item

