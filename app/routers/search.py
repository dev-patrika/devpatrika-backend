from fastapi import APIRouter, Query, Depends
from sqlmodel import Session, select
from typing import Dict, Any

from app.database import get_session
from app.models.news import NewsItem
from app.models.github_radar import GitHubRadar
from app.services.vectorstore.vector_service import semantic_search_wiki

router = APIRouter(prefix="/search", tags=["Unified Search"])

@router.get("")
def unified_search(
    q: str = Query(..., min_length=1, description="Query search string"),
    session: Session = Depends(get_session)
):
    """
    Unified cross-search endpoint.
    Retrieves keyword matching news items and GitHub repos from SQL,
    and returns semantically matching wiki entries from Chroma DB.
    """
    # 1. SQL search over NewsItem (title, content, summary)
    news_statement = select(NewsItem).where(
        NewsItem.title.like(f"%{q}%") | 
        (NewsItem.raw_content.like(f"%{q}%") if NewsItem.raw_content is not None else False) | 
        (NewsItem.summary.like(f"%{q}%") if NewsItem.summary is not None else False)
    ).order_by(NewsItem.published_at.desc()).limit(10)
    news_results = session.exec(news_statement).all()

    # 2. SQL search over GitHub Radar (repo_name, description)
    github_statement = select(GitHubRadar).where(
        GitHubRadar.repo_name.like(f"%{q}%") | 
        (GitHubRadar.description.like(f"%{q}%") if GitHubRadar.description is not None else False)
    ).order_by(GitHubRadar.stars_count.desc()).limit(10)
    github_results = session.exec(github_statement).all()

    # 3. Chroma semantic similarity search over wiki entries
    wiki_results = semantic_search_wiki(session, query=q, limit=5, threshold=0.5)

    return {
        "query": q,
        "results": {
            "news": news_results,
            "wiki": wiki_results,
            "github_radar": github_results
        }
    }
