"""
Dev Patrika — Ingestion Orchestrator (LangGraph Refactor)

Converts the old sequential orchestrator.py into a LangGraph StateGraph.

Graph Structure:
    START → load_recent_context → fetch_hn → fetch_devto → fetch_arxiv → fetch_github → commit_all → END

Note: Sources are sequential (not parallel) to maintain dedup consistency
within the same ingestion cycle.
"""

import logging
from datetime import datetime, timedelta
from typing import TypedDict
from sqlmodel import Session, select
from langgraph.graph import StateGraph, START, END

from app.models.news import NewsItem
from app.models.github_radar import GitHubRadar
from app.core.constants import NewsSource
from app.services.ingestion.hn import HackerNewsLoader
from app.services.ingestion.devto import DevToLoader
from app.services.ingestion.arxiv import ArxivLoader
from app.services.ingestion.github_trending import GitHubTrendingLoader
from app.services.ingestion.helpers import (
    get_freshness_tag,
    check_title_similarity,
    map_category_by_keywords,
)

logger = logging.getLogger("dev-patrika.ingestion")

# =====================================================================
# LangGraph State Definition
# =====================================================================

class IngestionState(TypedDict):
    """Shared state flowing through the ingestion graph."""
    session: Session
    recent_titles: list           # List[str]
    recent_urls: set              # Set of known news URLs
    recent_repo_urls: set         # Set of known GitHub repo URLs
    stats: dict                   # Nested stats dict per source

# =====================================================================
# Helper Functions
# =====================================================================

def _is_title_duplicate(title: str, recent_titles: list) -> bool:
    for r_title in recent_titles:
        if check_title_similarity(title, r_title) > 0.8:
            return True
    return False

def _extract_title(page_content: str) -> str:
    return page_content.split("\n")[0] if "\n" in page_content else page_content

# =====================================================================
# Graph Node Functions
# =====================================================================

def load_recent_context_node(state: IngestionState) -> dict:
    """Node: Load recently stored items for deduplication."""
    session = state["session"]
    
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_items = session.exec(
        select(NewsItem).where(NewsItem.created_at >= yesterday)
    ).all()
    
    recent_titles = [item.title for item in recent_items]
    recent_urls = set(session.exec(select(NewsItem.url)).all())
    recent_repo_urls = set(session.exec(select(GitHubRadar.repo_url)).all())
    
    stats = {
        "hacker_news": {"fetched": 0, "inserted": 0, "skipped_dup": 0},
        "dev_to": {"fetched": 0, "inserted": 0, "skipped_dup": 0},
        "arxiv": {"fetched": 0, "inserted": 0, "skipped_dup": 0},
        "github_trending": {"fetched": 0, "inserted": 0, "skipped_dup": 0},
    }
    
    logger.info(f"Loaded {len(recent_titles)} recent titles and {len(recent_urls)} recent URLs for dedup.")
    
    return {
        "recent_titles": recent_titles,
        "recent_urls": recent_urls,
        "recent_repo_urls": recent_repo_urls,
        "stats": stats,
    }

def fetch_hn_node(state: IngestionState) -> dict:
    """Node: Ingest Hacker News items."""
    session = state["session"]
    recent_urls = state["recent_urls"]
    recent_titles = state["recent_titles"]
    stats = state["stats"]
    
    logger.info("Ingesting Hacker News...")
    hn_loader = HackerNewsLoader(limit=10)
    
    for doc in hn_loader.lazy_load():
        stats["hacker_news"]["fetched"] += 1
        url = doc.metadata.get("url")
        title = _extract_title(doc.page_content)
        
        if url in recent_urls:
            stats["hacker_news"]["skipped_dup"] += 1
            continue
        if _is_title_duplicate(title, recent_titles):
            stats["hacker_news"]["skipped_dup"] += 1
            continue
        
        category = map_category_by_keywords(doc.page_content, "")
        news_item = NewsItem(
            title=title, url=url, summary=None, category=category,
            source=NewsSource.HACKER_NEWS,
            published_at=doc.metadata.get("published_at"),
            raw_content=doc.page_content,
            freshness_tag=get_freshness_tag(doc.metadata.get("published_at")),
        )
        session.add(news_item)
        recent_urls.add(url)
        recent_titles.append(title)
        stats["hacker_news"]["inserted"] += 1
    
    return {"recent_urls": recent_urls, "recent_titles": recent_titles, "stats": stats}

def fetch_devto_node(state: IngestionState) -> dict:
    """Node: Ingest Dev.to items."""
    session = state["session"]
    recent_urls = state["recent_urls"]
    recent_titles = state["recent_titles"]
    stats = state["stats"]
    
    logger.info("Ingesting Dev.to...")
    devto_loader = DevToLoader(limit_per_tag=3)
    
    for doc in devto_loader.lazy_load():
        stats["dev_to"]["fetched"] += 1
        url = doc.metadata.get("url")
        title = _extract_title(doc.page_content)
        
        if url in recent_urls:
            stats["dev_to"]["skipped_dup"] += 1
            continue
        if _is_title_duplicate(title, recent_titles):
            stats["dev_to"]["skipped_dup"] += 1
            continue
        
        category = map_category_by_keywords(doc.page_content, "")
        news_item = NewsItem(
            title=title, url=url, summary=None, category=category,
            source=NewsSource.DEV_TO,
            published_at=doc.metadata.get("published_at"),
            raw_content=doc.page_content,
            freshness_tag=get_freshness_tag(doc.metadata.get("published_at")),
        )
        session.add(news_item)
        recent_urls.add(url)
        recent_titles.append(title)
        stats["dev_to"]["inserted"] += 1
    
    return {"recent_urls": recent_urls, "recent_titles": recent_titles, "stats": stats}

def fetch_arxiv_node(state: IngestionState) -> dict:
    """Node: Ingest arXiv preprints."""
    session = state["session"]
    recent_urls = state["recent_urls"]
    recent_titles = state["recent_titles"]
    stats = state["stats"]
    
    logger.info("Ingesting arXiv preprints...")
    arxiv_loader = ArxivLoader(max_results=5)
    
    for doc in arxiv_loader.lazy_load():
        stats["arxiv"]["fetched"] += 1
        url = doc.metadata.get("url")
        title = _extract_title(doc.page_content)
        
        if url in recent_urls:
            stats["arxiv"]["skipped_dup"] += 1
            continue
        if _is_title_duplicate(title, recent_titles):
            stats["arxiv"]["skipped_dup"] += 1
            continue
        
        category = map_category_by_keywords(doc.page_content, "")
        news_item = NewsItem(
            title=title, url=url, summary=None, category=category,
            source=NewsSource.ARXIV,
            published_at=doc.metadata.get("published_at"),
            raw_content=doc.page_content,
            freshness_tag=get_freshness_tag(doc.metadata.get("published_at")),
        )
        session.add(news_item)
        recent_urls.add(url)
        recent_titles.append(title)
        stats["arxiv"]["inserted"] += 1
    
    return {"recent_urls": recent_urls, "recent_titles": recent_titles, "stats": stats}

def fetch_github_node(state: IngestionState) -> dict:
    """Node: Ingest GitHub Trending repos."""
    session = state["session"]
    recent_repo_urls = state["recent_repo_urls"]
    stats = state["stats"]
    
    logger.info("Ingesting GitHub Trending repos...")
    github_loader = GitHubTrendingLoader(since="daily")
    
    for doc in github_loader.lazy_load():
        stats["github_trending"]["fetched"] += 1
        url = doc.metadata.get("url")
        repo_name = doc.metadata.get("repo_name")
        
        if url in recent_repo_urls:
            stats["github_trending"]["skipped_dup"] += 1
            continue
        
        desc = doc.page_content.replace(f"Repository: {repo_name}\nDescription: ", "")
        desc = desc.split("\nLanguage:")[0] if "\nLanguage:" in desc else desc
        
        github_item = GitHubRadar(
            repo_name=repo_name, repo_url=url, description=desc.strip(),
            why_it_matters_summary=None,
            stars_count=doc.metadata.get("stars_count", 0),
            created_at=datetime.utcnow(),
        )
        session.add(github_item)
        recent_repo_urls.add(url)
        stats["github_trending"]["inserted"] += 1
    
    return {"recent_repo_urls": recent_repo_urls, "stats": stats}

def commit_all_node(state: IngestionState) -> dict:
    """Node: Commit all ingested items to the database."""
    session = state["session"]
    stats = state["stats"]
    
    try:
        session.commit()
        logger.info(f"Ingestion cycle committed successfully. Stats: {stats}")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to commit ingestion cycle: {str(e)}")
        raise e
    
    return {"stats": stats}

# =====================================================================
# Build the LangGraph StateGraph
# =====================================================================

def build_ingestion_graph() -> StateGraph:
    """Construct and compile the ingestion orchestrator graph."""
    graph = StateGraph(IngestionState)
    
    # Add nodes
    graph.add_node("load_recent_context", load_recent_context_node)
    graph.add_node("fetch_hn", fetch_hn_node)
    graph.add_node("fetch_devto", fetch_devto_node)
    graph.add_node("fetch_arxiv", fetch_arxiv_node)
    graph.add_node("fetch_github", fetch_github_node)
    graph.add_node("commit_all", commit_all_node)
    
    # Sequential edges (maintains dedup consistency within cycle)
    graph.add_edge(START, "load_recent_context")
    graph.add_edge("load_recent_context", "fetch_hn")
    graph.add_edge("fetch_hn", "fetch_devto")
    graph.add_edge("fetch_devto", "fetch_arxiv")
    graph.add_edge("fetch_arxiv", "fetch_github")
    graph.add_edge("fetch_github", "commit_all")
    graph.add_edge("commit_all", END)
    
    return graph.compile()

# =====================================================================
# Public API (backward compatible with scheduler.py)
# =====================================================================

_ingestion_graph = build_ingestion_graph()

def run_all_ingestions(session: Session) -> dict:
    """
    Public entry point — backward compatible with old orchestrator API.
    Invokes the LangGraph ingestion pipeline.
    """
    logger.info("Running LangGraph ingestion pipeline...")
    
    initial_state = {
        "session": session,
        "recent_titles": [],
        "recent_urls": set(),
        "recent_repo_urls": set(),
        "stats": {},
    }
    
    final_state = _ingestion_graph.invoke(initial_state)
    
    logger.info(f"LangGraph ingestion pipeline completed.")
    return final_state.get("stats", {})
