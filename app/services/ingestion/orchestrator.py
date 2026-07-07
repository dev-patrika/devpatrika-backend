import logging
from datetime import datetime, timezone, timedelta
from sqlmodel import Session, select
from typing import List

from app.models.news import NewsItem
from app.models.github_radar import GitHubRadar
from app.core.constants import NewsSource, TechCategory
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

def run_all_ingestions(session: Session) -> dict:
    """
    Orchestrate the parsing of all feeds, perform URL & fuzzy-title 
    deduplication, run classifications, and store them in SQLite.
    """
    logger.info("Starting ingestion cycle...")
    
    stats = {
        "hacker_news": {"fetched": 0, "inserted": 0, "skipped_dup": 0},
        "dev_to": {"fetched": 0, "inserted": 0, "skipped_dup": 0},
        "arxiv": {"fetched": 0, "inserted": 0, "skipped_dup": 0},
        "github_trending": {"fetched": 0, "inserted": 0, "skipped_dup": 0},
    }
    
    # 1. Fetch recently stored items to do in-memory fuzzy title deduplication
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_items = session.exec(
        select(NewsItem).where(NewsItem.created_at >= yesterday)
    ).all()
    
    recent_titles = [item.title for item in recent_items]
    recent_urls = set(session.exec(select(NewsItem.url)).all())
    recent_repo_urls = set(session.exec(select(GitHubRadar.repo_url)).all())

    def is_title_duplicate(title: str) -> bool:
        for r_title in recent_titles:
            if check_title_similarity(title, r_title) > 0.8:
                return True
        return False

    # --- Hacker News Ingestion ---
    logger.info("Ingesting Hacker News...")
    hn_loader = HackerNewsLoader(limit=20)
    for doc in hn_loader.lazy_load():
        stats["hacker_news"]["fetched"] += 1
        url = doc.metadata.get("url")
        title = doc.page_content.split("\n")[0] if "\n" in doc.page_content else doc.page_content
        
        if url in recent_urls:
            stats["hacker_news"]["skipped_dup"] += 1
            continue
            
        if is_title_duplicate(title):
            stats["hacker_news"]["skipped_dup"] += 1
            continue
            
        category = map_category_by_keywords(doc.page_content, "")
        news_item = NewsItem(
            title=title,
            url=url,
            summary=None,
            category=category,
            source=NewsSource.HACKER_NEWS,
            published_at=doc.metadata.get("published_at"),
            raw_content=doc.page_content,
            freshness_tag=get_freshness_tag(doc.metadata.get("published_at")),
        )
        session.add(news_item)
        recent_urls.add(url)
        recent_titles.append(title)
        stats["hacker_news"]["inserted"] += 1

    # --- Dev.to Ingestion ---
    logger.info("Ingesting Dev.to...")
    devto_loader = DevToLoader(limit_per_tag=8)
    for doc in devto_loader.lazy_load():
        stats["dev_to"]["fetched"] += 1
        url = doc.metadata.get("url")
        title = doc.page_content.split("\n")[0] if "\n" in doc.page_content else doc.page_content
        
        if url in recent_urls:
            stats["dev_to"]["skipped_dup"] += 1
            continue
            
        if is_title_duplicate(title):
            stats["dev_to"]["skipped_dup"] += 1
            continue
            
        category = map_category_by_keywords(doc.page_content, "")
        news_item = NewsItem(
            title=title,
            url=url,
            summary=None,
            category=category,
            source=NewsSource.DEV_TO,
            published_at=doc.metadata.get("published_at"),
            raw_content=doc.page_content,
            freshness_tag=get_freshness_tag(doc.metadata.get("published_at")),
        )
        session.add(news_item)
        recent_urls.add(url)
        recent_titles.append(title)
        stats["dev_to"]["inserted"] += 1

    # --- arXiv Ingestion ---
    logger.info("Ingesting arXiv preprints...")
    arxiv_loader = ArxivLoader(max_results=15)
    for doc in arxiv_loader.lazy_load():
        stats["arxiv"]["fetched"] += 1
        url = doc.metadata.get("url")
        title = doc.page_content.split("\n")[0] if "\n" in doc.page_content else doc.page_content
        
        if url in recent_urls:
            stats["arxiv"]["skipped_dup"] += 1
            continue
            
        if is_title_duplicate(title):
            stats["arxiv"]["skipped_dup"] += 1
            continue
            
        category = map_category_by_keywords(doc.page_content, "")
        news_item = NewsItem(
            title=title,
            url=url,
            summary=None,
            category=category,
            source=NewsSource.ARXIV,
            published_at=doc.metadata.get("published_at"),
            raw_content=doc.page_content,
            freshness_tag=get_freshness_tag(doc.metadata.get("published_at")),
        )
        session.add(news_item)
        recent_urls.add(url)
        recent_titles.append(title)
        stats["arxiv"]["inserted"] += 1

    # --- GitHub Trending Ingestion ---
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
            repo_name=repo_name,
            repo_url=url,
            description=desc.strip(),
            why_it_matters_summary=None, # Will be generated in v0.3.0
            stars_count=doc.metadata.get("stars_count", 0),
            created_at=datetime.utcnow(),
        )
        session.add(github_item)
        recent_repo_urls.add(url)
        stats["github_trending"]["inserted"] += 1

    try:
        session.commit()
        logger.info(f"Ingestion cycle completed successfully. Stats: {stats}")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to commit ingestion cycle to database: {str(e)}")
        raise e

    return stats
