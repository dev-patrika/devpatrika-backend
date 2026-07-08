import logging
from datetime import datetime, timedelta, timezone
from sqlmodel import Session, select

from app.models.news import NewsItem
from app.models.wiki import WikiEntry
from app.models.trending_topic import TrendingTopic

logger = logging.getLogger("dev-patrika.trending.engine")

def analyze_trending_topics(session: Session, days: int = 7) -> dict:
    """
    Scans recent news items, counts mentions of existing wiki terms,
    compares them to previous counts, updates trending directions,
    and returns a summary of the trending items.
    """
    logger.info(f"Running Trending Topics analysis for the last {days} days...")
    stats = {"scanned_terms": 0, "updated_trends": 0}
    
    try:
        # 1. Fetch all registered Wiki terms
        wiki_statement = select(WikiEntry)
        wiki_entries = session.exec(wiki_statement).all()
        if not wiki_entries:
            logger.info("No wiki entries found in SQLite to track for trends.")
            return stats
            
        # 2. Fetch news items processed in the last 'days' days
        since_time = datetime.utcnow() - timedelta(days=days)
        news_statement = select(NewsItem).where(NewsItem.created_at >= since_time)
        recent_news = session.exec(news_statement).all()
        
        logger.info(f"Loaded {len(recent_news)} news items for trend evaluation.")
        
        # 3. Process frequencies for each Wiki term
        for wiki in wiki_entries:
            term_lower = wiki.term.lower()
            mention_count = 0
            
            for news in recent_news:
                text_block = f"{news.title} {news.summary or ''} {news.raw_content or ''}".lower()
                if term_lower in text_block:
                    mention_count += 1
            
            # Fetch existing trend record
            trend_statement = select(TrendingTopic).where(TrendingTopic.term == wiki.term)
            existing_trend = session.exec(trend_statement).first()
            
            if existing_trend:
                # Determine direction based on frequency change
                if mention_count > existing_trend.frequency:
                    direction = "up"
                elif mention_count < existing_trend.frequency:
                    direction = "down"
                else:
                    direction = "stable"
                    
                existing_trend.frequency = mention_count
                existing_trend.trend_direction = direction
                existing_trend.updated_at = datetime.utcnow()
                session.add(existing_trend)
            else:
                # First time seeing this term in trends
                direction = "up" if mention_count > 0 else "stable"
                new_trend = TrendingTopic(
                    term=wiki.term,
                    frequency=mention_count,
                    trend_direction=direction,
                    updated_at=datetime.utcnow()
                )
                session.add(new_trend)
                
            stats["scanned_terms"] += 1
            stats["updated_trends"] += 1
            
        session.commit()
        logger.info(f"Trending Topics analysis complete. Stats: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Error during Trending Topics analysis: {str(e)}")
        session.rollback()
        return stats
