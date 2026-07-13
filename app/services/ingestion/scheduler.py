import asyncio
import logging
from datetime import datetime
from sqlmodel import Session, select
from app.database import engine
from app.services.ingestion.orchestrator import run_all_ingestions
from app.services.processing.pipeline import process_pending_items
from app.services.wiki_curator.pipeline import curate_wiki_from_news
from app.services.vectorstore.vector_service import index_all_news_items
from app.services.trending.trending_engine import analyze_trending_topics
from app.services.reports.weekly_compiler import compile_weekly_report
from app.models.weekly_report import WeeklyReport

logger = logging.getLogger("dev-patrika.scheduler")

async def ingestion_scheduler_loop():
    logger.info("Background Ingestion Scheduler starting up...")
    
    # Wait for 5 seconds to let the server boot cleanly
    await asyncio.sleep(5)
    
    while True:
        try:
            logger.info("Executing periodic feed ingestion and processing cycle...")
            
            # Wrap the entirely synchronous heavy process in a separate thread
            # so it does not block the FastAPI event loop.
            def run_sync_tasks():
                with Session(engine) as session:
                    # 1. Run the orchestrator to poll and save documents
                    stats = run_all_ingestions(session)
                    logger.info(f"Feed ingestion loop finished. Stats: {stats}")
                    
                    # 2. Run LLM news summarization
                    logger.info("Executing scheduled AI summarization...")
                    ai_stats = process_pending_items(session)
                    logger.info(f"AI summarization finished. Stats: {ai_stats}")
                    
                    # Index new news items in Neon pgvector database
                    logger.info("Synchronizing news vectors to pgvector...")
                    index_all_news_items(session)
                    
                    # 3. Run Wiki curator auto-curation
                    logger.info("Executing scheduled Wiki curation...")
                    wiki_stats = curate_wiki_from_news(session)
                    logger.info(f"Wiki curation finished. Stats: {wiki_stats}")
                    
                    # 4. Run Trending Topics analysis
                    logger.info("Executing scheduled Trending Topics analysis...")
                    trending_stats = analyze_trending_topics(session)
                    logger.info(f"Trending analysis finished. Stats: {trending_stats}")
                    
                    # 5. Check if Weekly Report compilation is needed
                    latest_report = session.exec(
                        select(WeeklyReport).order_by(WeeklyReport.created_at.desc())
                    ).first()
                    
                    if not latest_report or (datetime.utcnow() - latest_report.created_at).days >= 7:
                        logger.info("Generating weekly compiled developer report...")
                        compile_weekly_report(session)
            
            await asyncio.to_thread(run_sync_tasks)

        except Exception as e:
            logger.error(f"Error during scheduled feed ingestion/processing: {str(e)}")
            
        logger.info("Scheduler sleeping for 4 hours before next cycle.")
        # Sleep for 4 hours (4 * 3600 seconds)
        await asyncio.sleep(4 * 3600)

def start_scheduler():
    """Register the scheduler loop task in the running event loop."""
    logger.info("Scheduling background ingestion task loop...")
    asyncio.create_task(ingestion_scheduler_loop())
