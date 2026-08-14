import asyncio
import logging
from datetime import datetime, timedelta
from sqlmodel import Session, select
from app.database import engine
from app.agents.daily_brief import run_daily_brief_agent
from app.agents.wiki_curator import run_wiki_curator_agent
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
                # 1. Execute the stateful Daily Brief Agent (Ingestion + AI Summarization)
                with Session(engine) as session:
                    logger.info("Executing scheduled stateful Daily Brief Agent...")
                    brief_stats = run_daily_brief_agent(session)
                    logger.info(f"Daily Brief Agent finished. Stats: {brief_stats}")
                    
                # 2. Index new news items in Neon pgvector database
                with Session(engine) as session:
                    logger.info("Synchronizing news vectors to pgvector...")
                    index_all_news_items(session)
                    
                # 3. Execute the stateful Wiki Curator Agent (Extraction + Conflict Resolution / Merging)
                with Session(engine) as session:
                    logger.info("Executing scheduled stateful Wiki Curator Agent...")
                    wiki_stats = run_wiki_curator_agent(session)
                    logger.info(f"Wiki Curator Agent finished. Stats: {wiki_stats}")
                    
                # 4. Run Trending Topics analysis
                with Session(engine) as session:
                    logger.info("Executing scheduled Trending Topics analysis...")
                    trending_stats = analyze_trending_topics(session)
                    logger.info(f"Trending analysis finished. Stats: {trending_stats}")
                    
                # 5. Check if Weekly Report compilation is needed
                with Session(engine) as session:
                    latest_report = session.exec(
                        select(WeeklyReport).order_by(WeeklyReport.created_at.desc())
                    ).first()
                    
                    # ✅ FIX: Use timedelta comparison, not .days
                    # .days strips hours/minutes — so 6d 23h 59m = .days of 6, never triggers
                    should_compile = (
                        not latest_report or
                        (datetime.utcnow() - latest_report.created_at) >= timedelta(days=7)
                    )
                    if should_compile:
                        logger.info("Generating weekly compiled developer report...")
                        report = compile_weekly_report(session)
                        if report:
                            logger.info(f"Weekly report compiled successfully: ID {report.id}")
                        else:
                            logger.warning("Weekly report compilation returned None (insufficient data?)")
                    else:
                        days_since = (datetime.utcnow() - latest_report.created_at).days
                        logger.info(f"Weekly report skipped — last compiled {days_since} day(s) ago.")
            
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
