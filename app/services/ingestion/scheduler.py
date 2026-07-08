import asyncio
import logging
from sqlmodel import Session
from app.database import engine
from app.services.ingestion.orchestrator import run_all_ingestions
from app.services.processing.pipeline import process_pending_items
from app.services.wiki_curator.pipeline import curate_wiki_from_news

logger = logging.getLogger("dev-patrika.scheduler")

async def ingestion_scheduler_loop():
    logger.info("Background Ingestion Scheduler starting up...")
    
    # Wait for 5 seconds to let the server boot cleanly
    await asyncio.sleep(5)
    
    while True:
        try:
            logger.info("Executing periodic feed ingestion and processing cycle...")
            with Session(engine) as session:
                # 1. Run the orchestrator to poll and save documents
                stats = run_all_ingestions(session)
                logger.info(f"Feed ingestion loop finished. Stats: {stats}")
                
                # 2. Run LLM news summarization
                logger.info("Executing scheduled AI summarization...")
                ai_stats = process_pending_items(session)
                logger.info(f"AI summarization finished. Stats: {ai_stats}")
                
                # 3. Run Wiki curator auto-curation
                logger.info("Executing scheduled Wiki curation...")
                wiki_stats = curate_wiki_from_news(session)
                logger.info(f"Wiki curation finished. Stats: {wiki_stats}")
        except Exception as e:
            logger.error(f"Error during scheduled feed ingestion/processing: {str(e)}")
            
        logger.info("Scheduler sleeping for 6 hours before next cycle.")
        # Sleep for 6 hours (6 * 3600 seconds)
        await asyncio.sleep(6 * 3600)

def start_scheduler():
    """Register the scheduler loop task in the running event loop."""
    logger.info("Scheduling background ingestion task loop...")
    asyncio.create_task(ingestion_scheduler_loop())
