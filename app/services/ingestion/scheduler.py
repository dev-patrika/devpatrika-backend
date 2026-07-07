import asyncio
import logging
from sqlmodel import Session, select
from app.database import engine
from app.models.news import NewsItem
from app.services.ingestion.orchestrator import run_all_ingestions

logger = logging.getLogger("dev-patrika.scheduler")

async def ingestion_scheduler_loop():
    logger.info("Background Ingestion Scheduler starting up...")
    
    # Wait for 5 seconds to let the server boot cleanly
    await asyncio.sleep(5)
    
    while True:
        try:
            logger.info("Executing periodic feed ingestion...")
            with Session(engine) as session:
                # Run the orchestrator to poll and save documents
                stats = run_all_ingestions(session)
                logger.info(f"Feed ingestion loop finished. Stats: {stats}")
        except Exception as e:
            logger.error(f"Error during scheduled feed ingestion: {str(e)}")
            
        logger.info("Scheduler sleeping for 6 hours before next cycle.")
        # Sleep for 6 hours (6 * 3600 seconds)
        await asyncio.sleep(6 * 3600)

def start_scheduler():
    """Register the scheduler loop task in the running event loop."""
    logger.info("Scheduling background ingestion task loop...")
    asyncio.create_task(ingestion_scheduler_loop())
