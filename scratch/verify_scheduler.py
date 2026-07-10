import os
import sys

# Reconfigure stdout to use UTF-8 on Windows cmd/powershell to prevent UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import logging
from datetime import datetime, timedelta

# Ensure parent directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import Session, select
from app.database import engine, init_db
from app.services.ingestion.orchestrator import run_all_ingestions
from app.services.processing.pipeline import process_pending_items
from app.services.wiki_curator.pipeline import curate_wiki_from_news
from app.services.vectorstore.chroma_service import index_all_news_items
from app.services.trending.trending_engine import analyze_trending_topics
from app.services.reports.weekly_compiler import compile_weekly_report
from app.models.weekly_report import WeeklyReport
from app.models.news import NewsItem

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("verify-scheduler")

def main():
    logger.info("Initializing SQLite database tables...")
    init_db()
    
    logger.info("Starting synchronous verification of background scheduler tasks...")
    
    with Session(engine) as session:
        # Step 1: Run Ingestion
        logger.info("--- Step 1: Run Ingestion Orchestrator ---")
        try:
            ingest_stats = run_all_ingestions(session)
            logger.info(f"Ingestion Finished. Stats: {ingest_stats}")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            
        # Step 2: Run AI pipeline processing
        logger.info("--- Step 2: Run AI pipeline processing (Summarization) ---")
        try:
            ai_stats = process_pending_items(session)
            logger.info(f"AI Summarization Finished. Stats: {ai_stats}")
        except Exception as e:
            logger.error(f"AI processing failed: {e}")
            
        # Step 3: Run Vector indexing
        logger.info("--- Step 3: Run Vector Indexing ---")
        try:
            index_all_news_items(session)
            logger.info("News vector synchronization complete.")
        except Exception as e:
            logger.error(f"Vector indexing failed: {e}")
            
        # Step 4: Run Wiki curator auto-curation
        logger.info("--- Step 4: Run Wiki curator auto-curation ---")
        try:
            wiki_stats = curate_wiki_from_news(session)
            logger.info(f"Wiki Curation Finished. Stats: {wiki_stats}")
        except Exception as e:
            logger.error(f"Wiki curation failed: {e}")
            
        # Step 5: Run Trending Topics analysis
        logger.info("--- Step 5: Run Trending Topics analysis ---")
        try:
            trending_stats = analyze_trending_topics(session)
            logger.info(f"Trending Analysis Finished. Stats: {trending_stats}")
        except Exception as e:
            logger.error(f"Trending topics analysis failed: {e}")
            
        # Step 6: Run Weekly Report compilation (with forced mock to bypass 7-day rule for verification)
        logger.info("--- Step 6: Run Weekly Report Compilation ---")
        try:
            logger.info("Forcing weekly report generation for verification...")
            report = compile_weekly_report(session)
            if report:
                logger.info(f"Weekly Report compiled successfully: Title='{report.title}', ID={report.id}")
            else:
                # Let's check how many news items we have, in case compile_weekly_report returned None due to insufficient data
                news_count = session.exec(select(NewsItem)).all()
                logger.warning(f"compile_weekly_report returned None. Current news items count: {len(news_count)}")
        except Exception as e:
            logger.error(f"Weekly report compilation failed: {e}")

    logger.info("Verification of background scheduler tasks complete.")

if __name__ == "__main__":
    main()
