"""
Dev Patrika — Phase 1 Data Migration Script
SQLite → Neon (Postgres)

This script reads all rows from the local SQLite database and bulk-inserts
them into the Neon Postgres database. Run this AFTER `create_all()` has
created the tables on Neon.

Usage:
    python -m app.scripts.migrate_sqlite_to_neon
"""

import json
import logging
from datetime import datetime
from sqlmodel import create_engine, Session, SQLModel, select

# Import all models to register metadata
from app.models.news import NewsItem
from app.models.github_radar import GitHubRadar
from app.models.wiki import WikiEntry
from app.models.chat_history import ChatMessage
from app.models.trending_topic import TrendingTopic
from app.models.weekly_report import WeeklyReport
from app.models.notes import PersonalNote

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("migration")

# ============================================================
# Connection Strings
# ============================================================

SQLITE_URL = "sqlite:///./dev_patrika.db"

# Read Neon connection string from .env
from app.config import settings
NEON_URL = settings.DATABASE_URL

if not NEON_URL or "sqlite" in NEON_URL:
    raise RuntimeError(
        "DATABASE_URL in .env is still pointing to SQLite! "
        "Update it to the Neon Postgres connection string before running migration."
    )

# ============================================================
# Engine Setup
# ============================================================

sqlite_engine = create_engine(SQLITE_URL, echo=False, connect_args={"check_same_thread": False})
neon_engine = create_engine(NEON_URL, echo=False, pool_pre_ping=True)

def migrate_table(model_class, transform_fn=None):
    """
    Generic migration function: reads all rows from SQLite,
    optionally transforms them, and inserts into Neon.
    """
    table_name = model_class.__tablename__
    logger.info(f"Migrating table: {table_name}...")
    
    with Session(sqlite_engine) as sqlite_session:
        rows = sqlite_session.exec(select(model_class)).all()
    
    if not rows:
        logger.info(f"  ↳ No data in {table_name}. Skipping.")
        return 0
    
    inserted = 0
    with Session(neon_engine) as neon_session:
        for row in rows:
            # Create a new detached instance for Neon
            data = {}
            for col_name in model_class.__fields__:
                val = getattr(row, col_name, None)
                data[col_name] = val
            
            # Apply transformation if provided (e.g. related_links TEXT→JSONB)
            if transform_fn:
                data = transform_fn(data)
            
            new_row = model_class(**data)
            neon_session.add(new_row)
            inserted += 1
        
        neon_session.commit()
    
    logger.info(f"  ↳ Migrated {inserted} rows from {table_name}.")
    return inserted

def transform_wiki_links(data: dict) -> dict:
    """Convert related_links from JSON string to native Python list for JSONB."""
    links_val = data.get("related_links")
    if isinstance(links_val, str):
        try:
            data["related_links"] = json.loads(links_val)
        except (json.JSONDecodeError, TypeError):
            data["related_links"] = []
    elif links_val is None:
        data["related_links"] = []
    return data

def main():
    logger.info("=" * 60)
    logger.info("Dev Patrika — SQLite to Neon Migration")
    logger.info("=" * 60)
    logger.info(f"Source: {SQLITE_URL}")
    logger.info(f"Target: {NEON_URL[:50]}...")
    logger.info("")
    
    # Step 1: Create all tables on Neon
    logger.info("Creating tables on Neon (if not exist)...")
    SQLModel.metadata.create_all(neon_engine)
    logger.info("Tables created successfully.\n")
    
    # Step 2: Migrate each table
    stats = {}
    stats["news_items"] = migrate_table(NewsItem)
    stats["github_radar"] = migrate_table(GitHubRadar)
    stats["wiki_entries"] = migrate_table(WikiEntry, transform_fn=transform_wiki_links)
    stats["chat_messages"] = migrate_table(ChatMessage)
    stats["trending_topics"] = migrate_table(TrendingTopic)
    stats["weekly_reports"] = migrate_table(WeeklyReport)
    stats["personal_notes"] = migrate_table(PersonalNote)
    
    # Step 3: Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Migration Complete!")
    logger.info("=" * 60)
    total = sum(stats.values())
    for table, count in stats.items():
        logger.info(f"  {table}: {count} rows")
    logger.info(f"  TOTAL: {total} rows migrated")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start the FastAPI server: uvicorn app.main:app --reload")
    logger.info("  2. Hit /api/health to verify Neon connectivity")
    logger.info("  3. Test all API endpoints")

if __name__ == "__main__":
    main()
