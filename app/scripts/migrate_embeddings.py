import logging
from sqlmodel import Session, text
from app.database import engine
from app.services.vectorstore.vector_service import index_all_wiki_entries, index_all_news_items

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dev-patrika.migrate-embeddings")

def migrate():
    logger.info("Starting embeddings migration...")
    
    with Session(engine) as session:
        # Drop existing pgvector tables CASCADE so that the vector column dimension is reset
        logger.info("Dropping old pgvector tables to reset dimensions...")
        try:
            session.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;"))
            session.execute(text("DROP TABLE IF EXISTS langchain_pg_collection CASCADE;"))
            session.commit()
            logger.info("Old vector tables dropped successfully.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to drop old vector tables: {e}")
            raise e
            
        # Re-index everything
        logger.info("Re-indexing all Wiki Glossary entries using Hugging Face embeddings...")
        try:
            index_all_wiki_entries(session)
        except Exception as e:
            logger.error(f"Error during wiki entry re-indexing: {e}")
            
        logger.info("Re-indexing all News Items using Hugging Face embeddings...")
        try:
            index_all_news_items(session)
        except Exception as e:
            logger.error(f"Error during news items re-indexing: {e}")
            
    logger.info("Embeddings migration completed successfully!")

if __name__ == "__main__":
    migrate()
