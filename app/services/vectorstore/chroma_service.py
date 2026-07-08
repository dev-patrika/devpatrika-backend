import os
import logging
from typing import List, Tuple
from sqlmodel import Session, select
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from app.config import settings
from app.models.wiki import WikiEntry
from app.models.news import NewsItem

logger = logging.getLogger("dev-patrika.vectorstore.chroma")

# Path to persistent Chroma DB storage
CHROMA_PERSIST_DIR = os.path.abspath("chroma_db")

# Initialize Embeddings model (Google gemini-embedding-2)
# Sync API keys to environment if needed
if settings.GEMINI_API_KEY and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
)

def get_vectorstore(collection_name: str = "wiki_entries") -> Chroma:
    """Initialize or load the local persistent Chroma vector database for a specific collection."""
    # Ensure directory exists
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

# =====================================================================
# Wiki Entries Vector Operations
# =====================================================================

def index_wiki_entry(entry: WikiEntry):
    """
    Format, embed, and upsert a single WikiEntry into the Chroma vectorstore.
    """
    if not entry or not entry.id:
        logger.warning("Attempted to index an empty or unsaved wiki entry.")
        return
        
    logger.info(f"Indexing WikiEntry ID {entry.id} ('{entry.term}') in Chroma...")
    try:
        db = get_vectorstore(collection_name="wiki_entries")
        
        # Format semantic text content
        content = (
            f"Term: {entry.term}\n\n"
            f"Definition: {entry.definition}\n\n"
            f"Why it is trending: {entry.why_trending}"
        )
        
        doc = Document(
            page_content=content,
            metadata={
                "wiki_entry_id": entry.id,
                "term": entry.term
            }
        )
        
        # Check if document already exists and delete it first to prevent duplicates
        doc_id = str(entry.id)
        try:
            db.delete(ids=[doc_id])
        except Exception:
            # First insertion won't find it, ignore delete errors
            pass
            
        db.add_documents(documents=[doc], ids=[doc_id])
        logger.info(f"Successfully indexed WikiEntry '{entry.term}' in Chroma.")
    except Exception as e:
        logger.error(f"Failed to index WikiEntry ID {entry.id} in Chroma: {str(e)}")

def index_all_wiki_entries(session: Session):
    """
    Batch index all SQLite WikiEntry records into the Chroma DB.
    """
    logger.info("Starting batch wiki vector synchronization...")
    try:
        statement = select(WikiEntry)
        entries = session.exec(statement).all()
        
        if not entries:
            logger.info("No wiki entries found in SQLite to index.")
            return
            
        logger.info(f"Indexing {len(entries)} wiki entries...")
        for entry in entries:
            index_wiki_entry(entry)
            
        logger.info("Batch wiki vector synchronization complete.")
    except Exception as e:
        logger.error(f"Failed during batch wiki vector synchronization: {str(e)}")

def semantic_search_wiki(session: Session, query: str, limit: int = 3, threshold: float = 0.3) -> List[WikiEntry]:
    """
    Search Chroma DB semantically using cosine distance.
    Returns SQLite matching WikiEntry database records sorted by relevance.
    """
    logger.info(f"Running semantic wiki search for query: '{query}'")
    
    try:
        db = get_vectorstore(collection_name="wiki_entries")
        raw_results = db.similarity_search_with_score(query, k=limit)
        
        valid_ids = []
        scores_map = {}
        
        for doc, distance in raw_results:
            wiki_id = doc.metadata.get("wiki_entry_id")
            similarity = 1.0 - distance
            
            logger.info(f"Semantic match (Wiki): '{doc.metadata.get('term')}' (Similarity score: {similarity:.4f})")
            
            if similarity >= threshold:
                valid_ids.append(wiki_id)
                scores_map[wiki_id] = similarity
                
        if not valid_ids:
            logger.info("No semantic matches passed the similarity threshold for Wiki.")
            return []
            
        statement = select(WikiEntry).where(WikiEntry.id.in_(valid_ids))
        db_entries = session.exec(statement).all()
        db_entries.sort(key=lambda x: scores_map.get(x.id, 0.0), reverse=True)
        return db_entries
    except Exception as e:
        logger.error(f"Error during semantic wiki search: {str(e)}")
        return []

# =====================================================================
# News Items Vector Operations
# =====================================================================

def index_news_item(item: NewsItem):
    """
    Format, embed, and upsert a single NewsItem into the Chroma vectorstore.
    """
    if not item or not item.id:
        logger.warning("Attempted to index an empty or unsaved news item.")
        return
        
    logger.info(f"Indexing NewsItem ID {item.id} ('{item.title}') in Chroma...")
    try:
        db = get_vectorstore(collection_name="news_items")
        
        # Format semantic text content
        content = (
            f"Category: {item.category}\n"
            f"Title: {item.title}\n"
            f"Summary: {item.summary or ''}\n"
            f"Content: {item.raw_content or ''}"
        )
        
        doc = Document(
            page_content=content[:8000],  # Clamp length to keep vector size reasonable
            metadata={
                "news_item_id": item.id,
                "title": item.title,
                "category": item.category
            }
        )
        
        doc_id = str(item.id)
        try:
            db.delete(ids=[doc_id])
        except Exception:
            pass
            
        db.add_documents(documents=[doc], ids=[doc_id])
        logger.info(f"Successfully indexed NewsItem '{item.title}' in Chroma.")
    except Exception as e:
        logger.error(f"Failed to index NewsItem ID {item.id} in Chroma: {str(e)}")

def index_all_news_items(session: Session):
    """
    Batch index all SQLite NewsItem records into the Chroma DB.
    """
    logger.info("Starting batch news vector synchronization...")
    try:
        statement = select(NewsItem).where(NewsItem.summary != None)
        items = session.exec(statement).all()
        
        if not items:
            logger.info("No summarized news items found in SQLite to index.")
            return
            
        logger.info(f"Indexing {len(items)} news items...")
        for item in items:
            index_news_item(item)
            
        logger.info("Batch news vector synchronization complete.")
    except Exception as e:
        logger.error(f"Failed during batch news vector synchronization: {str(e)}")

def get_related_articles(session: Session, item_id: int, limit: int = 3, threshold: float = 0.3) -> List[NewsItem]:
    """
    Find semantically similar NewsItems given an item ID.
    Excludes the article itself from recommendations.
    """
    logger.info(f"Fetching related articles for NewsItem ID: {item_id}...")
    try:
        # Fetch target article
        statement = select(NewsItem).where(NewsItem.id == item_id)
        target_item = session.exec(statement).first()
        if not target_item or not target_item.summary:
            logger.warning(f"Target news item {item_id} not found or has no summary context.")
            return []
            
        db = get_vectorstore(collection_name="news_items")
        
        # Query Chroma using the target item's summary
        query_text = f"Title: {target_item.title}\nSummary: {target_item.summary}"
        # Request k=limit+1 because the article itself will likely match first
        raw_results = db.similarity_search_with_score(query_text, k=limit + 1)
        
        valid_ids = []
        scores_map = {}
        
        for doc, distance in raw_results:
            news_id = doc.metadata.get("news_item_id")
            
            # Exclude current item
            if news_id == item_id:
                continue
                
            similarity = 1.0 - distance
            logger.info(f"Related match: ID {news_id} '{doc.metadata.get('title')}' (Similarity: {similarity:.4f})")
            
            if similarity >= threshold:
                valid_ids.append(news_id)
                scores_map[news_id] = similarity
                
        if not valid_ids:
            logger.info("No related articles passed the similarity threshold.")
            return []
            
        related_statement = select(NewsItem).where(NewsItem.id.in_(valid_ids[:limit]))
        related_items = session.exec(related_statement).all()
        related_items.sort(key=lambda x: scores_map.get(x.id, 0.0), reverse=True)
        return related_items
    except Exception as e:
        logger.error(f"Error while fetching related articles: {str(e)}")
        return []

def semantic_search_news(session: Session, query: str, limit: int = 3, threshold: float = 0.3) -> List[NewsItem]:
    """
    Search Chroma DB semantically for matching news items.
    """
    logger.info(f"Running semantic news search for query: '{query}'")
    try:
        db = get_vectorstore(collection_name="news_items")
        raw_results = db.similarity_search_with_score(query, k=limit)
        
        valid_ids = []
        scores_map = {}
        
        for doc, distance in raw_results:
            news_id = doc.metadata.get("news_item_id")
            similarity = 1.0 - distance
            
            logger.info(f"Semantic match (News): '{doc.metadata.get('title')}' (Similarity score: {similarity:.4f})")
            
            if similarity >= threshold:
                valid_ids.append(news_id)
                scores_map[news_id] = similarity
                
        if not valid_ids:
            return []
            
        statement = select(NewsItem).where(NewsItem.id.in_(valid_ids))
        db_items = session.exec(statement).all()
        db_items.sort(key=lambda x: scores_map.get(x.id, 0.0), reverse=True)
        return db_items
    except Exception as e:
        logger.error(f"Error during semantic news search: {str(e)}")
        return []
