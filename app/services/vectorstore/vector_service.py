import os
import time
import logging
import requests
from typing import List, Tuple
from sqlmodel import Session, select
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from app.config import settings
from app.models.wiki import WikiEntry
from app.models.news import NewsItem

logger = logging.getLogger("dev-patrika.vectorstore.vector")

# Initialize Embeddings model (Hugging Face BAAI/bge-small-en-v1.5)
class HuggingFaceInferenceAPIEmbeddings(Embeddings):
    """Custom Embeddings class using Hugging Face's Cloud Inference API."""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _call_api(self, inputs: List[str]) -> List[List[float]]:
        # Retry mechanism in case model is loading (503 Service Unavailable)
        retries = 5
        backoff = 5
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": inputs, "options": {"wait_for_model": True}},
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    logger.warning(f"Hugging Face model is loading (attempt {attempt + 1}/{retries}). Sleeping {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    err_msg = f"Hugging Face API error {response.status_code}: {response.text}"
                    logger.error(err_msg)
                    raise ValueError(err_msg)
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                time.sleep(backoff)
                backoff *= 2
        raise TimeoutError("Hugging Face model failed to load in time.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._call_api(texts)

    def embed_query(self, text: str) -> List[float]:
        res = self._call_api([text])
        return res[0]

embeddings = HuggingFaceInferenceAPIEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    api_key=settings.HUGGINGFACE_API_KEY or os.environ.get("HUGGINGFACE_API_KEY", "")
)

# Convert connection string from psycopg2 format to standard postgresql format for psycopg3 (langchain-postgres)
connection_string = settings.DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

def get_vectorstore(collection_name: str = "wiki_entries") -> PGVector:
    """Initialize or load the Neon pgvector store for a specific collection."""
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True
    )

# =====================================================================
# Wiki Entries Vector Operations
# =====================================================================

def index_wiki_entry(entry: WikiEntry):
    """
    Format, embed, and upsert a single WikiEntry into the pgvector store.
    """
    if not entry or not entry.id:
        logger.warning("Attempted to index an empty or unsaved wiki entry.")
        return
        
    logger.info(f"Indexing WikiEntry ID {entry.id} ('{entry.term}') in pgvector...")
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
        
        doc_id = str(entry.id)
        try:
            db.delete(ids=[doc_id])
        except Exception:
            pass
            
        db.add_documents(documents=[doc], ids=[doc_id])
        logger.info(f"Successfully indexed WikiEntry '{entry.term}' in pgvector.")
    except Exception as e:
        logger.error(f"Failed to index WikiEntry ID {entry.id} in pgvector: {str(e)}")
        raise e

def index_all_wiki_entries(session: Session, batch_size: int = 50):
    """
    Batch index all SQL WikiEntry records into pgvector (optimized for API limits with auto-retry).
    """
    logger.info("Starting batch wiki vector synchronization...")
    try:
        statement = select(WikiEntry)
        entries = session.exec(statement).all()
        
        if not entries:
            logger.info("No wiki entries found in database to index.")
            return
            
        logger.info(f"Indexing {len(entries)} wiki entries in batches of {batch_size}...")
        db = get_vectorstore(collection_name="wiki_entries")
        
        for idx in range(0, len(entries), batch_size):
            batch = entries[idx:idx + batch_size]
            documents = []
            doc_ids = []
            
            for entry in batch:
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
                documents.append(doc)
                doc_ids.append(str(entry.id))
            
            batch_num = idx//batch_size + 1
            logger.info(f"Uploading wiki batch {batch_num} (size {len(batch)})...")
            
            # Retry mechanism for robust API calls
            retries = 3
            backoff = 30
            for attempt in range(1, retries + 1):
                try:
                    # Delete old entries to prevent duplication
                    try:
                        db.delete(ids=doc_ids)
                    except Exception:
                        pass
                        
                    db.add_documents(documents=documents, ids=doc_ids)
                    logger.info(f"Successfully indexed wiki batch {batch_num}")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt}/{retries} failed for wiki batch {batch_num}: {e}")
                    if attempt == retries:
                        logger.error(f"Failed to index wiki batch {batch_num} after {retries} attempts.")
                        raise e
                    logger.info(f"Sleeping {backoff} seconds before next retry...")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
            
            time.sleep(5.0)  # Rate limit safety delay between batches
            
        logger.info("Batch wiki vector synchronization complete.")
    except Exception as e:
        logger.error(f"Failed during batch wiki vector synchronization: {str(e)}")

def semantic_search_wiki(session: Session, query: str, limit: int = 3, threshold: float = 0.3) -> List[WikiEntry]:
    """
    Search pgvector store semantically using cosine distance.
    Returns database matching WikiEntry records sorted by relevance.
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
    Format, embed, and upsert a single NewsItem into the pgvector store.
    """
    if not item or not item.id:
        logger.warning("Attempted to index an empty or unsaved news item.")
        return
        
    logger.info(f"Indexing NewsItem ID {item.id} ('{item.title}') in pgvector...")
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
        logger.info(f"Successfully indexed NewsItem '{item.title}' in pgvector.")
    except Exception as e:
        logger.error(f"Failed to index NewsItem ID {item.id} in pgvector: {str(e)}")
        raise e

def index_all_news_items(session: Session, batch_size: int = 50):
    """
    Batch index all SQL NewsItem records into pgvector (optimized for API limits with auto-retry).
    """
    logger.info("Starting batch news vector synchronization...")
    try:
        statement = select(NewsItem).where(NewsItem.summary != None)
        items = session.exec(statement).all()
        
        if not items:
            logger.info("No summarized news items found in database to index.")
            return
            
        logger.info(f"Indexing {len(items)} news items in batches of {batch_size}...")
        db = get_vectorstore(collection_name="news_items")
        
        for idx in range(0, len(items), batch_size):
            batch = items[idx:idx + batch_size]
            documents = []
            doc_ids = []
            
            for item in batch:
                content = (
                    f"Category: {item.category}\n"
                    f"Title: {item.title}\n"
                    f"Summary: {item.summary or ''}\n"
                    f"Content: {item.raw_content or ''}"
                )
                doc = Document(
                    page_content=content[:8000],
                    metadata={
                        "news_item_id": item.id,
                        "title": item.title,
                        "category": item.category
                    }
                )
                documents.append(doc)
                doc_ids.append(str(item.id))
                
            batch_num = idx//batch_size + 1
            logger.info(f"Uploading news batch {batch_num} (size {len(batch)})...")
            
            # Retry mechanism for robust API calls
            retries = 3
            backoff = 30
            for attempt in range(1, retries + 1):
                try:
                    # Delete old entries to prevent duplication
                    try:
                        db.delete(ids=doc_ids)
                    except Exception:
                        pass
                        
                    db.add_documents(documents=documents, ids=doc_ids)
                    logger.info(f"Successfully indexed news batch {batch_num}")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt}/{retries} failed for news batch {batch_num}: {e}")
                    if attempt == retries:
                        logger.error(f"Failed to index news batch {batch_num} after {retries} attempts.")
                        raise e
                    logger.info(f"Sleeping {backoff} seconds before next retry...")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
            
            time.sleep(8.0)  # Rate limit safety delay between batches (slightly longer to protect Gemini limits)
            
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
        
        # Query pgvector using the target item's summary
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
    Search pgvector store semantically for matching news items.
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
