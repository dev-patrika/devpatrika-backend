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

# Initialize Embeddings model (Hugging Face with Multi-Key Rotation & Cloud Model Fallback)
class HuggingFaceInferenceAPIEmbeddings(Embeddings):
    """Custom Embeddings class using Hugging Face Cloud API with automatic key rotation and model fallback."""
    
    def __init__(self, primary_model: str, api_keys: List[str], fallback_models: List[str] = None):
        self.primary_model = primary_model
        self.fallback_models = fallback_models or [
            "BAAI/bge-small-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "thenlper/gte-small"
        ]
        # Clean and filter non-empty API keys
        self.api_keys = [k.strip() for k in api_keys if k and k.strip()]
        self.current_key_idx = 0

    def _call_api_with_fallback(self, inputs: List[str]) -> List[List[float]]:
        if not self.api_keys:
            logger.warning("No Hugging Face API keys configured. Embedding generation will fail.")
            raise ValueError("No Hugging Face API keys provided.")
            
        models_to_try = [self.primary_model] + [m for m in self.fallback_models if m != self.primary_model]
        last_exception = None

        for model in models_to_try:
            api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
            
            # Try all available API keys for this model
            for idx in range(len(self.api_keys)):
                key_index = (self.current_key_idx + idx) % len(self.api_keys)
                key_to_use = self.api_keys[key_index]
                headers = {"Authorization": f"Bearer {key_to_use}"}
                
                try:
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json={"inputs": inputs, "options": {"wait_for_model": True}},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        if isinstance(res_data, list):
                            # Success! Remember working key index
                            self.current_key_idx = key_index
                            return res_data
                            
                    elif response.status_code in (429, 403, 400, 402):
                        logger.warning(
                            f"Hugging Face Cloud API key limit/credits depleted (Status {response.status_code}) "
                            f"for model '{model}' with Key #{key_index + 1}. Rotating to next available API key..."
                        )
                        last_exception = ValueError(f"Cloud API limit reached ({response.status_code}): {response.text}")
                        continue  # Try next key
                        
                    elif response.status_code == 503:
                        logger.warning(f"Hugging Face model '{model}' is loading (503). Retrying...")
                        time.sleep(3)
                        continue
                    else:
                        logger.warning(f"Hugging Face API response {response.status_code}: {response.text}")
                        last_exception = ValueError(f"HF API Error {response.status_code}: {response.text}")
                except Exception as e:
                    logger.warning(f"Exception during HF embedding call for model '{model}': {e}")
                    last_exception = e
                    continue

        # If all cloud models & keys fail, raise the last exception
        raise last_exception or TimeoutError("All Hugging Face cloud embedding attempts failed.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._call_api_with_fallback(texts)

    def embed_query(self, text: str) -> List[float]:
        res = self._call_api_with_fallback([text])
        return res[0]

# Parse primary and fallback API keys from config/env (supports comma-separated string)
raw_keys = f"{settings.HUGGINGFACE_API_KEY},{getattr(settings, 'HUGGINGFACE_FALLBACK_API_KEYS', '')}"
hf_api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

embeddings = HuggingFaceInferenceAPIEmbeddings(
    primary_model="BAAI/bge-small-en-v1.5",
    api_keys=hf_api_keys or [os.environ.get("HUGGINGFACE_API_KEY", "")]
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
