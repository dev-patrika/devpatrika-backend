import os
import logging
from typing import List, Tuple
from sqlmodel import Session, select
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from app.config import settings
from app.models.wiki import WikiEntry

logger = logging.getLogger("dev-patrika.vectorstore.chroma")

# Path to persistent Chroma DB storage
CHROMA_PERSIST_DIR = os.path.abspath("chroma_db")

# Initialize Embeddings model (Google text-embedding-004)
# Sync API keys to environment if needed
if settings.GEMINI_API_KEY and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
)

def get_vectorstore() -> Chroma:
    """Initialize or load the local persistent Chroma vector database."""
    # Ensure directory exists
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    
    return Chroma(
        collection_name="wiki_entries",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

def index_wiki_entry(entry: WikiEntry):
    """
    Format, embed, and upsert a single WikiEntry into the Chroma vectorstore.
    """
    if not entry or not entry.id:
        logger.warning("Attempted to index an empty or unsaved wiki entry.")
        return
        
    logger.info(f"Indexing WikiEntry ID {entry.id} ('{entry.term}') in Chroma...")
    try:
        db = get_vectorstore()
        
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
    logger.info("Starting batch vector synchronization...")
    try:
        statement = select(WikiEntry)
        entries = session.exec(statement).all()
        
        if not entries:
            logger.info("No wiki entries found in SQLite to index.")
            return
            
        logger.info(f"Indexing {len(entries)} wiki entries...")
        for entry in entries:
            index_wiki_entry(entry)
            
        logger.info("Batch vector synchronization complete.")
    except Exception as e:
        logger.error(f"Failed during batch vector synchronization: {str(e)}")

def semantic_search_wiki(session: Session, query: str, limit: int = 3, threshold: float = 0.3) -> List[WikiEntry]:
    """
    Search Chroma DB semantically using cosine distance.
    Returns SQLite matching WikiEntry database records sorted by relevance.
    """
    logger.info(f"Running semantic wiki search for query: '{query}'")
    results_list = []
    
    try:
        db = get_vectorstore()
        
        # similarity_search_with_score returns Tuple[Document, float] where float is distance (L2 or cosine).
        # Chroma default is Cosine distance (0.0 is exact match, 1.0 is orthogonal).
        # Higher similarity means lower distance.
        raw_results = db.similarity_search_with_score(query, k=limit)
        
        # Filter by threshold (relaxed to 0.5-0.6 distance, meaning similarity >= 1 - distance)
        # Cosine similarity = 1 - distance. If we want similarity >= 0.5, then distance <= 0.5.
        valid_ids = []
        scores_map = {}
        
        for doc, distance in raw_results:
            wiki_id = doc.metadata.get("wiki_entry_id")
            # Calculate cosine similarity score (1 - distance)
            similarity = 1.0 - distance
            
            logger.info(f"Semantic match: '{doc.metadata.get('term')}' (Similarity score: {similarity:.4f}, Cosine distance: {distance:.4f})")
            
            # Use threshold filter
            if similarity >= threshold:
                valid_ids.append(wiki_id)
                scores_map[wiki_id] = similarity
                
        if not valid_ids:
            logger.info("No semantic matches passed the similarity threshold.")
            return []
            
        # Fetch actual DB objects
        statement = select(WikiEntry).where(WikiEntry.id.in_(valid_ids))
        db_entries = session.exec(statement).all()
        
        # Sort database entries by the similarity scores from Chroma (highest similarity first)
        db_entries.sort(key=lambda x: scores_map.get(x.id, 0.0), reverse=True)
        return db_entries
    except Exception as e:
        logger.error(f"Error during semantic wiki search: {str(e)}")
        return []
