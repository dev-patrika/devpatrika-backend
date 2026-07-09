import logging
import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import List
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from langchain_core.prompts import ChatPromptTemplate

from app.models.news import NewsItem
from app.models.wiki import WikiEntry
from app.services.processing.llm import get_llm
from app.services.processing.wiki_generator import generate_wiki_definition_async
from app.services.vectorstore.chroma_service import index_wiki_entry

logger = logging.getLogger("dev-patrika.wiki_curator.pipeline")

# =====================================================================
# Pydantic Schema for Structured Term Extraction
# =====================================================================

class ExtractedTerms(BaseModel):
    terms: List[str] = Field(
        description="A list of 3-5 technical frameworks, tools, libraries, APIs, protocols, or concepts extracted from the news items."
    )

# =====================================================================
# Wiki Curator Tasks
# =====================================================================

async def extract_terms_from_news_async(session: Session, hours: int = 24) -> List[str]:
    """
    Query recently processed news items and ask the LLM to extract 
    notable developer-centric technical terms, libraries, or concepts.
    """
    logger.info(f"Extracting technical terms from news processed in the last {hours} hours...")
    
    # 1. Fetch news items
    since_time = datetime.utcnow() - timedelta(hours=hours)
    statement = select(NewsItem).where(NewsItem.created_at >= since_time).where(NewsItem.summary != None)
    news_items = session.exec(statement).all()
    
    if not news_items:
        logger.info("No recently summarized news items found to extract terms from.")
        return []
        
    # 2. Compile news metadata text block
    content_block = []
    for item in news_items:
        content_block.append(f"Title: {item.title}\nCategory: {item.category}\nSummary:\n{item.summary}\n---")
    
    compiled_text = "\n\n".join(content_block)
    
    # 3. Call LLM to extract terms
    try:
        llm = get_llm(temperature=0.1)
        structured_llm = llm.with_structured_output(ExtractedTerms)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a technology trend spotter. Analyze the compiled list of developer news summaries "
                "and identify 3-5 key technical terms, libraries, frameworks, open-source projects, APIs, or "
                "concepts that are trending, new, or important for developers to understand. "
                "Exclude common, generic terms like 'Python', 'AI', 'JavaScript', or 'Git' unless they refer "
                "to a highly specific new release or project (e.g. 'Python 3.13' or 'Git LFS')."
            )),
            ("human", (
                "News Summaries:\n\n{text}\n\n"
                "Please extract the technical terms and return them in the structured format."
            ))
        ])
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"text": compiled_text[:12000]}) # Guard against token overflow
        
        if result and result.terms:
            extracted_list = [term.strip() for term in result.terms if term.strip()]
            logger.info(f"LLM extracted technical terms: {extracted_list}")
            return extracted_list
        return []
    except Exception as e:
        logger.error(f"Failed to extract terms from news items: {str(e)}")
        return []

def curate_wiki_from_news(session: Session, hours: int = 24) -> dict:
    """
    Automate the Dev Wiki curation workflow:
    1. Extract new terms from recent news.
    2. Check database presence (case-insensitive).
    3. Generate wiki entries for missing terms asynchronously.
    4. Index the new wiki entries in Chroma vector database.
    """
    logger.info("Starting automated Dev Wiki curation cycle...")
    stats = {"terms_extracted": 0, "entries_created": 0, "entries_indexed": 0, "errors": 0}
    
    # Step 1: Extract terms asynchronously using asyncio.run
    terms = asyncio.run(extract_terms_from_news_async(session, hours=hours))
    stats["terms_extracted"] = len(terms)
    
    if not terms:
        logger.info("No technical terms to process. Curation cycle complete.")
        return stats
        
    # Step 2: Query existing wiki terms to check duplicate records
    existing_wiki = session.exec(select(WikiEntry)).all()
    existing_terms_lower = {entry.term.lower() for entry in existing_wiki}
    
    # Find which terms are actually missing
    missing_terms = [t for t in terms if t.lower() not in existing_terms_lower]
    
    # Handle duplicate wiki entries (just re-index)
    for term in terms:
        if term.lower() in existing_terms_lower:
            logger.info(f"Wiki entry for '{term}' already exists in database. Re-indexing.")
            existing_entry = next((e for e in existing_wiki if e.term.lower() == term.lower()), None)
            if existing_entry:
                index_wiki_entry(existing_entry)
                stats["entries_indexed"] += 1

    if missing_terms:
        logger.info(f"Missing {len(missing_terms)} terms from Wiki. Generating concurrently...")
        
        async def generate_definitions(terms_list):
            tasks = [generate_wiki_definition_async(term) for term in terms_list]
            return await asyncio.gather(*tasks, return_exceptions=True)
            
        results = asyncio.run(generate_definitions(missing_terms))
        
        # Save generated terms to database and index in Chroma
        for term, result in zip(missing_terms, results):
            if isinstance(result, Exception) or result is None:
                logger.error(f"Failed to automatically curate WikiEntry for term '{term}'")
                stats["errors"] += 1
                continue
                
            try:
                # Double check to prevent race conditions
                statement = select(WikiEntry).where(WikiEntry.term == term)
                existing_entry = session.exec(statement).first()
                
                if existing_entry:
                    existing_entry.definition = result.definition
                    existing_entry.why_trending = result.why_trending
                    existing_entry.set_links(result.related_links)
                    existing_entry.updated_at = datetime.utcnow()
                    entry = existing_entry
                    logger.info(f"Updated existing WikiEntry for '{term}'")
                else:
                    entry = WikiEntry(
                        term=term,
                        definition=result.definition,
                        why_trending=result.why_trending,
                        related_links=json.dumps(result.related_links),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    logger.info(f"Created new WikiEntry for '{term}'")
                
                session.add(entry)
                session.commit()
                session.refresh(entry)
                
                stats["entries_created"] += 1
                # Index in Chroma DB
                index_wiki_entry(entry)
                stats["entries_indexed"] += 1
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save wiki entry for '{term}': {str(e)}")
                stats["errors"] += 1
                
    logger.info(f"Automated Wiki Curation cycle finished. Stats: {stats}")
    return stats
