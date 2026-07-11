"""
Dev Patrika — Wiki Curator Pipeline (LangGraph Refactor)

Converts the old wiki curator pipeline.py into a LangGraph StateGraph.

Graph Structure:
    START → extract_terms → filter_existing
        → [has_missing] → generate_definitions → save_and_index → END
        → [no_missing] → END
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, TypedDict
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from app.models.news import NewsItem
from app.models.wiki import WikiEntry
from app.services.processing.llm import get_llm
from app.services.processing.wiki_generator import generate_wiki_definition_async
from app.services.vectorstore.vector_service import index_wiki_entry

logger = logging.getLogger("dev-patrika.wiki_curator.pipeline")

# =====================================================================
# Pydantic Schema for Structured Term Extraction (unchanged)
# =====================================================================

class ExtractedTerms(BaseModel):
    terms: List[str] = Field(
        description="A list of 3-5 technical frameworks, tools, libraries, APIs, protocols, or concepts extracted from the news items."
    )

# =====================================================================
# LangGraph State Definition
# =====================================================================

class WikiCuratorState(TypedDict):
    """Shared state flowing through the wiki curation graph."""
    session: Session
    hours: int
    extracted_terms: list          # List[str]
    existing_terms_lower: set     # Set of lowercase existing wiki terms
    missing_terms: list           # List[str] — terms not yet in wiki
    existing_wiki: list           # List[WikiEntry]
    terms_extracted: int
    entries_created: int
    entries_indexed: int
    errors: int

# =====================================================================
# LLM Term Extraction (unchanged)
# =====================================================================

async def extract_terms_from_news_async(session: Session, hours: int = 24) -> List[str]:
    """
    Query recently processed news items and ask the LLM to extract 
    notable developer-centric technical terms, libraries, or concepts.
    """
    logger.info(f"Extracting technical terms from news processed in the last {hours} hours...")
    
    since_time = datetime.utcnow() - timedelta(hours=hours)
    statement = select(NewsItem).where(NewsItem.created_at >= since_time).where(NewsItem.summary != None)
    news_items = session.exec(statement).all()
    
    if not news_items:
        logger.info("No recently summarized news items found to extract terms from.")
        return []
    
    content_block = []
    for item in news_items:
        content_block.append(f"Title: {item.title}\nCategory: {item.category}\nSummary:\n{item.summary}\n---")
    
    compiled_text = "\n\n".join(content_block)
    
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
        result = await chain.ainvoke({"text": compiled_text[:12000]})
        
        if result and result.terms:
            extracted_list = [term.strip() for term in result.terms if term.strip()]
            logger.info(f"LLM extracted technical terms: {extracted_list}")
            return extracted_list
        return []
    except Exception as e:
        logger.error(f"Failed to extract terms from news items: {str(e)}")
        return []

# =====================================================================
# Graph Node Functions
# =====================================================================

def extract_terms_node(state: WikiCuratorState) -> dict:
    """Node: Extract technical terms from recent news using LLM."""
    session = state["session"]
    hours = state.get("hours", 24)
    
    terms = asyncio.run(extract_terms_from_news_async(session, hours=hours))
    
    return {
        "extracted_terms": terms,
        "terms_extracted": len(terms),
    }

def filter_existing_node(state: WikiCuratorState) -> dict:
    """Node: Check which extracted terms already exist in the wiki database."""
    session = state["session"]
    extracted_terms = state.get("extracted_terms", [])
    
    existing_wiki = session.exec(select(WikiEntry)).all()
    existing_terms_lower = {entry.term.lower() for entry in existing_wiki}
    
    missing_terms = [t for t in extracted_terms if t.lower() not in existing_terms_lower]
    
    # Re-index existing entries that were extracted (ensures vector store is up-to-date)
    indexed = 0
    for term in extracted_terms:
        if term.lower() in existing_terms_lower:
            logger.info(f"Wiki entry for '{term}' already exists. Re-indexing.")
            existing_entry = next((e for e in existing_wiki if e.term.lower() == term.lower()), None)
            if existing_entry:
                try:
                    index_wiki_entry(existing_entry)
                    indexed += 1
                except Exception as idx_err:
                    logger.warning(f"Failed to re-index existing WikiEntry '{term}': {idx_err}")
    
    logger.info(f"Missing {len(missing_terms)} terms from Wiki. Already indexed {indexed} existing entries.")
    
    return {
        "existing_wiki": list(existing_wiki),
        "existing_terms_lower": existing_terms_lower,
        "missing_terms": missing_terms,
        "entries_indexed": indexed,
    }

def generate_definitions_node(state: WikiCuratorState) -> dict:
    """Node: Generate wiki definitions for missing terms using LLM."""
    session = state["session"]
    missing_terms = state.get("missing_terms", [])
    
    if not missing_terms:
        return {"entries_created": 0}
    
    logger.info(f"Generating definitions for {len(missing_terms)} missing terms...")
    
    async def generate_definitions(terms_list):
        tasks = [generate_wiki_definition_async(term) for term in terms_list]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    results = asyncio.run(generate_definitions(missing_terms))
    
    created = 0
    errors = state.get("errors", 0)
    indexed = state.get("entries_indexed", 0)
    
    for term, result in zip(missing_terms, results):
        if isinstance(result, Exception) or result is None:
            logger.error(f"Failed to curate WikiEntry for term '{term}'")
            errors += 1
            continue
        
        try:
            # Double-check for race conditions
            statement = select(WikiEntry).where(WikiEntry.term == term)
            existing_entry = session.exec(statement).first()
            
            if existing_entry:
                existing_entry.definition = result.definition
                existing_entry.why_trending = result.why_trending
                existing_entry.related_links = result.related_links
                existing_entry.updated_at = datetime.utcnow()
                entry = existing_entry
                logger.info(f"Updated existing WikiEntry for '{term}'")
            else:
                entry = WikiEntry(
                    term=term,
                    definition=result.definition,
                    why_trending=result.why_trending,
                    related_links=result.related_links,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                logger.info(f"Created new WikiEntry for '{term}'")
            
            session.add(entry)
            session.commit()
            session.refresh(entry)
            
            created += 1
            
            # Index in pgvector
            index_wiki_entry(entry)
            indexed += 1
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save wiki entry for '{term}': {str(e)}")
            errors += 1
    
    return {
        "entries_created": created,
        "entries_indexed": indexed,
        "errors": errors,
    }

# =====================================================================
# Conditional Edge Functions
# =====================================================================

def route_after_filter(state: WikiCuratorState) -> str:
    """Route based on whether there are missing terms to generate."""
    terms = state.get("extracted_terms", [])
    missing = state.get("missing_terms", [])
    
    if not terms:
        return END
    if missing:
        return "generate_definitions"
    return END

# =====================================================================
# Build the LangGraph StateGraph
# =====================================================================

def build_wiki_curator_graph() -> StateGraph:
    """Construct and compile the wiki curator graph."""
    graph = StateGraph(WikiCuratorState)
    
    # Add nodes
    graph.add_node("extract_terms", extract_terms_node)
    graph.add_node("filter_existing", filter_existing_node)
    graph.add_node("generate_definitions", generate_definitions_node)
    
    # Add edges
    graph.add_edge(START, "extract_terms")
    graph.add_edge("extract_terms", "filter_existing")
    graph.add_conditional_edges("filter_existing", route_after_filter, {
        "generate_definitions": "generate_definitions",
        END: END,
    })
    graph.add_edge("generate_definitions", END)
    
    return graph.compile()

# =====================================================================
# Public API (backward compatible with scheduler.py)
# =====================================================================

_wiki_curator_graph = build_wiki_curator_graph()

def curate_wiki_from_news(session: Session, hours: int = 24) -> dict:
    """
    Public entry point — backward compatible with old pipeline API.
    Invokes the LangGraph wiki curation pipeline.
    """
    logger.info("Running LangGraph wiki curation pipeline...")
    
    initial_state = {
        "session": session,
        "hours": hours,
        "extracted_terms": [],
        "existing_wiki": [],
        "existing_terms_lower": set(),
        "missing_terms": [],
        "terms_extracted": 0,
        "entries_created": 0,
        "entries_indexed": 0,
        "errors": 0,
    }
    
    final_state = _wiki_curator_graph.invoke(initial_state)
    
    stats = {
        "terms_extracted": final_state.get("terms_extracted", 0),
        "entries_created": final_state.get("entries_created", 0),
        "entries_indexed": final_state.get("entries_indexed", 0),
        "errors": final_state.get("errors", 0),
    }
    
    logger.info(f"LangGraph wiki curation pipeline completed. Stats: {stats}")
    return stats
