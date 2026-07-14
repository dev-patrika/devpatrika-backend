import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, TypedDict, Set
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from app.models.news import NewsItem
from app.models.wiki import WikiEntry
from app.services.processing.llm import get_llm
from app.services.processing.wiki_generator import generate_wiki_definition_async
from app.services.vectorstore.vector_service import index_wiki_entry, semantic_search_wiki

logger = logging.getLogger("dev-patrika.agents.wiki-curator")

# =====================================================================
# Pydantic Schemas for Structured LLM Outputs
# =====================================================================

class ExtractedTerms(BaseModel):
    terms: List[str] = Field(
        description="A list of 3-5 technical frameworks, tools, libraries, APIs, protocols, or concepts extracted from the news items."
    )

class WikiMergeResult(BaseModel):
    definition: str = Field(
        description="The combined, updated definition of the technical term integrating both the existing definition and the new trends/updates. Keep it clear, concise, and developer-oriented."
    )
    why_trending: str = Field(
        description="The updated explanation of why this term is trending now, merging old reasons and new news events."
    )
    related_links: List[str] = Field(
        description="A list of consolidated, unique URLs for reference."
    )

# =====================================================================
# State Definition
# =====================================================================

class WikiCuratorAgentState(TypedDict):
    session: Session
    hours: int
    extracted_terms: List[str]
    missing_terms: List[str]
    conflicting_terms: List[Dict[str, Any]]  # list of {"term": str, "existing_entry": WikiEntry}
    entries_created: int
    entries_merged: int
    entries_indexed: int
    errors: int

# =====================================================================
# Graph Node Functions
# =====================================================================

def extract_terms_node(state: WikiCuratorAgentState) -> dict:
    """Node: Extract technical terms from recent news using LLM."""
    session = state["session"]
    hours = state.get("hours", 24)
    logger.info(f"Extracting technical terms from news processed in the last {hours} hours...")
    
    since_time = datetime.utcnow() - timedelta(hours=hours)
    statement = select(NewsItem).where(NewsItem.created_at >= since_time).where(NewsItem.summary != None)
    news_items = session.exec(statement).all()
    
    if not news_items:
        logger.info("No recently summarized news items found.")
        return {"extracted_terms": []}
    
    content_block = [f"Title: {item.title}\nCategory: {item.category}\nSummary:\n{item.summary}\n---" for item in news_items]
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
            ("human", "News Summaries:\n\n{text}\n\nPlease extract the technical terms and return them.")
        ])
        
        chain = prompt | structured_llm
        result = chain.invoke({"text": compiled_text[:12000]})
        
        terms = [t.strip() for t in result.terms if t.strip()] if result and result.terms else []
        logger.info(f"Extracted terms: {terms}")
        return {"extracted_terms": terms}
    except Exception as e:
        logger.error(f"Failed to extract terms: {str(e)}")
        return {"extracted_terms": [], "errors": state.get("errors", 0) + 1}

def check_conflicts_node(state: WikiCuratorAgentState) -> dict:
    """Node: Check database for name and semantic conflicts/overlaps of extracted terms."""
    session = state["session"]
    extracted_terms = state.get("extracted_terms", [])
    
    missing_terms = []
    conflicting_terms = []
    
    for term in extracted_terms:
        # 1. Exact Match Check (case-insensitive)
        stmt = select(WikiEntry).where(WikiEntry.term == term)
        existing = session.exec(stmt).first()
        
        if existing:
            logger.info(f"Conflict found: Exact name match for term '{term}'. Adding to conflict merge list.")
            conflicting_terms.append({"term": term, "existing_entry": existing})
            continue
            
        # 2. Semantic Overlap Check (similarity threshold 0.8)
        semantic_matches = semantic_search_wiki(session, query=term, limit=1, threshold=0.8)
        if semantic_matches:
            match = semantic_matches[0]
            logger.info(f"Conflict found: Semantic match for term '{term}' overlapping with existing term '{match.term}'. Adding to conflict merge list.")
            conflicting_terms.append({"term": term, "existing_entry": match})
            continue
            
        logger.info(f"No conflicts detected for term '{term}'. Adding to new terms list.")
        missing_terms.append(term)
        
    return {
        "missing_terms": missing_terms,
        "conflicting_terms": conflicting_terms
    }

def generate_definitions_node(state: WikiCuratorAgentState) -> dict:
    """Node: Generate wiki definitions for missing terms asynchronously."""
    session = state["session"]
    missing_terms = state.get("missing_terms", [])
    created = 0
    indexed = 0
    errors = 0
    
    if not missing_terms:
        return {"entries_created": 0}
        
    logger.info(f"Generating definitions for {len(missing_terms)} new wiki terms...")
    
    async def run_gens():
        tasks = [generate_wiki_definition_async(term) for term in missing_terms]
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    results = asyncio.run(run_gens())
    
    for term, result in zip(missing_terms, results):
        if isinstance(result, Exception) or result is None:
            logger.error(f"Failed to generate definition for term '{term}'")
            errors += 1
            continue
            
        try:
            entry = WikiEntry(
                term=term,
                definition=result.definition,
                why_trending=result.why_trending,
                related_links=result.related_links or [],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            created += 1
            
            # Index inside pgvector
            index_wiki_entry(entry)
            indexed += 1
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save wiki entry for '{term}': {str(e)}")
            errors += 1
            
    return {
        "entries_created": created,
        "entries_indexed": state.get("entries_indexed", 0) + indexed,
        "errors": state.get("errors", 0) + errors
    }

def merge_definitions_node(state: WikiCuratorAgentState) -> dict:
    """Node: Resolve term conflicts by merging old definition and new trends using LLM."""
    session = state["session"]
    conflicting_terms = state.get("conflicting_terms", [])
    merged = 0
    indexed = 0
    errors = 0
    
    if not conflicting_terms:
        return {"entries_merged": 0}
        
    logger.info(f"Merging and resolving conflicts for {len(conflicting_terms)} terms...")
    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(WikiMergeResult)
    
    for item in conflicting_terms:
        term = item["term"]
        existing_entry = item["existing_entry"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a technical editor. Your task is to merge and resolve overlapping concepts in a glossary. "
                "You are given an existing glossary entry and a new trending term context. "
                "Integrate any new technical context, features, or updates from the new trend into the existing definition. "
                "Keep the final output clean, precise, professional, and developer-centric. "
                "Consolidate and deduplicate any reference links.\n\n"
                "CRITICAL: Write in extremely simple, direct, and straightforward English. Use plain vocabulary and "
                "simple sentence structures. Do NOT use flowery, verbose, or poetic academic prose. Keep definitions "
                "clear and easy to understand."
            )),
            ("human", (
                "Existing Term: {existing_term}\n"
                "Existing Definition:\n{existing_definition}\n"
                "Existing Why Trending:\n{existing_why_trending}\n"
                "Existing Links: {existing_links}\n\n"
                "New Overlapping Term/Context: {new_term}\n\n"
                "Please output a merged and updated definition, why_trending explanation, and related links."
            ))
        ])
        
        try:
            chain = prompt | structured_llm
            result = chain.invoke({
                "existing_term": existing_entry.term,
                "existing_definition": existing_entry.definition,
                "existing_why_trending": existing_entry.why_trending,
                "existing_links": existing_entry.related_links or [],
                "new_term": term
            })
            
            if result:
                existing_entry.definition = result.definition
                existing_entry.why_trending = result.why_trending
                # Merge and unique-ify links
                existing_entry.related_links = list(set((existing_entry.related_links or []) + result.related_links))
                existing_entry.updated_at = datetime.utcnow()
                
                session.add(existing_entry)
                session.commit()
                session.refresh(existing_entry)
                merged += 1
                
                # Re-index inside pgvector
                index_wiki_entry(existing_entry)
                indexed += 1
                logger.info(f"Successfully resolved conflict and merged term '{existing_entry.term}'.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to merge and save entry for '{existing_entry.term}': {str(e)}")
            errors += 1
            
    return {
        "entries_merged": merged,
        "entries_indexed": state.get("entries_indexed", 0) + indexed,
        "errors": state.get("errors", 0) + errors
    }

# =====================================================================
# Conditional Edge Functions
# =====================================================================

def route_after_conflicts(state: WikiCuratorAgentState) -> List[str]:
    """Route after checking conflicts. Executes both merging and definition generation if needed."""
    routes = []
    if len(state.get("missing_terms", [])) > 0:
        routes.append("generate_definitions")
    if len(state.get("conflicting_terms", [])) > 0:
        routes.append("merge_definitions")
    return routes if routes else [END]

# =====================================================================
# Build and Compile Graph
# =====================================================================

def build_wiki_curator_graph() -> StateGraph:
    """Constructs the conflict-resolving Wiki Curator Agent graph."""
    graph = StateGraph(WikiCuratorAgentState)
    
    # Add Nodes
    graph.add_node("extract_terms", extract_terms_node)
    graph.add_node("check_conflicts", check_conflicts_node)
    graph.add_node("generate_definitions", generate_definitions_node)
    graph.add_node("merge_definitions", merge_definitions_node)
    
    # Define Edges
    graph.add_edge(START, "extract_terms")
    graph.add_edge("extract_terms", "check_conflicts")
    
    # Conditional branching to run merging and generating in parallel/conditional streams
    graph.add_conditional_edges("check_conflicts", route_after_conflicts, {
        "generate_definitions": "generate_definitions",
        "merge_definitions": "merge_definitions",
        END: END
    })
    
    graph.add_edge("generate_definitions", END)
    graph.add_edge("merge_definitions", END)
    
    return graph.compile()

# Compile graph singleton
wiki_curator_agent = build_wiki_curator_graph()

def run_wiki_curator_agent(session: Session, hours: int = 24) -> Dict[str, Any]:
    """
    Public API to invoke the Wiki Curator Agent.
    """
    logger.info("Invoking stateful Wiki Curator Agent...")
    initial_state = {
        "session": session,
        "hours": hours,
        "extracted_terms": [],
        "missing_terms": [],
        "conflicting_terms": [],
        "entries_created": 0,
        "entries_merged": 0,
        "entries_indexed": 0,
        "errors": 0
    }
    
    final_state = wiki_curator_agent.invoke(initial_state)
    logger.info("Wiki Curator Agent workflow finished execution.")
    return {
        "terms_extracted": len(final_state.get("extracted_terms", [])),
        "entries_created": final_state.get("entries_created", 0),
        "entries_merged": final_state.get("entries_merged", 0),
        "entries_indexed": final_state.get("entries_indexed", 0),
        "errors": final_state.get("errors", 0)
    }
