"""
Dev Patrika — Processing Pipeline (LangGraph Refactor)

Converts the old sequential pipeline.py into a LangGraph StateGraph.

Graph Structure:
    START → fetch_pending → route_items
        → [has_news] → process_news_batch → save_news
        → [has_github] → process_github_batch → save_github
        → [no_items] → END
    save_news → check_github
        → [has_github] → process_github_batch → save_github
        → [no_github] → END
    save_github → END
"""

import logging
import asyncio
from typing import List, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from app.models.news import NewsItem
from app.models.github_radar import GitHubRadar
from app.core.constants import TechCategory
from app.services.processing.llm import get_llm

logger = logging.getLogger("dev-patrika.processing.pipeline")

# =====================================================================
# Pydantic Schemas for Structured Output (unchanged)
# =====================================================================

class NewsAnalysis(BaseModel):
    overview: str = Field(
        description="A concise 1-2 sentence overview explaining what this news/tool is and how it works."
    )
    key_points: List[str] = Field(
        description="3-4 high-value bullet-point takeaways summarizing key aspects, updates, or details of the article."
    )
    community: str = Field(
        description="A brief sentence summarizing the community interest, popularity, stars, developer response, or adoption of the tech."
    )
    category: TechCategory = Field(
        description="The most appropriate category this news belongs to."
    )

class GitHubAnalysis(BaseModel):
    overview: str = Field(
        description="A brief paragraph (1-2 sentences) explaining what this repository is, what problem it solves, and its core functionality."
    )
    key_architectural_details: List[str] = Field(
        description="2-3 key technical or architectural features/takeaways (e.g. built in Rust, uses zero-dependency, etc.)."
    )
    community_traction: str = Field(
        description="A short sentence summarizing its current traction, stars count context, or growth on GitHub."
    )

# =====================================================================
# Helper Formatting Functions (unchanged)
# =====================================================================

def format_news_markdown(analysis: NewsAnalysis) -> str:
    """Format structured NewsAnalysis output into clean markdown."""
    bullets = "\n".join(f"- {point}" for point in analysis.key_points)
    return (
        f"**Overview**\n"
        f"{analysis.overview}\n\n"
        f"**Key Details**\n"
        f"{bullets}\n\n"
        f"**Community & Traction**\n"
        f"{analysis.community}"
    )

def format_github_markdown(analysis: GitHubAnalysis) -> str:
    """Format structured GitHubAnalysis output into clean markdown."""
    bullets = "\n".join(f"- {point}" for point in analysis.key_architectural_details)
    return (
        f"**Overview**\n"
        f"{analysis.overview}\n\n"
        f"**Key Details**\n"
        f"{bullets}\n\n"
        f"**Community & Traction**\n"
        f"{analysis.community_traction}"
    )

# =====================================================================
# LLM Analysis Functions (unchanged)
# =====================================================================

async def analyze_news_item_async(title: str, raw_content: str, source: str) -> NewsAnalysis:
    """Use async LLM chain with structured output to analyze a news item."""
    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(NewsAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert developer news analyst. Analyze the following tech news story and extract "
                "a structured summary and category. Be precise and avoid developer jargon where possible."
            )),
            ("human", (
                "Source: {source}\n"
                "Title: {title}\n"
                "Raw Content/Abstract: {raw_content}\n\n"
                "Please analyze the item and return the structured JSON output."
            ))
        ])
        
        chain = prompt | structured_llm
        truncated_content = raw_content[:8000] if raw_content else "No content description available."
        
        result = await chain.ainvoke({
            "source": source,
            "title": title,
            "raw_content": truncated_content
        })
        return result
    except Exception as e:
        logger.error(f"Failed to analyze news item '{title}': {str(e)}")
        raise e

async def analyze_github_repo_async(repo_name: str, description: str) -> GitHubAnalysis:
    """Use async LLM chain with structured output to analyze a trending repository."""
    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(GitHubAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a technical radar system. Analyze the following GitHub repository metadata and "
                "generate a structured summary explaining what it does and why it matters."
            )),
            ("human", (
                "Repository Name: {repo_name}\n"
                "Description: {description}\n\n"
                "Please analyze and return the structured JSON output."
            ))
        ])
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({
            "repo_name": repo_name,
            "description": description or "No description provided."
        })
        return result
    except Exception as e:
        logger.error(f"Failed to analyze GitHub repository '{repo_name}': {str(e)}")
        raise e

# =====================================================================
# LangGraph State Definition
# =====================================================================

class PipelineState(TypedDict):
    """Shared state flowing through the processing graph."""
    session: Session
    pending_news: list          # List[NewsItem]
    pending_repos: list         # List[GitHubRadar]
    news_results: list          # List[Optional[NewsAnalysis]]
    github_results: list        # List[Optional[GitHubAnalysis]]
    processed_news: int
    processed_github: int
    errors: int

# =====================================================================
# Graph Node Functions
# =====================================================================

def fetch_pending_node(state: PipelineState) -> dict:
    """Node: Query database for items needing LLM processing."""
    session = state["session"]
    
    logger.info("Checking for pending news items to analyze...")
    news_stmt = select(NewsItem).where(NewsItem.summary == None)
    pending_news = session.exec(news_stmt).all()
    
    logger.info("Checking for pending GitHub trending repos to analyze...")
    github_stmt = select(GitHubRadar).where(GitHubRadar.why_it_matters_summary == None)
    pending_repos = session.exec(github_stmt).all()
    
    logger.info(f"Found {len(pending_news)} news items and {len(pending_repos)} GitHub repos to process.")
    
    return {
        "pending_news": list(pending_news),
        "pending_repos": list(pending_repos),
    }

def process_news_node(state: PipelineState) -> dict:
    """Node: Run async LLM analysis on all pending news items with batching and retry."""
    pending_news = state["pending_news"]
    
    if not pending_news:
        logger.info("No pending news items found.")
        return {"news_results": [], "processed_news": 0}
    
    logger.info(f"Processing {len(pending_news)} news items asynchronously...")
    
    async def process_news_batch(items):
        chunk_size = 5
        results = []
        i = 0
        while i < len(items):
            chunk = items[i:i + chunk_size]
            logger.info(f"Processing news batch {i//chunk_size + 1}/{((len(items)-1)//chunk_size)+1} (size {len(chunk)})...")
            
            retry_count = 0
            max_retries = 1
            batch_success = False
            
            while retry_count <= max_retries and not batch_success:
                tasks = [analyze_news_item_async(item.title, item.raw_content, item.source) for item in chunk]
                try:
                    chunk_results = await asyncio.gather(*tasks)
                    results.extend(chunk_results)
                    batch_success = True
                    i += chunk_size
                    if i < len(items):
                        await asyncio.sleep(2)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                        retry_count += 1
                        if retry_count <= max_retries:
                            logger.warning(f"Rate limit hit: {e}. Sleeping 120s before retry {retry_count}/{max_retries}...")
                            await asyncio.sleep(120)
                        else:
                            logger.error(f"Rate limit exceeded after {max_retries} retries. Skipping batch.")
                            results.extend([None] * len(chunk))
                            i += chunk_size
                            batch_success = True
                    else:
                        logger.error(f"Non-rate-limit error: {e}. Skipping batch.")
                        results.extend([None] * len(chunk))
                        i += chunk_size
                        batch_success = True
        return results
    
    results = asyncio.run(process_news_batch(pending_news))
    return {"news_results": results}

def save_news_node(state: PipelineState) -> dict:
    """Node: Persist news analysis results to database."""
    session = state["session"]
    pending_news = state["pending_news"]
    news_results = state["news_results"]
    
    processed = 0
    errors = 0
    
    for item, analysis in zip(pending_news, news_results):
        if isinstance(analysis, Exception) or analysis is None:
            errors += 1
            continue
        
        item.summary = format_news_markdown(analysis)
        item.category = analysis.category
        session.add(item)
        processed += 1
    
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to commit news summaries: {str(e)}")
    
    logger.info(f"Saved {processed} news summaries ({errors} errors).")
    return {"processed_news": processed, "errors": state.get("errors", 0) + errors}

def process_github_node(state: PipelineState) -> dict:
    """Node: Run async LLM analysis on all pending GitHub repos with batching and retry."""
    pending_repos = state["pending_repos"]
    
    if not pending_repos:
        logger.info("No pending GitHub repos found.")
        return {"github_results": [], "processed_github": 0}
    
    logger.info(f"Processing {len(pending_repos)} GitHub repositories asynchronously...")
    
    async def process_repo_batch(repos):
        chunk_size = 5
        results = []
        i = 0
        while i < len(repos):
            chunk = repos[i:i + chunk_size]
            logger.info(f"Processing GitHub batch {i//chunk_size + 1}/{((len(repos)-1)//chunk_size)+1} (size {len(chunk)})...")
            
            retry_count = 0
            max_retries = 1
            batch_success = False
            
            while retry_count <= max_retries and not batch_success:
                tasks = [analyze_github_repo_async(repo.repo_name, repo.description) for repo in chunk]
                try:
                    chunk_results = await asyncio.gather(*tasks)
                    results.extend(chunk_results)
                    batch_success = True
                    i += chunk_size
                    if i < len(repos):
                        await asyncio.sleep(2)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                        retry_count += 1
                        if retry_count <= max_retries:
                            logger.warning(f"Rate limit hit: {e}. Sleeping 120s before retry {retry_count}/{max_retries}...")
                            await asyncio.sleep(120)
                        else:
                            logger.error(f"Rate limit exceeded after {max_retries} retries. Skipping batch.")
                            results.extend([None] * len(chunk))
                            i += chunk_size
                            batch_success = True
                    else:
                        logger.error(f"Non-rate-limit error: {e}. Skipping batch.")
                        results.extend([None] * len(chunk))
                        i += chunk_size
                        batch_success = True
        return results
    
    results = asyncio.run(process_repo_batch(pending_repos))
    return {"github_results": results}

def save_github_node(state: PipelineState) -> dict:
    """Node: Persist GitHub analysis results to database."""
    session = state["session"]
    pending_repos = state["pending_repos"]
    github_results = state["github_results"]
    
    processed = 0
    errors = 0
    
    for repo, analysis in zip(pending_repos, github_results):
        if isinstance(analysis, Exception) or analysis is None:
            errors += 1
            continue
        
        repo.why_it_matters_summary = format_github_markdown(analysis)
        session.add(repo)
        processed += 1
    
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to commit GitHub summaries: {str(e)}")
    
    logger.info(f"Saved {processed} GitHub summaries ({errors} errors).")
    return {"processed_github": processed, "errors": state.get("errors", 0) + errors}

# =====================================================================
# Conditional Edge Functions
# =====================================================================

def route_after_fetch(state: PipelineState) -> str:
    """Route based on what pending items exist."""
    has_news = len(state.get("pending_news", [])) > 0
    has_github = len(state.get("pending_repos", [])) > 0
    
    if has_news:
        return "process_news"
    elif has_github:
        return "process_github"
    else:
        return END

def route_after_save_news(state: PipelineState) -> str:
    """After saving news, check if GitHub repos also need processing."""
    if len(state.get("pending_repos", [])) > 0:
        return "process_github"
    return END

# =====================================================================
# Build the LangGraph StateGraph
# =====================================================================

def build_processing_graph() -> StateGraph:
    """Construct and compile the processing pipeline graph."""
    graph = StateGraph(PipelineState)
    
    # Add nodes
    graph.add_node("fetch_pending", fetch_pending_node)
    graph.add_node("process_news", process_news_node)
    graph.add_node("save_news", save_news_node)
    graph.add_node("process_github", process_github_node)
    graph.add_node("save_github", save_github_node)
    
    # Add edges
    graph.add_edge(START, "fetch_pending")
    graph.add_conditional_edges("fetch_pending", route_after_fetch, {
        "process_news": "process_news",
        "process_github": "process_github",
        END: END,
    })
    graph.add_edge("process_news", "save_news")
    graph.add_conditional_edges("save_news", route_after_save_news, {
        "process_github": "process_github",
        END: END,
    })
    graph.add_edge("process_github", "save_github")
    graph.add_edge("save_github", END)
    
    return graph.compile()

# =====================================================================
# Public API (backward compatible with scheduler.py)
# =====================================================================

# Compile graph once at module level
_processing_graph = build_processing_graph()

def process_pending_items(session: Session) -> dict:
    """
    Public entry point — backward compatible with old pipeline API.
    Invokes the LangGraph processing pipeline.
    """
    logger.info("Running LangGraph processing pipeline...")
    
    initial_state = {
        "session": session,
        "pending_news": [],
        "pending_repos": [],
        "news_results": [],
        "github_results": [],
        "processed_news": 0,
        "processed_github": 0,
        "errors": 0,
    }
    
    final_state = _processing_graph.invoke(initial_state)
    
    stats = {
        "processed_news": final_state.get("processed_news", 0),
        "processed_github": final_state.get("processed_github", 0),
        "errors": final_state.get("errors", 0),
    }
    
    logger.info(f"LangGraph processing pipeline completed. Stats: {stats}")
    return stats
