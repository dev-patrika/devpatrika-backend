import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from langchain_core.prompts import ChatPromptTemplate
from app.models.news import NewsItem
from app.models.github_radar import GitHubRadar
from app.core.constants import TechCategory
from app.services.processing.llm import get_llm

logger = logging.getLogger("dev-patrika.processing.pipeline")

# =====================================================================
# Pydantic Schemas for Structured Output
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
# Helper Formatting Functions
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
# Pipeline Processing Chains
# =====================================================================

def analyze_news_item(title: str, raw_content: str, source: str) -> Optional[NewsAnalysis]:
    """Use LLM chain with structured output to analyze a news item."""
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
        # Limit content size to prevent token overflow
        truncated_content = raw_content[:8000] if raw_content else "No content description available."
        
        result = chain.invoke({
            "source": source,
            "title": title,
            "raw_content": truncated_content
        })
        return result
    except Exception as e:
        logger.error(f"Failed to analyze news item '{title}': {str(e)}")
        return None

def analyze_github_repo(repo_name: str, description: str) -> Optional[GitHubAnalysis]:
    """Use LLM chain with structured output to analyze a trending repository."""
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
        result = chain.invoke({
            "repo_name": repo_name,
            "description": description or "No description provided."
        })
        return result
    except Exception as e:
        logger.error(f"Failed to analyze GitHub repository '{repo_name}': {str(e)}")
        return None

# =====================================================================
# Batch Processing Background Worker
# =====================================================================

def process_pending_items(session: Session) -> dict:
    """
    Fetch all items with un-generated summaries from SQLite and run them through the LLM pipeline.
    """
    stats = {"processed_news": 0, "processed_github": 0, "errors": 0}
    
    # 1. Process News Items
    logger.info("Checking for pending news items to analyze...")
    statement = select(NewsItem).where(NewsItem.summary == None)
    pending_news = session.exec(statement).all()
    
    if pending_news:
        logger.info(f"Found {len(pending_news)} news items to process.")
        for item in pending_news:
            analysis = analyze_news_item(item.title, item.raw_content, item.source)
            if analysis:
                item.summary = format_news_markdown(analysis)
                item.category = analysis.category
                session.add(item)
                stats["processed_news"] += 1
            else:
                stats["errors"] += 1
            
            # Commit after each item to prevent losing progress if one fails
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to commit news item summary for id {item.id}: {str(e)}")
                stats["errors"] += 1
    else:
        logger.info("No pending news items found.")

    # 2. Process GitHub Repositories
    logger.info("Checking for pending GitHub trending repos to analyze...")
    github_statement = select(GitHubRadar).where(GitHubRadar.why_it_matters_summary == None)
    pending_repos = session.exec(github_statement).all()
    
    if pending_repos:
        logger.info(f"Found {len(pending_repos)} GitHub repositories to process.")
        for repo in pending_repos:
            analysis = analyze_github_repo(repo.repo_name, repo.description)
            if analysis:
                repo.why_it_matters_summary = format_github_markdown(analysis)
                session.add(repo)
                stats["processed_github"] += 1
            else:
                stats["errors"] += 1
                
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to commit GitHub repo summary for id {repo.id}: {str(e)}")
                stats["errors"] += 1
    else:
        logger.info("No pending GitHub repositories found.")

    logger.info(f"AI processing cycle completed. Stats: {stats}")
    return stats
