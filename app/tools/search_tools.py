import logging
import requests
from typing import Optional
from bs4 import BeautifulSoup
from sqlmodel import Session, select
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.database import engine
from app.models.news import NewsItem
from app.core.constants import TechCategory
from app.services.processing.llm import get_llm

logger = logging.getLogger("dev-patrika.tools.search_tools")

@tool
def news_search_tool(query: str, category: Optional[str] = None, limit: int = 5) -> str:
    """
    Search for tech news articles in the local Dev Patrika database.
    Optionally filter by category (AI, Web Dev, Cybersecurity, Startups, Open Source, Cloud/DevOps).
    Returns a formatted string of matching news items.
    """
    logger.info(f"news_search_tool triggered with query='{query}', category={category}")
    try:
        with Session(engine) as session:
            statement = select(NewsItem)
            
            # Category filter
            if category:
                # Match enum string
                matched_category = None
                for cat in TechCategory:
                    if cat.value.lower() == category.lower() or cat.name.lower() == category.lower():
                        matched_category = cat
                        break
                if matched_category:
                    statement = statement.where(NewsItem.category == matched_category)
            
            # Content query filter
            statement = statement.where(
                NewsItem.title.like(f"%{query}%") | 
                NewsItem.raw_content.like(f"%{query}%") | 
                (NewsItem.summary.like(f"%{query}%") if NewsItem.summary is not None else False)
            )
            
            statement = statement.order_by(NewsItem.published_at.desc()).limit(limit)
            results = session.exec(statement).all()
            
            if not results:
                return f"No news articles found in local database matching query: '{query}'."
                
            formatted_results = []
            for item in results:
                formatted_results.append(
                    f"Title: {item.title}\n"
                    f"URL: {item.url}\n"
                    f"Category: {item.category.value if item.category else 'Uncategorized'}\n"
                    f"Source: {item.source.value}\n"
                    f"Published At: {item.published_at}\n"
                    f"Summary: {item.summary or 'No AI summary generated yet.'}\n"
                    f"---"
                )
            return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error executing news_search_tool: {str(e)}")
        return f"Error executing local database search: {str(e)}"

@tool
def github_search_tool(query: str, limit: int = 5) -> str:
    """
    Search for repositories on GitHub using the public GitHub Search API.
    Returns the top repository names, URLs, descriptions, and star counts.
    """
    logger.info(f"github_search_tool triggered with query='{query}'")
    url = f"https://api.github.com/search/repositories?q={query}&per_page={limit}"
    headers = {"User-Agent": "Dev-Patrika-App"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("items", [])
        if not items:
            return f"No GitHub repositories found matching query: '{query}'."
            
        formatted_results = []
        for item in items:
            formatted_results.append(
                f"Repository: {item.get('full_name')}\n"
                f"URL: {item.get('html_url')}\n"
                f"Stars: {item.get('stargazers_count', 0)}\n"
                f"Description: {item.get('description') or 'No description provided.'}\n"
                f"Language: {item.get('language') or 'Unknown'}\n"
                f"---"
            )
        return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error executing github_search_tool: {str(e)}")
        return f"Error querying GitHub Search API: {str(e)}"

@tool
def url_summarizer_tool(url: str) -> str:
    """
    Fetch a website/URL, scrape its main text content, and generate a concise summary of the page.
    Useful for reading and summarizing external web pages on-the-fly.
    """
    logger.info(f"url_summarizer_tool triggered with url='{url}'")
    headers = {"User-Agent": "Dev-Patrika-App/Crawler"}
    
    try:
        # 1. Scrape webpage content
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "lxml")
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.decompose()
            
        # Get raw text
        raw_text = soup.get_text(separator=" ")
        
        # Clean up whitespace
        lines = (line.strip() for line in raw_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = " ".join(chunk for chunk in chunks if chunk)
        
        if not clean_text or len(clean_text) < 100:
            return "Failed to extract readable text content from the target URL."
            
        # 2. Slice text to prevent token overflow (e.g. limit to first ~12,000 characters)
        truncated_text = clean_text[:12000]
        
        # 3. Summarize using fallback LLM
        llm = get_llm(temperature=0.0)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a professional research assistant. Read the following cleaned text content from an "
                "external webpage and write a clear, concise summary of it. Highlight key features, main "
                "conclusions, and technical details in bullet points."
            )),
            ("human", (
                "Source URL: {url}\n"
                "Content:\n{content}\n\n"
                "Please generate the summary now:"
            ))
        ])
        
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({
            "url": url,
            "content": truncated_text
        })
        return f"Summary of {url}:\n\n{summary}"
    except Exception as e:
        logger.error(f"Error executing url_summarizer_tool: {str(e)}")
        return f"Error fetching or summarizing URL '{url}': {str(e)}"
