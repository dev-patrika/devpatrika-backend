import io
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from langgraph.graph import StateGraph, START, END
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.news import NewsItem
from app.core.constants import NewsSource, TechCategory
from app.services.processing.llm import get_llm
from app.services.vectorstore.vector_service import index_news_item

logger = logging.getLogger("dev-patrika.agents.research-digest")

# =====================================================================
# State & Schemas
# =====================================================================

class ResearchDigestState(TypedDict):
    session: Session
    arxiv_id: str
    pdf_url: Optional[str]
    title: Optional[str]
    abstract: Optional[str]
    pdf_text: Optional[str]
    chunks: List[str]
    digest: Optional[str]
    news_item_id: Optional[int]
    errors: int

class DeveloperDigest(BaseModel):
    overview: str = Field(
        description="A 1-2 sentence non-technical overview of the research paper and why it is important for software development."
    )
    technical_architecture: List[str] = Field(
        description="2-3 paragraphs describing the core technical details, methodology, model, or algorithm introduced by the paper in a clear developer-friendly language."
    )
    key_takeaways: List[str] = Field(
        description="3-5 bullet points of key takeaways for developers, such as applications, skills, libraries, or tools."
    )
    relevance_to_industry: str = Field(
        description="A brief explanation of how this research impacts the industry and what developers should prepare for."
    )

# =====================================================================
# Helper Formatting
# =====================================================================

def format_digest_markdown(digest: DeveloperDigest, url: str) -> str:
    paragraphs = "\n\n".join(digest.technical_architecture)
    bullets = "\n".join(f"- {point}" for point in digest.key_takeaways)
    return (
        f"**Overview**\n"
        f"{digest.overview}\n\n"
        f"**Detailed Breakdown**\n"
        f"{paragraphs}\n\n"
        f"**Key Takeaways**\n"
        f"{bullets}\n\n"
        f"**Industry Impact**\n"
        f"{digest.relevance_to_industry}\n\n"
        f"🔗 [Read Full Research Paper]({url})"
    )

# =====================================================================
# Graph Node Functions
# =====================================================================

def fetch_metadata_node(state: ResearchDigestState) -> dict:
    """Node: Fetch paper title, abstract, and PDF link using the arXiv API."""
    arxiv_id = state["arxiv_id"]
    logger.info(f"Fetching arXiv metadata for ID: {arxiv_id}")
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        
        root = ET.fromstring(res.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        
        if entry is None:
            raise ValueError(f"arXiv entry not found for ID: {arxiv_id}")
            
        title_elem = entry.find("atom:title", ns)
        summary_elem = entry.find("atom:summary", ns)
        
        title = title_elem.text.strip() if title_elem is not None else ""
        abstract = " ".join(summary_elem.text.split()) if summary_elem is not None else ""
        
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or "pdf" in link.attrib.get("href", ""):
                pdf_url = link.attrib.get("href", "")
                break
                
        logger.info(f"Metadata loaded: '{title}'")
        return {
            "title": title,
            "abstract": abstract,
            "pdf_url": pdf_url
        }
    except Exception as e:
        logger.error(f"Failed to fetch metadata for {arxiv_id}: {e}")
        return {"errors": state.get("errors", 0) + 1}

def download_pdf_node(state: ResearchDigestState) -> dict:
    """Node: Download arXiv PDF preprint and extract text using pypdf."""
    pdf_url = state.get("pdf_url")
    if not pdf_url:
        return {"pdf_text": ""}
        
    logger.info(f"Downloading arXiv PDF from: {pdf_url}")
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        text_pages = []
        max_pages = min(8, len(reader.pages))
        for page_idx in range(max_pages):
            page_text = reader.pages[page_idx].extract_text()
            if page_text:
                text_pages.append(page_text)
                
        full_text = "\n".join(text_pages)
        logger.info(f"Successfully extracted {len(full_text)} characters from {max_pages} PDF pages.")
        return {"pdf_text": full_text}
    except Exception as e:
        logger.error(f"Failed to download/parse PDF: {e}. Falling back to abstract only.")
        return {"pdf_text": "", "errors": state.get("errors", 0) + 1}

def split_chunks_node(state: ResearchDigestState) -> dict:
    """Node: Split extracted text into semantic chunks for the translation pipeline."""
    pdf_text = state.get("pdf_text", "")
    abstract = state.get("abstract", "")
    
    text_to_split = pdf_text if pdf_text else abstract
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_text(text_to_split)
    
    logger.info(f"Split text into {len(chunks)} chunks.")
    return {"chunks": chunks}

def summarize_translate_node(state: ResearchDigestState) -> dict:
    """Node: Feed the paper chunks to the LLM and generate a developer-friendly digest."""
    title = state.get("title", "")
    abstract = state.get("abstract", "")
    chunks = state.get("chunks", [])
    
    context_text = f"Title: {title}\nAbstract: {abstract}\n\nKey Paper Content:\n"
    for c in chunks[:4]:
        context_text += f"- {c}\n"
        
    try:
        llm = get_llm(temperature=0.2)
        structured_llm = llm.with_structured_output(DeveloperDigest)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an AI research translation agent. Your task is to analyze academic research papers "
                "and translate complex mathematical, algorithmic, or ML jargon into non-technical, "
                "practical, and engaging summaries for software engineers and developers.\n\n"
                "CRITICAL: Write in extremely simple, direct, and straightforward English. Use plain vocabulary and "
                "simple sentence structures. Do NOT use flowery, verbose, or poetic academic prose. Keep explanations "
                "clear, brief, and easy to understand."
            )),
            ("human", "Paper Data:\n{context}\n\nPlease compile a structured developer-friendly digest.")
        ])
        
        chain = prompt | structured_llm
        result = chain.invoke({"context": context_text[:15000]})
        
        if result:
            digest = format_digest_markdown(result, state.get("pdf_url") or "")
            return {"digest": digest}
        raise ValueError("LLM returned empty digest")
    except Exception as e:
        logger.error(f"Failed to generate translation: {e}")
        return {"digest": "", "errors": state.get("errors", 0) + 1}

def publish_digest_node(state: ResearchDigestState) -> dict:
    """Node: Save the compiled research digest to the SQL database and pgvector store."""
    session = state["session"]
    digest = state.get("digest")
    title = state.get("title")
    pdf_url = state.get("pdf_url")
    
    if not digest or not title:
        logger.warning("Empty digest or title. Skipping publishing.")
        return {}
        
    try:
        stmt = select(NewsItem).where(NewsItem.url == pdf_url)
        existing = session.exec(stmt).first()
        
        if existing:
            existing.summary = digest
            existing.updated_at = datetime.utcnow()
            item = existing
            logger.info(f"Updated existing arXiv digest for '{title}'")
        else:
            item = NewsItem(
                title=f"arXiv: {title}",
                url=pdf_url,
                summary=digest,
                category=TechCategory.AI,
                source=NewsSource.ARXIV,
                published_at=datetime.utcnow(),
                created_at=datetime.utcnow()
            )
            logger.info(f"Created new arXiv digest for '{title}'")
            
        session.add(item)
        session.commit()
        session.refresh(item)
        
        # Index in pgvector
        index_news_item(item)
        
        return {"news_item_id": item.id}
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to publish research digest: {e}")
        return {"errors": state.get("errors", 0) + 1}

# =====================================================================
# Build and Compile Graph
# =====================================================================

def build_research_digest_graph() -> StateGraph:
    """Constructs the stateful Research Digest Agent graph."""
    graph = StateGraph(ResearchDigestState)
    
    graph.add_node("fetch_metadata", fetch_metadata_node)
    graph.add_node("download_pdf", download_pdf_node)
    graph.add_node("split_chunks", split_chunks_node)
    graph.add_node("summarize_translate", summarize_translate_node)
    graph.add_node("publish_digest", publish_digest_node)
    
    graph.add_edge(START, "fetch_metadata")
    graph.add_edge("fetch_metadata", "download_pdf")
    graph.add_edge("download_pdf", "split_chunks")
    graph.add_edge("split_chunks", "summarize_translate")
    graph.add_edge("summarize_translate", "publish_digest")
    graph.add_edge("publish_digest", END)
    
    return graph.compile()

# Compile graph singleton
research_digest_agent = build_research_digest_graph()

def run_research_digest_agent(session: Session, arxiv_id: str) -> Dict[str, Any]:
    """
    Public API to invoke the Research Digest Agent on a specific arXiv paper.
    """
    logger.info(f"Invoking stateful Research Digest Agent for arXiv ID: {arxiv_id}...")
    initial_state = {
        "session": session,
        "arxiv_id": arxiv_id,
        "pdf_url": None,
        "title": None,
        "abstract": None,
        "pdf_text": None,
        "chunks": [],
        "digest": None,
        "news_item_id": None,
        "errors": 0
    }
    
    final_state = research_digest_agent.invoke(initial_state)
    logger.info("Research Digest Agent finished execution.")
    return {
        "news_item_id": final_state.get("news_item_id"),
        "title": final_state.get("title"),
        "errors": final_state.get("errors", 0)
    }
