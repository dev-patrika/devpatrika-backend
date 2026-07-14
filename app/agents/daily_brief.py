import logging
from typing import TypedDict, List, Set, Dict, Any
from sqlmodel import Session
from langgraph.graph import StateGraph, START, END

# Import orchestrator nodes
from app.services.ingestion.orchestrator import (
    load_recent_context_node,
    fetch_hn_node,
    fetch_devto_node,
    fetch_arxiv_node,
    fetch_github_node,
    commit_all_node
)

# Import processing nodes and routing helpers
from app.services.processing.pipeline import (
    fetch_pending_node,
    process_news_node,
    save_news_node,
    process_github_node,
    save_github_node,
    route_after_fetch,
    route_after_save_news
)

logger = logging.getLogger("dev-patrika.agents.daily-brief")

# =====================================================================
# State Definition
# =====================================================================

class DailyBriefState(TypedDict):
    """Unified state for the entire Daily Brief agent workflow."""
    session: Session
    
    # Ingestion State Keys
    recent_titles: List[str]
    recent_urls: Set[str]
    recent_repo_urls: Set[str]
    stats: Dict[str, Any]
    
    # Processing State Keys
    pending_news: List[Any]
    pending_repos: List[Any]
    news_results: List[Any]
    github_results: List[Any]
    processed_news: int
    processed_github: int
    errors: int

# =====================================================================
# Build and Compile Graph
# =====================================================================

def build_daily_brief_graph() -> StateGraph:
    """Constructs the unified Daily Brief Agent graph."""
    graph = StateGraph(DailyBriefState)
    
    # 1. Ingestion Phase Nodes
    graph.add_node("load_recent_context", load_recent_context_node)
    graph.add_node("fetch_hn", fetch_hn_node)
    graph.add_node("fetch_devto", fetch_devto_node)
    graph.add_node("fetch_arxiv", fetch_arxiv_node)
    graph.add_node("fetch_github", fetch_github_node)
    graph.add_node("commit_ingested", commit_all_node)
    
    # 2. Processing Phase Nodes
    graph.add_node("fetch_pending", fetch_pending_node)
    graph.add_node("process_news", process_news_node)
    graph.add_node("save_news", save_news_node)
    graph.add_node("process_github", process_github_node)
    graph.add_node("save_github", save_github_node)
    
    # 3. Define Graph Edges
    # Sequential Ingestion Flow
    graph.add_edge(START, "load_recent_context")
    graph.add_edge("load_recent_context", "fetch_hn")
    graph.add_edge("fetch_hn", "fetch_devto")
    graph.add_edge("fetch_devto", "fetch_arxiv")
    graph.add_edge("fetch_arxiv", "fetch_github")
    graph.add_edge("fetch_github", "commit_ingested")
    
    # Hand-off to Processing Flow
    graph.add_edge("commit_ingested", "fetch_pending")
    
    # Conditional Processing Flow
    graph.add_conditional_edges("fetch_pending", route_after_fetch, {
        "process_news": "process_news",
        "process_github": "process_github",
        END: END
    })
    
    graph.add_edge("process_news", "save_news")
    graph.add_conditional_edges("save_news", route_after_save_news, {
        "process_github": "process_github",
        END: END
    })
    
    graph.add_edge("process_github", "save_github")
    graph.add_edge("save_github", END)
    
    return graph.compile()

# Compile the singleton graph
daily_brief_agent = build_daily_brief_graph()

def run_daily_brief_agent(session: Session) -> Dict[str, Any]:
    """
    Public API to invoke the unified Daily Brief Agent.
    Runs ingestion, deduplication, and AI summarization in one stateful transaction.
    """
    logger.info("Invoking stateful Daily Brief Agent...")
    initial_state = {
        "session": session,
        "recent_titles": [],
        "recent_urls": set(),
        "recent_repo_urls": set(),
        "stats": {},
        "pending_news": [],
        "pending_repos": [],
        "news_results": [],
        "github_results": [],
        "processed_news": 0,
        "processed_github": 0,
        "errors": 0
    }
    
    final_state = daily_brief_agent.invoke(initial_state)
    logger.info("Daily Brief Agent workflow finished execution.")
    return {
        "ingestion_stats": final_state.get("stats", {}),
        "processed_news": final_state.get("processed_news", 0),
        "processed_github": final_state.get("processed_github", 0),
        "errors": final_state.get("errors", 0)
    }
