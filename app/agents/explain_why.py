import logging
from typing import TypedDict, List, Dict, Any, Tuple
from sqlmodel import Session, select
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END

from app.services.processing.llm import get_llm
from app.services.vectorstore.vector_service import semantic_search_news, semantic_search_wiki
from app.models.github_radar import GitHubRadar

logger = logging.getLogger("dev-patrika.agents.explain-why")

class ExplainWhyState(TypedDict):
    session: Session
    query: str
    chat_history: List[Any]
    contexts: List[str]
    citations: List[Dict[str, Any]]
    answer: str

# =====================================================================
# Graph Node Functions
# =====================================================================

def retrieve_local_knowledge_node(state: ExplainWhyState) -> dict:
    """Node: Run semantic searches across News & Wiki to find facts for the query."""
    session = state["session"]
    query = state["query"]
    
    logger.info(f"Retrieve context for query: '{query}'")
    news_matches = semantic_search_news(session, query, limit=3, threshold=0.25)
    wiki_matches = semantic_search_wiki(session, query, limit=2, threshold=0.25)
    
    contexts = []
    citations = []
    source_idx = 1
    
    for item in news_matches:
        contexts.append(
            f"Source [{source_idx}] (News Article):\n"
            f"Title: {item.title}\n"
            f"Summary: {item.summary or ''}\n"
        )
        citations.append({
            "id": source_idx,
            "title": item.title,
            "url": item.url,
            "source": "News Feed"
        })
        source_idx += 1
        
    for item in wiki_matches:
        contexts.append(
            f"Source [{source_idx}] (Dev Wiki Glossary):\n"
            f"Term: {item.term}\n"
            f"Definition: {item.definition}\n"
        )
        citations.append({
            "id": source_idx,
            "title": f"Wiki: {item.term}",
            "url": f"https://devpatrika.com/wiki/{item.term.replace(' ', '_')}",
            "source": "Dev Wiki"
        })
        source_idx += 1
        
    return {"contexts": contexts, "citations": citations}

def search_github_radar_node(state: ExplainWhyState) -> dict:
    """Node: Query trending repos in GitHub Radar that match terms in the query."""
    session = state["session"]
    query = state["query"]
    contexts = state.get("contexts", [])
    citations = state.get("citations", [])
    
    words = [w.strip() for w in query.lower().split() if len(w.strip()) > 3]
    if not words:
        return {}
        
    stmt = select(GitHubRadar)
    repos = session.exec(stmt).all()
    
    matched_repos = []
    for r in repos:
        name = r.repo_name.lower()
        desc = (r.description or "").lower()
        if any(word in name or word in desc for word in words):
            matched_repos.append(r)
            
    source_idx = len(citations) + 1
    for repo in matched_repos[:2]:
        contexts.append(
            f"Source [{source_idx}] (GitHub Radar Repository):\n"
            f"Repository: {repo.repo_name}\n"
            f"Description: {repo.description or ''}\n"
            f"Why it matters: {repo.why_it_matters_summary or ''}\n"
        )
        citations.append({
            "id": source_idx,
            "title": f"GitHub: {repo.repo_name}",
            "url": repo.repo_url,
            "source": "GitHub Radar"
        })
        source_idx += 1
        
    return {"contexts": contexts, "citations": citations}

def generate_deep_dive_node(state: ExplainWhyState) -> dict:
    """Node: Compile a structured explanation with inline source citations [1], [2], etc."""
    query = state["query"]
    contexts = state.get("contexts", [])
    chat_history = state.get("chat_history", [])
    
    if not contexts:
        compiled_context = "No specific reference documents found in Dev Patrika database."
    else:
        compiled_context = "\n\n".join(contexts)
        
    llm = get_llm(temperature=0.4)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the 'Explain Why' AI Agent for Dev Patrika. Your goal is to answer deep-dive developer queries "
            "using the compiled reference documents. "
            "Provide a thorough, highly technical, yet clean and engaging explanation. "
            "Strictly cite your facts using bracketed numbering corresponding to the source indexes in the context (e.g. [1], [2]). "
            "If the provided context does not contain enough info, utilize your pre-trained developer knowledge but clearly distinguish it from database facts."
        )),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Reference Documents:\n{context}\n\nUser Query: {query}")
    ])
    
    history_messages = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            history_messages.append(msg)
        elif isinstance(msg, dict):
            if msg.get("role") == "user":
                history_messages.append(HumanMessage(content=msg.get("content", "")))
            else:
                history_messages.append(AIMessage(content=msg.get("content", "")))
            
    chain = prompt | llm
    response = chain.invoke({
        "context": compiled_context,
        "query": query,
        "history": history_messages
    })
    
    return {"answer": response.content}

# =====================================================================
# Build and Compile Graph
# =====================================================================

def build_explain_why_graph() -> StateGraph:
    graph = StateGraph(ExplainWhyState)
    
    graph.add_node("retrieve_local_knowledge", retrieve_local_knowledge_node)
    graph.add_node("search_github_radar", search_github_radar_node)
    graph.add_node("generate_deep_dive", generate_deep_dive_node)
    
    graph.add_edge(START, "retrieve_local_knowledge")
    graph.add_edge("retrieve_local_knowledge", "search_github_radar")
    graph.add_edge("search_github_radar", "generate_deep_dive")
    graph.add_edge("generate_deep_dive", END)
    
    return graph.compile()

explain_why_agent = build_explain_why_graph()

def run_explain_why_agent(
    session: Session,
    query: str,
    chat_history: List[Any] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Public entry point to invoke the 'Explain Why' Agent.
    Returns: Tuple[answer, citations]
    """
    logger.info(f"Invoking Explain Why Agent for query: '{query}'")
    initial_state = {
        "session": session,
        "query": query,
        "chat_history": chat_history or [],
        "contexts": [],
        "citations": [],
        "answer": ""
    }
    
    final_state = explain_why_agent.invoke(initial_state)
    return final_state.get("answer", ""), final_state.get("citations", [])
