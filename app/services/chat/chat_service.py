import logging
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
from sqlmodel import Session, select
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.config import settings
from app.models.chat_history import ChatMessage
from app.services.vectorstore.chroma_service import semantic_search_wiki, semantic_search_news

logger = logging.getLogger("dev-patrika.chat.service")

# =====================================================================
# Model Selection Helper
# =====================================================================

def get_selected_llm(model_name: str, temperature: float = 0.4):
    """
    Returns the appropriate LangChain chat model based on the user's selection.
    """
    model_lower = model_name.lower()
    
    # Fallback checks to environment variables
    groq_key = settings.GROQ_API_KEY
    gemini_key = settings.GEMINI_API_KEY
    
    if "gemini" in model_lower:
        logger.info(f"Instantiating Google Gemini Chat model ('gemini-2.5-flash')")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            api_key=gemini_key or "placeholder_key"
        )
    else:
        # Default or explicit Groq
        selected_id = "openai/gpt-oss-120b"
        if "qwen" in model_lower:
            selected_id = "qwen/qwen3.6-27b"
        elif "gpt-oss-20b" in model_lower:
            selected_id = "openai/gpt-oss-20b"
            
        logger.info(f"Instantiating Groq Chat model ('{selected_id}')")
        return ChatGroq(
            model=selected_id,
            temperature=temperature,
            api_key=groq_key or "placeholder_key"
        )

# =====================================================================
# Chat & Citation Processor Service
# =====================================================================

def process_chat_message(
    session: Session,
    session_id: str,
    message_content: str,
    model_name: str = "openai/gpt-oss-120b"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executes RAG context retrieval, loads chat memory, runs the LLM,
    parses citations, and saves chat history in SQLite.
    Returns: Tuple[answer_text, citations_list]
    """
    logger.info(f"Processing chat message for session: '{session_id}' using model '{model_name}'")
    
    # 1. Load Chat Memory (latest 10 messages)
    history_statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
    )
    db_history = session.exec(history_statement).all()
    # Reverse to keep chronological order
    db_history.reverse()
    
    langchain_messages = []
    for msg in db_history:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        else:
            langchain_messages.append(AIMessage(content=msg.content))
            
    # 2. RAG Context Retrieval (Wiki & News collections)
    wiki_context = semantic_search_wiki(session, query=message_content, limit=3, threshold=0.3)
    news_context = semantic_search_news(session, query=message_content, limit=3, threshold=0.3)
    
    # 3. Format Context and compile mapping list
    context_docs = []
    citation_map = {}  # index -> metadata dictionary
    source_idx = 1
    
    for wiki in wiki_context:
        doc_text = f"Source [{source_idx}] (Dev Wiki Concept):\nTerm: {wiki.term}\nDefinition: {wiki.definition}\n"
        context_docs.append(doc_text)
        citation_map[source_idx] = {
            "id": source_idx,
            "title": f"Wiki: {wiki.term}",
            "url": f"https://devpatrika.com/wiki/{wiki.term.replace(' ', '_')}",
            "source": "Dev Wiki"
        }
        source_idx += 1
        
    for news in news_context:
        doc_text = f"Source [{source_idx}] (News Article):\nTitle: {news.title}\nCategory: {news.category}\nSummary: {news.summary or ''}\n"
        context_docs.append(doc_text)
        citation_map[source_idx] = {
            "id": source_idx,
            "title": news.title,
            "url": news.url,
            "source": news.source or "Tech Feed"
        }
        source_idx += 1
        
    compiled_context_text = "\n\n".join(context_docs) if context_docs else "No specific document references found."
    
    # 4. Build System & Human Prompts
    system_prompt = (
        "You are the Dev Patrika AI Assistant, an expert advisor for developers and technology managers.\n"
        "You answer developer questions using historical news, technical glossary entries, and research preprints.\n\n"
        "Here are the rules for your responses:\n"
        "1. Answer the user's question clearly, professionally, and technically using the provided Reference Documents context.\n"
        "2. If you use information from a Source Document, you MUST cite the source index in brackets, e.g. [1], [2] next to the text where it is used. "
        "Do not make up sources outside the provided reference list.\n"
        "3. If the context does not contain the answer, rely on your parametric knowledge to answer politely, but DO NOT include any citations in that case.\n\n"
        "Reference Documents Context:\n"
        f"{compiled_context_text}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # 5. Call LLM
    try:
        llm = get_selected_llm(model_name)
        chain = prompt | llm
        
        response = chain.invoke({
            "chat_history": langchain_messages,
            "question": message_content
        })
        
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        # 6. Parse Citations from Output Text
        # Find numeric references like [1], [2], [10]
        cited_indices = re.findall(r'\[(\d+)\]', answer_text)
        # Unique integer values
        cited_ints = sorted(list(set(int(idx) for idx in cited_indices)))
        
        citations_response = []
        for c_idx in cited_ints:
            if c_idx in citation_map:
                citations_response.append(citation_map[c_idx])
                
        # 7. Persist turns in SQLite Database
        user_message = ChatMessage(
            session_id=session_id,
            role="user",
            content=message_content,
            created_at=datetime.utcnow()
        )
        assistant_message = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=answer_text,
            created_at=datetime.utcnow()
        )
        
        session.add(user_message)
        session.add(assistant_message)
        session.commit()
        
        logger.info(f"Persisted messages for session '{session_id}'. Citations returned: {len(citations_response)}")
        return answer_text, citations_response
    except Exception as e:
        logger.error(f"Error during chatbot run: {str(e)}")
        session.rollback()
        return f"I encountered a technical error: {str(e)}", []
