from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import Session
from app.database import get_session
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat.chat_service import process_chat_message, stream_chat_message
from typing import List, Dict

router = APIRouter(prefix="/ai", tags=["AI Engine"])

@router.get("/models", response_model=List[Dict[str, str]])
def get_supported_models():
    """
    Retrieve lists of supported LLM models.
    """
    return [
        {"provider": "Google", "model": "gemini-2.5-flash", "status": "active"},
        {"provider": "Groq", "model": "llama-3.3-70b-versatile", "status": "active"},
        {"provider": "Groq", "model": "llama3-8b-8192", "status": "active"},
        {"provider": "Groq", "model": "mixtral-8x7b-32768", "status": "active"}
    ]

@router.post("/chat")
def assistant_chat(request: ChatRequest, session: Session = Depends(get_session)):
    """
    Conversational RAG chatbot with session persistent history memory
    and explicit citation mappings. Returns a StreamingResponse.
    """
    model_name = request.model or "openai/gpt-oss-120b"
    return StreamingResponse(
        stream_chat_message(
            session=session,
            session_id=request.session_id,
            message_content=request.message,
            model_name=model_name
        ),
        media_type="text/event-stream"
    )
