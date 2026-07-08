from fastapi import APIRouter
from app.schemas.chat_schema import ChatRequest, ChatResponse
from typing import List, Dict

router = APIRouter(prefix="/ai", tags=["AI Engine"])

@router.get("/models", response_model=List[Dict[str, str]])
def get_supported_models():
    """
    Placeholder endpoint to retrieve lists of supported LLM models.
    Fully implemented in v0.5.0.
    """
    return [
        {"provider": "Google", "model": "gemini-2.5-flash", "status": "active"},
        {"provider": "Groq", "model": "openai/gpt-oss-120b", "status": "inactive"},
        {"provider": "Hugging Face", "model": "mistral-7b-instruct", "status": "inactive"}
    ]

@router.post("/chat", response_model=ChatResponse)
def assistant_chat(request: ChatRequest):
    """
    Placeholder endpoint for conversational RAG assistant queries.
    Fully implemented in v0.5.0.
    """
    return ChatResponse(
        response=f"This is a placeholder reply for: '{request.message}'. AI orchestrators, vector database tools, and models will be wired in v0.5.0.",
        model_used=request.model or "gemini-1.5-flash",
        metadata={"status": "mock"}
    )
