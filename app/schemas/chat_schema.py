from pydantic import BaseModel
from typing import Optional, List, Dict

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gemini-2.5-flash"
    session_id: Optional[str] = "default-session"
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    model_used: str
    citations: Optional[List[Dict]] = []
    metadata: Optional[Dict] = None

    model_config = {
        "protected_namespaces": ()
    }
