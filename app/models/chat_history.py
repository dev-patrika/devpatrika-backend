from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    role: str = Field(description="user or assistant")
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
