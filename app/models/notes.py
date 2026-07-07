from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class PersonalNote(SQLModel, table=True):
    __tablename__ = "personal_notes"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
