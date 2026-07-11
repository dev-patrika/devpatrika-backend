from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Column
from sqlalchemy.dialects.postgresql import JSONB

class WikiEntry(SQLModel, table=True):
    __tablename__ = "wiki_entries"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    term: str = Field(index=True, unique=True)
    definition: str
    why_trending: str
    related_links: Optional[List[str]] = Field(default=[], sa_column=Column(JSONB, default=[]))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
