from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class WikiEntryBase(BaseModel):
    term: str
    definition: str
    why_trending: str
    related_links: Optional[List[str]] = []

class WikiEntryCreate(WikiEntryBase):
    pass

class WikiEntryRead(BaseModel):
    id: int
    term: str
    definition: str
    why_trending: str
    related_links: List[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
