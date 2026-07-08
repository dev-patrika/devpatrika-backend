from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional, List
import json

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

    @field_validator("related_links", mode="before")
    @classmethod
    def parse_links(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return []
        return v

    class Config:
        from_attributes = True
