from typing import Optional
from datetime import datetime
import json
from sqlmodel import SQLModel, Field

class WikiEntry(SQLModel, table=True):
    __tablename__ = "wiki_entries"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    term: str = Field(index=True, unique=True)
    definition: str
    why_trending: str
    related_links: Optional[str] = Field(default="[]") # JSON string containing references
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def links_list(self) -> list:
        try:
            return json.loads(self.related_links or "[]")
        except Exception:
            return []

    def set_links(self, links: list):
        self.related_links = json.dumps(links)
