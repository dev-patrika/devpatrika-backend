from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field
from app.core.constants import TechCategory, NewsSource

class NewsItem(SQLModel, table=True):
    __tablename__ = "news_items"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    url: str = Field(index=True, unique=True)
    summary: Optional[str] = Field(default=None)
    category: Optional[TechCategory] = Field(default=None, index=True)
    source: NewsSource = Field(index=True)
    published_at: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    raw_content: Optional[str] = Field(default=None)
    freshness_tag: Optional[str] = Field(default=None)
