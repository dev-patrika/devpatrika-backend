from pydantic import BaseModel, HttpUrl
from datetime import datetime
from typing import Optional
from app.core.constants import TechCategory, NewsSource

class NewsItemBase(BaseModel):
    title: str
    url: str
    summary: Optional[str] = None
    category: Optional[TechCategory] = None
    source: NewsSource
    published_at: datetime
    raw_content: Optional[str] = None
    freshness_tag: Optional[str] = None

class NewsItemCreate(NewsItemBase):
    pass

class NewsItemRead(NewsItemBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
