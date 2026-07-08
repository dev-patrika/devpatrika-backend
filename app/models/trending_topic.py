from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class TrendingTopic(SQLModel, table=True):
    __tablename__ = "trending_topics"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    term: str = Field(index=True, unique=True)
    frequency: int = Field(default=0)
    trend_direction: str = Field(default="stable")  # "up", "down", or "stable"
    updated_at: datetime = Field(default_factory=datetime.utcnow)
