from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class WeeklyReport(SQLModel, table=True):
    __tablename__ = "weekly_reports"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    content: str  # Compiled markdown summary of news, repos, and trends
    start_date: datetime
    end_date: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
