from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class GitHubRadar(SQLModel, table=True):
    __tablename__ = "github_radar"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    repo_name: str = Field(index=True)
    repo_url: str = Field(unique=True)
    description: Optional[str] = Field(default=None)
    why_it_matters_summary: Optional[str] = Field(default=None)
    stars_count: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
