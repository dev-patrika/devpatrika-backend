from pydantic import BaseModel, HttpUrl
from datetime import datetime
from typing import Optional

class GitHubRadarBase(BaseModel):
    repo_name: str
    repo_url: str
    description: Optional[str] = None
    why_it_matters_summary: Optional[str] = None
    stars_count: int = 0

class GitHubRadarCreate(GitHubRadarBase):
    pass

class GitHubRadarRead(GitHubRadarBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
