from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    name: Optional[str] = Field(default=None)
    avatar_url: Optional[str] = Field(default=None)
    auth_provider: str = Field(
        default="email",
        description="Last login method: google, github, or email"
    )
    provider_id: Optional[str] = Field(
        default=None,
        description="External OAuth user ID (Google sub / GitHub id)"
    )
    is_verified: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
