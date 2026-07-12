from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class OTPCode(SQLModel, table=True):
    __tablename__ = "otp_codes"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True)
    otp_hash: str = Field(description="SHA-256 hash of the 6-digit OTP")
    expires_at: datetime
    attempts: int = Field(default=0, description="Failed verification attempts (max 5)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
