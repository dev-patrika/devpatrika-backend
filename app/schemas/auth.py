from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


# ============================================================
# OTP Request/Response
# ============================================================

class OTPRequestSchema(BaseModel):
    """Request body for POST /api/auth/otp/request"""
    email: str = Field(..., description="User email to send OTP to")


class OTPVerifySchema(BaseModel):
    """Request body for POST /api/auth/otp/verify"""
    email: str = Field(..., description="Email the OTP was sent to")
    otp: str = Field(..., min_length=6, max_length=6, description="6-digit OTP code")


# ============================================================
# Auth Responses
# ============================================================

class AuthTokenResponse(BaseModel):
    """Response after successful login (any method)"""
    access_token: str
    token_type: str = "bearer"
    user: "UserProfile"


class UserProfile(BaseModel):
    """Public user profile returned by /api/auth/me"""
    id: int
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    auth_provider: str
    is_verified: bool
    created_at: datetime


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str


# Rebuild model to resolve forward references
AuthTokenResponse.model_rebuild()
