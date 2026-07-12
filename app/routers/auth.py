from fastapi import APIRouter, Depends, HTTPException, Response, Request
from sqlmodel import Session
from app.database import get_session
from app.schemas.auth import (
    OTPRequestSchema,
    OTPVerifySchema,
    AuthTokenResponse,
    UserProfile,
    MessageResponse
)
import logging

from app.services.auth.email_service import send_otp_email
from app.services.auth.otp_service import create_and_store_otp, verify_otp_logic
from app.services.auth.jwt_service import create_access_token, create_refresh_token
from app.models.user import User
from sqlmodel import select
from datetime import datetime

logger = logging.getLogger("dev-patrika.auth")

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============================================================
# Email OTP Endpoints (Phase 1)
# ============================================================

@router.post("/otp/request", response_model=MessageResponse)
async def request_otp(body: OTPRequestSchema, session: Session = Depends(get_session)):
    """Send a 6-digit OTP to the given email address."""
    email = body.email.lower().strip()
    
    # 1. Generate and store OTP
    plain_otp = create_and_store_otp(session, email)
    
    # 2. Send email
    success = await send_otp_email(email, plain_otp)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to send OTP email. Please try again.")
        
    return MessageResponse(message=f"OTP sent successfully to {email}")


@router.post("/otp/verify", response_model=AuthTokenResponse)
async def verify_otp(body: OTPVerifySchema, response: Response, session: Session = Depends(get_session)):
    """Verify OTP and issue JWT access + refresh tokens."""
    email = body.email.lower().strip()
    
    # 1. Verify OTP
    is_valid, error_msg = verify_otp_logic(session, email, body.otp)
    if not is_valid:
        raise HTTPException(status_code=401, detail=error_msg)
        
    # 2. Find or create user
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        # First time login -> Register
        user = User(email=email, auth_provider="email", is_verified=True)
        session.add(user)
        session.commit()
        session.refresh(user)
    else:
        # Update last login info
        if not user.is_verified:
            user.is_verified = True
        user.auth_provider = "email"
        session.add(user)
        session.commit()
        session.refresh(user)
        
    # 3. Generate tokens
    access_token = create_access_token(user.id, user.email)
    refresh_token = create_refresh_token(user.id)
    
    # 4. Set HttpOnly cookie for refresh token
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,          # True in prod, false in dev (will be handled gracefully by browsers usually)
        samesite="lax",
        max_age=7 * 24 * 60 * 60  # 7 days
    )
    
    # 5. Build user profile response
    profile = UserProfile(
        id=user.id,
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        auth_provider=user.auth_provider,
        is_verified=user.is_verified,
        created_at=user.created_at
    )
    
    return AuthTokenResponse(access_token=access_token, user=profile)


from fastapi.responses import RedirectResponse
from app.services.auth.oauth_service import oauth
from app.config import settings

# ============================================================
# OAuth Endpoints (Phase 2 & 3)
# ============================================================

@router.get("/google/login")
async def google_login(request: Request):
    """Redirect to Google OAuth consent screen."""
    if not oauth.google:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured")
        
    # Build absolute callback URL
    redirect_uri = str(request.url_for('google_callback'))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback")
async def google_callback(request: Request, response: Response, session: Session = Depends(get_session)):
    """Handle Google OAuth callback."""
    if not oauth.google:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured")
        
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        logger.error(f"Google OAuth error: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to authenticate with Google")
        
    user_info = token.get('userinfo')
    if not user_info:
        raise HTTPException(status_code=400, detail="Could not fetch user info from Google")
        
    email = user_info.get("email").lower()
    name = user_info.get("name")
    avatar = user_info.get("picture")
    provider_id = user_info.get("sub")
    
    # 1. Find or create user
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        # Register new user
        user = User(
            email=email,
            name=name,
            avatar_url=avatar,
            auth_provider="google",
            provider_id=provider_id,
            is_verified=True
        )
        session.add(user)
        session.commit()
        session.refresh(user)
    else:
        # Update existing user (link OAuth identity if logging in via Google)
        if not user.name:
            user.name = name
        if not user.avatar_url:
            user.avatar_url = avatar
        if user.auth_provider == "email":
            user.auth_provider = "google"
            user.provider_id = provider_id
        session.add(user)
        session.commit()
        session.refresh(user)
        
    # 2. Generate tokens
    access_token = create_access_token(user.id, user.email)
    refresh_token = create_refresh_token(user.id)
    
    # 3. Create a redirect response to the frontend
    redirect_url = f"{settings.FRONTEND_URL}/login/success?token={access_token}"
    redirect_response = RedirectResponse(url=redirect_url)
    
    # 4. Set HttpOnly cookie for refresh token on the response
    redirect_response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=7 * 24 * 60 * 60  # 7 days
    )
    
    return redirect_response


@router.get("/github/login")
async def github_login():
    """Redirect to GitHub OAuth consent screen."""
    # TODO: Phase 3
    raise HTTPException(status_code=501, detail="GitHub OAuth not yet implemented")


@router.get("/github/callback")
async def github_callback(request: Request, session: Session = Depends(get_session)):
    """Handle GitHub OAuth callback."""
    # TODO: Phase 3
    raise HTTPException(status_code=501, detail="GitHub OAuth callback not yet implemented")


# ============================================================
# Session Management
# ============================================================

from app.services.auth.jwt_service import decode_token

@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(request: Request, session: Session = Depends(get_session)):
    """Return the current authenticated user's profile."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.split(" ", 1)[1]
    payload = decode_token(token)
    
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user_id = int(payload["sub"])
    user = session.get(User, user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserProfile(
        id=user.id,
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        auth_provider=user.auth_provider,
        is_verified=user.is_verified,
        created_at=user.created_at
    )


@router.post("/refresh", response_model=AuthTokenResponse)
async def refresh_access_token(request: Request, response: Response, session: Session = Depends(get_session)):
    """Issue a new access token using the refresh token cookie."""
    refresh_tok = request.cookies.get("refresh_token")
    if not refresh_tok:
        raise HTTPException(status_code=401, detail="No refresh token")
    
    payload = decode_token(refresh_tok)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    
    user_id = int(payload["sub"])
    user = session.get(User, user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Issue new access token
    new_access_token = create_access_token(user.id, user.email)
    
    profile = UserProfile(
        id=user.id,
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        auth_provider=user.auth_provider,
        is_verified=user.is_verified,
        created_at=user.created_at
    )
    
    return AuthTokenResponse(access_token=new_access_token, user=profile)


@router.post("/logout", response_model=MessageResponse)
async def logout(response: Response):
    """Clear the refresh token cookie."""
    response.delete_cookie(
        key="refresh_token",
        httponly=True,
        secure=True,
        samesite="lax"
    )
    return MessageResponse(message="Logged out successfully")

