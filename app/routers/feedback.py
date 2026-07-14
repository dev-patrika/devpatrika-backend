from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session
from pydantic import BaseModel, Field

from app.database import get_session
from app.routers.auth import get_current_user_profile
from app.schemas.auth import UserProfile
from app.services.auth.email_service import send_feedback_email

router = APIRouter(prefix="/feedback", tags=["Feedback"])

class FeedbackRequest(BaseModel):
    subject: str = Field(..., min_length=3, max_length=150)
    message: str = Field(..., min_length=10, max_length=5000)

@router.post("", response_model=dict)
async def submit_feedback(
    body: FeedbackRequest,
    request: Request,
    session: Session = Depends(get_session)
):
    """
    Submits user feedback. Automatically fetches the sender's details from the JWT token
    and sends the email directly from the backend.
    """
    try:
        user_profile: UserProfile = await get_current_user_profile(request, session)
    except HTTPException as auth_err:
        raise auth_err
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")
        
    sender_name = user_profile.name or "User"
    sender_email = user_profile.email
    
    success = await send_feedback_email(
        sender_name=sender_name,
        sender_email=sender_email,
        subject=body.subject,
        message=body.message
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to send feedback email. Please try again.")
        
    return {
        "status": "success",
        "detail": "Feedback submitted successfully."
    }
