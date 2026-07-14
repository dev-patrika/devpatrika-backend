import httpx
import logging
from app.config import settings

logger = logging.getLogger("dev-patrika.auth.email")

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"

async def send_otp_email(to_email: str, otp_code: str) -> bool:
    """
    Send an OTP email via Brevo (Sendinblue) API.
    Falls back to console output if no API key is present or DEBUG=True.
    """
    # Local dev fallback (only if API key is not provided)
    if not settings.BREVO_API_KEY:
        logger.info(f"\n{'='*40}\n[DEV FALLBACK] OTP for {to_email}: {otp_code}\n{'='*40}")
        return True

    payload = {
        "sender": {"email": settings.BREVO_SENDER_EMAIL, "name": "Dev Patrika"},
        "to": [{"email": to_email}],
        "subject": f"Your Dev Patrika Login Code: {otp_code}",
        "htmlContent": f"""do
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2>Welcome to Dev Patrika!</h2>
            <p>Your one-time login code is:</p>
            <h1 style="color: #4f46e5; letter-spacing: 4px;">{otp_code}</h1>
            <p>This code will expire in 10 minutes. If you did not request this, please ignore this email.</p>
        </div>
        """
    }

    headers = {
        "accept": "application/json",
        "api-key": settings.BREVO_API_KEY,
        "content-type": "application/json"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(BREVO_API_URL, json=payload, headers=headers)
            
            if response.status_code in (200, 201, 202):
                logger.info(f"OTP email sent successfully to {to_email}")
                return True
            else:
                logger.error(f"Failed to send email via Brevo: {response.text}")
                return False
    except Exception as e:
        logger.error(f"Error calling Brevo API: {str(e)}")
        return False


async def send_feedback_email(sender_name: str, sender_email: str, subject: str, message: str) -> bool:
    """
    Send user feedback details to the BREVO_SENDER_EMAIL.
    Falls back to console output if no API key is present.
    """
    to_email = settings.BREVO_SENDER_EMAIL or "i.e.ishantiwari@gmail.com"
    
    if not settings.BREVO_API_KEY:
        logger.info(f"\n{'='*40}\n[DEV FALLBACK] Feedback from {sender_name} ({sender_email}):\nSubject: {subject}\nMessage: {message}\n{'='*40}")
        return True

    payload = {
        "sender": {"email": settings.BREVO_SENDER_EMAIL, "name": "Dev Patrika Feedback"},
        "to": [{"email": to_email}],
        "subject": f"Dev Patrika User Feedback: {subject}",
        "htmlContent": f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e5e7eb; border-radius: 12px;">
            <h2 style="color: #4f46e5; border-bottom: 1px solid #e5e7eb; padding-bottom: 10px;">New Feedback Received</h2>
            <p><strong>From User:</strong> {sender_name or 'Anonymous'} ({sender_email})</p>
            <p><strong>Subject:</strong> {subject}</p>
            <div style="background-color: #f9fafb; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #4f46e5;">
                <p style="margin: 0; white-space: pre-wrap;">{message}</p>
            </div>
        </div>
        """
    }

    headers = {
        "accept": "application/json",
        "api-key": settings.BREVO_API_KEY,
        "content-type": "application/json"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(BREVO_API_URL, json=payload, headers=headers)
            if response.status_code in (200, 201, 202):
                logger.info(f"Feedback email sent successfully to {to_email}")
                return True
            else:
                logger.error(f"Failed to send feedback email: {response.text}")
                return False
    except Exception as e:
        logger.error(f"Error calling Brevo API for feedback: {str(e)}")
        return False

