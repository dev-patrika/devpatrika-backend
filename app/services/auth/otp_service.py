import hashlib
import secrets
from datetime import datetime, timedelta
from sqlmodel import Session, select
from app.models.otp_code import OTPCode
import logging

logger = logging.getLogger("dev-patrika.auth.otp")

OTP_EXPIRY_MINUTES = 10
MAX_ATTEMPTS = 5

def generate_secure_otp() -> str:
    """Generate a secure 6-digit numeric OTP."""
    # secrets.randbelow(1000000) generates a secure random int between 0 and 999999
    # zfill ensures it's zero-padded to 6 digits (e.g., '004512')
    return str(secrets.randbelow(1000000)).zfill(6)


def hash_otp(otp: str) -> str:
    """Create a SHA-256 hash of the OTP for secure database storage."""
    return hashlib.sha256(otp.encode()).hexdigest()


def create_and_store_otp(session: Session, email: str) -> str:
    """Generate an OTP, store its hash in the DB, and return the plain OTP."""
    # 1. Clean up any existing OTPs for this email to prevent clutter
    existing_otps = session.exec(select(OTPCode).where(OTPCode.email == email)).all()
    for existing in existing_otps:
        session.delete(existing)
    
    # 2. Generate new OTP
    plain_otp = generate_secure_otp()
    otp_hash = hash_otp(plain_otp)
    
    # 3. Save to database
    expires_at = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    db_otp = OTPCode(email=email, otp_hash=otp_hash, expires_at=expires_at)
    session.add(db_otp)
    session.commit()
    
    return plain_otp


def verify_otp_logic(session: Session, email: str, plain_otp: str) -> tuple[bool, str]:
    """
    Verify the provided OTP against the database.
    Returns: (is_valid: bool, error_message: str)
    """
    # 1. Fetch the latest OTP for this email
    db_otp = session.exec(
        select(OTPCode).where(OTPCode.email == email).order_by(OTPCode.created_at.desc())
    ).first()
    
    if not db_otp:
        return False, "No OTP found for this email. Please request a new one."
    
    # 2. Check if expired
    if datetime.utcnow() > db_otp.expires_at:
        return False, "OTP has expired. Please request a new one."
        
    # 3. Check attempt limit
    if db_otp.attempts >= MAX_ATTEMPTS:
        return False, "Too many failed attempts. Please request a new OTP."
        
    # 4. Verify hash
    input_hash = hash_otp(plain_otp)
    if input_hash != db_otp.otp_hash:
        # Increment attempt counter
        db_otp.attempts += 1
        session.add(db_otp)
        session.commit()
        return False, "Invalid OTP code."
        
    # 5. Success! Delete the OTP so it can't be reused
    session.delete(db_otp)
    session.commit()
    
    return True, ""
