from fastapi import APIRouter, Depends
from sqlmodel import Session, text
from app.database import get_session
import logging

logger = logging.getLogger("dev-patrika")
router = APIRouter(prefix="/health", tags=["Health"])

@router.api_route("", methods=["GET", "HEAD"])
def health_check(session: Session = Depends(get_session)):
    try:
        # Execute simple query to verify database connection
        session.exec(text("SELECT 1")).first()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed database verification: {str(e)}")
        return {"status": "error", "database": "disconnected", "detail": str(e)}
