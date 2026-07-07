from fastapi import APIRouter, Depends
from typing import List
from app.schemas.github_schema import GitHubRadarRead
from sqlmodel import Session
from app.database import get_session

router = APIRouter(prefix="/github", tags=["GitHub Radar"])

@router.get("/trending", response_model=List[GitHubRadarRead])
def get_trending_repos(session: Session = Depends(get_session)):
    """
    Placeholder endpoint to retrieve trending repositories and "why they matter".
    Fully implemented in v0.2.0.
    """
    return []
