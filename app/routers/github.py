from fastapi import APIRouter, Depends
from typing import List
from app.schemas.github_schema import GitHubRadarRead
from sqlmodel import Session, select
from app.database import get_session
from app.models.github_radar import GitHubRadar

router = APIRouter(prefix="/github", tags=["GitHub Radar"])

@router.get("/trending", response_model=List[GitHubRadarRead])
def get_trending_repos(session: Session = Depends(get_session)):
    """
    Retrieve stored daily trending repositories sorted by star count.
    """
    statement = select(GitHubRadar).order_by(GitHubRadar.stars_count.desc())
    results = session.exec(statement).all()
    return results

