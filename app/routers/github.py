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

@router.get("/repo/{repo_id}", response_model=GitHubRadarRead)
def get_single_repo(repo_id: int, session: Session = Depends(get_session)):
    """
    Retrieve details of a single GitHub repository by its database ID.
    """
    from fastapi import HTTPException
    repo = session.get(GitHubRadar, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo

