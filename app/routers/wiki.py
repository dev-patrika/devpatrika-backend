from fastapi import APIRouter, Depends
from typing import List
from app.schemas.wiki_schema import WikiEntryRead
from sqlmodel import Session
from app.database import get_session

router = APIRouter(prefix="/wiki", tags=["Wiki"])

@router.get("", response_model=List[WikiEntryRead])
def get_wiki_entries(session: Session = Depends(get_session)):
    """
    Placeholder endpoint to retrieve trending terms wiki definitions.
    Fully implemented in v0.4.0.
    """
    return []

@router.get("/{term}", response_model=WikiEntryRead)
def get_wiki_entry(term: str, session: Session = Depends(get_session)):
    """
    Placeholder endpoint to retrieve a single term definition.
    Fully implemented in v0.4.0.
    """
    # Return placeholder error or mock since returning empty won't match WikiEntryRead schema (requires id etc)
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Term '{term}' not found. Wiki generation functionality arrives in v0.4.0.")
