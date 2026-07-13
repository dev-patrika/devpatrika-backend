from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Optional
from app.schemas.wiki_schema import WikiEntryRead
from sqlmodel import Session, select
from app.database import get_session
from app.models.wiki import WikiEntry

router = APIRouter(prefix="/wiki", tags=["Wiki"])

@router.get("", response_model=List[WikiEntryRead])
def get_wiki_entries(
    q: Optional[str] = Query(default=None, description="Search query for wiki terms"),
    session: Session = Depends(get_session)
):
    """
    Retrieve stored Dev Wiki term definitions.
    Supports autocompleting and filtering terms.
    """
    statement = select(WikiEntry)
    if q:
        statement = statement.where(WikiEntry.term.ilike(f"%{q}%"))
    
    results = session.exec(statement.order_by(WikiEntry.term.asc())).all()
    return results

@router.get("/{term}", response_model=WikiEntryRead)
def get_wiki_entry(term: str, session: Session = Depends(get_session)):
    """
    Retrieve a detailed wiki entry (definition, why it's trending, links) for a concept.
    Matches case-insensitively.
    """
    # Direct check
    statement = select(WikiEntry).where(WikiEntry.term == term)
    result = session.exec(statement).first()
    
    if not result:
        # Case-insensitive fallback lookup
        all_entries = session.exec(select(WikiEntry)).all()
        for entry in all_entries:
            if entry.term.lower() == term.lower():
                return entry
        raise HTTPException(
            status_code=404, 
            detail=f"Wiki definition for term '{term}' not found. You can trigger it via POST /wiki/generate."
        )
    return result

@router.post("/generate", response_model=dict)
def trigger_wiki_generation(
    term: str,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """
    Manually request/trigger the LangChain Wiki curator agent to write a wiki entry for a technical term.
    """
    def run_wiki_generation_in_background():
        from app.database import engine
        from app.services.processing.wiki_generator import generate_wiki_definition
        with Session(engine) as bg_session:
            generate_wiki_definition(term, bg_session)

    background_tasks.add_task(run_wiki_generation_in_background)
    return {
        "status": "wiki_generation_triggered",
        "detail": f"Wiki generation task for term '{term}' has been scheduled to run in the background."
    }

@router.get("/{term}/timeline", response_model=dict)
def get_wiki_timeline(term: str, session: Session = Depends(get_session)):
    """
    Generate a chronological evolution timeline (Announcement -> Adoption -> Production -> Growth)
    for a technical term.
    """
    from app.services.wiki_curator.timeline_generator import generate_technology_timeline
    timeline = generate_technology_timeline(term, session)
    return {"term": term, "timeline": timeline}
