import logging
import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from langchain_core.prompts import ChatPromptTemplate
from app.models.wiki import WikiEntry
from app.services.processing.llm import get_llm

logger = logging.getLogger("dev-patrika.processing.wiki_generator")

class WikiDefinition(BaseModel):
    definition: str = Field(
        description="A detailed technical definition of the technology, concept, or tool. Format using clean markdown (can include bolding, code backticks, etc.)."
    )
    why_trending: str = Field(
        description="A 2-3 sentence explanation of why this concept is trending or why developers should care about it right now."
    )
    related_links: List[str] = Field(
        description="A list of 2-3 reputable URLs (e.g. official documentation, GitHub page, or Wikipedia) related to this term."
    )

async def generate_wiki_definition_async(term: str) -> Optional[WikiDefinition]:
    """
    Async LLM generation of a WikiEntry definition.
    Returns the raw Pydantic model without database operations.
    """
    logger.info(f"Generating Wiki definition for term asynchronously: '{term}'")
    try:
        llm = get_llm(temperature=0.2)
        structured_llm = llm.with_structured_output(WikiDefinition)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert software engineering encyclopedia editor. Write a structured wiki page "
                "definition for the requested technical term or library."
            )),
            ("human", (
                "Term: {term}\n\n"
                "Please generate the definition, why it's trending, and related resource links."
            ))
        ])
        
        chain = prompt | structured_llm
        return await chain.ainvoke({"term": term})
    except Exception as e:
        logger.error(f"Failed to async generate wiki entry for '{term}': {str(e)}")
        return None

def generate_wiki_definition(term: str, session: Session) -> Optional[WikiEntry]:
    """
    Generate or update a WikiEntry for a specific technical term.
    Uses the fallback LLM chain to extract structured definitions.
    """
    logger.info(f"Generating Wiki definition for term: '{term}'")
    try:
        llm = get_llm(temperature=0.2)
        structured_llm = llm.with_structured_output(WikiDefinition)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert software engineering encyclopedia editor. Write a structured wiki page "
                "definition for the requested technical term or library."
            )),
            ("human", (
                "Term: {term}\n\n"
                "Please generate the definition, why it's trending, and related resource links."
            ))
        ])
        
        chain = prompt | structured_llm
        result = chain.invoke({"term": term})
        
        if not result:
            logger.error(f"LLM returned empty output for wiki generation of term: '{term}'")
            return None
            
        # Check if term already exists in database (case-insensitive check for term equivalence in Postgres)
        from sqlmodel import func
        statement = select(WikiEntry).where(func.lower(WikiEntry.term) == func.lower(term))
        existing_entry = session.exec(statement).first()
        
        if existing_entry:
            existing_entry.definition = result.definition
            existing_entry.why_trending = result.why_trending
            existing_entry.related_links = result.related_links
            existing_entry.updated_at = datetime.utcnow()
            entry = existing_entry
            logger.info(f"Updated existing WikiEntry for '{term}'")
        else:
            entry = WikiEntry(
                term=term,
                definition=result.definition,
                why_trending=result.why_trending,
                related_links=result.related_links,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            logger.info(f"Created new WikiEntry for '{term}'")
            
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to generate wiki entry for '{term}': {str(e)}")
        return None
