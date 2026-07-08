import logging
from sqlmodel import Session, select
from langchain_core.prompts import ChatPromptTemplate

from app.models.news import NewsItem
from app.models.wiki import WikiEntry
from app.services.processing.llm import get_llm

logger = logging.getLogger("dev-patrika.wiki_curator.timeline_generator")

def generate_technology_timeline(term: str, session: Session) -> str:
    """
    Constructs a chronological evolution timeline (Announcement -> Adoption -> Production -> Growth)
    for a given technology term, grounded in local DB references and LLM intelligence.
    """
    logger.info(f"Generating evolution timeline for: '{term}'...")
    
    # 1. Fetch matching local database context
    wiki_statement = select(WikiEntry).where(WikiEntry.term.like(f"%{term}%"))
    wiki_matches = session.exec(wiki_statement).all()
    
    news_statement = select(NewsItem).where(
        NewsItem.title.like(f"%{term}%") |
        (NewsItem.summary.like(f"%{term}%") if NewsItem.summary is not None else False)
    ).limit(5)
    news_matches = session.exec(news_statement).all()
    
    # 2. Format context text block
    context_blocks = []
    if wiki_matches:
        context_blocks.append("--- Wiki Glossary Concept definitions ---")
        for w in wiki_matches:
            context_blocks.append(f"Term: {w.term}\nDefinition: {w.definition}\nWhy Trending: {w.why_trending}\n")
            
    if news_matches:
        context_blocks.append("--- Recent News Articles mentions ---")
        for n in news_matches:
            context_blocks.append(f"Title: {n.title}\nSummary: {n.summary}\n")
            
    context_text = "\n".join(context_blocks)
    
    # 3. Prompt LLM to compile the timeline
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert technology historian and developer advocate. "
                "Analyze the provided local database references (if any) and compile a highly structured, "
                "professional, and insightful technology evolution timeline for: '{term}'. "
                "Describe the progression in 4 key phases:\n"
                "1. **Phase 1: Announcement (Origin & Concepts)** - The launch, problems solved, initial code architecture.\n"
                "2. **Phase 2: Adoption (Community & Open Source Interest)** - Early developer reception, GitHub traction, initial frameworks.\n"
                "3. **Phase 3: Production (Stability & Enterprise Readiness)** - Integration milestones, production stability, commercial platforms.\n"
                "4. **Phase 4: Growth & Future Ecosystem (Evolution)** - Newest releases, integrations (e.g. MCP, agent architectures), and long-term outlook.\n\n"
                "Format the response strictly in Markdown under appropriate headings. Supplement with your parametric knowledge where needed."
            )),
            ("human", (
                "Technology Term: {term}\n\n"
                "Local DB context references:\n{context}\n\n"
                "Please generate the Markdown technology timeline."
            ))
        ])
        
        llm = get_llm(temperature=0.2)
        chain = prompt | llm
        
        result = chain.invoke({
            "term": term,
            "context": context_text if context_text else "No local database entries found."
        })
        
        timeline_markdown = result.content if hasattr(result, "content") else str(result)
        logger.info(f"Successfully generated evolution timeline for '{term}'.")
        return timeline_markdown
    except Exception as e:
        logger.error(f"Failed to generate timeline for '{term}': {str(e)}")
        return f"### Technology Timeline Error\nFailed to compile evolution stages for *{term}* due to LLM error: {str(e)}"
