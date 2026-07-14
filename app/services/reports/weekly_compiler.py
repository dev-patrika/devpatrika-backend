import logging
from datetime import datetime, timedelta
from sqlmodel import Session, select
from langchain_core.prompts import ChatPromptTemplate

from app.models.news import NewsItem
from app.models.github_radar import GitHubRadar
from app.models.trending_topic import TrendingTopic
from app.models.weekly_report import WeeklyReport
from app.services.processing.llm import get_llm

logger = logging.getLogger("dev-patrika.reports.weekly_compiler")

def compile_weekly_report(session: Session, days: int = 7) -> WeeklyReport:
    """
    Query tech news, github radar, and trending topics from the last 7 days,
    run the LLM compiler chain, and save a weekly markdown report in the database.
    """
    logger.info("Initializing Weekly AI Report compilation...")
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    try:
        # 1. Fetch top news items from the last 7 days
        news_statement = select(NewsItem).where(NewsItem.created_at >= start_date).where(NewsItem.summary != None)
        news_items = session.exec(news_statement).all()
        
        # 2. Fetch trending repositories
        github_statement = select(GitHubRadar).where(GitHubRadar.created_at >= start_date).order_by(GitHubRadar.stars_count.desc()).limit(10)
        github_items = session.exec(github_statement).all()
        
        # 3. Fetch active trending topics
        trends_statement = select(TrendingTopic).where(TrendingTopic.frequency > 0).order_by(TrendingTopic.frequency.desc()).limit(10)
        trending_topics = session.exec(trends_statement).all()
        
        if not news_items and not github_items:
            logger.warning("Insufficient news and repository data to generate a weekly report. Skipping.")
            return None
            
        # 4. Format information block
        news_blocks = []
        for i, item in enumerate(news_items, 1):
            news_blocks.append(f"{i}. [{item.category}] {item.title}\nSummary: {item.summary}\n")
        news_text = "\n".join(news_blocks)
        
        github_blocks = []
        for i, repo in enumerate(github_items, 1):
            github_blocks.append(f"{i}. {repo.repo_name} - {repo.stars_count} stars\nDescription: {repo.description}\nWhy it matters: {repo.why_it_matters_summary or ''}\n")
        github_text = "\n".join(github_blocks)
        
        trends_blocks = []
        for i, trend in enumerate(trending_topics, 1):
            trends_blocks.append(f"{i}. {trend.term} (Mention frequency: {trend.frequency}, Trend direction: {trend.trend_direction})")
        trends_text = "\n".join(trends_blocks)
        
        # 5. Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are the senior editorial director of Dev Patrika, a premium developer intelligence platform. "
                "Your job is to synthesize raw tech feeds into a beautiful, highly engaging, and professional "
                "weekly developer report in Markdown. "
                "Your audience consists of senior software engineering managers, tech leads, and developers. "
                "Keep the tone professional, insightful, and technical (avoid fluffy marketing speech).\n\n"
                "CRITICAL: Write in extremely simple, direct, and straightforward English. Use plain vocabulary and "
                "simple sentence structures. Do NOT use flowery, verbose, or poetic academic prose. Keep technical explanations "
                "clear, brief, and easy to understand.\n\n"
                "Structure the output exactly into these sections:\n"
                "1. **Weekly Executive Summary**: A concise, 1-paragraph overview highlighting the main shifts/trends in tech this week.\n"
                "2. **Top Tech Stories Deep-Dive**: Select the top 3 stories from the news logs, explain what they are, and add a brief 'Developer Impact Analysis' for each.\n"
                "3. **Trending Open-Source Radar**: Highlight the top 2-3 trending repositories from the logs, explaining their architectural significance.\n"
                "4. **Emerging Glossary / Concepts**: Explain the top trending terms from the logs and how developers can utilize them."
            )),
            ("human", (
                "Report coverage: {start} to {end}\n\n"
                "--- WEEKLY DATA LOGS ---\n\n"
                "News Stories:\n{news_logs}\n\n"
                "GitHub Repositories:\n{github_logs}\n\n"
                "Emerging Tech Topics:\n{trends_logs}\n\n"
                "Please compile this data and output the final markdown report directly."
            ))
        ])
        
        llm = get_llm(temperature=0.2)
        chain = prompt | llm
        
        logger.info("Calling LLM to compile report...")
        result = chain.invoke({
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "news_logs": news_text[:10000],
            "github_logs": github_text[:10000],
            "trends_logs": trends_text[:5000]
        })
        
        report_markdown = result.content if hasattr(result, "content") else str(result)
        
        # 6. Save in database
        report_title = f"Weekly Developer Intelligence Report ({start_date.strftime('%B %d')} - {end_date.strftime('%B %d, %Y')})"
        
        report = WeeklyReport(
            title=report_title,
            content=report_markdown,
            start_date=start_date,
            end_date=end_date,
            created_at=datetime.utcnow()
        )
        
        session.add(report)
        session.commit()
        session.refresh(report)
        
        logger.info(f"Successfully compiled and saved WeeklyReport ID {report.id} in database.")
        return report
    except Exception as e:
        logger.error(f"Failed to compile weekly report: {str(e)}")
        session.rollback()
        return None
