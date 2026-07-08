from .news import NewsItem
from .wiki import WikiEntry
from .github_radar import GitHubRadar
from .notes import PersonalNote
from .weekly_report import WeeklyReport
from .trending_topic import TrendingTopic
from .chat_history import ChatMessage

# Export all models so SQLModel metadata registers them
__all__ = [
    "NewsItem", 
    "WikiEntry", 
    "GitHubRadar", 
    "PersonalNote",
    "WeeklyReport",
    "TrendingTopic",
    "ChatMessage"
]
