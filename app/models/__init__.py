from .news import NewsItem
from .wiki import WikiEntry
from .github_radar import GitHubRadar
from .notes import PersonalNote

# Export all models so SQLModel metadata registers them
__all__ = ["NewsItem", "WikiEntry", "GitHubRadar", "PersonalNote"]
