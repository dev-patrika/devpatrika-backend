from enum import Enum

class TechCategory(str, Enum):
    AI = "AI"
    WEB_DEV = "Web Dev"
    CYBERSECURITY = "Cybersecurity"
    STARTUPS = "Startups"
    OPEN_SOURCE = "Open Source"
    CLOUD_DEVOPS = "Cloud/DevOps"

class NewsSource(str, Enum):
    HACKER_NEWS = "Hacker News"
    DEV_TO = "Dev.to"
    GITHUB_TRENDING = "GitHub Trending"
    ARXIV = "arXiv"
