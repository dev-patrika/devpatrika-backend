from .health import router as health_router
from .news import router as news_router
from .wiki import router as wiki_router
from .github import router as github_router
from .search import router as search_router
from .ai import router as ai_router

# List of all API routers to be registered in main.py
all_routers = [
    (health_router, "/api"),
    (news_router, "/api"),
    (wiki_router, "/api"),
    (github_router, "/api"),
    (search_router, "/api"),
    (ai_router, "/api")
]
