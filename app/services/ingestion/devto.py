from typing import Iterator, List
import requests
from datetime import datetime, timezone
# pyrefly: ignore [missing-import]
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class DevToLoader(BaseLoader):
    """
    Custom LangChain Document Loader to fetch the latest articles 
    from Dev.to REST API for specified tags.
    """
    def __init__(self, tags: List[str] = None, limit_per_tag: int = 10):
        self.tags = tags or ["python", "javascript", "webdev", "ai", "security", "devops"]
        self.limit = limit_per_tag
        self.base_url = "https://dev.to/api/articles"

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load articles from Dev.to API."""
        for tag in self.tags:
            try:
                url = f"{self.base_url}?tag={tag}&per_page={self.limit}"
                response = requests.get(url, headers={"User-Agent": "Dev-Patrika-Ingestion-Bot"}, timeout=10)
                response.raise_for_status()
                articles = response.json()
                
                for article in articles:
                    title = article.get("title", "")
                    description = article.get("description", "")
                    article_url = article.get("url")
                    
                    # Parse published timestamp (usually ISO 8601 string)
                    pub_str = article.get("published_at") or article.get("created_at")
                    if pub_str:
                        # Dev.to returns string like "2026-07-07T08:52:00Z"
                        # Replace Z with +00:00 for fromisoformat
                        pub_str = pub_str.replace("Z", "+00:00")
                        published_at = datetime.fromisoformat(pub_str)
                    else:
                        published_at = datetime.now(timezone.utc)
                        
                    yield Document(
                        page_content=f"{title}\n{description}".strip(),
                        metadata={
                            "source": "Dev.to",
                            "url": article_url,
                            "published_at": published_at,
                            "devto_id": article.get("id")
                        }
                    )
            except Exception as e:
                # Proceed to the next tag if one fails
                continue
