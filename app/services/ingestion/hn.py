from typing import Iterator
import requests
from datetime import datetime, timezone
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class HackerNewsLoader(BaseLoader):
    """
    Custom LangChain Document Loader to fetch the latest top stories 
    from Hacker News using the Firebase REST API.
    """
    def __init__(self, limit: int = 20):
        self.limit = limit
        self.base_url = "https://hacker-news.firebaseio.com/v0"

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load top stories from Hacker News."""
        try:
            # 1. Fetch top stories list
            top_stories_url = f"{self.base_url}/topstories.json"
            response = requests.get(top_stories_url, timeout=10)
            response.raise_for_status()
            story_ids = response.json()[:self.limit]
            
            # 2. Fetch details for each story ID
            for story_id in story_ids:
                try:
                    item_url = f"{self.base_url}/item/{story_id}.json"
                    item_resp = requests.get(item_url, timeout=5)
                    item_resp.raise_for_status()
                    item = item_resp.json()
                    
                    if not item or item.get("type") != "story":
                        continue
                        
                    title = item.get("title", "")
                    text = item.get("text", "")
                    url = item.get("url")
                    
                    # If story has no external URL, link to the Hacker News comments thread
                    if not url:
                        url = f"https://news.ycombinator.com/item?id={story_id}"
                        
                    # Parse timestamp (seconds since epoch)
                    timestamp = item.get("time")
                    published_at = (
                        datetime.fromtimestamp(timestamp, tz=timezone.utc)
                        if timestamp
                        else datetime.now(timezone.utc)
                    )
                    
                    # Yield structured Document
                    yield Document(
                        page_content=f"{title}\n{text}".strip(),
                        metadata={
                            "source": "Hacker News",
                            "url": url,
                            "published_at": published_at,
                            "hn_id": story_id
                        }
                    )
                except Exception as e:
                    # Skip problematic items and proceed
                    continue
        except Exception as e:
            # Log or yield empty if the API is down
            return
