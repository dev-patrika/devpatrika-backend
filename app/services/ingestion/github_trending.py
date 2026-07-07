from typing import Iterator
import requests
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger("dev-patrika.ingestion.github")

class GitHubTrendingLoader(BaseLoader):
    """
    Custom LangChain Document Loader to scrape daily trending repositories 
    from GitHub Trending.
    """
    def __init__(self, since: str = "daily"):
        # since can be "daily", "weekly", or "monthly"
        self.since = since
        self.url = f"https://github.com/trending?since={since}"

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load trending repositories from GitHub."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        try:
            response = requests.get(self.url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            articles = soup.select("article.Box-row")
            
            for article in articles:
                # 1. Parse repository name and path
                link_elem = article.select_one("h2 a") or article.select_one("h1 a")
                if not link_elem:
                    continue
                
                repo_path = link_elem.get("href", "").strip()
                if not repo_path.startswith("/"):
                    # Fallback check
                    repo_path = "/" + repo_path
                
                repo_url = f"https://github.com/trending{repo_path}" if "github.com" in repo_path else f"https://github.com{repo_path}"
                
                # Cleanup repo name (e.g., "owner / name" -> "owner/name")
                repo_name = link_elem.get_text(strip=True).replace(" ", "").replace("\n", "")
                
                # 2. Parse description
                desc_elem = article.select_one("p.col-9")
                description = desc_elem.get_text(strip=True) if desc_elem else ""
                
                # 3. Parse stars count
                stars = 0
                stars_elem = article.select_one('a[href*="/stargazers"]')
                if not stars_elem:
                    # Fallback to class search
                    muted_links = article.select("a.Link--muted")
                    if muted_links:
                        stars_elem = muted_links[0]
                        
                if stars_elem:
                    stars_text = stars_elem.get_text(strip=True).replace(",", "")
                    if "k" in stars_text.lower():
                        try:
                            stars = int(float(stars_text.lower().replace("k", "")) * 1000)
                        except ValueError:
                            pass
                    else:
                        try:
                            stars = int(stars_text)
                        except ValueError:
                            pass
                
                # 4. Parse language
                lang_elem = article.select_one('[itemprop="programmingLanguage"]')
                language = lang_elem.get_text(strip=True) if lang_elem else "Unknown"
                
                yield Document(
                    page_content=f"Repository: {repo_name}\nDescription: {description}\nLanguage: {language}".strip(),
                    metadata={
                        "source": "GitHub Trending",
                        "repo_name": repo_name,
                        "url": repo_url,
                        "stars_count": stars,
                        "language": language,
                        "published_at": datetime.now(timezone.utc)
                    }
                )
        except Exception as e:
            logger.error(f"Error while scraping GitHub Trending: {str(e)}")
            return
