from typing import Iterator
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class ArxivLoader(BaseLoader):
    """
    Custom LangChain Document Loader to fetch the latest CS / ML preprints 
    from arXiv via their public Atom feed/API.
    """
    def __init__(self, categories: list = None, max_results: int = 15):
        self.categories = categories or ["cs.AI", "cs.LG", "cs.SE"]
        self.max_results = max_results
        self.base_url = "http://export.arxiv.org/api/query"

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load preprints from arXiv."""
        # Join categories with OR logic
        cat_query = " OR ".join(f"cat:{cat}" for cat in self.categories)
        query_url = (
            f"{self.base_url}?search_query={cat_query}"
            f"&sortBy=submittedDate&sortOrder=descending"
            f"&max_results={self.max_results}"
        )
        
        try:
            response = requests.get(query_url, timeout=15)
            response.raise_for_status()
            
            # Parse Atom XML
            root = ET.fromstring(response.content)
            
            # Atom namespace prefix is usually {http://www.w3.org/2005/Atom}
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                id_elem = entry.find("atom:id", ns)  # Link to abstract page
                pub_elem = entry.find("atom:published", ns)
                
                title = title_elem.text.strip() if title_elem is not None else ""
                # Clean up excess whitespace and newlines from the abstract
                summary = " ".join(summary_elem.text.split()) if summary_elem is not None else ""
                url = id_elem.text.strip() if id_elem is not None else ""
                
                # Check for direct PDF link under alternate links
                pdf_url = url
                for link in entry.findall("atom:link", ns):
                    if link.attrib.get("title") == "pdf" or "pdf" in link.attrib.get("href", ""):
                        pdf_url = link.attrib.get("href", "")
                        break
                
                # Parse published datetime
                published_at = datetime.now(timezone.utc)
                if pub_elem is not None:
                    try:
                        # arXiv returns strings like "2026-07-06T15:20:00Z"
                        pub_str = pub_elem.text.strip().replace("Z", "+00:00")
                        published_at = datetime.fromisoformat(pub_str)
                    except ValueError:
                        pass
                
                yield Document(
                    page_content=f"{title}\n{summary}".strip(),
                    metadata={
                        "source": "arXiv",
                        "url": pdf_url or url,
                        "published_at": published_at,
                        "arxiv_id": url.split("/abs/")[-1] if "/abs/" in url else url
                    }
                )
        except Exception as e:
            return
