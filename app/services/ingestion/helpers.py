import re
from datetime import datetime, timezone
from typing import Set
from app.core.constants import TechCategory

def get_freshness_tag(published_at: datetime) -> str:
    """
    Generate a human-readable relative time tag.
    E.g., '10 minutes ago', '4 hours ago', 'Yesterday', or '3 days ago'.
    """
    if not published_at:
        return "Updated recently"
    
    # Ensure published_at is timezone-naive UTC for comparison
    if published_at.tzinfo is not None:
        published_at = published_at.astimezone(timezone.utc).replace(tzinfo=None)
        
    now = datetime.utcnow()
    diff = now - published_at
    
    seconds = diff.total_seconds()
    if seconds < 0:
        return "Just now"
    
    minutes = int(seconds // 60)
    hours = int(seconds // 3600)
    days = diff.days
    
    if minutes < 60:
        if minutes <= 1:
            return "Just now"
        return f"{minutes} minutes ago"
    elif hours < 24:
        if hours == 1:
            return "1 hour ago"
        return f"{hours} hours ago"
    elif days < 2:
        return "Yesterday"
    else:
        return f"{days} days ago"

def _tokenize(text: str) -> Set[str]:
    """Clean text and split into a set of lowercased alphanumeric words."""
    if not text:
        return set()
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return set(cleaned.split())

def check_title_similarity(title1: str, title2: str) -> float:
    """
    Calculate Jaccard similarity score between two titles.
    Returns a float between 0.0 and 1.0.
    """
    words1 = _tokenize(title1)
    words2 = _tokenize(title2)
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def map_category_by_keywords(title: str, description: str) -> TechCategory:
    """
    Assign a TechCategory based on keyword matching in title and description.
    Defaults to TechCategory.OPEN_SOURCE if no specific category is matched.
    """
    text = f"{title or ''} {description or ''}".lower()
    
    # Priority mapping matching our categories: AI, Web Dev, Cybersecurity, Startups, Open Source, Cloud/DevOps
    
    # 1. Cybersecurity keywords
    cyber_keywords = ["vulnerability", "exploit", "cve-", "malware", "hack", "breach", "zero-day", "ransomware", "security bypass", "phishing", "encryption", "auth"]
    if any(k in text for k in cyber_keywords):
        return TechCategory.CYBERSECURITY

    # 2. AI keywords
    ai_keywords = ["ai", "ml", "llm", "gpt", "deep learning", "neural network", "transformer", "claude", "gemini", "model weights", "parameters", "fine-tuning", "rag", "artificial intelligence"]
    # Be careful to match 'ai' as a word, not as a substring inside words like "again" or "main"
    words = set(re.findall(r"\b\w+\b", text))
    if any(k in words for k in ["ai", "ml", "llm", "gpt"]) or any(k in text for k in ["deep learning", "neural network", "transformer", "artificial intelligence", "fine-tuning"]):
        return TechCategory.AI

    # 3. Cloud/DevOps keywords
    cloud_keywords = ["docker", "kubernetes", "k8s", "aws", "azure", "gcp", "cloud", "devops", "terraform", "ci/cd", "ansible", "jenkins", "pipeline", "serverless", "microservices"]
    if any(k in text for k in cloud_keywords):
        return TechCategory.CLOUD_DEVOPS

    # 4. Web Dev keywords
    web_keywords = ["react", "next.js", "nextjs", "vue", "angular", "javascript", "typescript", "css", "html", "frontend", "tailwindcss", "svelte", "node.js", "nodejs", "web dev", "django", "fastapi"]
    if any(k in text for k in web_keywords):
        return TechCategory.WEB_DEV

    # 5. Startups keywords
    startup_keywords = ["funding", "startup", "venture", "seed round", "raised $", "acquisition", "acquire", "ipo", "y combinator", "yc"]
    if any(k in text for k in startup_keywords):
        return TechCategory.STARTUPS

    # 6. Default to Open Source
    return TechCategory.OPEN_SOURCE
