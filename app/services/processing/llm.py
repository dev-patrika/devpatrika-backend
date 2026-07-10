import os
import logging
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings

logger = logging.getLogger("dev-patrika.processing.llm")

def get_llm(temperature: float = 0.0):
    """
    Initialize LLM engines with fallback mechanism.
    Defaults to Groq (openai/gpt-oss-120b) and falls back to Gemini (gemini-2.5-flash).
    """
    # Sync settings keys to environment variables for LangChain internal resolution if needed
    if settings.GROQ_API_KEY and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
    if settings.GEMINI_API_KEY and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY

    # Retrieve keys from settings or environment
    groq_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    gemini_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")

    if not groq_key and not gemini_key:
        logger.warning("No API keys found for either Groq or Gemini in settings/environment.")

    # Initialize Groq Llama 3 Model
    # Use placeholder if key is missing to avoid instant instantiation crashes, allowing fallback to try next
    groq_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        api_key=groq_key or "placeholder_key"
    )

    # Initialize Gemini Flash Model
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        api_key=gemini_key or "placeholder_key"
    )

    # Return LLM chain with fallback to Gemini, catching all exceptions (including Groq-specific rate limit errors)
    return groq_llm.with_fallbacks([gemini_llm], exceptions_to_handle=(Exception,))
