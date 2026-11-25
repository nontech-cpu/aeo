import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "AEO/GEO Intelligence Engine"
    API_V1_STR: str = "/api/v1"
    
    # LLM & Search Grounding Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")

    # Industry-Standard Google SERP & AI Overview Keys (SerpApi / Serper.dev)
    SERPAPI_API_KEY: Optional[str] = os.getenv("SERPAPI_API_KEY", "")
    SERPER_API_KEY: Optional[str] = os.getenv("SERPER_API_KEY", "")
    
    # Defaults
    DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"
    DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Crawler Settings
    CRAWLER_TIMEOUT_SECONDS: int = 25
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
