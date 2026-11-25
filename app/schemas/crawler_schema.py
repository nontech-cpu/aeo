from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Optional

class WebsiteCrawlRequest(BaseModel):
    url: str = Field(..., description="Target website URL to crawl and analyze")

class WebsiteCrawlResult(BaseModel):
    url: str
    title: str = ""
    meta_description: str = ""
    headings: List[str] = Field(default_factory=list)
    clean_markdown: str = Field(..., description="Main content extracted in clean Markdown format")
    extracted_metadata: Dict[str, str] = Field(default_factory=dict)
    character_count: int = 0
