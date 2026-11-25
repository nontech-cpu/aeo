from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class AgentChoice(str, Enum):
    CHATGPT = "chatgpt"
    GEMINI = "gemini"
    ALL = "all"

class ModelExecutionResult(BaseModel):
    engine: str = Field(..., description="'chatgpt' or 'gemini'")
    prompt_id: int
    prompt_text: str
    response_text: str = Field(..., description="Normalized clean text output from the AI search engine")
    response_html: str = Field(default="", description="Sanitized semantic HTML output preserving lists, headings, and tables")
    cited_sources: List[str] = Field(default_factory=list, description="All citation URLs or grounding links cited")
    error: Optional[str] = None
