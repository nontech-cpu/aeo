from pydantic import BaseModel, Field
from typing import List, Optional

class SummaryPointRow(BaseModel):
    """Structured concise row extracted from an AI engine's bullet/numbered response."""
    rank: Optional[int] = None
    point_title: str
    summary: str
    is_target_brand: bool = False

class SingleModelEvaluation(BaseModel):
    """
    LLM-as-a-judge + deterministic ground-truth verification output for a single engine's response.
    Zero-regex guarantee: Evaluates brand presence, rank, sentiment, competitors, sources, and summary table rows.
    """
    engine: str
    is_brand_present: bool = Field(..., description="True ONLY if target brand or its exact domain actually appears in answer or cited URLs")
    mention_rank: Optional[int] = Field(None, description="Position/rank if explicitly recommended in a list (e.g. 1, 2, 7). None if not listed.")
    sentiment: str = Field(default="neutral", description="Positive, Neutral, Critical, or Not Mentioned")
    competitors_cited: List[str] = Field(default_factory=list, description="Names of competing companies or alternative brands mentioned in the answer")
    key_source_domains: List[str] = Field(default_factory=list, description="Domains from which the model gathered the answer")
    summary_rows: List[SummaryPointRow] = Field(default_factory=list, description="Structured tabular breakdown of key recommendations/points in the response")

class GooglePage1Result(BaseModel):
    rank: int
    title: str
    url: str
    domain: str
    is_target_brand: bool = False

class GoogleSerpSnapshot(BaseModel):
    """Structured snapshot of Google Search Page 1 Organic + Google AI Overview for one query."""
    status: str = Field(default="ok", description="'ok', 'captcha_blocked', 'api_key_required', or 'error'")
    status_note: str = ""
    ai_overview_supported: bool = True
    has_ai_overview: bool = False
    brand_in_ai_overview: bool = False
    ai_overview_text: Optional[str] = None
    ai_overview_sources: List[str] = Field(default_factory=list)
    brand_page1_rank: Optional[int] = None
    brand_page1_url: Optional[str] = None
    page1_top_results: List[GooglePage1Result] = Field(default_factory=list)

class PromptAuditItem(BaseModel):
    prompt_id: int
    prompt_text: str
    intent_category: str
    
    # Results per AI engine
    chatgpt_result: Optional[str] = None
    chatgpt_html: Optional[str] = None
    chatgpt_sources: List[str] = Field(default_factory=list)
    chatgpt_eval: Optional[SingleModelEvaluation] = None
    
    gemini_result: Optional[str] = None
    gemini_html: Optional[str] = None
    gemini_sources: List[str] = Field(default_factory=list)
    gemini_eval: Optional[SingleModelEvaluation] = None

    # Google AI Overview + Google Page 1 Organic SERP
    google_serp: Optional[GoogleSerpSnapshot] = None
