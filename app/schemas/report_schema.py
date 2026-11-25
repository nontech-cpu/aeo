from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from app.schemas.evaluation_schema import PromptAuditItem
from app.schemas.search_schema import AgentChoice

class AuditRequest(BaseModel):
    url: str = Field(..., description="Target website link to audit")
    agent_choice: AgentChoice = Field(default=AgentChoice.ALL, description="Which agents to run: 'chatgpt', 'gemini', or 'all'")
    execution_mode: str = Field(default="browser", description="'browser' for autonomous Playwright multi-tab agent, 'api' for official search APIs")
    
    # Pass already-extracted context from Step 1 so Step 2 doesn't re-generate prompts
    brand_name: Optional[str] = None
    business_category: Optional[str] = None
    business_summary: Optional[str] = None
    prompts_to_run: Optional[List[Dict]] = None
    
    # Custom User Controls
    custom_prompts: List[str] = Field(default_factory=list, description="Additional custom prompts entered by user")
    custom_competitors: List[str] = Field(default_factory=list, description="Competitor website links or brand names added by user")
    selected_prompt_ids: Optional[List[int]] = Field(default=None, description="Subset of auto-prompts selected by user")
    include_google_ai_overview: bool = Field(default=False, description="Optionally run Google AI Overview checks across prompts")
    include_google_page1: bool = Field(default=False, description="Optionally run Google Search Page 1 (Top 10 organic) checks across prompts")

class CompetitorStat(BaseModel):
    name: str
    is_custom_added: bool = False
    mention_count: int
    winning_prompts: List[int] = Field(default_factory=list)

class ActionableFix(BaseModel):
    priority: str = Field(default="High", description="High, Medium, or Low")
    category: str = Field(..., description="e.g., Technical Schema, Digital PR, Content Gap, Local Citations")
    issue: str = Field(..., description="What gap was detected compared to winning competitors")
    fix: str = Field(..., description="Exact actionable steps to fix the website")

class FullAuditReport(BaseModel):
    audit_id: str
    website_url: str
    brand_name: str
    business_category: str
    business_summary: str = ""
    timestamp: str
    agent_choice: str
    execution_mode: str = "browser"
    
    # User additions tracked
    tracked_competitors: List[str] = Field(default_factory=list)
    custom_prompts_count: int = 0
    include_google_ai_overview: bool = False
    include_google_page1: bool = False
    
    # Quantitative Scores
    total_prompts: int
    overall_presence_rate: float = Field(..., description="Percentage of prompt queries where brand was successfully mentioned (0-100%)")
    chatgpt_presence_rate: Optional[float] = None
    gemini_presence_rate: Optional[float] = None
    ai_overview_presence_rate: Optional[float] = None
    google_page1_presence_rate: Optional[float] = None
    
    # Competitive & Citation Intelligence
    top_competitors_winning: List[CompetitorStat] = Field(default_factory=list)
    top_authoritative_sources: List[str] = Field(default_factory=list)
    competitor_strategy_analysis: str = Field(default="", description="Detailed explanation of who won and how they are ranking")
    actionable_fixes: List[ActionableFix] = Field(default_factory=list, description="Clean, prioritized fixes for the user's website")
    
    # Detailed Prompt Audits
    prompt_audits: List[PromptAuditItem] = Field(default_factory=list)
    
    # Ready-to-copy Prompt for ChatGPT / Consultant
    chatgpt_consultant_prompt: str = Field(..., description="Formatted prompt for ChatGPT to get custom recommendations")

class AuditJobStatus(BaseModel):
    audit_id: str
    status: str = Field(..., description="'queued', 'running', 'evaluating', 'completed', 'failed'")
    progress_percent: int = 0
    current_message: str = "Initializing..."
    logs: List[str] = Field(default_factory=list)
    partial_items: List[PromptAuditItem] = Field(default_factory=list, description="Live intermediate responses as prompts complete")
    error: Optional[str] = None
    report: Optional[FullAuditReport] = None
