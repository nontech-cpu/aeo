from pydantic import BaseModel, Field
from typing import List

class SearchPrompt(BaseModel):
    id: int = Field(..., description="1-indexed sequence number (1 to 10)")
    prompt_text: str = Field(..., description="Natural language prompt that a high-intent user/buyer would ask ChatGPT or Gemini")
    intent_category: str = Field(..., description="Intent type: Transactional, Feature Specific, Comparison, Pricing, or Consultative")
    reasoning: str = Field(..., description="Why this prompt is crucial for the business")

class GeneratedPromptsResponse(BaseModel):
    brand_name: str = Field(..., description="Auto-detected brand or business name")
    business_category: str = Field(..., description="Specific business niche / service domain")
    business_summary: str = Field(default="", description="Concise summary of what the website does")
    key_services: List[str] = Field(default_factory=list, description="Core offerings or products")
    detected_location: str = Field(default="Global", description="Location or city if geographically specific")
    suggested_competitors: List[str] = Field(default_factory=list, description="3 to 6 peer/direct competitors in the same tier")
    prompts: List[SearchPrompt] = Field(..., min_length=5, max_length=15, description="High-intent search prompts without mentioning the brand name")
