import json
import logging
from app.config import settings
from app.schemas.crawler_schema import WebsiteCrawlResult
from app.schemas.prompt_schema import GeneratedPromptsResponse
from app.services.json_utils import extract_json_object
from app.services.browser_agent.browser_context import BrowserContextManager
from app.services.browser_agent.chatgpt_driver import ChatGPTDriver
from app.services.browser_agent.gemini_driver import GeminiDriver

logger = logging.getLogger(__name__)

PROMPT_GENERATION_INSTRUCTION = """
You are an expert Generative Engine Optimization (GEO) & Answer Engine Optimization (AEO) Strategist.
Analyze the crawled website content below and return ONLY a valid JSON object (no markdown commentary) with this exact structure:

{
  "brand_name": "Exact brand name detected from the website",
  "business_category": "Specific business niche and service domain",
  "business_summary": "2-3 sentence summary of what this business does, its offerings, and target market",
  "key_services": ["Service 1", "Service 2", "Service 3"],
  "detected_location": "City/Region/Country detected on website, or Global",
  "suggested_competitors": ["3 to 5 real peer competitors of similar size in this exact industry and location"],
  "prompts": [
    {
      "id": 1,
      "prompt_text": "Realistic buyer query without mentioning the brand name",
      "intent_category": "Transactional | Feature Specific | Comparison | Pricing | Consultative",
      "reasoning": "Why a buyer asks this"
    }
  ]
}

CRITICAL RULES:
1. Generate EXACTLY 10 prompts numbered id 1 to 10.
2. NEVER INCLUDE THE TARGET BRAND NAME OR WEBSITE DOMAIN IN ANY PROMPT! Real buyers search by problem, service, and city/location before knowing the brand.
3. Base every prompt strictly on what you actually read in the crawled website content (do NOT invent unrelated industries).
4. Include local intent (city/country) in several prompts if a geographic location appears on the site.
5. Output ONLY valid JSON.
"""

class PromptGeneratorService:
    """
    Generates 10 tailored, unbiased buyer prompts and peer competitors strictly from crawled site content.
    Zero hardcoded templates: uses configured LLM APIs or live Browser Agent (ChatGPT/Gemini tab).
    """

    @classmethod
    async def generate_10_prompts(cls, crawl_result: WebsiteCrawlResult) -> GeneratedPromptsResponse:
        user_content = f"""
Website URL: {crawl_result.url}
Page Title: {crawl_result.title}
Meta Description: {crawl_result.meta_description}
Key Headings: {", ".join(crawl_result.headings[:20])}

Crawled Website Content:
{crawl_result.clean_markdown[:5000]}
"""

        # 1. Try Official Gemini API if key is set
        if settings.GEMINI_API_KEY:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                response = client.models.generate_content(
                    model=settings.DEFAULT_GEMINI_MODEL,
                    contents=f"{PROMPT_GENERATION_INSTRUCTION}\n\n{user_content}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeneratedPromptsResponse,
                        temperature=0.2,
                    ),
                )
                return GeneratedPromptsResponse.model_validate(json.loads(response.text))
            except Exception as e:
                logger.warning(f"Gemini API prompt generation skipped/failed: {e}")

        # 2. Try Official OpenAI API if key is set
        if settings.OPENAI_API_KEY:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                completion = await client.beta.chat.completions.parse(
                    model=settings.DEFAULT_OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": PROMPT_GENERATION_INSTRUCTION},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=GeneratedPromptsResponse,
                    temperature=0.2,
                )
                return completion.choices[0].message.parsed
            except Exception as e:
                logger.warning(f"OpenAI API prompt generation skipped/failed: {e}")

        # 3. Live Browser LLM Mode (Zero API Key Needed! Uses real ChatGPT / Gemini web tab)
        full_prompt = f"{PROMPT_GENERATION_INSTRUCTION}\n\n{user_content}"
        return await BrowserContextManager.run(cls._generate_via_browser, full_prompt)

    @classmethod
    async def _generate_via_browser(cls, full_prompt: str) -> GeneratedPromptsResponse:
        ctx = await BrowserContextManager.get_context(headless=True)

        last_err = None
        for driver_name, driver in [("ChatGPT", ChatGPTDriver), ("Gemini", GeminiDriver)]:
            try:
                logger.info(f"Analyzing crawled website via live {driver_name} browser tab...")
                page = await driver.get_or_create_page(ctx)
                raw_text, _ = await driver.execute_query(page, full_prompt, fresh_chat=True, timeout_seconds=75)
                parsed_json = extract_json_object(raw_text)
                validated = GeneratedPromptsResponse.model_validate(parsed_json)
                validated.brand_name = " ".join(validated.brand_name.split())
                validated.business_category = " ".join(validated.business_category.split())
                validated.business_summary = " ".join(validated.business_summary.split())
                validated.detected_location = " ".join(validated.detected_location.split())
                validated.key_services = [" ".join(s.split()) for s in validated.key_services]
                validated.suggested_competitors = [" ".join(c.split()) for c in validated.suggested_competitors]
                validated.prompts = validated.prompts[:10]
                for idx, p in enumerate(validated.prompts, start=1):
                    p.id = idx
                    p.prompt_text = " ".join(p.prompt_text.split())
                    p.intent_category = " ".join(p.intent_category.split())
                    p.reasoning = " ".join(p.reasoning.split())
                return validated
            except Exception as e:
                logger.warning(f"{driver_name} browser prompt generation attempt failed: {type(e).__name__}: {e}")
                last_err = e

        raise RuntimeError(
            f"Could not analyze website using API or Live Browser Agent: {type(last_err).__name__}: {last_err}"
        )
