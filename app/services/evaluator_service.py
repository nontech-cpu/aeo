import asyncio
import json
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from app.config import settings
from app.schemas.evaluation_schema import SingleModelEvaluation, PromptAuditItem
from app.schemas.search_schema import ModelExecutionResult
from app.schemas.prompt_schema import SearchPrompt
from app.schemas.report_schema import ActionableFix
from app.services.json_utils import extract_json_object
from app.services.browser_agent.browser_context import BrowserContextManager
from app.services.browser_agent.chatgpt_driver import ChatGPTDriver
from app.services.browser_agent.gemini_driver import GeminiDriver

logger = logging.getLogger(__name__)

class BatchPromptEvalItem(BaseModel):
    prompt_id: int
    chatgpt_brand_present: bool = False
    chatgpt_rank: Optional[int] = None
    chatgpt_sentiment: str = "not_mentioned"
    chatgpt_competitors: List[str] = Field(default_factory=list)
    gemini_brand_present: bool = False
    gemini_rank: Optional[int] = None
    gemini_sentiment: str = "not_mentioned"
    gemini_competitors: List[str] = Field(default_factory=list)

class BatchEvaluationBundle(BaseModel):
    evaluations: List[BatchPromptEvalItem]
    competitor_strategy_analysis: str
    actionable_fixes: List[ActionableFix]

BATCH_EVAL_INSTRUCTION = """
You are an expert AEO/GEO Audit Analyst.
Review the Target Brand, Website Summary, and the live AI Search responses from ChatGPT and Google Gemini below.

Return ONLY a valid JSON object matching this exact schema:
{
  "evaluations": [
    {
      "prompt_id": 1,
      "chatgpt_brand_present": false,
      "chatgpt_rank": null,
      "chatgpt_sentiment": "positive | neutral | critical | not_mentioned",
      "chatgpt_competitors": ["Exact company/brand names recommended by ChatGPT"],
      "gemini_brand_present": false,
      "gemini_rank": null,
      "gemini_sentiment": "positive | neutral | critical | not_mentioned",
      "gemini_competitors": ["Exact company/brand names recommended by Gemini"]
    }
  ],
  "competitor_strategy_analysis": "Detailed 2-paragraph analysis explaining which competitors won most consistently, what patterns/sources gave them visibility (Google reviews, local directories, Clutch, niche landing pages, etc.), and how they compare to our target brand.",
  "actionable_fixes": [
    {
      "priority": "High | Medium | Low",
      "category": "On-Page Content | Technical Schema & llms.txt | Local Citations & Reviews | Digital PR & Directories",
      "issue": "Specific visibility gap found in the audit",
      "fix": "Concrete, practical step-by-step fix for the target website"
    }
  ]
}

Rules:
- Extract ALL real competitor brands/agencies/companies mentioned in each ChatGPT and Gemini answer (never fabricate names).
- Set brand_present = true ONLY if the target brand or domain actually appears in that engine's response.
- Provide 4 to 6 high-impact actionable_fixes tailored strictly to this business and the audit evidence.
- Output ONLY valid JSON.
"""

class LLMEvaluatorService:
    """
    Industry-grade Batch Evaluator & Strategic Fix Generator.
    Uses LLM-as-a-judge (API or Browser Tab) with Pydantic validation and zero regex.
    """

    @classmethod
    async def _evaluate_via_browser(cls, eval_input: str) -> Optional[BatchEvaluationBundle]:
        is_cloud = BrowserContextManager.is_cloud_runtime()
        for driver_name, driver in [("ChatGPT", ChatGPTDriver), ("Gemini", GeminiDriver)]:
            ctx = None
            try:
                if is_cloud:
                    ctx = await BrowserContextManager.create_isolated_context(headless=True)
                    page = await ctx.new_page()
                else:
                    ctx = await BrowserContextManager.get_context(headless=True)
                    page = await driver.get_or_create_page(ctx)

                logger.info(f"Running LLM-as-a-judge batch evaluation via {driver_name} tab...")
                raw_eval_text, _ = await driver.execute_query(page, eval_input, fresh_chat=True, timeout_seconds=90)
                parsed_json = extract_json_object(raw_eval_text)
                return BatchEvaluationBundle.model_validate(parsed_json)
            except Exception as e:
                logger.warning(f"{driver_name} batch evaluation failed: {type(e).__name__}: {e}")
                err_low = str(e).lower()
                if is_cloud or "crashed" in err_low or "closed" in err_low:
                    await BrowserContextManager.close_persistent_context()
                    await BrowserContextManager.restart_browser(headless=True)
            finally:
                if is_cloud and ctx is not None:
                    try:
                        await asyncio.wait_for(ctx.close(), timeout=5)
                    except Exception:
                        pass
        return None

    @classmethod
    async def evaluate_and_strategize(
        cls,
        prompts: List[SearchPrompt],
        search_results: List[ModelExecutionResult],
        target_brand: str,
        target_url: str,
        business_category: str,
        business_summary: str,
        custom_competitors: List[str]
    ) -> Tuple[List[PromptAuditItem], str, List[ActionableFix]]:

        parsed_url = urlparse(target_url)
        target_domain = parsed_url.netloc or parsed_url.path

        # Group raw results by prompt_id
        audit_items: Dict[int, PromptAuditItem] = {
            p.id: PromptAuditItem(
                prompt_id=p.id,
                prompt_text=p.prompt_text,
                intent_category=p.intent_category,
            )
            for p in prompts
        }

        for res in search_results:
            item = audit_items.get(res.prompt_id)
            if not item:
                continue
            if res.engine == "chatgpt":
                item.chatgpt_result = res.response_text
                item.chatgpt_html = res.response_html
                item.chatgpt_sources = res.cited_sources
            elif res.engine == "gemini":
                item.gemini_result = res.response_text
                item.gemini_html = res.response_html
                item.gemini_sources = res.cited_sources

        # Build compact evaluation dossier for batch LLM-as-a-judge analysis
        dossier_lines = [
            f"Target Brand: {target_brand}",
            f"Target Domain: {target_domain}",
            f"Business Category: {business_category}",
            f"Business Summary: {business_summary}",
            f"Custom Competitors Watched: {', '.join(custom_competitors) if custom_competitors else 'None'}",
            "\n--- SEARCH RESULTS PER PROMPT ---",
        ]

        for item in audit_items.values():
            dossier_lines.append(f"\n[Prompt #{item.prompt_id}]: {item.prompt_text}")
            if item.chatgpt_result:
                dossier_lines.append(f"ChatGPT Answer:\n{item.chatgpt_result[:1200]}")
            if item.gemini_result:
                dossier_lines.append(f"Gemini Answer:\n{item.gemini_result[:1200]}")

        eval_input = f"{BATCH_EVAL_INSTRUCTION}\n\n" + "\n".join(dossier_lines)
        bundle: Optional[BatchEvaluationBundle] = None

        # 1. Try Gemini API
        if settings.GEMINI_API_KEY:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                res = client.models.generate_content(
                    model=settings.DEFAULT_GEMINI_MODEL,
                    contents=eval_input,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BatchEvaluationBundle,
                        temperature=0.1,
                    ),
                )
                bundle = BatchEvaluationBundle.model_validate(json.loads(res.text))
            except Exception as e:
                logger.warning(f"Gemini API batch evaluation skipped/failed: {e}")

        # 2. Try OpenAI API
        if bundle is None and settings.OPENAI_API_KEY:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                completion = await client.beta.chat.completions.parse(
                    model=settings.DEFAULT_OPENAI_MODEL,
                    messages=[{"role": "user", "content": eval_input}],
                    response_format=BatchEvaluationBundle,
                    temperature=0.1,
                )
                bundle = completion.choices[0].message.parsed
            except Exception as e:
                logger.warning(f"OpenAI API batch evaluation skipped/failed: {e}")

        # 3. Live Browser LLM Evaluator (Zero API key required!)
        if bundle is None:
            bundle = await BrowserContextManager.run(cls._evaluate_via_browser, eval_input)

        # Apply structured bundle results to PromptAuditItems
        if bundle:
            eval_map = {ev.prompt_id: ev for ev in bundle.evaluations}
            for pid, item in audit_items.items():
                ev = eval_map.get(pid)
                if item.chatgpt_result is not None:
                    item.chatgpt_eval = SingleModelEvaluation(
                        engine="chatgpt",
                        is_brand_present=ev.chatgpt_brand_present if ev else False,
                        mention_rank=ev.chatgpt_rank if ev else None,
                        sentiment=ev.chatgpt_sentiment if ev else "not_mentioned",
                        competitors_cited=ev.chatgpt_competitors if ev else [],
                        key_source_domains=[urlparse(u).netloc for u in item.chatgpt_sources if urlparse(u).netloc],
                    )
                if item.gemini_result is not None:
                    item.gemini_eval = SingleModelEvaluation(
                        engine="gemini",
                        is_brand_present=ev.gemini_brand_present if ev else False,
                        mention_rank=ev.gemini_rank if ev else None,
                        sentiment=ev.gemini_sentiment if ev else "not_mentioned",
                        competitors_cited=ev.gemini_competitors if ev else [],
                        key_source_domains=[urlparse(u).netloc for u in item.gemini_sources if urlparse(u).netloc],
                    )
            return list(audit_items.values()), bundle.competitor_strategy_analysis, bundle.actionable_fixes

        # Minimal token-containment safety net (only if all LLM evaluators fail)
        brand_low = target_brand.lower().strip()
        dom_low = target_domain.lower().replace("www.", "").strip()
        for item in audit_items.values():
            if item.chatgpt_result is not None:
                low = item.chatgpt_result.lower()
                found = (len(brand_low) > 2 and brand_low in low) or (len(dom_low) > 2 and dom_low in low)
                item.chatgpt_eval = SingleModelEvaluation(
                    engine="chatgpt",
                    is_brand_present=found,
                    mention_rank=None,
                    sentiment="positive" if found else "not_mentioned",
                    competitors_cited=[c for c in custom_competitors if c.lower() in low],
                    key_source_domains=[urlparse(u).netloc for u in item.chatgpt_sources if urlparse(u).netloc],
                )
            if item.gemini_result is not None:
                low = item.gemini_result.lower()
                found = (len(brand_low) > 2 and brand_low in low) or (len(dom_low) > 2 and dom_low in low)
                item.gemini_eval = SingleModelEvaluation(
                    engine="gemini",
                    is_brand_present=found,
                    mention_rank=None,
                    sentiment="positive" if found else "not_mentioned",
                    competitors_cited=[c for c in custom_competitors if c.lower() in low],
                    key_source_domains=[urlparse(u).netloc for u in item.gemini_sources if urlparse(u).netloc],
                )

        return (
            list(audit_items.values()),
            "Automated LLM strategy summary could not be generated; see full raw responses below.",
            [],
        )
