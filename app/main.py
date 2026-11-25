import asyncio
import uuid
import logging
from datetime import datetime
from typing import Dict, List
from urllib.parse import urlparse
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas.crawler_schema import WebsiteCrawlRequest
from app.schemas.prompt_schema import GeneratedPromptsResponse, SearchPrompt
from app.schemas.evaluation_schema import PromptAuditItem
from app.schemas.report_schema import AuditRequest, FullAuditReport, CompetitorStat, AuditJobStatus
from app.schemas.search_schema import AgentChoice, ModelExecutionResult
from app.services.crawler_service import WebsiteCrawler
from app.services.prompt_service import PromptGeneratorService
from app.services.search_service import MultiEngineSearchService
from app.services.browser_agent.browser_context import BrowserContextManager
from app.services.browser_agent.runner import BrowserAuditRunner
from app.services.evaluator_service import LLMEvaluatorService
from app.services.pdf_service import PDFReportGenerator
from app.services.serp_service import GoogleSerpService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await BrowserContextManager.close()

app = FastAPI(title=settings.PROJECT_NAME, version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracking & completed audit reports store
AUDIT_JOBS: Dict[str, AuditJobStatus] = {}
AUDIT_REPORTS_STORE: Dict[str, FullAuditReport] = {}
BACKGROUND_TASKS: Dict[str, asyncio.Task] = {}

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the modern, single-page Hermes AEO audit dashboard."""
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend template not found")
    return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))

@app.get("/health")
async def health_check():
    """Lightweight health check endpoint for Railway and UptimeRobot."""
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": "2.0.0",
    }

@app.post("/api/v1/crawl-and-suggest", response_model=GeneratedPromptsResponse)
async def crawl_and_suggest_prompts(payload: WebsiteCrawlRequest):
    """
    Step 1: Deeply crawls target website (with JS rendering & internal sub-pages),
    and generates 10 real, custom prompts + suggested peer competitors.
    """
    try:
        logger.info(f"Initiating deep multi-page crawl for: {payload.url}")
        crawl_result = await WebsiteCrawler.crawl(payload.url)

        logger.info(f"Analyzing website '{crawl_result.title}' ({crawl_result.character_count} chars) to formulate 10 custom prompts...")
        prompts_response = await PromptGeneratorService.generate_10_prompts(crawl_result)
        return prompts_response
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e) or repr(e)}"
        logger.error(f"Error in crawl-and-suggest: {err_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to crawl and analyze website: {err_msg}")

async def _execute_audit_job(audit_id: str, payload: AuditRequest):
    """Background worker orchestrating multi-tab browser execution, evaluation, and report synthesis."""
    job = AUDIT_JOBS[audit_id]

    def update_progress(pct: int, msg: str):
        job.progress_percent = pct
        job.current_message = msg
        job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    try:
        job.status = "running"
        update_progress(2, f"Preparing audit for {payload.url}...")

        # Reuse Step 1 prompts/metadata when passed by frontend; otherwise crawl fresh
        brand_name = payload.brand_name
        business_category = payload.business_category
        business_summary = payload.business_summary or ""
        base_prompts: List[SearchPrompt] = []

        if payload.prompts_to_run and brand_name and business_category:
            base_prompts = [SearchPrompt.model_validate(p) for p in payload.prompts_to_run]
        else:
            update_progress(4, "Crawling website & generating 10 custom prompts...")
            crawl_res = await WebsiteCrawler.crawl(payload.url)
            meta = await PromptGeneratorService.generate_10_prompts(crawl_res)
            brand_name = meta.brand_name
            business_category = meta.business_category
            business_summary = meta.business_summary
            base_prompts = meta.prompts

        # Filter selected prompts
        active_prompts: List[SearchPrompt] = []
        if payload.selected_prompt_ids is not None:
            active_prompts = [p for p in base_prompts if p.id in payload.selected_prompt_ids]
        else:
            active_prompts = list(base_prompts)

        # Append custom user prompts
        next_id = max([p.id for p in active_prompts], default=0) + 1
        for cp in payload.custom_prompts:
            if cp.strip():
                active_prompts.append(SearchPrompt(
                    id=next_id,
                    prompt_text=cp.strip(),
                    intent_category="Custom User Query",
                    reasoning="Added manually by user",
                ))
                next_id += 1

        if not active_prompts:
            raise ValueError("No prompts selected for audit.")

        # Track partial live items as each prompt completes
        partial_map: Dict[int, PromptAuditItem] = {
            p.id: PromptAuditItem(prompt_id=p.id, prompt_text=p.prompt_text, intent_category=p.intent_category)
            for p in active_prompts
        }

        def on_partial_result(res: ModelExecutionResult):
            item = partial_map.get(res.prompt_id)
            if not item:
                return
            if res.engine == "chatgpt":
                item.chatgpt_result = res.response_text
                item.chatgpt_html = res.response_html
                item.chatgpt_sources = res.cited_sources
            elif res.engine == "gemini":
                item.gemini_result = res.response_text
                item.gemini_html = res.response_html
                item.gemini_sources = res.cited_sources
            job.partial_items = [
                it for it in partial_map.values()
                if it.chatgpt_result is not None or it.gemini_result is not None
            ]

        # Execute AI searches AND Google AI Overview + Page 1 SERP checks concurrently
        async def _run_ai_searches():
            if payload.execution_mode == "api":
                update_progress(15, f"Executing {len(active_prompts)} prompts via official Search Grounding APIs...")
                search_engine = MultiEngineSearchService(concurrency_limit=4)
                return await search_engine.execute_batch(active_prompts, payload.agent_choice)
            return await BrowserAuditRunner.run_browser_audit(
                prompts=active_prompts,
                agent_choice=payload.agent_choice,
                headless=True,
                concurrency_per_engine=2,
                on_progress=update_progress,
                on_result=on_partial_result,
            )

        async def _run_google_serp():
            if not (payload.include_google_ai_overview or payload.include_google_page1):
                return {}
            update_progress(12, "Checking selected Google Search visibility options across all queries...")
            return await GoogleSerpService.check_queries(
                active_prompts,
                brand_name,
                payload.url,
                include_ai_overview=payload.include_google_ai_overview,
                include_page1=payload.include_google_page1,
            )

        raw_results, serp_snapshots = await asyncio.gather(_run_ai_searches(), _run_google_serp())

        # Evaluate all responses + generate competitive insights & clean fixes
        job.status = "evaluating"
        update_progress(92, "Evaluating all responses, extracting competitors, and generating clean fixes...")
        prompt_audits, strategy_analysis, actionable_fixes = await LLMEvaluatorService.evaluate_and_strategize(
            prompts=active_prompts,
            search_results=raw_results,
            target_brand=brand_name,
            target_url=payload.url,
            business_category=business_category,
            business_summary=business_summary,
            custom_competitors=payload.custom_competitors,
        )

        # Calculate quantitative scorecards (including Google AI Overview & Page 1)
        total = len(prompt_audits)
        chatgpt_hits = 0
        gemini_hits = 0
        overall_hits = 0
        ai_ov_hits = 0
        ai_ov_checked = 0
        ai_ov_supported_count = 0
        page1_hits = 0
        serp_ok_count = 0

        competitor_counter: Dict[str, CompetitorStat] = {
            c: CompetitorStat(name=c, is_custom_added=True, mention_count=0)
            for c in payload.custom_competitors
        }
        authoritative_sources = set()

        for item in prompt_audits:
            serp = serp_snapshots.get(item.prompt_id)
            item.google_serp = serp

            hit = False
            if item.chatgpt_eval and item.chatgpt_eval.is_brand_present:
                chatgpt_hits += 1
                hit = True
            if item.gemini_eval and item.gemini_eval.is_brand_present:
                gemini_hits += 1
                hit = True
            if serp and serp.status == "ok":
                serp_ok_count += 1
                if payload.include_google_ai_overview and serp.ai_overview_supported:
                    ai_ov_supported_count += 1
                    if serp.has_ai_overview:
                        ai_ov_checked += 1
                        if serp.brand_in_ai_overview:
                            ai_ov_hits += 1
                            hit = True
                if payload.include_google_page1 and serp.brand_page1_rank is not None:
                    page1_hits += 1
                if payload.include_google_ai_overview:
                    for s in serp.ai_overview_sources:
                        dom = urlparse(s).netloc
                        if dom:
                            authoritative_sources.add(dom)
            if hit:
                overall_hits += 1

            for ev in [item.chatgpt_eval, item.gemini_eval]:
                if ev and ev.competitors_cited:
                    for c in ev.competitors_cited:
                        c_clean = c.strip()
                        if not c_clean:
                            continue
                        if c_clean not in competitor_counter:
                            competitor_counter[c_clean] = CompetitorStat(name=c_clean, is_custom_added=False, mention_count=0)
                        competitor_counter[c_clean].mention_count += 1
                        if item.prompt_id not in competitor_counter[c_clean].winning_prompts:
                            competitor_counter[c_clean].winning_prompts.append(item.prompt_id)

            for s in item.chatgpt_sources + item.gemini_sources:
                domain = urlparse(s).netloc
                if domain:
                    authoritative_sources.add(domain)

        overall_rate = (overall_hits / total * 100.0) if total > 0 else 0.0
        chatgpt_rate = (chatgpt_hits / total * 100.0) if total > 0 and payload.agent_choice in [AgentChoice.CHATGPT, AgentChoice.ALL] else None
        gemini_rate = (gemini_hits / total * 100.0) if total > 0 and payload.agent_choice in [AgentChoice.GEMINI, AgentChoice.ALL] else None
        ai_ov_rate = (
            (ai_ov_hits / ai_ov_checked * 100.0)
            if payload.include_google_ai_overview and ai_ov_checked > 0
            else (0.0 if payload.include_google_ai_overview and ai_ov_supported_count > 0 else None)
        )
        page1_rate = (page1_hits / serp_ok_count * 100.0) if payload.include_google_page1 and serp_ok_count > 0 else None

        sorted_comps = sorted(competitor_counter.values(), key=lambda x: x.mention_count, reverse=True)[:10]

        summary_lines = [f"- Overall AI Visibility: {overall_rate:.1f}%"]
        if payload.include_google_ai_overview:
            summary_lines.append(
                f"- Google AI Overview Presence: {f'{ai_ov_rate:.1f}%' if ai_ov_rate is not None else 'Not available (requires SERPAPI_API_KEY)'}"
            )
        if payload.include_google_page1:
            summary_lines.append(
                f"- Google Page 1 Organic Presence: {f'{page1_rate:.1f}%' if page1_rate is not None else 'Not available'}"
            )
        summary_lines.extend([
            f"- ChatGPT Presence: {f'{chatgpt_rate:.1f}%' if chatgpt_rate is not None else 'N/A'}",
            f"- Google Gemini Presence: {f'{gemini_rate:.1f}%' if gemini_rate is not None else 'N/A'}",
            f"- Top Winning Competitors: {', '.join([c.name for c in sorted_comps[:5]]) if sorted_comps else 'None'}",
            f"- Cited Sources: {', '.join(list(authoritative_sources)[:6]) if authoritative_sources else 'None'}",
        ])

        consultant_prompt = (
            f"I ran a complete AEO/GEO Search Audit on my website ({payload.url}) for brand '{brand_name}' ({business_category}).\n\n"
            f"Audit Summary:\n" + "\n".join(summary_lines) + "\n\n"
            f"Competitive Analysis Detected:\n{strategy_analysis}\n\n"
            f"Please review the attached full audit PDF and give me exact code (JSON-LD schema, llms.txt), content rewrites, and PR/directory steps to outrank these competitors."
        )

        report = FullAuditReport(
            audit_id=audit_id,
            website_url=payload.url,
            brand_name=brand_name,
            business_category=business_category,
            business_summary=business_summary,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            agent_choice=payload.agent_choice.value,
            execution_mode=payload.execution_mode,
            tracked_competitors=payload.custom_competitors,
            custom_prompts_count=len(payload.custom_prompts),
            include_google_ai_overview=payload.include_google_ai_overview,
            include_google_page1=payload.include_google_page1,
            total_prompts=total,
            overall_presence_rate=overall_rate,
            chatgpt_presence_rate=chatgpt_rate,
            gemini_presence_rate=gemini_rate,
            ai_overview_presence_rate=ai_ov_rate,
            google_page1_presence_rate=page1_rate,
            top_competitors_winning=sorted_comps,
            top_authoritative_sources=list(authoritative_sources)[:15],
            competitor_strategy_analysis=strategy_analysis,
            actionable_fixes=actionable_fixes,
            prompt_audits=prompt_audits,
            chatgpt_consultant_prompt=consultant_prompt,
        )

        AUDIT_REPORTS_STORE[audit_id] = report
        job.report = report
        job.status = "completed"
        update_progress(100, "I have completely analyzed your website across all prompts!")

    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e) or repr(e)}"
        logger.error(f"Audit job {audit_id} failed: {err_msg}", exc_info=True)
        job.status = "failed"
        job.error = err_msg
        job.current_message = f"Audit failed: {err_msg}"

@app.post("/api/v1/start-audit")
async def start_audit(payload: AuditRequest):
    """Starts the multi-tab audit asynchronously and returns an audit_id immediately for live polling."""
    audit_id = str(uuid.uuid4())
    AUDIT_JOBS[audit_id] = AuditJobStatus(
        audit_id=audit_id,
        status="queued",
        progress_percent=1,
        current_message="Queued audit job...",
    )
    task = asyncio.create_task(_execute_audit_job(audit_id, payload))
    BACKGROUND_TASKS[audit_id] = task
    return {"audit_id": audit_id}

@app.get("/api/v1/audit-status/{audit_id}", response_model=AuditJobStatus)
async def get_audit_status(audit_id: str):
    """Returns real-time progress logs, completion status, and final FullAuditReport."""
    job = AUDIT_JOBS.get(audit_id)
    if not job:
        raise HTTPException(status_code=404, detail="Audit job not found")
    return job

@app.get("/api/v1/download-pdf/{audit_id}")
async def download_pdf_report(audit_id: str):
    """Generates and streams the comprehensive multi-page PDF audit dossier."""
    report = AUDIT_REPORTS_STORE.get(audit_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit report not found or not yet completed")

    try:
        pdf_bytes = await PDFReportGenerator.generate_pdf_bytes_async(report)
        clean_brand = "".join(c for c in report.brand_name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        filename = f"Hermes_AEO_Audit_{clean_brand or 'Report'}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e) or repr(e)}"
        logger.error(f"Error generating PDF: {err_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {err_msg}")

if __name__ == "__main__":
    import sys
    import uvicorn
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
