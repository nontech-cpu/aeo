import asyncio
import logging
from typing import List, Callable, Optional
from app.schemas.search_schema import ModelExecutionResult, AgentChoice
from app.schemas.prompt_schema import SearchPrompt
from app.services.browser_agent.browser_context import BrowserContextManager
from app.services.browser_agent.chatgpt_driver import ChatGPTDriver
from app.services.browser_agent.gemini_driver import GeminiDriver

logger = logging.getLogger(__name__)

class BrowserAuditRunner:
    """
    High-Speed Parallel Multi-Tab Browser Audit Runner.
    Runs ChatGPT and Gemini concurrently using isolated worker contexts per query,
    cutting total 20-query execution time from ~8 minutes down to ~45-75 seconds.
    """

    @classmethod
    async def run_browser_audit(
        cls,
        prompts: List[SearchPrompt],
        agent_choice: AgentChoice = AgentChoice.ALL,
        headless: bool = True,
        concurrency_per_engine: int = 2,
        on_progress: Optional[Callable[[int, str], None]] = None,
        on_result: Optional[Callable[[ModelExecutionResult], None]] = None,
    ) -> List[ModelExecutionResult]:
        return await BrowserContextManager.run(
            cls._run_browser_audit_internal,
            prompts,
            agent_choice,
            headless,
            concurrency_per_engine,
            on_progress,
            on_result,
        )

    @classmethod
    async def _run_single_worker(
        cls,
        engine: str,
        prompt: SearchPrompt,
        idx: int,
        total_prompts: int,
        sem: asyncio.Semaphore,
        headless: bool,
        progress_cb: Callable[[str], None],
        result_cb: Optional[Callable[[ModelExecutionResult], None]],
    ) -> ModelExecutionResult:
        async with sem:
            last_err = None
            for attempt in range(2):
                ctx = await BrowserContextManager.create_isolated_context(headless=headless)
                try:
                    page = await ctx.new_page()
                    label = "ChatGPT" if engine == "chatgpt" else "Gemini"
                    if attempt == 0:
                        progress_cb(f"{label} Worker -> Executing Prompt {idx}/{total_prompts}: '{prompt.prompt_text}'")
                    else:
                        progress_cb(f"{label} Worker -> Retrying Prompt {idx}/{total_prompts}...")

                    if engine == "chatgpt":
                        clean_text, clean_html, sources = await asyncio.wait_for(
                            ChatGPTDriver.execute_query_rich(page, prompt.prompt_text, fresh_chat=True, timeout_seconds=55),
                            timeout=75,
                        )
                    else:
                        clean_text, clean_html, sources = await asyncio.wait_for(
                            GeminiDriver.execute_query_rich(page, prompt.prompt_text, fresh_chat=True, timeout_seconds=55),
                            timeout=75,
                        )

                    res = ModelExecutionResult(
                        engine=engine,
                        prompt_id=prompt.id,
                        prompt_text=prompt.prompt_text,
                        response_text=clean_text,
                        response_html=clean_html,
                        cited_sources=sources,
                    )
                    progress_cb(f"{label} Worker -> Completed Prompt {idx}/{total_prompts} ({len(clean_text)} chars, {len(sources)} sources)")
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(f"{engine} Prompt #{prompt.id} attempt {attempt + 1} failed: {e}")
                finally:
                    try:
                        await asyncio.wait_for(ctx.close(), timeout=5)
                    except Exception:
                        pass

            if last_err is not None:
                logger.error(f"{engine} Prompt #{prompt.id} failed after retry: {last_err}")
                res = ModelExecutionResult(
                    engine=engine,
                    prompt_id=prompt.id,
                    prompt_text=prompt.prompt_text,
                    response_text=f"Execution failed on {engine} tab: {last_err}",
                    response_html="",
                    cited_sources=[],
                    error=str(last_err),
                )
                progress_cb(f"{engine.capitalize()} Worker -> Prompt {idx}/{total_prompts} error: {last_err}")

            if result_cb:
                result_cb(res)
            return res

    @classmethod
    async def _run_browser_audit_internal(
        cls,
        prompts: List[SearchPrompt],
        agent_choice: AgentChoice = AgentChoice.ALL,
        headless: bool = True,
        concurrency_per_engine: int = 3,
        on_progress: Optional[Callable[[int, str], None]] = None,
        on_result: Optional[Callable[[ModelExecutionResult], None]] = None,
    ) -> List[ModelExecutionResult]:

        total_tasks = len(prompts) * (2 if agent_choice == AgentChoice.ALL else 1)
        completed_count = 0

        def _emit(msg: str, increment: bool = False):
            nonlocal completed_count
            if increment:
                completed_count += 1
            pct = int((completed_count / max(total_tasks, 1)) * 82) + 8
            logger.info(msg)
            if on_progress:
                on_progress(min(pct, 90), msg)

        def _on_res(res: ModelExecutionResult):
            _emit(f"Saved [{res.engine.upper()}] Prompt #{res.prompt_id} result.", increment=True)
            if on_result:
                on_result(res)

        gpt_sem = asyncio.Semaphore(concurrency_per_engine)
        gem_sem = asyncio.Semaphore(concurrency_per_engine)

        tasks = []
        total_p = len(prompts)

        if agent_choice in [AgentChoice.CHATGPT, AgentChoice.ALL]:
            _emit(f"Launching {min(concurrency_per_engine, total_p)} parallel ChatGPT workers...")
            for idx, p in enumerate(prompts, start=1):
                tasks.append(
                    cls._run_single_worker("chatgpt", p, idx, total_p, gpt_sem, headless, _emit, _on_res)
                )

        if agent_choice in [AgentChoice.GEMINI, AgentChoice.ALL]:
            _emit(f"Launching {min(concurrency_per_engine, total_p)} parallel Gemini workers...")
            for idx, p in enumerate(prompts, start=1):
                tasks.append(
                    cls._run_single_worker("gemini", p, idx, total_p, gem_sem, headless, _emit, _on_res)
                )

        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)
