import asyncio
import logging
import time
from typing import Tuple, List
from playwright.async_api import BrowserContext, Page, TimeoutError
from app.services.html_sanitizer import sanitize_ai_response_html

logger = logging.getLogger(__name__)

class GeminiDriver:
    """
    High-speed Google Gemini Browser Driver.
    - Uses element-scoped fill/press (safe for parallel isolated contexts).
    - Zero arbitrary post-goto sleeps.
    - Extracts clean semantic HTML (preserving comparison tables) + normalized plain text + citations.
    """

    GEMINI_URL = "https://gemini.google.com/app"

    @classmethod
    async def get_or_create_page(cls, context: BrowserContext) -> Page:
        for page in context.pages:
            if "gemini.google.com" in page.url:
                return page
        return await context.new_page()

    @classmethod
    async def execute_query_rich(
        cls,
        page: Page,
        prompt_text: str,
        fresh_chat: bool = True,
        timeout_seconds: int = 75
    ) -> Tuple[str, str, List[str]]:
        try:
            logger.info(f"[Gemini Driver] Executing query (fresh={fresh_chat}): {prompt_text[:65]}...")

            if fresh_chat or "gemini.google.com" not in page.url:
                await page.goto(cls.GEMINI_URL, wait_until="domcontentloaded", timeout=30000)

            input_selector = 'rich-textarea div[contenteditable="true"], div[role="textbox"]'
            await page.wait_for_selector(input_selector, state="visible", timeout=20000)
            input_box = page.locator(input_selector).first

            response_selector = 'message-content'
            prev_count = await page.locator(response_selector).count()

            # Element-scoped fill + Angular change-detection tick + Enter submit
            await input_box.click()
            await input_box.fill(prompt_text)
            await asyncio.sleep(0.35)
            await input_box.press("Enter")

            # Verify generation started within 2.5s; if input still has text, re-submit safely
            await asyncio.sleep(2.0)
            stop_btn_selector = 'button[aria-label*="Stop"]'
            if not await page.locator(stop_btn_selector).first.is_visible() and await page.locator(response_selector).count() == prev_count:
                send_btn = page.locator('button[aria-label="Send message"]').first
                if await send_btn.is_visible() and await send_btn.is_enabled():
                    await send_btn.click()
                else:
                    await input_box.press("Enter")

            deadline = time.monotonic() + timeout_seconds
            poll_interval = 0.45
            last_text = ""
            stable_ticks = 0

            while time.monotonic() < deadline:
                await asyncio.sleep(poll_interval)

                stop_visible = await page.locator(stop_btn_selector).first.is_visible()
                current_count = await page.locator(response_selector).count()

                if current_count > prev_count:
                    current_txt = (await page.locator(response_selector).last.inner_text(timeout=5000)).strip()
                    if not stop_visible and current_txt:
                        if current_txt == last_text:
                            stable_ticks += 1
                        else:
                            stable_ticks = 0
                            last_text = current_txt
                        if stable_ticks >= 2:
                            break
                    elif current_txt:
                        last_text = current_txt

            if not last_text:
                raise TimeoutError("Timed out waiting for Gemini model response")

            # Extract clean semantic HTML & citations from model-response
            last_msg_node = page.locator(response_selector).last
            raw_html = await last_msg_node.inner_html()
            native_text = (await last_msg_node.inner_text()).strip() or last_text
            clean_html, clean_text, sources = sanitize_ai_response_html(raw_html, native_inner_text=native_text)

            # Check outer model-response container for additional grounding links
            container = page.locator("model-response").last
            if await container.count() > 0:
                _, _, outer_sources = sanitize_ai_response_html(await container.inner_html())
                for s in outer_sources:
                    if s not in sources:
                        sources.append(s)

            logger.info(f"[Gemini Driver] Captured response ({len(clean_text)} chars, {len(sources)} sources)")
            return clean_text, clean_html, sources

        except Exception as e:
            logger.error(f"[Gemini Driver] Query failed: {e}")
            raise RuntimeError(f"Gemini browser execution error: {e}")

    @classmethod
    async def execute_query(
        cls,
        page: Page,
        prompt_text: str,
        fresh_chat: bool = True,
        timeout_seconds: int = 75
    ) -> Tuple[str, List[str]]:
        clean_text, _, sources = await cls.execute_query_rich(page, prompt_text, fresh_chat, timeout_seconds)
        return clean_text, sources
