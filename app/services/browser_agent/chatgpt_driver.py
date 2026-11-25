import asyncio
import logging
import time
from typing import Tuple, List
from playwright.async_api import BrowserContext, Page, TimeoutError
from app.services.html_sanitizer import sanitize_ai_response_html

logger = logging.getLogger(__name__)

class ChatGPTDriver:
    """
    High-speed ChatGPT Browser Driver.
    - Uses element-scoped fill/press (safe for parallel isolated contexts).
    - Zero arbitrary post-goto sleeps.
    - Extracts clean semantic HTML + normalized plain text + cited URLs via BeautifulSoup.
    """

    CHATGPT_URL = "https://chatgpt.com"

    @classmethod
    async def get_or_create_page(cls, context: BrowserContext) -> Page:
        for page in context.pages:
            if "chatgpt.com" in page.url:
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
            logger.info(f"[ChatGPT Driver] Executing query (fresh={fresh_chat}): {prompt_text[:65]}...")

            if fresh_chat or "chatgpt.com" not in page.url:
                await page.goto(cls.CHATGPT_URL, wait_until="domcontentloaded", timeout=30000)

            input_selector = 'textarea[name="prompt"], #mobile-composer-prompt, #prompt-textarea, div[contenteditable="true"][role="textbox"]'
            await page.wait_for_selector(input_selector, state="visible", timeout=20000)
            input_box = page.locator(input_selector).first

            assistant_sel = 'li[data-message-role="assistant"], div[data-message-author-role="assistant"]'
            prev_count = await page.locator(assistant_sel).count()

            # Element-scoped instant fill + small React state tick + submit
            await input_box.click()
            await input_box.fill(prompt_text)
            await asyncio.sleep(0.35)

            send_btn_selector = 'button[aria-label="Send message"], button[data-testid="send-button"], button[aria-label="Send prompt"]'
            send_btn = page.locator(send_btn_selector).first
            if await send_btn.is_visible() and await send_btn.is_enabled():
                await send_btn.click()
            else:
                await input_box.press("Enter")

            # Ensure submission registered within 2.5s; if not, re-submit once
            await asyncio.sleep(2.0)
            if await page.locator(assistant_sel).count() == prev_count:
                try:
                    if await input_box.is_visible() and (await input_box.input_value()).strip():
                        await input_box.press("Enter")
                except Exception:
                    pass

            # Wait dynamically using real wall-clock deadline (never drifts under heavy DOM load)
            deadline = time.monotonic() + timeout_seconds
            poll_interval = 0.45
            last_text = ""
            stable_ticks = 0

            while time.monotonic() < deadline:
                await asyncio.sleep(poll_interval)

                guest_complete_sel = 'li[data-message-role="assistant"][data-message-complete]'
                if await page.locator(guest_complete_sel).count() > prev_count:
                    txt = (await page.locator(guest_complete_sel).last.inner_text(timeout=5000)).strip()
                    if txt:
                        last_text = txt
                        break

                current_count = await page.locator(assistant_sel).count()
                if current_count > prev_count:
                    current_txt = (await page.locator(assistant_sel).last.inner_text(timeout=5000)).strip()
                    stop_visible = await page.locator('button[data-testid="stop-button"], button[aria-label="Stop generating"], button[aria-label="Stop streaming"]').first.is_visible()
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
                raise TimeoutError("Timed out waiting for ChatGPT assistant response")

            # Extract clean semantic HTML & layout-aware normalized text from assistant node
            last_msg_node = page.locator(assistant_sel).last
            raw_html = await last_msg_node.inner_html()
            native_text = (await last_msg_node.inner_text()).strip() or last_text
            clean_html, clean_text, sources = sanitize_ai_response_html(raw_html, native_inner_text=native_text)

            logger.info(f"[ChatGPT Driver] Captured response ({len(clean_text)} chars, {len(sources)} clean sources)")
            return clean_text, clean_html, sources

        except Exception as e:
            logger.error(f"[ChatGPT Driver] Query failed: {e}")
            raise RuntimeError(f"ChatGPT browser execution error: {e}")

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
