import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional
from playwright.async_api import async_playwright, Browser, BrowserContext
from app.config import settings

logger = logging.getLogger(__name__)

STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined
});
"""

BROWSER_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-infobars",
    "--disable-dev-shm-usage",
    "--disable-extensions",
    "--disable-gpu",
    "--disable-software-rasterizer",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-features=TranslateUI",
]

class BrowserContextManager:
    """
    Industry-grade Playwright Browser Context Manager.
    - Pins all Playwright operations to a dedicated ProactorEventLoop worker thread
      so Windows SelectorEventLoop servers never hit NotImplementedError.
    - Supports both a shared persistent context AND lightweight isolated parallel worker contexts.
    - Renders HTML templates directly to Executive PDF via Chromium print engine.
    """

    _playwright = None
    _browser: Optional[Browser] = None
    _context: Optional[BrowserContext] = None
    _user_data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "browser_session"

    _loop: Optional[asyncio.AbstractEventLoop] = None
    _thread: Optional[threading.Thread] = None
    _init_lock = threading.Lock()

    @classmethod
    def _ensure_worker_loop(cls) -> asyncio.AbstractEventLoop:
        with cls._init_lock:
            if cls._loop is not None and cls._loop.is_running():
                return cls._loop

            ready_event = threading.Event()

            def _loop_runner():
                if sys.platform == "win32":
                    loop = asyncio.ProactorEventLoop()
                else:
                    loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                cls._loop = loop
                ready_event.set()
                loop.run_forever()

            cls._thread = threading.Thread(target=_loop_runner, name="playwright-proactor-loop", daemon=True)
            cls._thread.start()
            ready_event.wait()
            return cls._loop

    @classmethod
    async def run(cls, async_fn: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any) -> Any:
        """Dispatches an async Playwright coroutine onto the dedicated ProactorEventLoop thread."""
        worker_loop = cls._ensure_worker_loop()
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is worker_loop:
            return await async_fn(*args, **kwargs)

        cf_future = asyncio.run_coroutine_threadsafe(async_fn(*args, **kwargs), worker_loop)
        return await asyncio.wrap_future(cf_future)

    @classmethod
    async def _ensure_playwright(cls):
        if cls._playwright is None:
            cls._playwright = await async_playwright().start()

    @classmethod
    async def get_browser(cls, headless: bool = True) -> Browser:
        """Returns a shared Chromium browser instance for spawning isolated parallel worker contexts."""
        await cls._ensure_playwright()
        if cls._browser is None or not cls._browser.is_connected():
            cls._browser = await cls._playwright.chromium.launch(headless=headless, args=BROWSER_ARGS)
        return cls._browser

    @classmethod
    async def create_isolated_context(cls, headless: bool = True) -> BrowserContext:
        """Creates a lightweight isolated BrowserContext (ideal for parallel concurrent tabs)."""
        browser = await cls.get_browser(headless=headless)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=settings.USER_AGENT,
            ignore_https_errors=True,
        )
        await ctx.add_init_script(STEALTH_SCRIPT)
        return ctx

    @classmethod
    async def get_context(cls, headless: bool = True) -> BrowserContext:
        """Returns the persistent context (preserves login cookies in data/browser_session)."""
        cls._user_data_dir.mkdir(parents=True, exist_ok=True)
        await cls._ensure_playwright()

        if cls._context is None:
            try:
                cls._context = await cls._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(cls._user_data_dir),
                    headless=headless,
                    viewport={"width": 1280, "height": 720},
                    user_agent=settings.USER_AGENT,
                    args=BROWSER_ARGS,
                    ignore_https_errors=True,
                )
            except Exception as lock_err:
                logger.warning(f"Persistent session directory busy ({lock_err}); using isolated stealth context.")
                cls._context = await cls.create_isolated_context(headless=headless)

            await cls._context.add_init_script(STEALTH_SCRIPT)
            logger.info(f"Initialized Playwright browser context in {cls._user_data_dir}")

        return cls._context

    @classmethod
    async def close_persistent_context(cls):
        """Closes any open persistent context to free Chromium RAM before worker runs."""
        if cls._context is not None:
            try:
                await asyncio.wait_for(cls._context.close(), timeout=5)
            except Exception as e:
                logger.warning(f"Error closing persistent context: {e}")
            cls._context = None

    @classmethod
    async def restart_browser(cls, headless: bool = True):
        """Recycles the shared Chromium browser instance to recover from renderer OOM crashes."""
        if cls._browser is not None:
            try:
                await asyncio.wait_for(cls._browser.close(), timeout=5)
            except Exception as e:
                logger.warning(f"Error closing shared browser during restart: {e}")
            cls._browser = None
        await cls.get_browser(headless=headless)

    @classmethod
    async def _render_pdf_internal(cls, html_content: str, header_html: str = "", footer_html: str = "") -> bytes:
        ctx = await cls.create_isolated_context(headless=True)
        page = await ctx.new_page()
        try:
            await page.set_content(html_content, wait_until="load")
            pdf_bytes = await page.pdf(
                format="A4",
                print_background=True,
                display_header_footer=bool(header_html or footer_html),
                header_template=header_html or "<span></span>",
                footer_template=footer_html or "<span></span>",
                margin={"top": "18mm", "bottom": "18mm", "left": "14mm", "right": "14mm"},
            )
            return pdf_bytes
        finally:
            await ctx.close()

    @classmethod
    async def render_html_to_pdf(cls, html_content: str, header_html: str = "", footer_html: str = "") -> bytes:
        return await cls.run(cls._render_pdf_internal, html_content, header_html, footer_html)

    @classmethod
    async def _close_internal(cls):
        if cls._context:
            try:
                await cls._context.close()
            except Exception as e:
                logger.warning(f"Error closing context: {e}")
            cls._context = None

        if cls._browser:
            try:
                await cls._browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            cls._browser = None

        if cls._playwright:
            try:
                await cls._playwright.stop()
            except Exception as e:
                logger.warning(f"Error stopping playwright: {e}")
            cls._playwright = None

    @classmethod
    async def close(cls):
        if cls._loop is not None and cls._loop.is_running():
            await cls.run(cls._close_internal)
            cls._loop.call_soon_threadsafe(cls._loop.stop)
            if cls._thread and cls._thread.is_alive():
                cls._thread.join(timeout=3)
            cls._loop = None
            cls._thread = None
