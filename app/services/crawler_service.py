import asyncio
import logging
from typing import Dict, List, Set
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import httpx
from app.config import settings
from app.schemas.crawler_schema import WebsiteCrawlResult
from app.services.browser_agent.browser_context import BrowserContextManager

logger = logging.getLogger(__name__)

KEY_PAGE_KEYWORDS = ("about", "service", "product", "solution", "pricing", "feature", "contact", "work", "portfolio")

class WebsiteCrawler:
    """
    Industry-grade Multi-Page Web Crawler for AEO ingestion.
    Renders JavaScript via Playwright and extracts semantic DOM trees without brittle regexes.
    """

    @staticmethod
    def _clean_dom_text(html_content: str) -> tuple[str, str, List[str], List[str], bool, str]:
        soup = BeautifulSoup(html_content, "html.parser")

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("meta", property="og:title"):
            title = soup.find("meta", property="og:title").get("content", "").strip()

        meta_description = ""
        meta_desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
        if meta_desc_tag and meta_desc_tag.get("content"):
            meta_description = meta_desc_tag["content"].strip()

        has_schema = bool(soup.find("script", type="application/ld+json"))

        headings: List[str] = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            text = h.get_text(" ", strip=True)
            if text and len(text) > 2 and text not in headings:
                headings.append(text)

        links: List[str] = []
        for a in soup.find_all("a", href=True):
            links.append(a["href"].strip())

        # Strip non-content boilerplate
        for tag in soup(["script", "style", "noscript", "svg", "nav", "aside"]):
            tag.decompose()

        raw_lines = soup.get_text(separator="\n", strip=True).splitlines()
        clean_lines = []
        prev = ""
        for line in raw_lines:
            line = line.strip()
            if line and line != prev and len(line) > 3:
                clean_lines.append(line)
                prev = line

        return title, meta_description, headings, links, has_schema, "\n".join(clean_lines)

    @classmethod
    async def _playwright_fetch(cls, url: str) -> str:
        ctx = await BrowserContextManager.create_isolated_context(headless=True)
        page = await ctx.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            await asyncio.sleep(2.5)
            return await page.content()
        finally:
            await ctx.close()

    @classmethod
    async def _fetch_rendered_html(cls, url: str, allow_playwright: bool = True) -> str:
        # Try fast HTTP first; if content looks like an empty SPA shell or fails, render with Playwright
        html = ""
        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=True, verify=False) as client:
                resp = await client.get(url, headers={"User-Agent": settings.USER_AGENT})
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            logger.info(f"HTTP crawl note for {url}: {e}")
            if not allow_playwright:
                return ""

        # Check if SPA shell (very little visible text)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for t in soup(["script", "style", "noscript"]):
                t.decompose()
            visible_len = len(soup.get_text(strip=True))
            if visible_len > 600 or not allow_playwright:
                return html

        # Full JS render via Playwright on dedicated worker loop (homepage/SPA shell only)
        try:
            html = await BrowserContextManager.run(cls._playwright_fetch, url)
        except Exception as e:
            logger.warning(f"Playwright crawl failed for {url}: {type(e).__name__}: {e}")
            if not html:
                raise RuntimeError(f"Unable to reach or render {url}: {type(e).__name__}: {e}")
        return html

    @classmethod
    async def crawl(cls, url: str) -> WebsiteCrawlResult:
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"

        parsed_root = urlparse(url)
        domain = parsed_root.netloc
        base_url = f"{parsed_root.scheme}://{domain}"

        # 1. Crawl Homepage
        home_html = await cls._fetch_rendered_html(url)
        title, meta_desc, headings, raw_links, has_schema, home_text = cls._clean_dom_text(home_html)

        all_sections = [f"=== HOMEPAGE ({url}) ===\n{home_text[:4500]}"]
        all_headings = list(headings)

        # 2. Discover up to 3 important internal sub-pages (About, Services, Pricing, Contact)
        candidate_urls: List[str] = []
        seen: Set[str] = {url.rstrip("/")}

        for href in raw_links:
            full_link = urljoin(base_url, href).split("#")[0].rstrip("/")
            p_link = urlparse(full_link)
            if p_link.netloc == domain and full_link not in seen:
                path_lower = p_link.path.lower()
                if any(kw in path_lower for kw in KEY_PAGE_KEYWORDS):
                    seen.add(full_link)
                    candidate_urls.append(full_link)
                    if len(candidate_urls) >= 3:
                        break

        # Fetch discovered internal pages cleanly via HTTP (never spawn parallel Playwright tabs for rate-limited subpages)
        if candidate_urls:
            sub_tasks = [cls._fetch_rendered_html(sub_url, allow_playwright=False) for sub_url in candidate_urls]
            sub_htmls = await asyncio.gather(*sub_tasks, return_exceptions=True)
            for sub_url, sub_res in zip(candidate_urls, sub_htmls):
                if isinstance(sub_res, str) and sub_res:
                    _, _, sub_h, _, sub_schema, sub_txt = cls._clean_dom_text(sub_res)
                    has_schema = has_schema or sub_schema
                    for h in sub_h:
                        if h not in all_headings:
                            all_headings.append(h)
                    all_sections.append(f"=== SUBPAGE ({sub_url}) ===\n{sub_txt[:2500]}")

        combined_markdown = "\n\n".join(all_sections)

        metadata: Dict[str, str] = {
            "domain": domain,
            "pages_crawled": str(1 + len(candidate_urls)),
            "has_json_ld_schema": str(has_schema),
            "headings_count": str(len(all_headings)),
        }

        return WebsiteCrawlResult(
            url=url,
            title=title or domain,
            meta_description=meta_desc,
            headings=all_headings[:25],
            clean_markdown=combined_markdown[:12000],
            extracted_metadata=metadata,
            character_count=len(combined_markdown),
        )
