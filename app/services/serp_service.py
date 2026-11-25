import asyncio
import logging
from typing import List, Dict, Optional
from urllib.parse import urlparse, quote_plus
import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.schemas.prompt_schema import SearchPrompt
from app.schemas.evaluation_schema import GooglePage1Result, GoogleSerpSnapshot
from app.services.browser_agent.browser_context import BrowserContextManager

logger = logging.getLogger(__name__)

class GoogleSerpService:
    """
    Industry-grade Google AI Overview & Page-1 Organic SERP Checker.
    Supports SerpApi, Serper.dev, and honest browser SERP inspection (stops cleanly on CAPTCHA).
    Strictly zero regex.
    """

    @staticmethod
    def _clean_domain(url: str) -> str:
        netloc = urlparse(url).netloc.lower()
        return netloc.removeprefix("www.")

    @classmethod
    def _is_brand_match(cls, text: str, url: str, target_brand: str, target_domain: str) -> bool:
        clean_target_dom = target_domain.lower().removeprefix("www.").strip()
        if clean_target_dom and len(clean_target_dom) > 3:
            if clean_target_dom in cls._clean_domain(url) or clean_target_dom in text.lower():
                return True
        clean_brand = target_brand.lower().strip()
        if clean_brand and len(clean_brand) > 2:
            if clean_brand in text.lower():
                return True
        return False

    @staticmethod
    def _flatten_serpapi_ai_blocks(blocks: list) -> str:
        lines: List[str] = []
        for blk in blocks or []:
            if isinstance(blk, dict):
                snippet = blk.get("snippet")
                if snippet:
                    lines.append(str(snippet).strip())
                for item in blk.get("list", []) or []:
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        snip = item.get("snippet", "")
                        combined = f"- {title}: {snip}".strip(" :-")
                        if combined:
                            lines.append(combined)
                    elif isinstance(item, str):
                        lines.append(f"- {item}")
        return "\n".join(lines).strip()

    @classmethod
    async def _fetch_via_serpapi(
        cls,
        client: httpx.AsyncClient,
        prompt_text: str,
        target_brand: str,
        target_domain: str,
        include_ai_overview: bool = True,
        include_page1: bool = True,
    ) -> GoogleSerpSnapshot:
        params = {
            "engine": "google",
            "q": prompt_text,
            "gl": "in",
            "hl": "en",
            "num": 10,
            "api_key": settings.SERPAPI_API_KEY,
        }
        resp = await client.get("https://serpapi.com/search.json", params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()

        # 1. AI Overview Parsing (only when requested)
        ai_text = ""
        ai_sources: List[str] = []
        has_ai_ov = False
        brand_in_ai_ov = False

        if include_ai_overview:
            ai_ov = data.get("ai_overview") or {}
            if ai_ov.get("page_token") and not ai_ov.get("text_blocks"):
                try:
                    ov_resp = await client.get(
                        "https://serpapi.com/search.json",
                        params={
                            "engine": "google_ai_overview",
                            "page_token": ai_ov["page_token"],
                            "api_key": settings.SERPAPI_API_KEY,
                        },
                        timeout=20,
                    )
                    if ov_resp.status_code == 200:
                        ai_ov = ov_resp.json().get("ai_overview") or ai_ov
                except Exception as e:
                    logger.warning(f"SerpApi AI overview token fetch failed: {e}")

            ai_text = cls._flatten_serpapi_ai_blocks(ai_ov.get("text_blocks", []))
            for ref in ai_ov.get("references", []) or []:
                link = ref.get("link") if isinstance(ref, dict) else None
                if link and link not in ai_sources:
                    ai_sources.append(link)

            has_ai_ov = bool(ai_ov) and (bool(ai_text) or bool(ai_sources))
            if has_ai_ov:
                if cls._is_brand_match(ai_text, "", target_brand, target_domain):
                    brand_in_ai_ov = True
                elif any(cls._is_brand_match("", src, target_brand, target_domain) for src in ai_sources):
                    brand_in_ai_ov = True

        # 2. Google Page 1 Organic Top 10 Parsing (only when requested)
        page1_results: List[GooglePage1Result] = []
        brand_rank = None
        brand_url = None

        if include_page1:
            for idx, org in enumerate(data.get("organic_results", [])[:10], start=1):
                link = org.get("link", "")
                title = org.get("title", "")
                pos = org.get("position") or idx
                dom = cls._clean_domain(link)
                is_target = cls._is_brand_match(title, link, target_brand, target_domain)
                if is_target and brand_rank is None:
                    brand_rank = pos
                    brand_url = link
                page1_results.append(GooglePage1Result(
                    rank=pos, title=title, url=link, domain=dom, is_target_brand=is_target
                ))

        return GoogleSerpSnapshot(
            status="ok",
            status_note="Verified via SerpApi",
            ai_overview_supported=include_ai_overview,
            has_ai_overview=has_ai_ov,
            brand_in_ai_overview=brand_in_ai_ov,
            ai_overview_text=ai_text or None,
            ai_overview_sources=ai_sources,
            brand_page1_rank=brand_rank,
            brand_page1_url=brand_url,
            page1_top_results=page1_results,
        )

    @classmethod
    async def _fetch_via_serper(
        cls,
        client: httpx.AsyncClient,
        prompt_text: str,
        target_brand: str,
        target_domain: str,
        include_ai_overview: bool = True,
        include_page1: bool = True,
    ) -> GoogleSerpSnapshot:
        headers = {
            "X-API-KEY": settings.SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {"q": prompt_text, "gl": "in", "hl": "en", "num": 10}
        resp = await client.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        ai_text = ""
        ai_sources: List[str] = []
        has_ai_ov = False
        brand_in_ai_ov = False

        if include_ai_overview:
            ai_ov = data.get("aiOverview") or data.get("answerBox") or {}
            if isinstance(ai_ov, dict):
                ai_text = str(ai_ov.get("snippet") or ai_ov.get("answer") or ai_ov.get("text") or "").strip()
                for src in ai_ov.get("sources", []) or []:
                    link = src.get("link") if isinstance(src, dict) else (src if isinstance(src, str) else None)
                    if link and link not in ai_sources:
                        ai_sources.append(link)
                if ai_ov.get("link") and ai_ov["link"] not in ai_sources:
                    ai_sources.append(ai_ov["link"])

            has_ai_ov = bool(ai_text)
            brand_in_ai_ov = has_ai_ov and (
                cls._is_brand_match(ai_text, "", target_brand, target_domain)
                or any(cls._is_brand_match("", s, target_brand, target_domain) for s in ai_sources)
            )

        page1_results: List[GooglePage1Result] = []
        brand_rank = None
        brand_url = None
        if include_page1:
            for idx, org in enumerate(data.get("organic", [])[:10], start=1):
                link = org.get("link", "")
                title = org.get("title", "")
                pos = org.get("position") or idx
                dom = cls._clean_domain(link)
                is_target = cls._is_brand_match(title, link, target_brand, target_domain)
                if is_target and brand_rank is None:
                    brand_rank = pos
                    brand_url = link
                page1_results.append(GooglePage1Result(
                    rank=pos, title=title, url=link, domain=dom, is_target_brand=is_target
                ))

        if include_ai_overview and not has_ai_ov:
            status_note = "Top-10 Page 1 live via Serper.dev (Serper API does not return Google AI Overview; add SERPAPI_API_KEY in .env for live AI Overview)"
        elif not include_ai_overview:
            status_note = "Google Page 1 (Top 10) verified via Serper.dev"
        else:
            status_note = "Verified via Serper.dev"

        return GoogleSerpSnapshot(
            status="ok",
            status_note=status_note,
            ai_overview_supported=bool(has_ai_ov) if include_ai_overview else False,
            has_ai_overview=has_ai_ov,
            brand_in_ai_overview=brand_in_ai_ov,
            ai_overview_text=ai_text or None,
            ai_overview_sources=ai_sources,
            brand_page1_rank=brand_rank,
            brand_page1_url=brand_url,
            page1_top_results=page1_results,
        )

    @classmethod
    async def _check_serp_via_browser(
        cls,
        prompts: List[SearchPrompt],
        target_brand: str,
        target_domain: str,
        include_ai_overview: bool = True,
        include_page1: bool = True,
    ) -> Dict[int, GoogleSerpSnapshot]:
        """
        Attempts clean SERP inspection using the persistent browser session.
        Respects anti-bot gates: if Google presents a CAPTCHA (/sorry/index), stops cleanly without bypassing.
        """
        async def _run_in_worker() -> Dict[int, GoogleSerpSnapshot]:
            snapshots: Dict[int, GoogleSerpSnapshot] = {}
            ctx = await BrowserContextManager.get_context(headless=True)
            page = await ctx.new_page()
            blocked = False

            try:
                for p in prompts:
                    if blocked:
                        snapshots[p.id] = GoogleSerpSnapshot(
                            status="captcha_blocked",
                            status_note="Google Search returned anti-bot gate (/sorry/index). Add SERPAPI_API_KEY or SERPER_API_KEY in .env for reliable Page 1 & AI Overview data."
                        )
                        continue

                    url = f"https://www.google.com/search?q={quote_plus(p.prompt_text)}&hl=en&gl=in"
                    await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    await asyncio.sleep(1.5)

                    if "/sorry/" in page.url.lower():
                        blocked = True
                        logger.warning("[Google SERP] Hit Google /sorry/ anti-bot gate; stopping browser SERP checks cleanly.")
                        snapshots[p.id] = GoogleSerpSnapshot(
                            status="captcha_blocked",
                            status_note="Google Search returned anti-bot gate (/sorry/index). Add SERPAPI_API_KEY or SERPER_API_KEY in .env for reliable Page 1 & AI Overview data."
                        )
                        continue

                    html = await page.content()
                    soup = BeautifulSoup(html, "html.parser")

                    # Extract Google AI Overview container if requested
                    ai_ov_text = ""
                    ai_ov_sources: List[str] = []
                    has_ai_ov = False
                    brand_in_ai_ov = False

                    if include_ai_overview:
                        ai_container = soup.find(attrs={"data-attrid": "ai_overview"}) or soup.find(attrs={"data-md": "ai_overview"})
                        if not ai_container:
                            for heading in soup.find_all(["h1", "h2", "div", "span"]):
                                if heading.get_text(strip=True) == "AI Overview" and heading.parent:
                                    ai_container = heading.parent.parent or heading.parent
                                    break

                        if ai_container:
                            ai_ov_text = " ".join(ai_container.get_text(" ", strip=True).split())[:1500]
                            for a in ai_container.find_all("a", href=True):
                                href = a["href"]
                                if href.startswith("http") and "google.com" not in urlparse(href).netloc:
                                    if href not in ai_ov_sources:
                                        ai_ov_sources.append(href)

                        has_ai_ov = bool(ai_ov_text and len(ai_ov_text) > 30)
                        brand_in_ai_ov = has_ai_ov and (
                            cls._is_brand_match(ai_ov_text, "", target_brand, target_domain)
                            or any(cls._is_brand_match("", s, target_brand, target_domain) for s in ai_ov_sources)
                        )

                    # Extract Google Page 1 Organic Top 10 if requested
                    page1_results: List[GooglePage1Result] = []
                    brand_rank = None
                    brand_url = None
                    if include_page1:
                        seen_urls = set()
                        search_root = soup.find(id="rso") or soup.find(id="search") or soup
                        for h3 in search_root.find_all("h3"):
                            a_tag = h3.find_parent("a", href=True)
                            if not a_tag:
                                continue
                            link = a_tag["href"].strip()
                            if not link.startswith("http") or link in seen_urls:
                                continue
                            dom = cls._clean_domain(link)
                            if not dom or "google." in dom:
                                continue
                            seen_urls.add(link)
                            rank = len(page1_results) + 1
                            title = h3.get_text(" ", strip=True)
                            is_target = cls._is_brand_match(title, link, target_brand, target_domain)
                            if is_target and brand_rank is None:
                                brand_rank = rank
                                brand_url = link
                            page1_results.append(GooglePage1Result(
                                rank=rank, title=title, url=link, domain=dom, is_target_brand=is_target
                            ))
                            if len(page1_results) >= 10:
                                break

                    has_data = bool(page1_results) if include_page1 else has_ai_ov
                    snapshots[p.id] = GoogleSerpSnapshot(
                        status="ok" if has_data else "api_key_required",
                        status_note="Verified via Google Search" if has_data else "No search results parsed (browser anti-bot protection active)",
                        ai_overview_supported=include_ai_overview,
                        has_ai_overview=has_ai_ov,
                        brand_in_ai_overview=brand_in_ai_ov,
                        ai_overview_text=ai_ov_text or None,
                        ai_overview_sources=ai_ov_sources,
                        brand_page1_rank=brand_rank,
                        brand_page1_url=brand_url,
                        page1_top_results=page1_results,
                    )
            finally:
                try:
                    await page.close()
                except Exception:
                    pass
            return snapshots

        return await BrowserContextManager.run(_run_in_worker)

    @classmethod
    async def check_queries(
        cls,
        prompts: List[SearchPrompt],
        target_brand: str,
        target_url: str,
        include_ai_overview: bool = True,
        include_page1: bool = True,
    ) -> Dict[int, GoogleSerpSnapshot]:
        """
        Checks selected Google SERP checks (AI Overview and/or Page 1 Organic Top 10).
        If only Page 1 is requested and SERPER_API_KEY is configured, prefers Serper.dev for speed/cost.
        If AI Overview is requested, prefers SerpApi -> Serper.dev -> Browser check.
        """
        if not include_ai_overview and not include_page1:
            return {}

        target_domain = cls._clean_domain(target_url)

        async def _run_serpapi() -> Dict[int, GoogleSerpSnapshot]:
            logger.info(f"Checking Google SERP via SerpApi (AIO={include_ai_overview}, Page1={include_page1}) for {len(prompts)} prompts...")
            async with httpx.AsyncClient() as client:
                tasks = [
                    cls._fetch_via_serpapi(client, p.prompt_text, target_brand, target_domain, include_ai_overview, include_page1)
                    for p in prompts
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return {
                    p.id: (GoogleSerpSnapshot(status="error", status_note=f"SerpApi error: {res}") if isinstance(res, Exception) else res)
                    for p, res in zip(prompts, results)
                }

        async def _run_serper() -> Dict[int, GoogleSerpSnapshot]:
            logger.info(f"Checking Google SERP via Serper.dev (AIO={include_ai_overview}, Page1={include_page1}) for {len(prompts)} prompts...")
            async with httpx.AsyncClient() as client:
                tasks = [
                    cls._fetch_via_serper(client, p.prompt_text, target_brand, target_domain, include_ai_overview, include_page1)
                    for p in prompts
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return {
                    p.id: (GoogleSerpSnapshot(status="error", status_note=f"Serper error: {res}") if isinstance(res, Exception) else res)
                    for p, res in zip(prompts, results)
                }

        if not include_ai_overview and include_page1:
            if settings.SERPER_API_KEY:
                return await _run_serper()
            if settings.SERPAPI_API_KEY:
                return await _run_serpapi()
        else:
            if settings.SERPAPI_API_KEY:
                return await _run_serpapi()
            if settings.SERPER_API_KEY:
                return await _run_serper()

        logger.info("No SERPAPI_API_KEY/SERPER_API_KEY configured; attempting honest browser Google Search check...")
        return await cls._check_serp_via_browser(prompts, target_brand, target_domain, include_ai_overview, include_page1)
