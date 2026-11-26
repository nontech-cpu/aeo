import asyncio
import json
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from app.config import settings
from app.schemas.evaluation_schema import SingleModelEvaluation, PromptAuditItem, SummaryPointRow
from app.schemas.search_schema import ModelExecutionResult
from app.schemas.prompt_schema import SearchPrompt
from app.schemas.report_schema import ActionableFix
from app.services.json_utils import extract_json_object
from app.services.browser_agent.browser_context import BrowserContextManager
from app.services.browser_agent.chatgpt_driver import ChatGPTDriver
from app.services.browser_agent.gemini_driver import GeminiDriver

logger = logging.getLogger(__name__)

GENERIC_BRAND_SUFFIXES = {
    "pvt", "ltd", "llp", "inc", "llc", "co", "corp",
    "official", "india", "store", "shop", "online",
}
CTA_LINK_PREFIXES = ("explore ", "shop ", "view ", "visit ", "see ", "learn more", "buy now", "check ")

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
- Extract ALL real competitor brands/agencies/companies mentioned in each ChatGPT and Gemini answer (never fabricate names, and never include the Target Brand itself as a competitor).
- Set brand_present = true ONLY if the target brand or domain actually appears in that engine's answer or Cited URLs. If an engine result starts with "Execution failed on", set brand_present = false.
- Provide 4 to 6 high-impact actionable_fixes tailored strictly to this business and the audit evidence.
- Output ONLY valid JSON.
"""

def _is_execution_error(text: Optional[str]) -> bool:
    if not text:
        return True
    cleaned = text.strip()
    return cleaned.startswith("Execution failed on ") or "browser execution error:" in cleaned.lower()

def _normalize_words(text: str) -> str:
    chars = [ch.lower() if ch.isalnum() else " " for ch in (text or "")]
    return " ".join("".join(chars).split())

def _compact_alnum(text: str) -> str:
    return "".join(ch.lower() for ch in (text or "") if ch.isalnum())

def _clean_host(url_or_domain: str) -> str:
    raw = (url_or_domain or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    host = (urlparse(raw).netloc or "").lower().removeprefix("www.").split(":")[0].strip()
    return host

def _build_brand_matchers(target_brand: str, target_url: str) -> Tuple[str, List[str], List[str]]:
    clean_domain = _clean_host(target_url)
    domain_root = clean_domain.split(".")[0] if "." in clean_domain else clean_domain

    brand_phrases: List[str] = []
    compact_tokens: List[str] = []

    norm_brand = _normalize_words(target_brand)
    if len(norm_brand) >= 3:
        brand_phrases.append(norm_brand)

    words = norm_brand.split()
    if words and words[0] == "the" and len(words) >= 2:
        no_the = " ".join(words[1:])
        if len(words[1:]) >= 2 or len(no_the) >= 5:
            brand_phrases.append(no_the)

    trimmed = list(words)
    while len(trimmed) > 1 and trimmed[-1] in GENERIC_BRAND_SUFFIXES:
        trimmed.pop()
    if trimmed:
        trimmed_phrase = " ".join(trimmed)
        if trimmed_phrase not in brand_phrases and (len(trimmed) >= 2 or len(trimmed_phrase) >= 5):
            brand_phrases.append(trimmed_phrase)

    compact_brand = _compact_alnum(target_brand)
    if len(compact_brand) >= 5:
        compact_tokens.append(compact_brand)
    if len(domain_root) >= 5 and domain_root not in compact_tokens:
        if len(domain_root) >= 6 or domain_root == compact_brand:
            compact_tokens.append(domain_root)

    return clean_domain, brand_phrases, compact_tokens

def _host_matches_domain(host: str, clean_domain: str) -> bool:
    if not host or not clean_domain:
        return False
    return host == clean_domain or host.endswith("." + clean_domain)

def _has_boundary_domain(text_lower: str, clean_domain: str) -> bool:
    if not clean_domain or len(clean_domain) < 4:
        return False
    start = 0
    dom_len = len(clean_domain)
    while True:
        idx = text_lower.find(clean_domain, start)
        if idx == -1:
            return False
        before_ok = idx == 0 or not text_lower[idx - 1].isalnum()
        after_idx = idx + dom_len
        after_ok = after_idx >= len(text_lower) or not text_lower[after_idx].isalnum()
        if before_ok and after_ok:
            return True
        start = idx + 1

def _contains_target_brand(
    text: Optional[str],
    html: Optional[str],
    sources: List[str],
    clean_domain: str,
    brand_phrases: List[str],
    compact_tokens: List[str],
) -> bool:
    if _is_execution_error(text) and not html:
        return False

    for src in sources or []:
        if _host_matches_domain(_clean_host(src), clean_domain):
            return True

    html_text = ""
    if html:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            if _host_matches_domain(_clean_host(a["href"]), clean_domain):
                return True
        html_text = soup.get_text(" ", strip=True)

    combined = f"{text or ''}\n{html_text}".strip()
    if not combined:
        return False
    combined_lower = combined.lower()

    if _has_boundary_domain(combined_lower, clean_domain):
        return True

    padded = f" {_normalize_words(combined_lower)} "
    for phrase in brand_phrases:
        if f" {phrase} " in padded:
            return True
    for tok in compact_tokens:
        if f" {tok} " in padded:
            return True
    return False

def _parse_numbered_prefix(line: str) -> Tuple[Optional[int], str]:
    s = " ".join((line or "").split()).strip()
    if not s:
        return None, ""
    if s.startswith("#"):
        i = 1
        while i < len(s) and s[i].isdigit():
            i += 1
        if i > 1 and i < len(s) and s[i] in (" ", ".", ":", ")", "-"):
            rest = s[i + 1:].lstrip(" .:-)").strip()
            return int(s[1:i]), rest or s
    i = 0
    while i < len(s) and i < 3 and s[i].isdigit():
        i += 1
    if i > 0 and i < len(s) and s[i] in (".", ")"):
        rest = s[i + 1:].strip()
        if rest and not (len(rest) > 0 and rest[0].isdigit()):
            return int(s[:i]), rest
    return None, s

def _truncate_clean(text: str, limit: int = 185) -> str:
    clean = " ".join((text or "").split()).strip()
    if len(clean) <= limit:
        return clean
    cut = clean[:limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "…"

def _is_cta_noise(line: str) -> bool:
    low = line.strip().lower()
    if not low or len(low) > 48:
        return False
    return any(low.startswith(pref) for pref in CTA_LINK_PREFIXES)

def _split_title_and_summary(raw_text: str) -> Tuple[str, str]:
    clean = " ".join((raw_text or "").split()).strip()
    for sep in (": ", " — ", " – ", " - "):
        idx = clean.find(sep)
        if 2 <= idx <= 75:
            return clean[:idx].strip(), clean[idx + len(sep):].strip()
    return _truncate_clean(clean, 70), ""

def _li_is_distinct_item(li_tag) -> bool:
    first_bold = li_tag.find(["strong", "b"], recursive=False)
    if not first_bold:
        children = [c for c in li_tag.children if str(c).strip()]
        if children and getattr(children[0], "name", None) in ("strong", "b"):
            first_bold = children[0]
    txt = " ".join(li_tag.get_text(" ", strip=True).split())
    if first_bold:
        bold_txt = " ".join(first_bold.get_text(" ", strip=True).split())
        if 2 <= len(bold_txt) <= 85 and len(txt) > len(bold_txt) + 12:
            return True
    return False

def _extract_summary_rows(
    response_text: Optional[str],
    response_html: Optional[str],
    sources: List[str],
    clean_domain: str,
    brand_phrases: List[str],
    compact_tokens: List[str],
    max_rows: int = 12,
) -> List[SummaryPointRow]:
    if _is_execution_error(response_text):
        return [SummaryPointRow(
            rank=None,
            point_title="Execution Failed",
            summary=_truncate_clean(response_text or "Browser execution error", 180),
            is_target_brand=False,
        )]

    rows: List[SummaryPointRow] = []

    if response_html:
        soup = BeautifulSoup(response_html, "html.parser")
        current_item: Optional[Dict] = None

        def _flush_current():
            nonlocal current_item
            if not current_item:
                return
            details = [d for d in current_item["details"] if d and not _is_cta_noise(d)]
            summary_txt = _truncate_clean(" · ".join(details), 185) if details else "Recommended in AI answer"
            title_txt = _truncate_clean(current_item["title"], 80)
            is_hit = _contains_target_brand(
                f"{title_txt} {summary_txt}",
                current_item.get("raw_html", ""),
                [],
                clean_domain,
                brand_phrases,
                compact_tokens,
            )
            rows.append(SummaryPointRow(
                rank=current_item["rank"],
                point_title=title_txt,
                summary=summary_txt,
                is_target_brand=is_hit,
            ))
            current_item = None

        top_nodes = list(soup.find_all(["h2", "h3", "h4", "p", "ol", "ul", "table"], recursive=False))
        if not top_nodes:
            top_nodes = list(soup.find_all(["h2", "h3", "h4", "p", "ol", "ul", "table"]))

        for node in top_nodes:
            if len(rows) >= max_rows:
                break
            tag = node.name.lower()

            if tag == "table":
                _flush_current()
                for tr in node.find_all("tr"):
                    tds = [ " ".join(td.get_text(" ", strip=True).split()) for td in tr.find_all("td") ]
                    if not tds:
                        continue
                    title_txt = _truncate_clean(tds[0], 75)
                    summary_txt = _truncate_clean(" · ".join(tds[1:]), 185) or title_txt
                    is_hit = _contains_target_brand(
                        " ".join(tds), str(tr), [], clean_domain, brand_phrases, compact_tokens
                    )
                    rows.append(SummaryPointRow(
                        rank=len(rows) + 1,
                        point_title=title_txt,
                        summary=summary_txt,
                        is_target_brand=is_hit,
                    ))
                    if len(rows) >= max_rows:
                        break
                continue

            if tag == "ol":
                _flush_current()
                for idx, li in enumerate(node.find_all("li", recursive=False), start=1):
                    full_txt = " ".join(li.get_text(" ", strip=True).split())
                    if not full_txt:
                        continue
                    title_txt, rest_txt = _split_title_and_summary(full_txt)
                    is_hit = _contains_target_brand(full_txt, str(li), [], clean_domain, brand_phrases, compact_tokens)
                    rows.append(SummaryPointRow(
                        rank=idx,
                        point_title=title_txt,
                        summary=_truncate_clean(rest_txt or full_txt, 185),
                        is_target_brand=is_hit,
                    ))
                    if len(rows) >= max_rows:
                        break
                continue

            if tag == "ul":
                lis = node.find_all("li", recursive=False)
                is_distinct_ul = bool(lis) and any(_li_is_distinct_item(li) for li in lis)
                if is_distinct_ul and (current_item is None or not current_item["details"]):
                    # Discard empty section headers (e.g., "1. Best D2C Websites") that only introduce a brand list
                    current_item = None
                    for li in lis:
                        full_txt = " ".join(li.get_text(" ", strip=True).split())
                        if not full_txt or _is_cta_noise(full_txt):
                            continue
                        first_bold = li.find(["strong", "b"])
                        if first_bold:
                            bold_txt = " ".join(first_bold.get_text(" ", strip=True).split()).rstrip(":-– ")
                            rem = full_txt
                            if full_txt.lower().startswith(bold_txt.lower()):
                                rem = full_txt[len(bold_txt):].lstrip(" :.-–")
                            title_txt, summary_txt = bold_txt, rem
                        else:
                            title_txt, summary_txt = _split_title_and_summary(full_txt)
                        is_hit = _contains_target_brand(full_txt, str(li), [], clean_domain, brand_phrases, compact_tokens)
                        rows.append(SummaryPointRow(
                            rank=len(rows) + 1,
                            point_title=_truncate_clean(title_txt, 80),
                            summary=_truncate_clean(summary_txt or full_txt, 185),
                            is_target_brand=is_hit,
                        ))
                        if len(rows) >= max_rows:
                            break
                elif current_item is not None:
                    for li in lis:
                        btxt = " ".join(li.get_text(" ", strip=True).split())
                        if btxt and not _is_cta_noise(btxt):
                            current_item["details"].append(btxt)
                            current_item["raw_html"] += str(li)
                continue

            txt = " ".join(node.get_text(" ", strip=True).split())
            if not txt:
                continue
            rank_num, rem = _parse_numbered_prefix(txt)
            if rank_num is not None and rem:
                _flush_current()
                title_part, rest_part = _split_title_and_summary(rem)
                current_item = {
                    "rank": rank_num,
                    "title": title_part,
                    "details": [rest_part] if rest_part else [],
                    "raw_html": str(node),
                }
            elif current_item is not None:
                if not _is_cta_noise(txt) and len(current_item["details"]) < 4:
                    current_item["details"].append(txt)
                    current_item["raw_html"] += str(node)

        _flush_current()

    # Fallback: parse clean plain text lines (numbered items or key sentences)
    if not rows and response_text:
        lines = [ln.strip() for ln in response_text.splitlines() if ln.strip()]
        current_item = None
        for ln in lines:
            if len(rows) >= max_rows:
                break
            rank_num, rem = _parse_numbered_prefix(ln)
            if rank_num is not None and rem:
                if current_item:
                    title_txt = _truncate_clean(current_item["title"], 80)
                    sum_txt = _truncate_clean(" · ".join(current_item["details"]), 185) or "Recommended in AI answer"
                    rows.append(SummaryPointRow(
                        rank=current_item["rank"],
                        point_title=title_txt,
                        summary=sum_txt,
                        is_target_brand=_contains_target_brand(f"{title_txt} {sum_txt}", "", [], clean_domain, brand_phrases, compact_tokens),
                    ))
                title_part, rest_part = _split_title_and_summary(rem)
                current_item = {"rank": rank_num, "title": title_part, "details": [rest_part] if rest_part else []}
            elif current_item is not None and not _is_cta_noise(ln) and len(current_item["details"]) < 4:
                clean_ln = ln.lstrip("•-* ").strip()
                if clean_ln:
                    current_item["details"].append(clean_ln)
        if current_item and len(rows) < max_rows:
            title_txt = _truncate_clean(current_item["title"], 80)
            sum_txt = _truncate_clean(" · ".join(current_item["details"]), 185) or "Recommended in AI answer"
            rows.append(SummaryPointRow(
                rank=current_item["rank"],
                point_title=title_txt,
                summary=sum_txt,
                is_target_brand=_contains_target_brand(f"{title_txt} {sum_txt}", "", [], clean_domain, brand_phrases, compact_tokens),
            ))

    # Ultimate fallback for pure prose answers with no lists
    if not rows and response_text:
        paras = [ " ".join(p.split()) for p in response_text.split("\n") if len(p.strip()) > 25 ]
        for idx, p in enumerate(paras[:4], start=1):
            is_hit = _contains_target_brand(p, "", [], clean_domain, brand_phrases, compact_tokens)
            rows.append(SummaryPointRow(
                rank=idx,
                point_title=f"Key Point #{idx}",
                summary=_truncate_clean(p, 185),
                is_target_brand=is_hit,
            ))

    # Also detect rank if target domain matched in cited sources but not in row text
    return rows

def _extract_deterministic_rank(summary_rows: List[SummaryPointRow], response_text: Optional[str], clean_domain: str, brand_phrases: List[str], compact_tokens: List[str]) -> Optional[int]:
    for row in summary_rows:
        if row.is_target_brand and row.rank is not None:
            return row.rank
    if response_text:
        for ln in response_text.splitlines():
            rank_num, rem = _parse_numbered_prefix(ln)
            if rank_num is not None and _contains_target_brand(rem, "", [], clean_domain, brand_phrases, compact_tokens):
                return rank_num
    return None

def _format_dossier_excerpt(text: Optional[str], summary_rows: List[SummaryPointRow], sources: List[str]) -> str:
    if _is_execution_error(text):
        return f"Execution failed: {text}"
    raw = (text or "").strip()
    if len(raw) <= 3200:
        body = raw
    else:
        body = f"{raw[:1600]}\n...[middle omitted]...\n{raw[-1400:]}"
    parts = [body]
    if summary_rows:
        idx_labels = [f"#{r.rank or '?'} {r.point_title}" for r in summary_rows[:12]]
        parts.append("Extracted Items: " + "; ".join(idx_labels))
    if sources:
        parts.append("Cited URLs: " + ", ".join(sources[:12]))
    return "\n".join(parts)

def _filter_non_target_competitors(competitors: List[str], clean_domain: str, brand_phrases: List[str], compact_tokens: List[str]) -> List[str]:
    filtered: List[str] = []
    for comp in competitors or []:
        c_clean = comp.strip()
        if not c_clean:
            continue
        if _contains_target_brand(c_clean, "", [], clean_domain, brand_phrases, compact_tokens):
            continue
        if c_clean not in filtered:
            filtered.append(c_clean)
    return filtered

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

    @staticmethod
    def _build_verified_engine_eval(
        engine: str,
        response_text: Optional[str],
        response_html: Optional[str],
        sources: List[str],
        summary_rows: List[SummaryPointRow],
        llm_rank: Optional[int],
        llm_sentiment: str,
        llm_competitors: List[str],
        custom_competitors: List[str],
        clean_domain: str,
        brand_phrases: List[str],
        compact_tokens: List[str],
    ) -> SingleModelEvaluation:
        if _is_execution_error(response_text):
            return SingleModelEvaluation(
                engine=engine,
                is_brand_present=False,
                mention_rank=None,
                sentiment="error",
                competitors_cited=[],
                key_source_domains=[],
                summary_rows=summary_rows,
            )

        verified_present = _contains_target_brand(
            response_text, response_html, sources, clean_domain, brand_phrases, compact_tokens
        )
        det_rank = (
            _extract_deterministic_rank(summary_rows, response_text, clean_domain, brand_phrases, compact_tokens)
            if verified_present else None
        )
        final_rank = det_rank or (llm_rank if (verified_present and llm_rank and llm_rank > 0) else None)

        if not verified_present:
            final_sentiment = "not_mentioned"
        elif llm_sentiment and llm_sentiment.lower() not in ("not_mentioned", "error", ""):
            final_sentiment = llm_sentiment
        else:
            final_sentiment = "positive"

        merged_comps = list(llm_competitors or [])
        low_txt = (response_text or "").lower()
        for custom in custom_competitors or []:
            c_clean = custom.strip()
            if c_clean and c_clean.lower() in low_txt and c_clean not in merged_comps:
                merged_comps.append(c_clean)

        clean_comps = _filter_non_target_competitors(merged_comps, clean_domain, brand_phrases, compact_tokens)

        return SingleModelEvaluation(
            engine=engine,
            is_brand_present=verified_present,
            mention_rank=final_rank,
            sentiment=final_sentiment,
            competitors_cited=clean_comps,
            key_source_domains=[_clean_host(u) for u in (sources or []) if _clean_host(u)],
            summary_rows=summary_rows,
        )

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
        clean_domain, brand_phrases, compact_tokens = _build_brand_matchers(target_brand, target_url)

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

        # Pre-extract structured answer summary rows for every prompt & engine
        rows_by_prompt: Dict[int, Dict[str, List[SummaryPointRow]]] = {}
        for pid, item in audit_items.items():
            rows_by_prompt[pid] = {
                "chatgpt": _extract_summary_rows(
                    item.chatgpt_result, item.chatgpt_html, item.chatgpt_sources,
                    clean_domain, brand_phrases, compact_tokens,
                ) if item.chatgpt_result is not None else [],
                "gemini": _extract_summary_rows(
                    item.gemini_result, item.gemini_html, item.gemini_sources,
                    clean_domain, brand_phrases, compact_tokens,
                ) if item.gemini_result is not None else [],
            }

        # Build complete untruncated-head+tail evaluation dossier (with extracted items + Cited URLs)
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
            if item.chatgpt_result is not None:
                excerpt = _format_dossier_excerpt(item.chatgpt_result, rows_by_prompt[item.prompt_id]["chatgpt"], item.chatgpt_sources)
                dossier_lines.append(f"ChatGPT Answer:\n{excerpt}")
            if item.gemini_result is not None:
                excerpt = _format_dossier_excerpt(item.gemini_result, rows_by_prompt[item.prompt_id]["gemini"], item.gemini_sources)
                dossier_lines.append(f"Gemini Answer:\n{excerpt}")

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

        eval_map = {ev.prompt_id: ev for ev in bundle.evaluations} if bundle else {}
        for pid, item in audit_items.items():
            ev = eval_map.get(pid)
            if item.chatgpt_result is not None:
                item.chatgpt_eval = cls._build_verified_engine_eval(
                    engine="chatgpt",
                    response_text=item.chatgpt_result,
                    response_html=item.chatgpt_html,
                    sources=item.chatgpt_sources,
                    summary_rows=rows_by_prompt[pid]["chatgpt"],
                    llm_rank=ev.chatgpt_rank if ev else None,
                    llm_sentiment=ev.chatgpt_sentiment if ev else "not_mentioned",
                    llm_competitors=ev.chatgpt_competitors if ev else [],
                    custom_competitors=custom_competitors,
                    clean_domain=clean_domain,
                    brand_phrases=brand_phrases,
                    compact_tokens=compact_tokens,
                )
            if item.gemini_result is not None:
                item.gemini_eval = cls._build_verified_engine_eval(
                    engine="gemini",
                    response_text=item.gemini_result,
                    response_html=item.gemini_html,
                    sources=item.gemini_sources,
                    summary_rows=rows_by_prompt[pid]["gemini"],
                    llm_rank=ev.gemini_rank if ev else None,
                    llm_sentiment=ev.gemini_sentiment if ev else "not_mentioned",
                    llm_competitors=ev.gemini_competitors if ev else [],
                    custom_competitors=custom_competitors,
                    clean_domain=clean_domain,
                    brand_phrases=brand_phrases,
                    compact_tokens=compact_tokens,
                )

        if bundle:
            return list(audit_items.values()), bundle.competitor_strategy_analysis, bundle.actionable_fixes

        return (
            list(audit_items.values()),
            "Automated LLM strategy summary could not be generated; see structured prompt tables below.",
            [],
        )
