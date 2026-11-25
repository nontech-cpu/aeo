from html import escape
from html.parser import HTMLParser
from typing import Tuple, List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from bs4 import BeautifulSoup, NavigableString, Tag

ALLOWED_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "strong", "b", "em", "i",
    "code", "pre", "table", "thead", "tbody", "tr", "th", "td",
    "a", "blockquote", "br", "hr"
}

BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "blockquote", "pre", "hr"}

WIDGET_MARKERS = ("<Entity", "<entity", "<badge", "<MapWidgetV2", "<mapwidgetv2", "<AsyncImageGroup", "<asyncimagegroup")

IGNORED_LINK_DOMAINS = {
    "www.mapbox.com", "mapbox.com", "apps.mapbox.com",
    "www.openstreetmap.org", "openstreetmap.org",
    "chatgpt.com", "openai.com", "gemini.google.com",
    "support.google.com", "policies.google.com", "accounts.google.com"
}

class _WidgetNormalizer(HTMLParser):
    """
    Zero-regex HTML/JSX widget tokenizer built on stdlib HTMLParser.
    Converts internal ChatGPT directives (<Entity>, <badge>, <MapWidgetV2>, <AsyncImageGroup>)
    into clean semantic HTML without swallowing trailing text after self-closing tags.
    """
    DROP_VOID_WIDGETS = {"mapwidgetv2", "asyncimagegroup"}

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.out: List[str] = []
        self._skip_depth = 0

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        t = tag.lower()
        if t in self.DROP_VOID_WIDGETS:
            return
        if t == "entity":
            val = dict(attrs).get("value") or ""
            if val:
                self.out.append(f"<strong>{escape(val)}</strong>")
            return
        if t == "badge":
            return
        self.out.append(f"<{tag}{self._fmt_attrs(attrs)}/>")

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        t = tag.lower()
        if t in self.DROP_VOID_WIDGETS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if t == "entity":
            val = dict(attrs).get("value") or ""
            if val:
                self.out.append(f"<strong>{escape(val)}</strong>")
            self._skip_depth += 1
            return
        if t == "badge":
            self.out.append("<em>")
            return
        self.out.append(f"<{tag}{self._fmt_attrs(attrs)}>")

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t in self.DROP_VOID_WIDGETS or t == "entity":
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if t == "badge":
            self.out.append("</em>")
            return
        self.out.append(f"</{tag}>")

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            self.out.append(data)

    def handle_entityref(self, name: str):
        if self._skip_depth == 0:
            self.out.append(f"&{name};")

    def handle_charref(self, name: str):
        if self._skip_depth == 0:
            self.out.append(f"&#{name};")

    @staticmethod
    def _fmt_attrs(attrs: List[Tuple[str, Optional[str]]]) -> str:
        if not attrs:
            return ""
        parts = []
        for k, v in attrs:
            parts.append(k if v is None else f'{k}="{escape(v, quote=True)}"')
        return " " + " ".join(parts)

def _normalize_widget_markup(fragment: str) -> str:
    parser = _WidgetNormalizer()
    parser.feed(fragment)
    parser.close()
    return "".join(parser.out)

def _has_widget_markup(text: str) -> bool:
    return any(m in text for m in WIDGET_MARKERS)

def _soup_to_clean_text(soup: BeautifulSoup) -> str:
    """Extracts block-aware clean plain text without splitting inline <strong>/<em>/<a> tokens."""
    text_copy = BeautifulSoup(str(soup), "html.parser")
    for br in text_copy.find_all("br"):
        br.replace_with("\n")
    for block in text_copy.find_all(BLOCK_TAGS):
        block.append("\n")
    for cell in text_copy.find_all(["td", "th"]):
        cell.append(" | ")
    return normalize_plain_text(text_copy.get_text())

def clean_citation_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if not parsed.netloc or parsed.netloc.lower() in IGNORED_LINK_DOMAINS:
        return ""
    query_params = parse_qs(parsed.query)
    query_params.pop("utm_source", None)
    clean_query = urlencode(query_params, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, clean_query, ""))

def normalize_plain_text(raw_text: str) -> str:
    """
    Normalizes plain text by collapsing 2+ consecutive blank lines
    down to a single blank line without splitting inline spans onto separate lines.
    """
    if not raw_text:
        return ""
    normalized_lines: List[str] = []
    blank_streak = 0
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            blank_streak += 1
            if blank_streak <= 1 and normalized_lines:
                normalized_lines.append("")
        else:
            blank_streak = 0
            normalized_lines.append(stripped)
    return "\n".join(normalized_lines).strip()

def sanitize_ai_response_html(raw_html: str, native_inner_text: str = "") -> Tuple[str, str, List[str]]:
    """
    Converts raw ChatGPT/Gemini assistant DOM HTML into clean semantic HTML + normalized plain text + cited URLs.
    Strips internal ChatGPT widget tags (<Entity>, <badge>, <MapWidgetV2>, <AsyncImageGroup>) cleanly.
    Strictly zero regex.
    """
    if not raw_html or not raw_html.strip():
        return "", normalize_plain_text(native_inner_text), []

    soup = BeautifulSoup(raw_html, "html.parser")

    # 1. Remove UI chrome, buttons, icons, maps, interactive quiz forms, scripts
    for junk in soup(["button", "svg", "style", "script", "audio", "video", "noscript", "iframe", "img", "form", "input", "label", "canvas", "select"]):
        junk.decompose()

    for hidden in soup.find_all(attrs={"aria-hidden": "true"}):
        hidden.decompose()

    # 2. Clean direct widget DOM tags if present
    for w_drop in soup(["mapwidgetv2", "asyncimagegroup"]):
        w_drop.decompose()
    for ent in soup.find_all("entity"):
        val = ent.get("value") or ent.get_text(strip=True)
        if val:
            strong = soup.new_tag("strong")
            strong.string = val
            ent.replace_with(strong)
        else:
            ent.decompose()
    for bd in soup.find_all("badge"):
        txt = bd.get_text(strip=True)
        if txt:
            em = soup.new_tag("em")
            em.string = txt
            bd.replace_with(em)
        else:
            bd.decompose()

    # 3. Clean literal escaped widget directives inside text nodes (skipping code/pre blocks)
    for text_node in list(soup.find_all(string=True)):
        if not isinstance(text_node, NavigableString):
            continue
        if any(parent.name in ("code", "pre", "script", "style") for parent in text_node.parents):
            continue
        node_str = str(text_node)
        if _has_widget_markup(node_str):
            cleaned_frag_str = _normalize_widget_markup(node_str)
            frag_soup = BeautifulSoup(cleaned_frag_str, "html.parser")
            text_node.replace_with(frag_soup)

    sources: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("http"):
            cleaned = clean_citation_url(href)
            if cleaned:
                a["href"] = cleaned
                a["target"] = "_blank"
                a["rel"] = "noopener noreferrer"
                if cleaned not in sources:
                    sources.append(cleaned)
            else:
                txt_low = a.get_text(strip=True).lower()
                if "mapbox" in txt_low or "openstreetmap" in txt_low or "improve this map" in txt_low:
                    a.decompose()
                    continue
                a.unwrap()

    # 4. Strip noisy framework attributes & unwrap non-semantic div/span wrappers
    for tag in list(soup.find_all(True)):
        if not isinstance(tag, Tag) or tag.attrs is None:
            continue
        if tag.name not in ALLOWED_TAGS:
            tag.unwrap()
        else:
            if tag.name == "a":
                href = tag.get("href")
                tag.attrs = {"href": href, "target": "_blank", "rel": "noopener noreferrer"} if href else {}
            else:
                tag.attrs = {}

    # Remove empty paragraphs / list items / headings
    for tag in list(soup.find_all(["p", "li", "h1", "h2", "h3", "h4"])):
        if not tag.get_text(strip=True):
            tag.decompose()

    clean_html = str(soup).strip()

    # 5. Extract clean plain text: prefer block-aware soup text if native_inner_text had widget directives
    if native_inner_text and not _has_widget_markup(native_inner_text):
        clean_text = normalize_plain_text(native_inner_text)
    else:
        clean_text = _soup_to_clean_text(soup)

    return clean_html, clean_text, sources
