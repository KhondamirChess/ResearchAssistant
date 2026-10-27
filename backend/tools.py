"""The two V1 tools: WebSearchTool and DocumentReaderTool.

Both return plain dicts. On failure they return {"error": "..."} instead of raising,
so the agent loop can show the problem and carry on (functional requirement F7).
"""
import ipaddress
import re
import socket
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx

from . import config

TIMEOUT = 15.0
MAX_BYTES = 5_000_000
MAX_REDIRECTS = 3
HEADERS = {"User-Agent": "ResearchAssistantPrototype/0.1 (student project)"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_safe_url(url: str) -> bool:
    """Block non-http(s) URLs and anything that resolves to a private/local address.

    The agent fetches URLs chosen by an LLM, so this stops it from reaching
    localhost, your router or cloud metadata endpoints (SSRF).
    """
    try:
        p = urlparse(url)
    except ValueError:
        return False
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


# ---------------------------------------------------------------- web search
def web_search(query: str, top_k: int = 5, date_range: str | None = None) -> dict:
    """Search the web through Tavily. date_range: day | week | month | year."""
    query = (query or "").strip()
    if not query:
        return {"error": "empty query"}
    if not config.TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY is not set"}
    top_k = max(1, min(int(top_k or 5), 10))
    body = {"query": query, "max_results": top_k, "search_depth": "basic"}
    if date_range in ("day", "week", "month", "year"):
        body["time_range"] = date_range
    try:
        r = httpx.post(
            "https://api.tavily.com/search",
            json=body,
            headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"search API returned HTTP {e.response.status_code}"}
    except (httpx.HTTPError, ValueError) as e:
        return {"error": f"search failed: {type(e).__name__}"}
    results = [
        {"title": x.get("title") or x.get("url", ""), "url": x.get("url", ""),
         "snippet": (x.get("content") or "")[:500]}
        for x in data.get("results", []) if x.get("url")
    ]
    return {"results": results[:top_k], "retrieved_at": _now()}


# ----------------------------------------------------------- document reader
def _fetch(url: str) -> tuple[bytes, str]:
    """GET with manual redirects so every hop is re-checked by is_safe_url."""
    with httpx.Client(timeout=TIMEOUT, headers=HEADERS, follow_redirects=False) as c:
        for _ in range(MAX_REDIRECTS + 1):
            if not is_safe_url(url):
                raise ValueError("URL is not allowed (private address or bad scheme)")
            with c.stream("GET", url) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers.get("location", ""))
                    continue
                r.raise_for_status()
                chunks, size = [], 0
                for chunk in r.iter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        break
                    chunks.append(chunk)
                return b"".join(chunks), r.headers.get("content-type", "").lower()
        raise ValueError("too many redirects")


def _extract_text(raw: bytes, content_type: str, url: str) -> tuple[str, str]:
    """Return (title, text) for a PDF or HTML page."""
    if "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
        import fitz  # pymupdf

        with fitz.open(stream=raw, filetype="pdf") as doc:
            title = (doc.metadata or {}).get("title") or url
            text = "\n\n".join(page.get_text() for page in doc)
        return title, text
    import trafilatura

    html = raw.decode("utf-8", errors="replace")
    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    title = url
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and meta.title:
            title = meta.title
    except Exception:
        pass
    return title, text


_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = {"the", "and", "for", "are", "what", "which", "with", "that", "this", "from",
         "how", "why", "does", "did", "have", "has", "was", "were", "about", "into"}


def select_passages(text: str, focus: str, n: int = 4, size: int = 700) -> list[dict]:
    """Pick the n paragraphs that share the most words with the focus question."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40]
    chunks: list[str] = []
    for p in paras:  # split very long paragraphs
        chunks += [p[i:i + size] for i in range(0, len(p), size)]
    words = {w for w in _WORD.findall((focus or "").lower()) if w not in _STOP}
    scored = []
    for idx, ch in enumerate(chunks):
        score = len(words & set(_WORD.findall(ch.lower())))
        scored.append((score, idx, ch))
    best = sorted(scored, key=lambda t: (-t[0], t[1]))[:n]
    best.sort(key=lambda t: t[1])  # keep document order
    return [{"text": ch, "location": f"chunk {idx + 1} of {len(chunks)}"}
            for _, idx, ch in best]


def read_document(url: str, focus_question: str = "") -> dict:
    """Fetch a page or PDF and return the passages most relevant to the question."""
    try:
        raw, ctype = _fetch(url)
        title, text = _extract_text(raw, ctype, url)
    except (httpx.HTTPError, ValueError) as e:
        return {"error": f"could not read {url}: {e}"}
    except Exception as e:  # parser problems (corrupt PDF etc.)
        return {"error": f"could not parse {url}: {type(e).__name__}"}
    if len(text.strip()) < 80:
        return {"error": "no readable text found (scanned PDF or JavaScript-only page?)"}
    return {"title": title, "url": url,
            "passages": select_passages(text, focus_question),
            "retrieved_at": _now()}


# ---------------------------------------------------- tool specs for the LLM
TOOL_SPECS = [
    {
        "name": "web_search",
        "description": "Search the web for pages and papers relevant to a sub-question. "
                       "Returns title, url and snippet for each result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                "date_range": {"type": "string", "enum": ["day", "week", "month", "year"]},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_document",
        "description": "Fetch a web page or PDF and return the passages most relevant "
                       "to focus_question. Only use URLs returned by web_search.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "focus_question": {"type": "string"}},
            "required": ["url", "focus_question"],
        },
    },
]

HANDLERS = {"web_search": web_search, "read_document": read_document}
