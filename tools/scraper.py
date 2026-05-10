import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import httpx
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ── Shared client ─────────────────────────────────────────────────────────────
_client: httpx.AsyncClient | None = None

_JSON_HEADERS = {"Accept": "application/json", "User-Agent": "python-httpx/0.27"}
_RSS_HEADERS  = {"Accept": "application/rss+xml, application/xml, text/xml"}

_REQUEST_DELAY = 0.5


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _client


async def close_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


# ── Public API ────────────────────────────────────────────────────────────────

async def scrape_news(ticker: str, max_articles: int = 10) -> list[dict]:
    """
    Fetch recent news for a ticker using a 3-source fallback chain:
      1. Finnhub company news API  — stable JSON, real timestamps (PRIMARY)
      2. Yahoo Finance RSS feed    — XML, no JS rendering needed  (FALLBACK 1)
      3. Yahoo Finance search API  — rate-limited but sometimes works (FALLBACK 2)

    Returns [] only if all three sources fail. Never raises.
    """
    ticker = ticker.upper().strip()

    # ── Source 1: Finnhub ─────────────────────────────────────────────────────
    if hasattr(settings, "FINNHUB_API_KEY") and settings.FINNHUB_API_KEY.get_secret_value():
        try:
            articles = await _fetch_finnhub(ticker, max_articles)
            if articles:
                logger.info(f"[{ticker}] Finnhub: {len(articles)} articles")
                return articles
            logger.info(f"[{ticker}] Finnhub returned 0 articles")
        except Exception as e:
            logger.warning(f"[{ticker}] Finnhub failed: {e}")
    else:
        logger.warning(f"[{ticker}] FINNHUB_API_KEY not set — skipping Finnhub source")

    await asyncio.sleep(_REQUEST_DELAY)

    # ── Source 2: Yahoo RSS ───────────────────────────────────────────────────
    # RSS is XML-based — no JavaScript rendering, no CSS selector fragility.
    # Much more stable than HTML scraping.
    try:
        articles = await _fetch_yahoo_rss(ticker, max_articles)
        if articles:
            logger.info(f"[{ticker}] Yahoo RSS: {len(articles)} articles")
            return articles
        logger.info(f"[{ticker}] Yahoo RSS returned 0 articles")
    except Exception as e:
        logger.warning(f"[{ticker}] Yahoo RSS failed: {e}")

    await asyncio.sleep(_REQUEST_DELAY)

    # ── Source 3: Yahoo JSON API (last resort) ────────────────────────────────
    try:
        articles = await _fetch_yahoo_api(ticker, max_articles)
        if articles:
            logger.info(f"[{ticker}] Yahoo API: {len(articles)} articles")
            return articles
        logger.info(f"[{ticker}] Yahoo API returned 0 articles")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning(f"[{ticker}] Yahoo API rate-limited (429) — all sources exhausted")
        else:
            logger.warning(f"[{ticker}] Yahoo API HTTP {e.response.status_code}")
    except Exception as e:
        logger.warning(f"[{ticker}] Yahoo API failed: {e}")

    logger.error(f"[{ticker}] All news sources failed — returning empty list")
    return []


# ── Source implementations ────────────────────────────────────────────────────

async def _fetch_finnhub(ticker: str, max_articles: int) -> list[dict]:
    """
    Finnhub /company-news endpoint.
    Returns articles from the last 7 days. Free tier: 60 req/min.
    Docs: https://finnhub.io/docs/api/company-news
    """
    today = datetime.now(timezone.utc)
    from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": settings.FINNHUB_API_KEY.get_secret_value(),
    }

    client = _get_client()
    resp = await client.get(url, params=params, headers=_JSON_HEADERS)
    resp.raise_for_status()
    items = resp.json()

    if not isinstance(items, list):
        logger.warning(f"[{ticker}] Finnhub returned unexpected shape: {type(items)}")
        return []

    articles = []
    for item in items[:max_articles]:
        headline = (item.get("headline") or "").strip()
        url_ = (item.get("url") or "").strip()
        if not headline or not url_:
            continue

        published_ts = item.get("datetime")  # Unix timestamp
        published_at = (
            datetime.fromtimestamp(published_ts, tz=timezone.utc).isoformat()
            if published_ts
            else datetime.now(timezone.utc).isoformat()
        )

        articles.append({
            "headline": headline,
            "url": url_,
            "ticker": ticker,
            "source": item.get("source", "finnhub"),
            "published_at": published_at,
            # Finnhub provides a summary — much richer than headline-only
            "content": (item.get("summary") or "").strip()[:1500],
        })

    return articles


async def _fetch_yahoo_rss(ticker: str, max_articles: int) -> list[dict]:
    """
    Yahoo Finance RSS feed — XML format, no JavaScript needed.
    URL pattern: https://feeds.finance.yahoo.com/rss/2.0/headline?s=TICKER
    This is far more stable than HTML scraping because RSS is a published format.
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline"
    params = {"s": ticker, "region": "US", "lang": "en-US"}

    client = _get_client()
    resp = await client.get(url, params=params, headers=_RSS_HEADERS)
    resp.raise_for_status()

    # Parse XML — no CSS selectors, no fragile DOM traversal
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        logger.warning(f"[{ticker}] RSS XML parse error: {e}")
        return []

    # RSS structure: <rss><channel><item>...</item></channel></rss>
    ns = {"dc": "http://purl.org/dc/elements/1.1/"}
    items = root.findall(".//item")

    articles = []
    for item in items[:max_articles]:
        headline = (item.findtext("title") or "").strip()
        url_ = (item.findtext("link") or "").strip()
        if not headline or not url_:
            continue

        # pubDate is RFC 2822 format: "Mon, 09 May 2026 14:30:00 +0000"
        pub_date_str = item.findtext("pubDate") or ""
        published_at = _parse_rfc2822(pub_date_str) or datetime.now(timezone.utc).isoformat()

        description = (item.findtext("description") or "").strip()

        articles.append({
            "headline": headline,
            "url": url_,
            "ticker": ticker,
            "source": "yahoo_rss",
            "published_at": published_at,
            "content": description[:1500],
        })

    return articles


async def _fetch_yahoo_api(ticker: str, max_articles: int) -> list[dict]:
    """
    Yahoo Finance internal search API — last resort only.
    Frequently rate-limits; use only when Finnhub and RSS both fail.
    """
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": ticker, "newsCount": max_articles, "quotesCount": 0}

    client = _get_client()
    resp = await client.get(url, params=params, headers=_JSON_HEADERS)
    resp.raise_for_status()
    data = resp.json()

    articles = []
    for item in data.get("news", [])[:max_articles]:
        headline = (item.get("title") or "").strip()
        url_ = (item.get("link") or "").strip()
        if not headline or not url_:
            continue

        published_ts = item.get("providerPublishTime")
        published_at = (
            datetime.fromtimestamp(published_ts, tz=timezone.utc).isoformat()
            if published_ts
            else datetime.now(timezone.utc).isoformat()
        )

        articles.append({
            "headline": headline,
            "url": url_,
            "ticker": ticker,
            "source": item.get("publisher", "yahoo_finance"),
            "published_at": published_at,
            "content": "",
        })

    return articles


# ── Utilities ─────────────────────────────────────────────────────────────────

def _parse_rfc2822(date_str: str) -> str | None:
    """
    Parse RSS pubDate (RFC 2822) to ISO 8601 string.
    Returns None on parse failure — caller uses now() as fallback.
    """
    if not date_str:
        return None
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


async def enrich_article_content(article: dict, max_chars: int = 1500) -> dict:
    """
    Best-effort article body extraction. Not needed for Finnhub articles
    (they include a summary already). Use selectively for RSS/Yahoo articles
    where content is empty.
    """
    # Skip enrichment if we already have substantial content
    if len(article.get("content", "")) > 200:
        return article

    url = article.get("url", "")
    if not url:
        return article

    try:
        from bs4 import BeautifulSoup
        client = _get_client()
        resp = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"},
            timeout=httpx.Timeout(connect=3.0, read=8.0, write=3.0, pool=3.0),
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["nav", "footer", "aside", "script", "style", "figure"]):
            tag.decompose()
        body = soup.find("article") or soup.find("main") or soup.body
        if body:
            article["content"] = body.get_text(separator=" ", strip=True)[:max_chars]
    except Exception as e:
        logger.debug(f"Content enrichment failed for {url}: {e}")

    return article