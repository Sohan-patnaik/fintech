import re
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from core.security import get_current_user_id
from core.limiter import limiter
from core.logger import get_logger
from tools.scraper import scrape_news
from tools.rag import store_article, retrieve_relevant_news

from schemas.all import ArticleOut,RelevantNewsResponse,NewsResponse

router = APIRouter(prefix="/api/v1/news", tags=["news"])
logger = get_logger(__name__)

_TICKER_RE = re.compile(r"^[A-Z]{1,6}$")





def _validate_ticker(ticker: str) -> str:
    """Uppercase and validate ticker. Raises 422 on invalid input."""
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid ticker format: {ticker!r}. Must be 1–6 uppercase letters.",
        )
    return t


@router.get(
    "/{ticker}",
    response_model=NewsResponse,
    summary="Fetch and index news for a ticker",
)
@limiter.limit("10/minute")
async def get_news(
    request: Request,
    ticker: str,
    max_articles: int = Query(default=10, ge=1, le=50),
    user_id: int = Depends(get_current_user_id),
):
    ticker = _validate_ticker(ticker)
    logger.info(f"[{ticker}] News fetch requested by user {user_id}")

    articles = await scrape_news(ticker, max_articles=max_articles)

    if not articles:
        logger.warning(f"[{ticker}] Scraper returned 0 articles")
        return NewsResponse(ticker=ticker, count=0, stored=0, failed=0, articles=[])

    stored = 0
    failed = 0

    for article in articles:

        success = store_article(article)
        if success:
            stored += 1
        else:
            failed += 1
            logger.debug(f"[{ticker}] Failed to store: {article.get('url', 'unknown')}")

    logger.info(f"[{ticker}] Indexed {stored}/{len(articles)} articles ({failed} failed)")

    return NewsResponse(
        ticker=ticker,
        count=len(articles),
        stored=stored,
        failed=failed,
        articles=[
            ArticleOut(
                headline=a.get("headline", ""),
                url=a.get("url", ""),
                source=a.get("source", "unknown"),
                published_at=a.get("published_at"),
                sentiment_score=a.get("sentiment_score"),
            )
            for a in articles
        ],
    )


@router.get(
    "/{ticker}/relevant",
    response_model=RelevantNewsResponse,
    summary="Retrieve semantically relevant news for a ticker and query",
)
@limiter.limit("20/minute")
async def get_relevant_news(
    request: Request,
    ticker: str,
    q: str = Query(default="", max_length=500),
    n: int = Query(default=5, ge=1, le=20),
    max_age_days: int = Query(default=30, ge=1, le=365),
    user_id: int = Depends(get_current_user_id),
):
    ticker = _validate_ticker(ticker)

    query = q.strip() or f"{ticker} stock price analysis earnings outlook"

    logger.info(f"[{ticker}] Relevant news query: {query[:80]!r} (user {user_id})")

    articles = retrieve_relevant_news(
        ticker=ticker,
        query=query,
        n=n,
        max_age_days=max_age_days,
    )

    if not articles:
        logger.info(f"[{ticker}] No relevant articles found for query: {query[:80]!r}")

    return RelevantNewsResponse(
        ticker=ticker,
        query=query,
        count=len(articles),
        articles=[
            ArticleOut(
                headline=a.get("headline", ""),
                url=a.get("url", ""),
                source=a.get("source", "unknown"),
                published_at=a.get("published_at"),
                sentiment_score=a.get("sentiment_score"),
                relevance_score=a.get("relevance_score"),
            )
            for a in articles
        ],
    )