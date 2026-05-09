import asyncio
import random
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import TypeVar, Callable, Any
from schemas.all import StockData, FundamentalsData
from core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

STOCK_CACHE_TTL_SECONDS = 60
FUNDAMENTALS_CACHE_TTL_SECONDS = 4 * 60 * 60

# ✅ Removed requests.Session entirely — newer yfinance manages its own
# curl_cffi session internally. Passing a requests.Session raises YFDataException.

_cache: dict[str, dict] = {}
_cache_locks: dict[str, asyncio.Lock] = {}


def _get_lock(key: str) -> asyncio.Lock:
    """Return (creating if needed) a per-key asyncio lock."""
    if key not in _cache_locks:
        _cache_locks[key] = asyncio.Lock()
    return _cache_locks[key]


def _get_from_cache(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and entry["expiry"] > datetime.now(timezone.utc):
        return entry["value"]
    return None


def _set_cache(key: str, value: Any, ttl_seconds: int) -> None:
    _cache[key] = {
        "value": value,
        "expiry": datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
    }


_RETRYABLE_ERRORS = (
    "Too Many Requests",
    "429",
    "Connection reset",
    "RemoteDisconnected",
    "Read timed out",
    "ConnectionError",
)


async def _fetch_with_retry(
    func: Callable[[], T],
    retries: int = 4,
    base_delay: float = 1.0,
) -> T | None:
    """
    Run a sync yfinance callable in a thread pool with async-safe exponential
    backoff. Uses asyncio.to_thread so the event loop is never blocked.

    Returns None if the ticker has no data (invalid/delisted).
    Raises only on unrecoverable non-data errors.
    """
    last_exc: Exception | None = None

    for attempt in range(retries):
        try:
            return await asyncio.to_thread(func)

        except ValueError as e:
            # ✅ "No historical data" is not retryable — ticker is invalid/delisted
            logger.warning(f"No data (attempt {attempt + 1}/{retries}): {e} — not retrying")
            return None

        except Exception as e:
            last_exc = e
            err_str = str(e)

            is_retryable = any(token in err_str for token in _RETRYABLE_ERRORS)
            is_last_attempt = attempt == retries - 1

            if is_last_attempt or not is_retryable:
                logger.error(
                    f"Fetch failed (attempt {attempt + 1}/{retries}): {e}",
                    exc_info=not is_retryable,
                )
                raise

            delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
            logger.warning(
                f"Retryable error on attempt {attempt + 1}/{retries}: {e}. "
                f"Backing off {delay:.2f}s"
            )
            await asyncio.sleep(delay)

    raise last_exc or RuntimeError("Max retries exceeded")


def _get_ticker(ticker: str) -> yf.Ticker:
    # ✅ No session argument — let yfinance manage curl_cffi internally
    return yf.Ticker(ticker)


def _safe_fast_info(ticker_obj: yf.Ticker, attr: str, default: Any = None) -> Any:
    """
    getattr wrapper for yfinance FastInfo objects.
    fast_info is NOT a dict — .get() raises AttributeError.
    """
    try:
        val = getattr(ticker_obj.fast_info, attr, default)
        return val if val is not None else default
    except Exception:
        return default


async def get_stock_data(ticker: str) -> StockData | None:
    """
    Fetch real-time price, volume, and 52-week range for a ticker.
    Results cached for STOCK_CACHE_TTL_SECONDS (default 60s).

    Returns None if the ticker is invalid or delisted.
    """
    ticker = ticker.upper().strip()
    cache_key = f"stock:{ticker}"

    async with _get_lock(cache_key):
        cached = _get_from_cache(cache_key)
        if cached:
            logger.debug(f"[{ticker}] stock cache hit")
            return cached

        def _fetch() -> StockData:
            t = _get_ticker(ticker)
            hist = t.history(period="5d")

            if hist.empty:
                # ✅ Raise ValueError — _fetch_with_retry catches this and returns None
                raise ValueError(f"No historical data returned for {ticker}")

            price = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
            change_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0.0

            return StockData(
                ticker=ticker,
                price=round(price, 2),
                change_pct=round(change_pct, 4),
                volume=float(_safe_fast_info(t, "last_volume", 0)),
                week_52_high=round(float(_safe_fast_info(t, "year_high", price)), 2),
                week_52_low=round(float(_safe_fast_info(t, "year_low", price)), 2),
                market_cap=_safe_fast_info(t, "market_cap"),
            )

        # ✅ Returns None for invalid tickers instead of raising
        data = await _fetch_with_retry(_fetch)

        if data is None:
            logger.warning(f"[{ticker}] No stock data — invalid or delisted ticker")
            return None

        _set_cache(cache_key, data, ttl_seconds=STOCK_CACHE_TTL_SECONDS)
        logger.info(f"[{ticker}] stock data fetched: ${data.price} ({data.change_pct:+.2f}%)")
        return data


async def get_fundamentals(ticker: str) -> FundamentalsData | None:
    """
    Fetch fundamental metrics (P/E, EPS, ROE, etc.) for a ticker.
    Results cached for FUNDAMENTALS_CACHE_TTL_SECONDS (default 4h).

    Returns None if the ticker is invalid or data is unavailable.
    """
    ticker = ticker.upper().strip()
    cache_key = f"fund:{ticker}"

    async with _get_lock(cache_key):
        cached = _get_from_cache(cache_key)
        if cached:
            logger.debug(f"[{ticker}] fundamentals cache hit")
            return cached

        def _fetch() -> FundamentalsData:
            t = _get_ticker(ticker)
            info = t.info

            if not info or info.get("symbol") is None:
                raise ValueError(f"No fundamentals data returned for {ticker}")

            return FundamentalsData(
                ticker=ticker,
                pe_ratio=info.get("trailingPE"),
                eps=info.get("trailingEps"),
                roe=info.get("returnOnEquity"),
                debt_to_equity=info.get("debtToEquity"),
                revenue_growth=info.get("revenueGrowth"),
                analyst_rating=info.get("recommendationKey"),
            )

        data = await _fetch_with_retry(_fetch)

        if data is None:
            logger.warning(f"[{ticker}] No fundamentals data — invalid or delisted ticker")
            return None

        _set_cache(cache_key, data, ttl_seconds=FUNDAMENTALS_CACHE_TTL_SECONDS)
        logger.info(f"[{ticker}] fundamentals fetched: P/E={data.pe_ratio}, EPS={data.eps}")
        return data


async def get_stock_and_fundamentals(ticker: str) -> tuple[StockData | None, FundamentalsData | None]:
    """
    Convenience function: fetch both in parallel.
    Used by the router agent when full analysis is requested.
    """
    stock, fundamentals = await asyncio.gather(
        get_stock_data(ticker),
        get_fundamentals(ticker),
    )
    return stock, fundamentals