from fastapi import APIRouter, Depends, HTTPException, Request, status
from datetime import datetime, timezone
from core.limiter import limiter
from core.logger import get_logger
from core.security import get_current_user_id
from db.session import get_supabase
from schemas.all import StockData, FundamentalsData
from tools.yahoo_finance import get_stock_data, get_fundamentals

router = APIRouter(prefix="/api/v1/stocks", tags=["stocks"])
logger = get_logger(__name__)


@router.get("/{ticker}/fundamentals", response_model=FundamentalsData)
@limiter.limit("3/minute")
async def stock_fundamentals(
    request: Request,
    ticker: str,
    user_id: int = Depends(get_current_user_id),
):
    ticker = ticker.upper().strip()

    try:
        # Fix: get_fundamentals is async — must be awaited
        return await get_fundamentals(ticker)
    except Exception as e:
        logger.warning(f"Fundamentals fetch failed for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not fetch fundamentals for {ticker}.",
        )


@router.get("/{ticker}", response_model=StockData)
@limiter.limit("3/minute")
async def stock_quote(
    request: Request,
    ticker: str,
    user_id: int = Depends(get_current_user_id),
):
    ticker = ticker.upper().strip()
    sb = get_supabase()

    # 1. Try cache first
    try:
        cached = (
            sb.table("cached_stock_data")
            .select("data, cached_at")
            .eq("ticker", ticker)
            .limit(1)
            .execute()
        )
        if cached.data:
            logger.debug(f"Cache hit for {ticker}")
            return StockData(**cached.data[0]["data"])
    except Exception as e:
        # Cache read failure is non-fatal — fall through to live fetch
        logger.warning(f"Cache read failed for {ticker}: {e}")

    # 2. Live fetch
    try:
        # Fix: get_stock_data is async — must be awaited
        data = await get_stock_data(ticker)
    except Exception as e:
        logger.error(f"Live fetch failed for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not fetch data for {ticker}.",
        )

    # 3. Write to cache — non-fatal if it fails
    try:
        sb.table("cached_stock_data").upsert({
            "ticker": ticker,
            "data": data.model_dump(),
            # Fix: datetime.utcnow() is deprecated — use timezone-aware UTC
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        logger.debug(f"Cache written for {ticker}")
    except Exception as e:
        logger.warning(f"Cache write failed for {ticker}: {e}")

    return data