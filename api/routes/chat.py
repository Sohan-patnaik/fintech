import uuid
import re
import json
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from schemas.all import ChatRequest, ChatResponse, RecommendationType
from graph.workflow import run_pipeline, stream_pipeline
from core.security import get_current_user_id
from core.logger import get_logger
from core.limiter import limiter

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
logger = get_logger(__name__)

_TICKER_BLACKLIST = {
    # 2-letter noise words (critical — these were causing "IS" to match before "AAPL")
    "IS", "IN", "AT", "BE", "DO", "GO", "IF", "MY", "NO", "OF",
    "ON", "OR", "SO", "TO", "UP", "AN", "AS", "BY", "HE", "ME",
    "WE", "AM", "ID", "IT", "US", "UK", "EU", "RE", "VC",
    # Question words
    "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "WHY", "HOW",
    # Action words
    "BUY", "SELL", "HOLD", "GET", "GIVE", "MAKE", "TAKE", "LOOK",
    # Finance acronyms that are NOT tickers
    "ETF", "IPO", "GDP", "CEO", "CFO", "CTO", "ROE", "EPS",
    "NAV", "AUM", "FII", "DII", "NFO", "SIP", "EMI",
    # Common English words
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
    "HAS", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "NEW", "NOW",
    "ITS", "MAY", "WILL", "WITH", "THIS", "THAT", "HAVE", "FROM",
    # Currency codes
    "USD", "INR", "GBP", "JPY",
    # Sector abbreviations
    "AI", "ML", "EV", "PE",
}


@router.post("", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    payload: ChatRequest,
    user_id: int = Depends(get_current_user_id),
):
    request_id = str(uuid.uuid4())[:8]
    log = logger.bind(request_id=request_id, user_id=user_id) if hasattr(logger, "bind") else logger

    query = payload.query.strip()
    log.info(f"[{request_id}] Chat request: {query[:80]!r}")

    ticker_hint = _extract_ticker_hint(query)
    if ticker_hint:
        log.debug(f"[{request_id}] Ticker hint extracted: {ticker_hint}")
    else:
        log.debug(f"[{request_id}] No ticker hint found — router agent will resolve")

    try:
        result = await run_pipeline(
            query=query,
            ticker=ticker_hint,
            holdings=[],  # chat endpoint has no portfolio context
        )
    except Exception as e:
        log.error(f"[{request_id}] Pipeline raised: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed. Reference ID: {request_id}",
        )

    return _build_response(result, request_id, log)


@router.post("/stream")
@limiter.limit("10/minute")
async def chat_stream(
    request: Request,
    payload: ChatRequest,
    user_id: int = Depends(get_current_user_id),
):
    request_id = str(uuid.uuid4())[:8]
    query = payload.query.strip()
    ticker_hint = _extract_ticker_hint(query)
    
    async def event_generator():
        async for chunk in stream_pipeline(query=query, ticker=ticker_hint, holdings=[]):
            event_type = chunk["event"]
            data_str = json.dumps(chunk["data"])
            yield f"event: {event_type}\ndata: {data_str}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _build_response(result: dict, request_id: str, log) -> ChatResponse:
    """
    Build a validated ChatResponse from the pipeline result dict.

    Handles two possible shapes from _format_response:
      Shape A — nested:  result["decision"]["recommendation"]  (correct)
      Shape B — flat:    result["recommendation"]              (pipeline bug workaround)

    ChatResponse validators enforce:
      - recommendation  -> lowercased via coerce_recommendation
      - confidence      -> 0.0 <= x <= 1.0
      - reasons         -> non-empty list
      - risks           -> non-empty list
      - data_sources    -> non-empty list
    So we always provide safe fallbacks before hitting Pydantic.
    """
    if result.get("is_fallback"):
        log.warning(f"[{request_id}] Pipeline returned fallback response — analysis incomplete")

    # Support both nested and flat pipeline output shapes
    decision = result.get("decision") or {}
    if not decision:
        # Flat shape: fields live directly on the state dict
        decision = {
            "recommendation": result.get("recommendation"),
            "confidence":     result.get("confidence"),
            "reasons":        result.get("reasons"),
            "risks":          result.get("risks"),
            "data_sources":   result.get("data_sources"),
        }

    if not decision.get("recommendation"):
        log.error(f"[{request_id}] No decision found — state keys: {list(result.keys())}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis could not be completed. Reference ID: {request_id}",
        )

    # Safe fallbacks — required because ChatResponse.non_empty_lists validator
    # raises ValueError if any of these are empty, causing a 500.
    reasons      = decision.get("reasons")      or ["Analysis based on available data."]
    risks        = decision.get("risks")        or ["Insufficient data to assess risks."]
    data_sources = decision.get("data_sources") or ["market_data"]

    # Clamp confidence to valid schema range before Pydantic sees it (ge=0.0, le=1.0)
    raw_confidence = decision.get("confidence", 0.0) or 0.0
    confidence = max(0.0, min(1.0, float(raw_confidence)))

    try:
        return ChatResponse(
            recommendation=decision.get("recommendation", RecommendationType.HOLD.value),
            confidence=confidence,
            reasons=reasons,
            risks=risks,
            data_sources=data_sources,
            raw_data=result.get("raw_data"),  # Optional[dict] — None is fine
        )
    except Exception as e:
        log.error(
            f"[{request_id}] ChatResponse validation failed: {e} | decision={decision}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Response formatting failed. Reference ID: {request_id}",
        )


def _extract_ticker_hint(query: str) -> str:
    """
    Best-effort ticker extraction from a natural language query.
    Returns a hint for the router agent — NOT authoritative.

    Strategy:
    1. Uppercase the full query and find all 2-6 char word tokens
    2. Filter against blacklist
    3. Sort survivors by length descending — real tickers (AAPL, MSFT, TSLA)
       are longer than noise words (IS, IN, AT), so longest match wins
    4. Return first survivor or "" if none found

    Examples:
      "what is aapl?"             -> "AAPL"  (IS filtered, AAPL wins by length)
      "should I buy MSFT?"        -> "MSFT"
      "AI outlook for the market" -> ""      (all filtered)
      "compare AAPL and MSFT"     -> "AAPL"  (first longest; agent handles multi-ticker)
    """
    candidates = re.findall(r"\b([A-Z]{2,6})\b", query.upper())

    filtered = [c for c in candidates if c not in _TICKER_BLACKLIST]

    # Longest candidates first — ticker symbols beat noise words
    filtered.sort(key=len, reverse=True)

    return filtered[0] if filtered else ""