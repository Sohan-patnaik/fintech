from fastapi import APIRouter, Depends, HTTPException, Request, status
from datetime import datetime, timezone
from core.limiter import limiter
from core.logger import get_logger
from db.session import get_supabase
from schemas.all import PortfolioCreate, PortfolioOut, TransactionCreate, TransactionOut, Holding, ChatResponse
from core.security import get_current_user_id
from graph.workflow import run_pipeline

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])
logger = get_logger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _require_portfolio(sb, portfolio_id: int, user_id: int) -> dict:
    """
    Fetch a portfolio and verify ownership.
    Always raises 404 (never 403) to avoid leaking existence of other users' portfolios.
    """
    try:
        result = sb.table("portfolios").select(
            "id, user_id, name, holdings, created_at"
        ).eq("id", portfolio_id).execute()
    except Exception as e:
        logger.error(f"DB error fetching portfolio {portfolio_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch portfolio.",
        )

    if not result.data or result.data[0]["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found.",
        )

    return result.data[0]


def _apply_transaction(holdings: list[Holding], txn: TransactionCreate) -> list[Holding]:
    """
    Apply a BUY or SELL to the current holdings list.
    Returns a new list — does NOT mutate the input.
    Raises ValueError on invalid operations.
    """
    holdings = [h.model_copy() for h in holdings]
    existing = next((h for h in holdings if h.ticker == txn.ticker), None)

    action = txn.action.value if hasattr(txn.action, "value") else txn.action

    if action == "buy":
        if existing:
            total_qty = existing.qty + txn.quantity
            existing.avg_price = round(
                (existing.avg_price * existing.qty + txn.price * txn.quantity) / total_qty,
                4,
            )
            existing.qty = round(total_qty, 8)
        else:
            holdings.append(Holding(
                ticker=txn.ticker,
                qty=txn.quantity,
                avg_price=txn.price,
            ))

    elif action == "sell":
        if not existing:
            raise ValueError(f"Cannot sell {txn.ticker}: no position held.")
        if txn.quantity > existing.qty:
            raise ValueError(
                f"Cannot sell {txn.quantity} of {txn.ticker}: only {existing.qty} held."
            )
        existing.qty = round(existing.qty - txn.quantity, 8)
        if existing.qty <= 0:
            holdings = [h for h in holdings if h.ticker != txn.ticker]

    return holdings


# ── routes ─────────────────────────────────────────────────────────────────────

@router.post("", response_model=PortfolioOut, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def create_portfolio(
    request: Request,
    payload: PortfolioCreate,
    user_id: int = Depends(get_current_user_id),
):
    sb = get_supabase()
    try:
        result = sb.table("portfolios").insert({
            "user_id": user_id,
            "name": payload.name,
            "holdings": [h.model_dump() for h in payload.holdings],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.error(f"Portfolio creation failed for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio creation failed.",
        )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio creation failed — no data returned.",
        )

    p = result.data[0]
    logger.info(f"Portfolio created: id={p['id']} user={user_id}")
    return PortfolioOut(
        id=p["id"],
        name=p["name"],
        holdings=p["holdings"] or [],
        created_at=p["created_at"],
    )


@router.get("", response_model=list[PortfolioOut])
async def list_portfolios(
    request: Request,
    user_id: int = Depends(get_current_user_id),
):
    sb = get_supabase()
    try:
        result = (
            sb.table("portfolios")
            .select("id, name, holdings, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as e:
        logger.error(f"Portfolio list failed for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch portfolios.",
        )

    return [
        PortfolioOut(
            id=p["id"],
            name=p["name"],
            holdings=p["holdings"] or [],
            created_at=p["created_at"],
        )
        for p in result.data
    ]


@router.get("/{portfolio_id}/analyze", response_model=ChatResponse)
async def analyze_portfolio(
    request: Request,
    portfolio_id: int,
    user_id: int = Depends(get_current_user_id),
):
    sb = get_supabase()
    p = _require_portfolio(sb, portfolio_id, user_id)

    holdings = p["holdings"] or []
    if not holdings:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Portfolio has no holdings to analyze.",
        )

    # Normalize to plain dicts with only the keys FinState expects.
    # ticker="" intentionally — run_pipeline calls _extract_ticker(query) itself
    # for portfolio-wide analysis where no single ticker is the focus.
    normalized_holdings = [
        {"ticker": h["ticker"], "qty": h["qty"], "avg_price": h["avg_price"]}
        for h in holdings
    ]

    try:
        result = await run_pipeline(
            query="Analyze this portfolio's risk, diversification, and overall health.",
            ticker="",
            holdings=normalized_holdings,
        )
    except Exception as e:
        logger.error(f"Pipeline failed for portfolio {portfolio_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio analysis failed.",
        )

    decision = result.get("decision", {})
    if not decision:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis could not be completed. Please try again.",
        )

    return ChatResponse(
        recommendation=decision.get("recommendation", "hold"),
        confidence=decision.get("confidence", 0.0),
        reasons=decision.get("reasons") or ["Analysis incomplete."],
        risks=decision.get("risks") or ["Unable to assess risks."],
        data_sources=decision.get("data_sources") or [],
    )


@router.post(
    "/{portfolio_id}/transaction",
    response_model=TransactionOut,
    status_code=status.HTTP_201_CREATED,
)
async def add_transaction(
    request: Request,
    portfolio_id: int,
    payload: TransactionCreate,
    user_id: int = Depends(get_current_user_id),
):
    sb = get_supabase()
    p = _require_portfolio(sb, portfolio_id, user_id)

    current_holdings = [Holding(**h) for h in (p["holdings"] or [])]
    try:
        updated_holdings = _apply_transaction(current_holdings, payload)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    executed_at = datetime.now(timezone.utc).isoformat()

    try:
        sb.table("portfolios").update({
            "holdings": [h.model_dump() for h in updated_holdings],
        }).eq("id", portfolio_id).execute()

        txn_result = sb.table("transactions").insert({
            "portfolio_id": portfolio_id,
            "ticker": payload.ticker,
            "action": payload.action.value if hasattr(payload.action, "value") else payload.action,
            "quantity": payload.quantity,
            "price": payload.price,
            "executed_at": executed_at,
        }).execute()
    except Exception as e:
        logger.error(f"Transaction failed for portfolio {portfolio_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transaction could not be recorded.",
        )

    if not txn_result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transaction recorded but response data missing.",
        )

    txn = txn_result.data[0]
    logger.info(
        f"Transaction recorded: {payload.action} "
        f"{payload.quantity} {payload.ticker} @ ${payload.price} "
        f"portfolio={portfolio_id} user={user_id}"
    )

    return TransactionOut(
        id=txn["id"],
        ticker=txn["ticker"],
        action=txn["action"],
        quantity=txn["quantity"],
        price=txn["price"],
        executed_at=txn["executed_at"],
    )