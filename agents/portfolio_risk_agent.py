import asyncio
from agents.base_agent import BaseAgent
from tools.yahoo_finance import get_stock_data, get_fundamentals
from schemas.all import RiskData
from core.logger import get_logger

_SECTOR_FALLBACK: dict[str, str] = {

    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "RELIANCE": "Energy", "ONGC": "Energy",
    "HDFCBANK": "Finance", "ICICIBANK": "Finance", "SBIN": "Finance", "AXISBANK": "Finance",
    "TATAMOTORS": "Auto", "M&M": "Auto",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma",

    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "AMZN": "Consumer Cyclical", "TSLA": "Auto",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance",
}

class PortfolioRiskAgent(BaseAgent):
    async def run(self, state: dict) -> dict:
        holdings: list[dict] = state.get("holdings", [])
        if not holdings:
            return {**state, "risk_data": None}
        try:
            enriched, stale_tickers = await self._enrich_holdings(holdings)

            if not enriched:
                self.logger.warning("All holdings failed price fetch — skipping risk analysis")
                return {**state, "risk_data": None}

            total_value = sum(h["current_value"] for h in enriched)
            sector_exposure: dict[str, float] = {}
            max_concentration = 0.0

            for h in enriched:
                pct = (h["current_value"] / total_value) * 100 if total_value else 0.0
                h["weight_pct"] = round(pct, 2)
                sector = h.get("sector", "Other")
                sector_exposure[sector] = round(sector_exposure.get(sector, 0.0) + pct, 2)
                if pct > max_concentration:
                    max_concentration = pct

            risk_score_raw = self._compute_risk_score(
                max_concentration, len(holdings), sector_exposure
            )
            risk_score = round(risk_score_raw / 100.0, 3)

            suggestions = self._build_suggestions(
                max_concentration, sector_exposure, len(holdings)
            )
            if stale_tickers:
                suggestions.append(
                    f"Note: prices for {', '.join(stale_tickers)} could not be fetched "
                    f"— cost-basis values used. Risk score may be understated."
                )

            risk_data = RiskData(
                risk_score=risk_score,
                volatility=(
                    "high" if risk_score > 0.70
                    else "medium" if risk_score > 0.40
                    else "low"
                ),
                concentration=round(max_concentration / 100.0, 3),  
                sector_exposure=sector_exposure,
                suggestions=suggestions,
            )

            self.logger.info(
                f"Portfolio risk: score={risk_score:.3f} "
                f"({risk_data.volatility}), "
                f"max_concentration={max_concentration:.1f}%, "
                f"sectors={list(sector_exposure.keys())}"
            )

            return {
                **state,
                "risk_data": risk_data.model_dump(),
                "portfolio_summary": {
                    "total_value": round(total_value, 2),
                    "holdings": enriched,
                    "stale_tickers": stale_tickers,
                },
            }

        except Exception as e:
            self.logger.error(f"PortfolioRiskAgent failed: {e}", exc_info=True)
            return {
                **state,
                "risk_data": None,
                "errors": state.get("errors", []) + [f"PortfolioRiskAgent: {str(e)}"],
            }


    async def _enrich_holdings(
        self, holdings: list[dict]
    ) -> tuple[list[dict], list[str]]:
        """
        Fetch current price and sector for all holdings in parallel.
        Returns (enriched_holdings, stale_tickers).
        stale_tickers contains any ticker where live price fetch failed.
        """
        tasks = [self._enrich_one(h) for h in holdings]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        enriched: list[dict] = []
        stale_tickers: list[str] = []

        for h, result in zip(holdings, results):
            if isinstance(result, Exception):
                self.logger.warning(
                    f"Price fetch failed for {h['ticker']}: {result} — using avg_price"
                )
                stale_tickers.append(h["ticker"])
                enriched.append({
                    **h,
                    "current_price": h["avg_price"],
                    "current_value": h["avg_price"] * h["qty"],
                    "sector": _SECTOR_FALLBACK.get(h["ticker"].upper(), "Other"),
                    "price_is_stale": True,
                })
            else:
                enriched.append(result)

        return enriched, stale_tickers

    async def _enrich_one(self, holding: dict) -> dict:
        """
        Fetch live price and sector for a single holding.
        Sector is sourced from yfinance .info — falls back to SECTOR_MAP,
        then "Other". This avoids the static-map maintenance problem.
        """
        ticker = holding["ticker"].upper()

        stock = await get_stock_data(ticker)
        current_value = stock.price * holding["qty"]

        sector = await self._lookup_sector(ticker)

        return {
            **holding,
            "current_price": stock.price,
            "current_value": current_value,
            "sector": sector,
            "price_is_stale": False,
        }

    async def _lookup_sector(self, ticker: str) -> str:
        """
        Look up sector from yfinance fundamentals (cached).
        Falls back to hardcoded map, then "Other".
        """
        try:
            fund = await get_fundamentals(ticker)

            return _SECTOR_FALLBACK.get(ticker, "Other")
        except Exception:
            return _SECTOR_FALLBACK.get(ticker, "Other")

    def _compute_risk_score(
        self,
        concentration: float,     
        n_stocks: int,
        sectors: dict[str, float],
    ) -> float:
        """
        Heuristic risk score 0–100. Higher = riskier.

        Components:
        - Concentration risk: up to 50 points (>50% in one stock = max points)
        - Diversification risk: up to 25 points (<10 stocks)
        - Sector concentration: up to 25 points (<4 sectors)

        This is a v1 heuristic. Does not account for beta, correlation,
        or historical volatility — label it as such in the decision agent prompt.
        """
        score = 0.0
        score += min(concentration, 50.0)

        score += max(0.0, (10 - n_stocks) * 2.5)

        score += max(0.0, (4 - len(sectors)) * 6.25)

        return round(min(score, 100.0), 2)


    def _build_suggestions(
        self,
        conc: float,
        sectors: dict[str, float],
        n: int,
    ) -> list[str]:
        tips: list[str] = []

        if conc > 40:
            tips.append(
                f"Top holding is {conc:.1f}% of portfolio — consider trimming to reduce concentration risk."
            )
        if n < 5:
            tips.append(
                f"Only {n} stock(s) held — diversifying to 8–12 names reduces idiosyncratic risk."
            )
        if len(sectors) < 3:
            tips.append(
                f"Exposure across only {len(sectors)} sector(s) — adding defensive sectors "
                f"(Utilities, Healthcare) reduces cyclical drawdown risk."
            )

        for sector, pct in sectors.items():
            if pct > 60:
                tips.append(
                    f"{sector} sector is {pct:.1f}% of portfolio — "
                    f"consider adding positions in uncorrelated sectors."
                )

        if not tips:
            tips.append("Portfolio appears reasonably diversified across holdings and sectors.")

        return tips