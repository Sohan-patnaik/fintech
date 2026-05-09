from agents.base_agent import BaseAgent
from tools.yahoo_finance import get_stock_data


class MarketDataAgent(BaseAgent):
    async def run(self, state: dict) -> dict:
        ticker = state.get("ticker", "")
        if not ticker:
            return {**state, "market_data": None}
        try:
            data = await get_stock_data(ticker)

            # ✅ Guard: invalid/delisted ticker returns None instead of raising
            if data is None:
                self.logger.warning(f"[{ticker}] No market data — invalid or delisted ticker")
                return {
                    **state,
                    "market_data": None,
                    "errors": state.get("errors", []) + [f"No market data for '{ticker}' — invalid or delisted ticker"],
                }

            self.logger.info(f"Market data fetched for {ticker}: ${data.price}")
            return {**state, "market_data": data.model_dump()}

        except Exception as e:
            self.logger.error(f"MarketDataAgent error: {e}")
            return {**state, "market_data": None, "errors": state.get("errors", []) + [str(e)]}