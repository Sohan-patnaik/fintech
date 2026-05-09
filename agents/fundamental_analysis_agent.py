import json
from agents.base_agent import BaseAgent
from tools.yahoo_finance import get_fundamentals
from tools.llm_client import chat_complete

SYSTEM = """You are a fundamental stock analyst. Given financial ratios, return ONLY valid JSON:
{
  "quality": "<poor|fair|good|excellent>",
  "highlights": ["<highlight1>", "<highlight2>"],
  "concerns": ["<concern1>"]
}"""


class FundamentalAnalysisAgent(BaseAgent):
    async def run(self, state: dict) -> dict:
        ticker = state.get("ticker", "")
        if not ticker:
            return {**state, "fundamentals_data": None}
        try:
            fund = await get_fundamentals(ticker)

            # ✅ Guard: yfinance returns None on 429 / empty response
            if fund is None:
                self.logger.warning(f"[{ticker}] Fundamentals unavailable (rate-limited or empty)")
                return {
                    **state,
                    "fundamentals_data": {
                        "quality": "unavailable",
                        "highlights": [],
                        "concerns": ["Fundamentals temporarily unavailable — rate limited"],
                    },
                }

            ratios_text = (
                f"P/E: {fund.pe_ratio}, EPS: {fund.eps}, ROE: {fund.roe}, "
                f"Debt/Equity: {fund.debt_to_equity}, Revenue Growth: {fund.revenue_growth}, "
                f"Analyst Rating: {fund.analyst_rating}"
            )
            raw = await chat_complete(SYSTEM, f"Ticker: {ticker}\nRatios: {ratios_text}")

            # ✅ Guard: LLM may return empty string or markdown-wrapped JSON
            if not raw or not raw.strip():
                raise ValueError("LLM returned empty response for fundamentals")

            # ✅ Strip markdown code fences if LLM wraps response in ```json ... ```
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            analysis = json.loads(cleaned)
            result = {**fund.model_dump(), **analysis}
            self.logger.info(f"Fundamentals for {ticker}: {analysis['quality']}")
            return {**state, "fundamentals_data": result}

        except json.JSONDecodeError as e:
            self.logger.error(f"[{ticker}] Fundamentals JSON parse failed: {e} | raw={raw!r}")
            return {
                **state,
                "fundamentals_data": {**fund.model_dump(), "quality": "unavailable", "highlights": [], "concerns": []},
                "errors": state.get("errors", []) + [f"Fundamentals parse error: {e}"],
            }
        except Exception as e:
            self.logger.error(f"FundamentalAnalysisAgent error: {e}")
            return {**state, "fundamentals_data": None, "errors": state.get("errors", []) + [str(e)]}