import re
from agents.base_agent import BaseAgent

INTENT_MAP = {
    "full_analysis": [
        "should i", "buy", "sell", "invest", "recommendation", "worth it",
        "good investment", "long term", "short term", "entry point",
        "exit strategy", "hold or sell", "is it good", "future potential",
        "upside", "downside", "target price", "analysis", "opinion"
    ],
    "news_only": [
        "news", "sentiment", "latest", "recent",
        "what's happening", "updates", "headlines",
        "market news", "company news", "any news",
        "current events", "buzz", "trend"
    ],
    "price_only": [
        "price", "quote", "current", "how much", "trading at",
        "stock price", "live price", "today price",
        "current value", "market price", "last traded",
        "ltp", "cmp"
    ],
    "portfolio": [
        "portfolio", "my holdings", "risk", "diversif",
        "allocation", "asset allocation", "rebalanc",
        "overweight", "underweight", "exposure",
        "concentration", "my stocks", "review my portfolio",
        "portfolio analysis", "holdings review"
    ],
    "fundamentals": [
        "pe ratio", "eps", "roe", "fundamentals", "earnings",
        "balance sheet", "income statement", "cash flow",
        "valuation", "book value", "debt", "profit",
        "revenue", "growth", "margin", "financials",
        "quarter results", "annual report"
    ],
}


class RouterAgent(BaseAgent):
    async def run(self, state: dict) -> dict:
        query = state.get("query", "").lower()
        intent = self._classify(query)
        agents_to_run = self._resolve_agents(intent)
        self.logger.info(f"Intent: {intent} → agents: {agents_to_run}")
        return {**state, "intent": intent, "agents_to_run": agents_to_run}

    def _classify(self, query: str) -> str:
        for intent, keywords in INTENT_MAP.items():
            if any(kw in query for kw in keywords):
                return intent
            return "full_analysis"

    def _resolve_agents(self, intent: str) -> list[str]:
        mapping = {
            "full_analysis": ["market_data", "news_analysis", "fundamentals", "decision"],
            "news_only":     ["news_analysis"],
            "price_only":    ["market_data"],
            "portfolio":     ["market_data", "portfolio_risk"],
            "fundamentals":  ["fundamentals", "market_data"],
        }
        return mapping.get(intent, ["market_data", "news_analysis", "fundamentals", "decision"])
