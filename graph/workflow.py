import asyncio
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.router_agent import RouterAgent
from agents.market_data_agent import MarketDataAgent
from agents.news_analysis_agent import NewsAnalysisAgent
from agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from agents.portfolio_risk_agent import PortfolioRiskAgent
from agents.decision_agent import DecisionAgent
from core.logger import get_logger

logger = get_logger(__name__)


class FinState(TypedDict, total=False):
    query: str
    ticker: str
    holdings: str
    intent: str
    agents_to_run: list[str]
    market_data: Optional[dict]
    news_data: Optional[dict]
    fundamentals_data: Optional[dict]
    risk_data: Optional[dict]
    decision: Optional[dict]
    errors: list[str]


router = RouterAgent()
market = MarketDataAgent()
news = NewsAnalysisAgent()
fundamentals = FundamentalAnalysisAgent()
risk = PortfolioRiskAgent()
decision_agent = DecisionAgent()


async def route_node(state: FinState) -> FinState:
    return await router.run(state)


async def parallel_analysis_node(state: FinState) -> FinState:
    agents_to_run = state.get("agents_to_run", [])
    tasks = {}

    if "market_data" in agents_to_run:
        tasks["market"] = market.run(state)
    if "news_analysis" in agents_to_run:
        tasks["news"] = news.run(state)
    if "fundamentals" in agents_to_run:
        tasks["fundamentals"] = fundamentals.run(state)
    if "portfolio_risk" in agents_to_run:
        tasks["risk"] = risk.run(state)

    if not tasks:
        return state

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    merged = dict(state)

    for key, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.error(f"Agent {key} raised: {result}")
            merged.setdefault("errors", []).append(str(result))
        else:
            merged.update(result)

    return merged


async def decision_node(state: FinState) -> FinState:
    if "decision" not in state.get("agents_to_run", []):
        return state
    return await decision_agent.run(state)


def should_decide(state: FinState) -> str:
    if state.get("intent") in ("full_analysis", "fundamentals"):
        return "decision_node"
    return END


def build_graph() -> StateGraph:
    g = StateGraph(FinState)

    g.add_node("router", route_node)
    g.add_node("analysis", parallel_analysis_node)
    g.add_node("decision_node", decision_node)

    g.set_entry_point("router")
    g.add_edge("router", "analysis")
    g.add_conditional_edges("analysis", should_decide, {
        "decision_node": "decision_node",
        END: END
    })
    g.add_edge("decision_node", END)

    return g.compile()


graph = build_graph()


async def run_pipeline(query: str, ticker: str = "", holdings: list[dict] = None):
    ticker = ticker.upper().strip() if ticker else _extract_ticker(query)
    initial_state: FinState = {
        "query": query,
        "ticker": ticker,
        "holdings": holdings or [],
        "errors": [],
    }
    final_state = await graph.ainvoke(initial_state)
    return _format_response(final_state)


def _extract_ticker(query: str) -> str:
    import re
    match = re.search(r'\b([A-Z]{1,5})\b', query.upper())
    return match.group(1) if match else ""


def _format_response(state: FinState) -> dict:
    decision = state.get("decision") or {}
    return {
        "recommendation": decision.get("recommendation", "INFO"),
        "confidence": decision.get("confidence", 0.0),
        "reasons": decision.get("reasons", []),
        "risks": decision.get("risks", []),
        "data_sources": decision.get("data_sources", []),
        "market_data": state.get("market_data"),
        "news_data": state.get("news_data"),
        "fundamentals_data": state.get("fundamentals_data"),
        "risk_data": state.get("risk_data"),
        "errors": state.get("errors", []),
    }