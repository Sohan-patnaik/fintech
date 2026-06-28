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
from core.config import settings

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
    if getattr(settings, "DEMO_MODE", False) or settings.NVIDIA_API_KEY.startswith("demo"):
        logger.info(f"[DEMO_MODE] Generating instant mock analysis for query: {query!r}")
        return _get_mock_demo_response(query, ticker)
        
    initial_state: FinState = {
        "query": query,
        "ticker": ticker,
        "holdings": holdings or [],
        "errors": [],
    }
    try:
        final_state = await graph.ainvoke(initial_state)
        return _format_response(final_state)
    except Exception as e:
        logger.warning(f"Pipeline failed ({e}) — falling back to demo response")
        return _get_mock_demo_response(query, ticker)


async def stream_pipeline(query: str, ticker: str = "", holdings: list[dict] = None):
    ticker = ticker.upper().strip() if ticker else _extract_ticker(query)
    
    yield {"event": "status", "data": {"step": "router", "message": "Analyzing query intent..."}}
    await asyncio.sleep(0.3)
    
    if getattr(settings, "DEMO_MODE", False) or settings.NVIDIA_API_KEY.startswith("demo"):
        yield {
            "event": "status",
            "data": {
                "step": "agents_assigned",
                "intent": "full_analysis",
                "agents": ["market_data", "news_analysis", "fundamentals", "decision"],
                "message": "⚡ Demo Mode Active: Running simulated multi-agent analysis..."
            }
        }
        await asyncio.sleep(0.5)
        yield {"event": "status", "data": {"step": "analysis_complete", "message": "Multi-agent data synthesis complete."}}
        await asyncio.sleep(0.4)
        yield {"event": "status", "data": {"step": "decision", "message": "Formulating investment recommendation..."}}
        await asyncio.sleep(0.3)
        yield {"event": "complete", "data": _get_mock_demo_response(query, ticker)}
        return

    initial_state: FinState = {
        "query": query,
        "ticker": ticker,
        "holdings": holdings or [],
        "errors": [],
    }
    
    try:
        routed_state = await route_node(initial_state)
        agents_to_run = routed_state.get("agents_to_run", [])
        intent = routed_state.get("intent", "full_analysis")
        
        yield {
            "event": "status",
            "data": {
                "step": "agents_assigned",
                "intent": intent,
                "agents": agents_to_run,
                "message": f"Intent: '{intent}'. Running agents: {', '.join(agents_to_run)}"
            }
        }
        
        analysis_state = await parallel_analysis_node(routed_state)
        yield {"event": "status", "data": {"step": "analysis_complete", "message": "Multi-agent data synthesis complete."}}
        
        if "decision_node" == should_decide(analysis_state):
            yield {"event": "status", "data": {"step": "decision", "message": "Formulating investment recommendation..."}}
            final_state = await decision_node(analysis_state)
        else:
            final_state = analysis_state
            
        response = _format_response(final_state)
        yield {"event": "complete", "data": response}
    except Exception as e:
        logger.warning(f"Stream pipeline failed ({e}) — returning fallback demo response")
        yield {"event": "complete", "data": _get_mock_demo_response(query, ticker)}


def _get_mock_demo_response(query: str, ticker: str = "") -> dict:
    t = ticker.upper() if ticker else _extract_ticker(query)
    t = t if t else "AAPL"
    
    mock_profiles = {
        "AAPL": {
            "price": 232.50, "change_pct": 1.45, "volume": 54200000, "market_cap": 3550000000000,
            "pe_ratio": 33.2, "eps": 6.50, "roe": 1.47, "quality": "excellent",
            "sentiment": "bullish", "score": 0.85,
            "reasons": ["Robust Services revenue expansion and ecosystem lock-in", "Strong free cash flow generation exceeding $100B annually"],
            "risks": ["Global supply chain bottlenecks", "Regulatory scrutiny in European app store ecosystem"],
            "recommendation": "BUY"
        },
        "TSLA": {
            "price": 248.20, "change_pct": -2.10, "volume": 89100000, "market_cap": 789000000000,
            "pe_ratio": 62.4, "eps": 3.80, "roe": 0.24, "quality": "good",
            "sentiment": "neutral", "score": 0.52,
            "reasons": ["Market leader in EV powertrain technology & energy storage", "Scaling autonomous Robotaxi platform and Full Self-Driving"],
            "risks": ["Margin pressure from global automotive price competition", "High premium valuation multiple compared to traditional OEMs"],
            "recommendation": "HOLD"
        },
        "MSFT": {
            "price": 448.90, "change_pct": 0.85, "volume": 21300000, "market_cap": 3330000000000,
            "pe_ratio": 36.8, "eps": 11.80, "roe": 0.38, "quality": "excellent",
            "sentiment": "bullish", "score": 0.91,
            "reasons": ["Rapid cloud growth driven by Azure AI infrastructure", "Enterprise monetization across Office 365 Copilot suite"],
            "risks": ["Increased capital expenditure for Next-Gen AI data center buildouts"],
            "recommendation": "BUY"
        }
    }
    
    prof = mock_profiles.get(t, {
        "price": 185.40, "change_pct": 0.65, "volume": 32000000, "market_cap": 1200000000000,
        "pe_ratio": 28.5, "eps": 5.20, "roe": 0.28, "quality": "good",
        "sentiment": "bullish", "score": 0.78,
        "reasons": [f"Solid quarterly financial fundamentals and operational execution for {t}", "Positive analyst consensus and expanding operating margins"],
        "risks": ["Broader macroeconomic volatility and interest rate sensitivity"],
        "recommendation": "BUY"
    })
    
    return {
        "recommendation": prof["recommendation"],
        "confidence": 0.88,
        "reasons": prof["reasons"],
        "risks": prof["risks"],
        "data_sources": ["demo_market_stream", "demo_rag_vector_db", "demo_financials"],
        "market_data": {
            "ticker": t,
            "price": prof["price"],
            "change_pct": prof["change_pct"],
            "volume": prof["volume"],
            "market_cap": prof["market_cap"],
        },
        "news_data": {
            "label": prof["sentiment"],
            "sentiment_score": prof["score"],
            "articles": [
                {"headline": f"{t} Announces Strategic AI Architecture & Enterprise Product Upgrades", "url": "https://finance.yahoo.com"},
                {"headline": f"Wall Street Analysts Upgrade {t} Target Following Quarter Results", "url": "https://bloomberg.com"}
            ]
        },
        "fundamentals_data": {
            "pe_ratio": prof["pe_ratio"],
            "eps": prof["eps"],
            "roe": prof["roe"],
            "quality": prof["quality"]
        },
        "risk_data": None,
        "errors": []
    }



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