import pytest
from agents.router_agent import RouterAgent

@pytest.mark.asyncio
async def test_router_eval_intents():
    router = RouterAgent()
    
    test_cases = [
        ("What is the current stock price of AAPL?", "price_only", ["market_data"]),
        ("What is the latest news and sentiment for TSLA?", "news_only", ["news_analysis"]),
        ("Review my portfolio holdings and asset allocation risk", "portfolio", ["market_data", "portfolio_risk"]),
        ("Show me the PE ratio and balance sheet fundamentals for MSFT", "fundamentals", ["fundamentals", "market_data"]),
        ("Should I buy or sell NVDA for long term?", "full_analysis", ["market_data", "news_analysis", "fundamentals", "decision"]),
    ]
    
    for query, expected_intent, expected_agents in test_cases:
        state = {"query": query}
        result = await router.run(state)
        
        assert result["intent"] == expected_intent, f"Failed for query '{query}': expected {expected_intent}, got {result['intent']}"
        assert result["agents_to_run"] == expected_agents, f"Failed for query '{query}': expected {expected_agents}, got {result['agents_to_run']}"
