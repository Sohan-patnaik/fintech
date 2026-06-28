import pytest
from graph.workflow import run_pipeline, _format_response

@pytest.mark.asyncio
async def test_pipeline_eval_fallback():
    # Test execution with mock query to verify formatting and safety defaults
    result = await run_pipeline("What is the live price of AAPL?", ticker="AAPL")
    
    assert "recommendation" in result
    assert "confidence" in result
    assert "reasons" in result
    assert "risks" in result
    assert "data_sources" in result
    assert isinstance(result["reasons"], list)
    assert isinstance(result["risks"], list)

def test_format_response_unit():
    mock_state = {
        "decision": {
            "recommendation": "BUY",
            "confidence": 0.85,
            "reasons": ["Strong earnings growth"],
            "risks": ["Market volatility"],
            "data_sources": ["yfinance", "news_rag"]
        },
        "errors": []
    }
    formatted = _format_response(mock_state)
    assert formatted["recommendation"] == "BUY"
    assert formatted["confidence"] == 0.85
    assert len(formatted["reasons"]) == 1
