import json
from agents.base_agent import BaseAgent
from tools.rag import retrieve_relevant_news
from tools.scraper import scrape_news
from tools.rag import store_article
from tools.llm_client import chat_complete

SYSTEM = """You are a financial news analyst. Given news headlines, return ONLY valid JSON:
{
  "score": <float -1.0 to 1.0>,
  "label": "<bearish|neutral|bullish>",
  "summary": "<2 sentence summary>",
  "key_points": ["<point1>", "<point2>", "<point3>"]
}"""


class NewsAnalysisAgent(BaseAgent):
    async def run(self, state: dict) -> dict:
        ticker = state.get("ticker", "")
        query = state.get("query", ticker)
        if not ticker:
            return {**state, "news_data": None}
        try:
            articles = retrieve_relevant_news(ticker, query)
            if not articles:
                fresh = await scrape_news(ticker)
                for a in fresh:
                    store_article(a)
                articles = fresh

            if not articles:
                return {**state, "news_data": {"score": 0, "label": "neutral", "articles": []}}

            headlines = "\n".join(f"- {a['headline']}" for a in articles[:5])
            raw = await chat_complete(SYSTEM, f"Ticker: {ticker}\nHeadlines:\n{headlines}")
            sentiment = json.loads(raw)
            sentiment["articles"] = articles[:5]
            self.logger.info(
                f"News sentiment for {ticker}: {sentiment['label']} ({sentiment['score']})")
            return {**state, "news_data": sentiment}
        except Exception as e:
            self.logger.error(f"NewsAnalysisAgent error: {e}")
            return {**state, "news_data": None, "errors": state.get("errors", []) + [str(e)]}
