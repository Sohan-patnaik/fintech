import json
import re
from typing import Any
from agents.base_agent import BaseAgent
from tools.llm_client import chat_complete
from schemas.all import ChatResponse, RecommendationType
from core.logger import get_logger

SYSTEM = """You are a senior equity analyst. Analyze the provided market data and return ONLY valid JSON.

Output schema:
{
  "recommendation": "buy" | "hold" | "sell",
  "confidence": float between 0.0 and 1.0,
  "reasons": [at least 2 specific, data-backed strings],
  "risks": [at least 1 specific risk string],
  "data_sources": [list of sources actually used]
}

Example output:
{
  "recommendation": "hold",
  "confidence": 0.61,
  "reasons": [
    "P/E of 27.4 is elevated relative to sector average of 22.1, suggesting limited upside.",
    "News sentiment is neutral (score: 0.08) with no material catalyst in recent headlines."
  ],
  "risks": [
    "Debt-to-equity of 1.82 increases vulnerability to rate hikes."
  ],
  "data_sources": ["market_data", "fundamentals", "news"]
}

Rules:
- Base recommendation strictly on the data provided. Do not hallucinate metrics.
- If a data source is missing or None, exclude it from data_sources and do not reference it.
- Return ONLY the JSON object. No markdown fences, no explanation, no preamble."""

MAX_CONTEXT_CHARS = 3000


class DecisionAgent(BaseAgent):
    async def run(self, state: dict) -> dict:
        ticker = state.get("ticker", "UNKNOWN")

        context = self._build_context(state, ticker)
        self.logger.debug(f"[{ticker}] Decision context ({len(context)} chars):\n{context}")

        try:
            raw = await chat_complete(SYSTEM, context)
            decision = self._parse_and_validate(raw, ticker)
            self.logger.info(
                f"[{ticker}] Decision: {decision['recommendation'].upper()} "
                f"(confidence={decision['confidence']:.2f})"
            )
            return {**state, "decision": decision, "decision_error": None}

        except Exception as e:
            self.logger.error(f"[{ticker}] DecisionAgent failed: {e}", exc_info=True)
            return {**state, "decision": self._fallback(), "decision_error": str(e)}


    def _parse_and_validate(self, raw: str, ticker: str) -> dict:
        """
        Parse LLM output into a validated decision dict.
        Handles markdown fences and validates against ChatResponse schema.
        Raises ValueError if the output is unrecoverable.
        """
        cleaned = self._strip_markdown(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            self.logger.warning(f"[{ticker}] Raw LLM output was not valid JSON: {raw[:200]}")
            raise ValueError(f"LLM returned non-JSON output: {e}") from e

        try:
            validated = ChatResponse(
                recommendation=parsed.get("recommendation", "hold"),
                confidence=parsed.get("confidence", 0.0),
                reasons=parsed.get("reasons") or ["No reasons provided."],
                risks=parsed.get("risks") or ["No risks identified."],
                data_sources=parsed.get("data_sources") or [],
            )
        except Exception as e:
            self.logger.warning(
                f"[{ticker}] LLM output failed schema validation: {parsed} | error: {e}"
            )
            raise ValueError(f"LLM output failed Pydantic validation: {e}") from e

        return validated.model_dump()

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """
        Remove markdown code fences that LLMs add despite instructions.
        Handles ```json ... ```, ``` ... ```, and leading/trailing whitespace.
        """
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()


    def _build_context(self, state: dict, ticker: str) -> str:
        """
        Build the user-facing prompt context from agent state.
        Handles both Pydantic model and plain dict inputs gracefully.
        Enforces MAX_CONTEXT_CHARS to prevent silent LLM truncation.
        """
        parts = [
            f"Stock: {ticker}",
            f"Query: {state.get('query', 'General analysis requested')}",
        ]

        if md := state.get("market_data"):
            md = self._to_dict(md)
            price = md.get("price")
            chg = md.get("change_pct")
            high = md.get("week_52_high")
            low = md.get("week_52_low")
            cap = md.get("market_cap")

            line = f"Market: price=${price}"
            if chg is not None:
                line += f" ({chg:+.2f}% today)"
            if high and low:
                line += f" | 52w range ${low}–${high}"
            if cap:
                line += f" | mkt cap ${cap:,.0f}"
            parts.append(line)

        if nd := state.get("news_data"):
            nd = self._to_dict(nd)
            score = nd.get("score")
            label = nd.get("label", "unknown")
            if score is not None:
                parts.append(
                    f"News sentiment: {label} (score={score:.2f})"
                )
            articles = nd.get("articles", [])[:2]
            for art in articles:
                art = self._to_dict(art)
                headline = art.get("headline", "")
                if headline:
                    parts.append(f"  • {headline}")

        if fd := state.get("fundamentals_data"):
            fd = self._to_dict(fd)
            metrics = []
            for key, label in [
                ("pe_ratio", "P/E"),
                ("eps", "EPS"),
                ("roe", "ROE"),
                ("debt_to_equity", "D/E"),
                ("revenue_growth", "Rev growth"),
            ]:
                val = fd.get(key)
                if val is not None:
                    metrics.append(f"{label}={val}")
            if metrics:
                parts.append(f"Fundamentals: {' | '.join(metrics)}")
            if rating := fd.get("analyst_rating"):
                parts.append(f"Analyst consensus: {rating}")

        if rd := state.get("risk_data"):
            rd = self._to_dict(rd)
            score = rd.get("risk_score")
            if score is not None:
                parts.append(f"Portfolio risk score: {score:.2f}/1.0")
            suggestions = rd.get("suggestions", [])[:2] 
            for s in suggestions:
                parts.append(f"  • {s}")

        context = "\n".join(parts)

        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]
            context += "\n[Context truncated due to length — prioritize market_data and fundamentals]"
            self.logger.warning(
                f"[{ticker}] Context truncated to {MAX_CONTEXT_CHARS} chars"
            )

        return context

    def _fallback(self) -> dict:
        """
        Safe fallback when the agent crashes.
        is_fallback=True lets callers distinguish a genuine HOLD from a failure.
        Internal error details are NOT included — they're logged, not leaked to users.
        """
        return {
            "recommendation": RecommendationType.HOLD.value,
            "confidence": 0.0,
            "reasons": ["Analysis could not be completed. Please try again."],
            "risks": ["System error — result may be unreliable."],
            "data_sources": [],
            "is_fallback": True,
        }


    @staticmethod
    def _to_dict(obj: Any) -> dict:
        """
        Normalize Pydantic models and plain dicts to dict uniformly.
        Agents upstream may store either — this makes _build_context robust to both.
        """
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, dict):
            return obj
        return {}