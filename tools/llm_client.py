from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
from core.config import settings
from core.logger import get_logger
import asyncio

logger = get_logger(__name__)

_client = None


def get_llm_client() -> ChatNVIDIA:
    global _client
    if not settings.NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY not configured")

    if _client is None:
        _client = ChatNVIDIA(
            model="nvidia/nemotron-3-nano-30b-a3b",
            api_key=settings.NVIDIA_API_KEY,
            temperature=0.2,
            top_p=1,
            max_tokens=16384,
        )
    return _client


async def chat_complete(system: str, user: str) -> str:
    client = get_llm_client()

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]

    def _call():
        try:
            resp = client.invoke(messages)
            return resp.content.strip()
        except Exception:
            logger.exception("LLM call failed")
            raise

    return await asyncio.to_thread(_call)