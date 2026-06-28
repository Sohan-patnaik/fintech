import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import chromadb
from chromadb import EmbeddingFunction
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dateutil.parser import parse as parse_dt

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


_lock = threading.Lock()
_client: Optional[chromadb.PersistentClient] = None
_collection = None


class NVIDIAEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = NVIDIAEmbeddings(model_name=settings.EMBEDDING_MODEL)

    def __call__(self, input: list[str]) -> list:
        return self.model.embed_documents(input)


def _get_collection():
    global _client, _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                _client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
                _collection = _client.get_or_create_collection(
                    name=settings.CHROMA_COLLECTION,
                    embedding_function=NVIDIAEmbeddingFunction(),
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "ChromaDB collection initialized",
                    extra={"collection": settings.CHROMA_COLLECTION}
                )
    return _collection


def _to_timestamp(value) -> float:
    """Convert any datetime-like value to a Unix timestamp float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return parse_dt(value).timestamp()
        except Exception:
            pass
    return datetime.now(timezone.utc).timestamp()


def store_article(article: dict) -> bool:
    """
    Embed and store a news article. Returns True on success, False on failure.
    """
    required = {"url", "headline"}
    if missing := required - article.keys():
        logger.error(f"store_article missing required fields: {missing}")
        return False

    col = _get_collection()

    metadata = {
        "ticker": article.get("ticker", "").upper(),
        "headline": article["headline"],
        "url": article["url"],
        "source": article.get("source", "unknown"),
        "published_at": _to_timestamp(article.get("published_at")),
        "sentiment_score": float(article.get("sentiment_score", 0.0)),
    }

    document = (
        f"{article['headline']} {article['headline']} "
        f"{article.get('content', '')}"
    ).strip()

    try:
        col.upsert(
            ids=[article["url"]],
            documents=[document],
            metadatas=[metadata],
        )
        return True
    except Exception as e:
        logger.error(f"ChromaDB upsert failed for {article['url']}: {e}")
        return False


def retrieve_relevant_news(
    ticker: str,
    query: str,
    n: int = 5,
    max_age_days: int = 30,
    min_results: int = 0,
) -> list[dict]:
    """
    Retrieve top-n relevant articles for a ticker + query.
    """
    col = _get_collection()

    # Guard against empty collection
    total = col.count()
    if total == 0:
        logger.info("Collection is empty, skipping query")
        return []

    actual_n = min(n, total)

    where_clause: dict = {"ticker": ticker.upper()}
    if max_age_days > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).timestamp()
        where_clause = {
            "$and": [
                {"ticker": {"$eq": ticker.upper()}},
                {"published_at": {"$gte": cutoff}},
            ]
        }

    try:
        results = col.query(
            query_texts=[query],
            n_results=actual_n,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not documents:
            logger.info(f"No articles found for {ticker} | query='{query}'")
            return []

        if len(documents) < min_results:
            logger.warning(
                f"Low retrieval count for {ticker}: "
                f"got {len(documents)}, expected >= {min_results}"
            )

        articles = []
        seen_headlines = set()

        for doc, meta, dist in zip(documents, metadatas, distances):
            headline = meta["headline"]
            headline_key = headline.lower().strip()[:80]

            if headline_key in seen_headlines:
                logger.debug(f"Skipping duplicate headline: {headline[:60]}")
                continue
            seen_headlines.add(headline_key)

            articles.append({
                "headline": headline,
                "url": meta["url"],
                "content": doc,
                "source": meta.get("source", "unknown"),
                "published_at": datetime.fromtimestamp(
                    meta["published_at"], tz=timezone.utc
                ).isoformat() if meta.get("published_at") else None,
                "sentiment_score": meta.get("sentiment_score", 0.0),
                "relevance_score": round(1 - dist, 4),
            })

        articles.sort(key=lambda a: a["relevance_score"], reverse=True)
        return articles

    except Exception as e:
        logger.error(
            f"RAG retrieval failed for {ticker}: {e}",
            extra={"ticker": ticker, "query": query}
        )
        return []


def delete_stale_articles(max_age_days: int = 30) -> int:
    """
    Delete articles older than max_age_days.
    Returns the number of deleted documents.
    """
    col = _get_collection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).timestamp()

    try:
        stale = col.get(
            where={"published_at": {"$lt": cutoff}},
            include=[],
        )
        stale_ids = stale["ids"]

        if not stale_ids:
            logger.info("No stale articles to delete")
            return 0

        col.delete(ids=stale_ids)
        logger.info(f"Deleted {len(stale_ids)} stale articles older than {max_age_days} days")
        return len(stale_ids)

    except Exception as e:
        logger.error(f"Stale article cleanup failed: {e}")
        return 0


def collection_stats() -> dict:
    """
    Return basic health stats about the collection.
    """
    try:
        col = _get_collection()
        count = col.count()
        return {
            "status": "ok",
            "document_count": count,
            "collection": settings.CHROMA_COLLECTION
        }
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        return {"status": "error", "error": str(e)}