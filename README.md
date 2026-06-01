# FinSight AI — Investment Analysis Backend

> A production-grade FastAPI backend powering an AI investment analyst. Ask a question in plain English, get a structured Buy / Hold / Sell recommendation backed by real-time market data, news sentiment, and fundamental analysis — all orchestrated through a multi-agent LangGraph pipeline.

---

## What This System Does

Retail investors have access to data but rarely have the tools to interpret it. FinSight bridges that gap. A user asks *"Should I buy TSLA right now?"* and the system:

1. Routes the intent through a **LangGraph state machine**
2. Fires three specialist agents **in parallel** — market data, news sentiment, and fundamental analysis
3. Aggregates all outputs and passes them to a **Decision Agent** that produces a structured, explainable recommendation
4. Returns a typed JSON response with a confidence score, specific reasons, and identified risks

---

## Architecture

```
User Query (REST API)
        │
   FastAPI Gateway  ←─ JWT Auth, Rate Limiting, Request Validation
        │
   Router Agent     ←─ Classifies intent, decides which agents to invoke
        │
┌───────┼────────────────────┐
│       │                    │
▼       ▼                    ▼
Market  News Analysis    Fundamentals     ← run in parallel via asyncio.gather
Data    Agent            Agent
Agent   (RAG + LLM)      (yfinance + LLM)
│       │                    │
└───────┴────────────────────┘
        │
   Aggregator Node
        │
   Decision Agent   ←─ Produces final structured recommendation
        │
   ChatResponse (Pydantic-validated JSON)
```

---

## Key Technical Decisions

### Multi-Agent Orchestration with LangGraph
Agents are nodes in a typed state graph with conditional edges. The router agent inspects the query and fans out only to the agents it needs — a price-only query never touches the news or fundamentals pipeline. Parallel execution via `asyncio.gather` keeps total latency close to the slowest single agent rather than the sum of all agents.

### RAG Pipeline (ChromaDB + sentence-transformers)
News articles are scraped, embedded using `sentence-transformers`, and stored in ChromaDB with metadata including `published_at`, `source`, and pre-computed `sentiment_score`. Retrieval uses ticker-scoped vector search with a configurable time window (`max_age_days`) so stale sentiment doesn't influence recommendations. A nightly cleanup job prunes expired articles.

### Structured LLM Output with Pydantic Validation
Every agent that calls the LLM enforces a strict JSON schema. The Decision Agent strips markdown fences, parses the output, and validates it through the `ChatResponse` Pydantic model before writing to state. A field validator normalises casing (`"Buy"` → `"buy"`) before enum parsing. If validation fails, the agent returns an explicit fallback with `is_fallback: True` — the API never silently returns fabricated data.

### Async-Safe yfinance with Per-Key Locking
yfinance is synchronous. All market data calls are wrapped in `asyncio.to_thread` so the FastAPI event loop is never blocked. A per-cache-key `asyncio.Lock` prevents thundering-herd: concurrent requests for the same ticker share a single in-flight yfinance call rather than issuing independent fetches. Stock price TTL is 60s; fundamentals TTL is 4h (they're quarterly, not tick-by-tick).

### Three-Source News Pipeline with Graceful Fallback
```
Finnhub API (primary)  →  Yahoo Finance RSS (fallback)  →  Yahoo JSON API (last resort)
```
The HTML scraper was replaced entirely — Yahoo's React-rendered pages require a headless browser to scrape reliably. The RSS feed is XML-based, stable, and doesn't need CSS selectors. Finnhub provides article summaries alongside headlines, making embeddings significantly richer.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI (async, Python 3.13) |
| **AI Orchestration** | LangGraph, LangChain Core |
| **LLM** | Nvidia Nemotron 70B via Nvidia API |
| **Vector Store** | ChromaDB + sentence-transformers |
| **Market Data** | yfinance (async-wrapped) |
| **News** | Finnhub API, Yahoo Finance RSS |
| **Database** | Supabase (PostgreSQL) |
| **Auth** | JWT (python-jose + passlib bcrypt) |
| **HTTP Client** | httpx (shared async client, connection pooling) |
| **Validation** | Pydantic v2 with field validators |
| **Rate Limiting** | slowapi |
| **Logging** | Structured logging with per-request correlation IDs |

---

## Project Structure

```
├── agents/
│   ├── base_agent.py          # Abstract base with shared logger + error contract
│   ├── router_agent.py        # Intent classification, conditional graph edges
│   ├── market_data_agent.py   # Real-time price, volume, 52-week range
│   ├── news_analysis_agent.py # RAG retrieval + LLM sentiment scoring
│   ├── fundamentals_agent.py  # P/E, EPS, ROE, D/E interpretation
│   ├── decision_agent.py      # Aggregation + structured recommendation
│   └── portfolio_risk_agent.py# Concentration, sector exposure, risk scoring
│
├── api/routes/
│   ├── auth.py                # Register, login, JWT token issuance
│   ├── chat.py                # Primary query endpoint
│   ├── news.py                # News fetch + RAG retrieval endpoints
│   └── portfolio.py           # Portfolio CRUD + transaction log + analysis
│
├── graph/
│   └── workflow.py            # LangGraph state machine definition
│
├── tools/
│   ├── llm_client.py          # Nvidia API wrapper
│   ├── yahoo_finance.py       # Async market data + fundamentals
│   ├── scraper.py             # Three-source news pipeline
│   └── rag.py                 # ChromaDB store + retrieval + cleanup
│
├── schemas/
│   └── all.py                 # Pydantic v2 models for all I/O contracts
│
├── core/
│   ├── config.py              # Settings (pydantic-settings, .env)
│   ├── security.py            # Password hashing, JWT, auth dependency
│   ├── logger.py              # Structured logger factory
│   └── limiter.py             # Rate limiter instance
│
└── db/
    └── session.py             # Supabase client factory
```

---

## API Reference

### Auth
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create account — unique email enforced at DB level |
| `POST` | `/api/v1/auth/token` | Login — returns JWT bearer token |

### Chat (AI Analysis)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/chat` | Natural language investment query → structured recommendation |

**Request**
```json
{
  "query": "Should I buy AAPL given current market conditions?",
  "portfolio_id": 3
}
```

**Response**
```json
{
  "recommendation": "hold",
  "confidence": 0.72,
  "reasons": [
    "P/E of 27.4 is elevated relative to sector average of 22.1.",
    "News sentiment is neutral (score: 0.08) with no material catalyst."
  ],
  "risks": [
    "Debt-to-equity of 1.82 increases vulnerability to rate hikes."
  ],
  "data_sources": ["market_data", "fundamentals", "news"],
  "timestamp": "2026-05-09T20:58:44Z"
}
```

### Portfolio
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/portfolio` | Create portfolio with initial holdings |
| `GET` | `/api/v1/portfolio` | List all portfolios for authenticated user |
| `GET` | `/api/v1/portfolio/{id}/analyze` | Full AI risk + diversification analysis |
| `POST` | `/api/v1/portfolio/{id}/transaction` | Record BUY/SELL — updates holdings with weighted avg cost |

### News
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/news/{ticker}` | Fetch + index latest articles for a ticker |
| `GET` | `/api/v1/news/{ticker}/relevant?q=...` | Semantic RAG retrieval against ChromaDB |

---

## Security

- **Passwords** hashed with bcrypt (cost factor 12) via passlib
- **JWT** tokens signed with HS256, expiry configurable via env
- **Timing-safe login** — `verify_password` always runs (even on unknown email) to prevent user enumeration via response time
- **Ownership enforcement** on all portfolio endpoints — 404 returned whether resource is missing or belongs to another user (never 403)
- **Input validation** on all path parameters — ticker regex `^[A-Z]{1,6}$` blocks injection attempts
- **Internal errors never exposed** — all `except Exception` handlers log the real error and return a generic message
- **Rate limiting** on auth endpoints (5/min login, 3/min register) and all AI endpoints

---

## Running Locally

**Prerequisites:** Python 3.13, a [Supabase](https://supabase.com) project, a [Finnhub](https://finnhub.io) API key (free), and a [Nvidia API](https://build.nvidia.com) key.

```bash
# 1. Clone and install
git clone <repo-url>
cd fintech-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in: SUPABASE_URL, SUPABASE_KEY, NVIDIA_API_KEY, FINNHUB_API_KEY, SECRET_KEY

# 3. Run
uvicorn main:app --reload --port 8000
```

Interactive API docs available at `http://localhost:8000/docs`

### Environment Variables

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Supabase service role key |
| `NVIDIA_API_KEY` | Nvidia NIM API key for Nemotron |
| `FINNHUB_API_KEY` | Finnhub news API key (free tier works) |
| `SECRET_KEY` | Random string for JWT signing (min 32 chars) |
| `CHROMA_PATH` | Local path for ChromaDB persistence (default: `./chroma_db`) |
| `EMBEDDING_MODEL` | sentence-transformers model (default: `all-MiniLM-L6-v2`) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT lifetime (default: `60`) |

---

## Production Considerations

**What's implemented:**
- Async-safe parallel agent execution
- Per-key cache locking (prevents thundering herd)
- Graceful agent fallbacks with `is_fallback` signalling
- Per-request correlation IDs for log tracing
- Structured error handling — no raw exceptions exposed to clients
- ChromaDB TTL cleanup (call `delete_stale_articles()` on a daily schedule)
- Connection pooling on both `httpx` and `requests` clients with shutdown hooks

**What's scoped for next iteration:**
- Semantic caching with Redis (avoid redundant LLM calls for similar queries)
- DeepEval test suite for RAG retrieval quality (MRR, NDCG) and decision agent output
- LangSmith tracing for per-node observability
- Prompt versioning and A/B testing infrastructure
- vLLM self-hosted inference for the fundamentals analysis step

---

## Design Choices Worth Noting

**Why LangGraph over LangChain agents?** LangChain's agent executor is a black box — you can't inspect or test individual steps. LangGraph exposes the full state at every node, making debugging, testing, and observability straightforward. Conditional edges give deterministic routing rather than LLM-decided tool calls.

**Why ChromaDB over Pinecone/Weaviate?** ChromaDB runs locally with zero infrastructure for development and can be swapped for a hosted vector DB by changing one config line. For this project's scale it's the right tradeoff — no managed service costs, no network round-trip for embeddings.

**Why Supabase over raw PostgreSQL?** The REST client eliminates the `asyncpg` SSL handshake bug on Python 3.13 + Windows (documented in the architecture notes). For production, the connection pool and row-level security policies are a bonus.

---

## Author

Built as a full-stack AI engineering project demonstrating end-to-end LLM system design — from data pipelines and RAG through multi-agent orchestration to production API hardening.
