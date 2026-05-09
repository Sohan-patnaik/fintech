from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from typing import Optional, Any, Union
from datetime import datetime, timezone
from enum import Enum


class ActionType(str, Enum):
    BUY = "buy"
    SELL = "sell"


class RecommendationType(str, Enum):
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)
    full_name: Optional[str] = None


class UserOut(BaseModel):
    id: int
    email: str
    full_name: Optional[str] = None
    created_at: datetime
    model_config = {"from_attributes": True}


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class Holding(BaseModel):
    ticker: str
    qty: float = Field(gt=0)
    avg_price: float = Field(gt=0)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class PortfolioCreate(BaseModel):
    name: str = "My Portfolio"
    holdings: list[Holding] = Field(default_factory=list)


class PortfolioOut(BaseModel):
    id: int
    name: str
    holdings: list[Holding]
    created_at: datetime
    model_config = {"from_attributes": True}


class TransactionCreate(BaseModel):
    ticker:   str
    action:   ActionType
    quantity: float = Field(gt=0)
    price:    float = Field(gt=0)
 
    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()
 
    @field_validator("action", mode="before")
    @classmethod
    def coerce_action(cls, v) -> str:
        # Accepts "BUY", "buy", "Buy", "SELL", "sell" etc.
        if isinstance(v, str):
            return v.lower()
        return v
 
 
class TransactionOut(BaseModel):
    id:          int
    ticker:      str
    action:      ActionType
    quantity:    float
    price:       float
    executed_at: datetime
 
    model_config = {"from_attributes": True}



class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    portfolio_id: Optional[int] = None
    user_context: Optional[dict[str, Any]] = None  
    mode: Optional[str] = None                     


class AgentResult(BaseModel):
    agent: str
    data: Union[dict[str, Any], "StockData", "FundamentalsData", "SentimentData", "RiskData"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "v1"


class ChatResponse(BaseModel):
    recommendation: RecommendationType
    confidence: float = Field(ge=0.0, le=1.0) 
    reasons: list[str]
    risks: list[str]
    data_sources: list[str]
    raw_data: Optional[dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "v1"

    @field_validator("recommendation", mode="before")
    @classmethod
    def coerce_recommendation(cls, v: str) -> str:
      
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("reasons", "risks", "data_sources")
    @classmethod
    def non_empty_lists(cls, v: list) -> list:
        if not v:
            raise ValueError("must contain at least one item")
        return v


class StockData(BaseModel):
    ticker: str
    price: float = Field(gt=0)
    change_pct: float
    volume: float 
    week_52_high: float = Field(gt=0)
    week_52_low: float = Field(gt=0)
    market_cap: Optional[float] = None

    @model_validator(mode="after")
    def validate_52_week_range(self) -> "StockData":
        if self.week_52_low > self.week_52_high:
            raise ValueError("week_52_low cannot exceed week_52_high")
        return self


class FundamentalsData(BaseModel):
    ticker: str
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None
    analyst_rating: Optional[str] = None


class Article(BaseModel):
    title: str
    url: str
    sentiment: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


class SentimentData(BaseModel):
    ticker: str
    score: float = Field(ge=-1.0, le=1.0) 
    label: str
    articles: list[Article]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label(cls, v: str) -> str:
        return v.lower()


class RiskData(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)  
    volatility: str
    concentration: float = Field(ge=0.0, le=1.0)
    sector_exposure: dict[str, float]
    suggestions: list[str]


class ArticleOut(BaseModel):
    headline: str
    url: str
    source: str = "unknown"
    published_at: Optional[str] = None
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None  


class NewsResponse(BaseModel):
    ticker: str
    count: int
    stored: int
    failed: int
    articles: list[ArticleOut]


class RelevantNewsResponse(BaseModel):
    ticker: str
    query: str
    count: int
    articles: list[ArticleOut]    