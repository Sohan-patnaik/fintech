from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey,
    Text, JSON, Boolean, func
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), default="My Portfolio")
    holdings = Column(JSON, default=list)   # [{"ticker": "TCS", "qty": 10, "avg_price": 3500}]
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    user = relationship("User", back_populates="portfolios")
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)   # BUY | SELL
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    executed_at = Column(DateTime, server_default=func.now())

    portfolio = relationship("Portfolio", back_populates="transactions")


class CachedStockData(Base):
    __tablename__ = "cached_stock_data"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), unique=True, index=True, nullable=False)
    data = Column(JSON, nullable=False)
    cached_at = Column(DateTime, server_default=func.now())


class NewsEmbedding(Base):
    __tablename__ = "news_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(1024), unique=True)
    ticker = Column(String(20), index=True)
    headline = Column(Text)
    content = Column(Text)
    sentiment_score = Column(Float)   # -1.0 to +1.0
    published_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
