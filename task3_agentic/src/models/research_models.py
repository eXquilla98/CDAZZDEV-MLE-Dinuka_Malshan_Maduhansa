from typing import Literal

from pydantic import BaseModel, Field


Sentiment = Literal["positive", "negative", "neutral"]


class Risk(BaseModel):
    risk: str
    evidence: list[str]
    severity: Literal["low", "medium", "high"]


class HedgeStrategy(BaseModel):
    strategy: str
    rationale: str
    data_support: list[str]


class ResearchReport(BaseModel):
    ticker: str
    financial_health_summary: str
    market_sentiment: Sentiment
    sentiment_score: float = Field(ge=-1.0, le=1.0)

    top_three_risks: list[Risk] = Field(
        min_length=3,
        max_length=3,
    )

    hedge_strategy_recommendation: HedgeStrategy