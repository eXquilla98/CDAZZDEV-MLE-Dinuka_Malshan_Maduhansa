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

class DataAnalystOutput(BaseModel):
    """
    Structured handoff from the Data Analyst agent to
    the Research Writer agent.
    """

    ticker: str

    latest_price: float

    price_indicators: dict[str, float | None]

    annualized_volatility: float

    market_sentiment: Sentiment

    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    sentiment_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    quantitative_findings: list[str]


class ClarificationRequest(BaseModel):
    """
    Structured clarification request from the
    Research Writer agent to the Data Analyst agent.
    """

    question: str


class ClarificationResponse(BaseModel):
    """
    Structured response from the Data Analyst agent
    to the Research Writer agent.
    """

    answer: str

    supporting_data: list[str]