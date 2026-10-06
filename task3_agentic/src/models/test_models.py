from src.models.research_models import (
    HedgeStrategy,
    ResearchReport,
    Risk,
)


report = ResearchReport(
    ticker="AAPL",
    financial_health_summary="Apple shows relatively strong financial health.",
    market_sentiment="positive",
    sentiment_score=0.6,
    top_three_risks=[
        Risk(
            risk="Demand weakness",
            evidence=["Recent headlines indicate concerns about iPhone demand."],
            severity="medium",
        ),
        Risk(
            risk="AI competition",
            evidence=["Competitors continue investing heavily in AI capabilities."],
            severity="medium",
        ),
        Risk(
            risk="Market valuation",
            evidence=["The stock is trading near elevated valuation levels."],
            severity="high",
        ),
    ],
    hedge_strategy_recommendation=HedgeStrategy(
        strategy="Protective put",
        rationale="A protective put can limit downside while maintaining equity exposure.",
        data_support=[
            "30-day annualized volatility is approximately 21%.",
        ],
    ),
)

print("\n=== RESEARCH REPORT MODEL ===")
print(report.model_dump_json(indent=2))