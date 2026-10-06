from src.tools.sentiment import llm_sentiment


headlines = [
    "Apple shares surge after strong quarterly earnings report",
    "Apple faces growing concerns over declining iPhone demand",
    "Apple announces a new product launch next month",
]

result = llm_sentiment(headlines)

print("\n=== SENTIMENT ANALYSIS ===")
print(f"Overall sentiment: {result['overall_sentiment']}")
print(f"Sentiment score: {result['sentiment_score']}")
print(f"Headline count: {result['headline_count']}")

print("\n=== HEADLINE ANALYSIS ===")

for item in result["headline_analysis"]:
    print(f"\nHeadline: {item['headline']}")
    print(f"Sentiment: {item['sentiment']}")
    print(f"Score: {item['score']}")
    