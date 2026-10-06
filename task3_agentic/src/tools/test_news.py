from src.tools.news import get_news


news = get_news("AAPL", 10)

print(f"\nRetrieved {len(news)} headlines\n")

for index, item in enumerate(news, start=1):
    print(f"{index}. {item['title']}")
    print(f"   Publisher: {item['publisher']}")
    print(f"   Published: {item['published_at']}")
    print(f"   URL: {item['url']}")
    print()