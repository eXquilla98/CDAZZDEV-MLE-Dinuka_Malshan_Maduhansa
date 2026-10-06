from src.tools.web_search import web_search


result = web_search(
    "Apple stock latest financial outlook",
    max_results=5,
)

print("\n=== WEB SEARCH ===")
print(f"Results: {len(result)}")

for index, item in enumerate(result, start=1):
    print(f"\n--- Result {index} ---")
    print(f"Title: {item['title']}")
    print(f"URL: {item['url']}")
    print(f"Snippet: {item['snippet']}")