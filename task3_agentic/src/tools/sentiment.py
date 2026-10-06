import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()


class HeadlineSentiment(BaseModel):
    headline: str
    sentiment: str
    score: float = Field(ge=-1.0, le=1.0)


class SentimentResult(BaseModel):
    overall_sentiment: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    headline_count: int
    headline_analysis: list[HeadlineSentiment]


def llm_sentiment(headlines: list[Any]) -> dict[str, Any]:
    """
    Analyze financial sentiment using an LLM.

    Accepts either:
    - a list of headline strings
    - a list of news dictionaries containing a "title" field

    Returns validated structured sentiment data.
    """

    if not headlines:
        raise ValueError("headlines must not be empty")

    valid_headlines: list[str] = []

    for item in headlines:
        if isinstance(item, str):
            title = item.strip()

        elif isinstance(item, dict):
            raw_title = item.get("title")
            title = str(raw_title).strip() if raw_title else ""

        else:
            title = ""

        if title:
            valid_headlines.append(title)

    if not valid_headlines:
        raise ValueError(
            "No valid headlines provided. "
            "Expected strings or dictionaries containing a 'title' field."
        )    
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    )

    client = Groq(api_key=api_key)

    headlines_text = "\n".join(
        f"{index + 1}. {headline}"
        for index, headline in enumerate(valid_headlines)
    )

    system_prompt = """
You are a financial news sentiment analyst.

Analyze the provided financial news headlines specifically
for their likely impact on the referenced company's stock.

For each headline:
- classify sentiment as positive, negative, or neutral
- provide a sentiment score from -1.0 to 1.0

Then provide:
- overall sentiment
- overall sentiment score
- confidence from 0.0 to 1.0

Do not invent facts.
Judge sentiment only from the information contained
in the headlines.
"""

    user_prompt = f"""
Analyze these financial headlines:

{headlines_text}

Return ONLY valid JSON with this structure:

{{
  "overall_sentiment": "positive|negative|neutral",
  "sentiment_score": 0.0,
  "confidence": 0.0,
  "headline_count": 0,
  "headline_analysis": [
    {{
      "headline": "headline text",
      "sentiment": "positive|negative|neutral",
      "score": 0.0
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("LLM returned an empty response")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned invalid JSON: {content}"
        ) from exc

    result = SentimentResult.model_validate(parsed)

    return result.model_dump()