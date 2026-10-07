import json
import os
import time

from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from src.llm.provider import build_llm
from src.memory.research_cache import load_research
from src.observability.trace import write_trace


load_dotenv()


# def build_llm() -> ChatOpenAI:
#     """Create the LLM used for cached follow-up questions."""
#     return ChatOpenAI(
#         model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
#         temperature=0,
#     )


def answer_followup(
    ticker: str,
    question: str,
) -> str:
    """
    Answer a short-term follow-up using only cached research.

    No external research tools are available to this agent.
    """

    start_time = time.perf_counter()
    cached_research = load_research(ticker)
    duration_ms = (time.perf_counter() - start_time) * 1000

    if cached_research is None:
        raise ValueError(
            f"No cached research available for ticker: {ticker}"
        )

    prompt = f"""
You are a follow-up research assistant.

You MUST answer the user's question using ONLY the cached
research provided below.

Do NOT perform new research.
Do NOT assume new market information.
Do NOT call external tools.

If the cached research does not contain enough information
to answer the question, explicitly say that the information
is not available in the cached research.

Cached research:

{json.dumps(cached_research, indent=2, ensure_ascii=False)}

User follow-up question:

{question}
"""

    llm = build_llm()

    result = llm.invoke(prompt)

    answer = result.content

    write_trace(
        tool_name="cache_read",
        inputs={
            "ticker": ticker,
            "question": question,
        },
        output=cached_research,
        duration_ms=duration_ms,
        success=True,
        agent="followup_agent",
        event_type="cache_read",
    )

    return answer