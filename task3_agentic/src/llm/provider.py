import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()


def build_llm() -> BaseChatModel:
    """
    Create the configured LLM provider.

    Supported providers:
    - openai
    - groq
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()

    if provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            temperature=0,
        )

    if provider == "groq":
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. "
        "Expected 'openai' or 'groq'."
    )