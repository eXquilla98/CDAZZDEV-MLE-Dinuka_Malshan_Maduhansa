# CITATIONS

This document records external libraries, frameworks, APIs, documentation, and AI assistance used during the development of Task 3 — Agentic Workflows.

---

## 1. Python Libraries and Frameworks

### LangChain

Used for LLM integration, tool definitions, message handling, and structured agent workflows.

Repository / documentation:

https://www.langchain.com/

https://docs.langchain.com/

---

### LangGraph

Used to implement the stateful Agent A workflow and its tool-calling loop.

Repository / documentation:

https://langchain-ai.github.io/langgraph/

---

### Pydantic

Used to define and validate structured outputs exchanged between agents.

Documentation:

https://docs.pydantic.dev/

---

### yfinance

Used to retrieve historical stock market data and calculate technical indicators and volatility.

Repository:

https://github.com/ranaroussi/yfinance

---

### Groq Python SDK

Used for LLM-based sentiment analysis through the Groq API.

Repository:

https://github.com/groq/groq-python

---

### LangChain Groq Integration

Used to integrate Groq models with the LangChain-based agents.

Documentation:

https://docs.langchain.com/oss/python/integrations/providers/groq

---

### OpenAI Python SDK

Used as an alternative LLM provider during development and testing.

Documentation:

https://platform.openai.com/docs/

---

### DuckDuckGo Search / DDGS

Used to retrieve external web research and analyst commentary.

Repository:

https://github.com/deedy5/ddgs

---

### Google News RSS

Used to retrieve recent financial news headlines.

The implementation uses Google News RSS feeds rather than an authenticated news API.

---

## 2. AI-Assisted Development

Generative AI tools were used during development as permitted by the technical assessment instructions.

AI assistance was primarily used for:

- Brainstorming and refining the agent architecture
- Reviewing implementation approaches
- Debugging Python and LangChain/LangGraph issues
- Improving prompts
- Reviewing structured Pydantic models
- Debugging tool-calling workflows
- Reviewing error messages
- Suggesting test strategies
- Improving documentation

The generated suggestions were reviewed, adapted, tested, and integrated into the implementation manually.

The final implementation was tested locally using the project's virtual environment and API integrations.

---

## 3. Original Implementation

The following components were implemented and adapted specifically for this assessment:

- Financial price-data tool
- Technical indicator calculations
- Historical volatility tool
- Financial news retrieval
- LLM-based sentiment analysis
- Web research tool
- Single-agent research workflow
- Multi-agent Data Analyst / Research Writer architecture
- Pydantic agent handoff models
- Agent clarification loop
- Persistent JSON research cache
- Cache-based follow-up agent
- Agent execution tracing
- Configurable OpenAI / Groq provider abstraction

---

## 4. Source Attribution

External libraries and frameworks listed above are used through their documented public APIs.

No proprietary source code was copied into this repository.

Where implementation approaches were informed by documentation or examples, they were adapted to the requirements of this assessment.

---

## 5. API Credentials

API credentials are stored locally through environment variables.

The `.env` file is excluded from version control and no API keys are included in this repository.
