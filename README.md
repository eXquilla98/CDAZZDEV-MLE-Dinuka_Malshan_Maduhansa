
```markdown
# CDAZZDEV-MLE-Dinuka_Malshan_Maduhansa

Technical assessment submission for the **Senior Machine Learning Engineer** role at **Ceylon Dazzling Dev Holding (Pvt.) Ltd.**

This repository focuses on **Task 3 — Agentic Workflows**.

---

# Task 3 — Agentic Financial Research Workflow

This project implements an agentic financial research system that analyses a stock ticker using market data, technical indicators, financial news, web research, LLM-based sentiment analysis, multi-agent collaboration, persistent memory, and execution tracing.

The system supports both **OpenAI** and **Groq** as configurable LLM providers. The final workflow was validated using **Groq**.

---

# Architecture

The workflow is divided into specialized agents with explicit tool boundaries.

```text
                         User Query
                             |
                             v
                  +----------------------+
                  |      Agent B          |
                  |   Research Writer     |
                  +----------------------+
                       |            |
                       v            v
                  News Search    Web Search
                       |            |
                       +-----+------+
                             |
                       Research Evidence
                             |
                             v
                  +----------------------+
                  |      Agent A          |
                  |     Data Analyst      |
                  +----------------------+
                    |       |        |
                    v       v        v
                 Price   Volatility Sentiment
                  Data               |
                    |       |        |
                    +---+---+--------+
                        |
                 Structured Output
                  (Pydantic Model)
                        |
                        v
                  Agent B reviews
                  quantitative data
                        |
                        v
              Clarification Request
                        |
                        v
                  Agent A responds
                        |
                        v
                 Final Research Report
                        |
                        v
              Persistent JSON Cache
                        |
                        v
                 Follow-up Agent
                        |
                        v
             Cache-only Follow-up
```

---

# Task 3A — Single Research Agent

The initial research agent demonstrates an agentic tool-selection workflow.

The research agent can dynamically use the following tools:

- `get_price_data(ticker, period)`
- `get_news(ticker, n)`
- `calculate_volatility(ticker, window)`
- `llm_sentiment(headlines)`
- `web_search(query)`

The research process follows a:

```text
Tool → Observation → Decision
```

loop rather than executing a fixed sequence of tool calls.

The agent evaluates the available observations and determines whether additional research is required.

The final report contains:

- Financial Health Summary
- Market Sentiment
- Top Three Risks
- Evidence for each risk
- Hedge Strategy Recommendation

---

# Task 3B — Multi-Agent Workflow

Task 3B separates responsibilities between two specialized agents.

## Agent A — Data Analyst

Agent A is responsible for quantitative and sentiment analysis.

Agent A is restricted to:

- Price data
- Technical indicators
- Historical volatility
- Sentiment analysis

Agent A cannot perform:

- Web searches
- News retrieval
- External research

The following tools are available exclusively to Agent A:

```text
price_data_tool
volatility_tool
sentiment_tool
```

Agent A produces a structured `DataAnalystOutput` using Pydantic validation.

---

## Agent B — Research Writer

Agent B is responsible for external research and final report generation.

Agent B is restricted to:

- Financial news retrieval
- Web research
- External financial commentary
- Final report generation

Agent B cannot directly access Agent A's quantitative tools.

The following tools are available exclusively to Agent B:

```text
news_tool
web_search_tool
```

---

# Structured Agent Handoff

Agent A and Agent B communicate using structured Pydantic models.

The workflow follows:

```text
User Query
    |
    v
Agent B — External Research
    |
    v
News + Web Evidence
    |
    v
Agent A — Quantitative Analysis
    |
    v
DataAnalystOutput
    |
    v
Agent B — Review
    |
    v
Clarification Request
    |
    v
Agent A — Clarification Response
    |
    v
Agent B — Final Research Report
```

The clarification process is performed automatically without manual intervention.

---

# Critique / Clarification Loop

Agent B is required to identify an important ambiguity or missing quantitative detail in Agent A's analysis.

Agent B asks exactly one clarification question.

Agent A responds using the quantitative and sentiment information already available in its analysis.

The response is then incorporated into Agent B's final report.

Example workflow:

```text
Agent A
   |
   | Structured quantitative analysis
   v
Agent B
   |
   | Identifies missing clarification
   v
Clarification Request
   |
   v
Agent A
   |
   | Clarification Response
   v
Agent B
   |
   v
Final Research Report
```

---

# Task 3C — Persistent Memory and Follow-up

The system implements persistent research memory using a JSON cache.

The cache is stored in:

```text
memory/research_cache.json
```

Research is stored by ticker and research date.

The structure is:

```text
Ticker
  └── Research Date
        ├── Analyst Output
        ├── News Results
        ├── Web Results
        └── Final Report
```

For example:

```text
AAPL
└── 2026-10-07
    ├── analyst_output
    ├── news_results
    ├── web_results
    └── final_report
```

---

# Cache-Based Follow-up

A follow-up agent can answer short-term questions using the previously cached research.

The follow-up agent:

- Reads the persistent JSON cache
- Uses the cached research as its context
- Does not perform new web research
- Does not retrieve new news
- Does not call the financial data tools

The workflow is:

```text
Previous Research
       |
       v
Persistent JSON Cache
       |
       v
Follow-up Question
       |
       v
Follow-up Agent
       |
       v
Cache-only Answer
```

This allows the system to reuse previously generated research instead of repeating the complete research workflow.

---

# Observability and Execution Tracing

Agent and tool execution is recorded in:

```text
agent_trace.jsonl
```

Each trace record contains:

- Timestamp
- Agent
- Event type
- Tool name
- Tool inputs
- Truncated output
- Execution duration
- Success/failure status

The trace provides execution evidence for the agentic workflow.

A complete execution can contain events such as:

```text
agent_b          → news_tool
agent_b          → web_search_tool

agent_a          → price_data_tool
agent_a          → volatility_tool
agent_a          → sentiment_tool

agent_b          → clarification_request
agent_a          → clarification_response

agent_b          → final_report

followup_agent   → cache_read
```

The trace is intentionally retained in the repository as execution evidence for the assessment.

---

# LLM Provider Configuration

The project supports two LLM providers:

- OpenAI
- Groq

Provider selection is centralized through:

```text
src/llm/provider.py
```

The provider can be selected through the `.env` configuration.

For Groq:

```env
LLM_PROVIDER=groq
```

For OpenAI:

```env
LLM_PROVIDER=openai
```

The same agent implementation can therefore be executed with either provider without changing the agent logic.

---

# Groq Configuration

The final Task 3 workflow was validated using Groq.

Example configuration:

```env
LLM_PROVIDER=groq

GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
```

---

# OpenAI Configuration

OpenAI can also be used for development and testing.

Example configuration:

```env
LLM_PROVIDER=openai

OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-5.4-mini
```

API keys are stored locally through environment variables.

The `.env` file is excluded from version control and is not committed to the repository.

---

# Technology Stack

## Programming

- Python

## Agentic Framework

- LangChain
- LangGraph

## LLM Providers

- Groq
- OpenAI

## Data and Financial Analysis

- yfinance
- pandas
- NumPy

## Structured Outputs

- Pydantic

## News and Web Research

- Google News RSS
- DDGS / DuckDuckGo Search

## Version Control

- Git
- GitHub

---

# Project Structure

```text
task3_agentic/
│
├── memory/
│   └── research_cache.json
│
├── src/
│   │
│   ├── agents/
│   │   ├── data_analyst_agent.py
│   │   ├── research_agent.py
│   │   ├── research_writer_agent.py
│   │   ├── followup_agent.py
│   │   ├── multi_agent_workflow.py
│   │   └── test_task3c.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── provider.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   └── research_cache.py
│   │
│   ├── models/
│   │   └── research_models.py
│   │
│   ├── observability/
│   │   └── trace.py
│   │
│   └── tools/
│       ├── price_data.py
│       ├── news.py
│       ├── volatility.py
│       ├── sentiment.py
│       └── web_search.py
│
├── agent_trace.jsonl
├── README.md
└── requirements.txt
```

---

# Main Components

## Financial Data

```text
src/tools/price_data.py
```

Provides:

- Current/latest price
- OHLCV data
- SMA indicators
- RSI
- MACD
- Bollinger Bands

---

## News Retrieval

```text
src/tools/news.py
```

Retrieves recent relevant financial news headlines.

---

## Volatility

```text
src/tools/volatility.py
```

Calculates annualized historical volatility from market price data.

---

## LLM Sentiment

```text
src/tools/sentiment.py
```

Performs structured financial sentiment analysis on supplied headlines.

The output includes:

- Overall sentiment
- Sentiment score
- Confidence
- Per-headline sentiment

The output is validated using Pydantic models.

---

## Web Search

```text
src/tools/web_search.py
```

Provides external financial research and analyst commentary through DDGS / DuckDuckGo Search.

---

## Agent Models

```text
src/models/research_models.py
```

Contains structured models for:

- Research reports
- Risks
- Hedge strategies
- Data analyst output
- Clarification requests
- Clarification responses

---

## Persistent Memory

```text
src/memory/research_cache.py
```

Provides functions for:

- Saving research
- Loading research
- Checking cache availability

Research is persisted in:

```text
memory/research_cache.json
```

---

## Observability

```text
src/observability/trace.py
```

Provides structured JSONL execution tracing with:

- Agent information
- Tool names
- Inputs
- Truncated outputs
- Execution duration
- Success/failure status

---

# Installation

## 1. Clone the repository

```powershell
git clone <repository-url>
cd CDAZZDEV-MLE-Dinuka_Malshan_Maduhansa\task3_agentic
```

---

## 2. Create the virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# Environment Configuration

Create a local `.env` file in the `task3_agentic` directory.

For Groq:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
```

For OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-5.4-mini
```

Do not commit `.env` or API keys.

---

# Running Task 3

## Run the Multi-Agent Workflow

From the `task3_agentic` directory:

```powershell
python -m src.agents.multi_agent_workflow
```

The workflow performs:

```text
1. Agent B — External Research
2. Agent A — Data Analysis
3. Agent B — Clarification Request
4. Agent A — Clarification Response
5. Agent B — Final Research Report
6. Workflow Complete
```

The research is also persisted in:

```text
memory/research_cache.json
```

---

# Run the Task 3C Follow-up

After research has been cached:

```powershell
python -m src.agents.test_task3c
```

The follow-up agent reads the persistent research cache and answers a short-term question without performing new external research.

---

# Example Execution

The workflow was validated using the ticker:

```text
AAPL
```

A successful Task 3B execution demonstrated:

```text
Agent B
  ├── News research
  └── Web research

Agent A
  ├── Price data
  ├── Volatility
  └── Sentiment

Agent B
  └── Clarification request

Agent A
  └── Clarification response

Agent B
  └── Final structured report

Persistent Cache
  └── research_cache.json

Follow-up Agent
  └── Cache-only response
```

The execution trace is retained in:

```text
agent_trace.jsonl
```

---

# Error Handling and Validation

The implementation includes validation and failure handling for:

- Missing API credentials
- Unsupported LLM providers
- Empty headline input
- Invalid headline data
- Unauthorized tool calls
- Failed tool execution
- Missing cached research
- Invalid structured LLM output

Pydantic models are used to validate structured outputs exchanged between agents.

---

# Design Considerations

## Specialized Agent Responsibilities

The agents intentionally have restricted tool access.

Agent A focuses on:

```text
Quantitative Data
+
Technical Indicators
+
Volatility
+
Sentiment
```

Agent B focuses on:

```text
News
+
Web Research
+
Report Writing
```

This separation reduces unnecessary tool access and makes each agent's responsibility explicit.

---

## Structured Agent Communication

Pydantic models are used for agent-to-agent communication rather than relying only on unstructured text.

This provides:

- Schema validation
- Consistent data structures
- Easier downstream processing
- Reduced ambiguity between agents

---

## Persistent Research Cache

The JSON cache allows research results to be reused across executions.

Instead of performing the complete research workflow again for every short-term follow-up question, the follow-up agent can use the previously generated research.

---

# Assessment Evidence

The repository intentionally retains the following generated artifacts:

```text
agent_trace.jsonl
memory/research_cache.json
```

These files provide inspectable evidence of:

- Tool execution
- Agent interactions
- Structured handoffs
- Clarification loop
- Final report generation
- Persistent research state
- Cache-based follow-up

---

# Security

API credentials are loaded through environment variables.

The following must remain local:

```text
.env
```

No API keys or secrets are stored in source code or committed to the repository.

---

# Project Status

## Task 3 — Agentic Workflows

**Implementation complete and tested.**

Implemented:

- [x] Task 3A — Single research agent
- [x] Financial data tools
- [x] News retrieval
- [x] Historical volatility
- [x] LLM sentiment analysis
- [x] Web research
- [x] Agent tool-selection loop
- [x] Task 3B — Multi-agent workflow
- [x] Agent A — Data Analyst
- [x] Agent B — Research Writer
- [x] Structured Pydantic handoff
- [x] Agent clarification loop
- [x] Task 3C — Persistent memory
- [x] Cache-based follow-up
- [x] Execution tracing
- [x] OpenAI provider support
- [x] Groq provider support
- [x] Groq end-to-end workflow validation

---

# Submission Scope

This repository submission focuses on:

**Task 3 — Agentic Workflows**

The implementation demonstrates an end-to-end multi-agent financial research workflow with tool use, structured agent communication, persistent memory, configurable LLM providers, and execution observability.


