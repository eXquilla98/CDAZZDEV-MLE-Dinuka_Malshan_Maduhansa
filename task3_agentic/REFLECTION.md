Perfect. Then let's create them now. We should keep them **honest and specific to what you actually used**, especially because the assessment allows AI assistance and expects it to be documented.

## 1. Create `CITATIONS.md`

From PowerShell:

```powershell
New-Item -ItemType File -Force .\CITATIONS.md
code .\CITATIONS.md
```

Paste this:

```markdown
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

```

---

## 2. Create `REFLECTION.md`

Then:

```powershell
New-Item -ItemType File -Force .\REFLECTION.md
code .\REFLECTION.md
```

Paste this:

```markdown
# REFLECTION

## Task 3 — Agentic Workflows

This reflection describes the engineering decisions, trade-offs, limitations, and potential improvements of the agentic financial research system implemented for the assessment.

---

## 1. Why an Agentic Workflow?

The main objective was to demonstrate a workflow where an LLM can decide which tools are required based on the information available rather than following a completely fixed sequence.

The research process therefore follows a:

```text
Tool → Observation → Decision
```

pattern.

The agent receives tool results and can determine whether additional information is required before producing its final analysis.

This was particularly useful for the financial research task because different parts of the analysis require different sources of information:

- Market data for quantitative analysis
- News for current events
- Web research for external commentary
- LLM reasoning for sentiment and synthesis

---

## 2. Why Separate Agent A and Agent B?

The multi-agent design separates quantitative analysis from external research.

### Agent A — Data Analyst

Agent A is responsible for:

- Price data
- Technical indicators
- Historical volatility
- Sentiment analysis

### Agent B — Research Writer

Agent B is responsible for:

- News research
- Web research
- External evidence
- Final report generation

This separation provides clear boundaries for tool access.

It also reduces the likelihood that one agent will use unrelated tools or mix quantitative data collection with external research responsibilities.

---

## 3. Why Restrict Tool Access?

Tool restrictions were intentionally implemented at the agent level.

Agent A cannot access news or web-search tools.

Agent B cannot access price, volatility, or sentiment tools.

The goal was to make the responsibilities of each agent explicit rather than allowing every agent to access every available capability.

This also makes the system easier to reason about and test.

---

## 4. Why Pydantic for Agent Handoffs?

Agent A passes its analysis to Agent B using structured Pydantic models.

For example, `DataAnalystOutput` contains fields for:

- Latest price
- Technical indicators
- Annualized volatility
- Market sentiment
- Sentiment score
- Sentiment confidence
- Quantitative findings

Using a schema provides validation between agents and reduces reliance on unstructured natural-language handoffs.

This also makes the output easier to inspect, test, and extend.

---

## 5. Why Add a Clarification Loop?

The clarification step was added to demonstrate agent-to-agent collaboration rather than simply running two independent agents.

Agent B reviews Agent A's structured analysis and identifies one important ambiguity or missing quantitative detail.

Agent B then asks Agent A exactly one clarification question.

Agent A responds using the information already available in its analysis.

Agent B incorporates that response into the final report.

This creates a simple critique-and-refinement cycle:

```text
Agent A
   ↓
Agent B Review
   ↓
Clarification
   ↓
Agent A Response
   ↓
Agent B Final Report
```

---

## 6. Why Persistent JSON Memory?

Task 3C required short-term follow-up capability without repeating the research process.

A JSON cache was chosen because the assessment scope is small and the stored research structure is straightforward.

The cache is organized by:

```text
ticker → research date → research data
```

It stores:

- Analyst output
- News results
- Web results
- Final report

This allows a follow-up agent to answer questions using previously generated research.

For example, after completing the AAPL research workflow, a follow-up question about the identified risks can be answered from the cached research without calling the research tools again.

---

## 7. Why Not Use a Database?

A production system would likely use a persistent database or document store instead of a local JSON file.

For this assessment, JSON provides several advantages:

- Minimal infrastructure
- Easy inspection
- Simple persistence
- Easy reproduction
- Human-readable assessment evidence

If this system were expanded into a production application, the cache layer could be replaced with a database without changing the higher-level agent workflow significantly.

---

## 8. Why Add Execution Tracing?

The `agent_trace.jsonl` file records the execution of tools and important agent events.

Each record contains:

- Timestamp
- Agent
- Event type
- Tool name
- Inputs
- Truncated output
- Execution duration
- Success/failure status

This makes the agent workflow inspectable rather than treating the LLM as a black box.

The trace is particularly useful for debugging:

- Unexpected tool selection
- Failed tool calls
- Slow operations
- Agent-to-agent interactions
- Follow-up cache access

---

## 9. LLM Provider Design

The implementation supports both OpenAI and Groq.

The provider selection is centralized in:

```text
src/llm/provider.py
```

This allows the same agent implementation to be tested with different providers without changing the agent logic.

The final Task 3 workflow was validated using Groq.

This design also makes it easier to replace the LLM provider in the future.

---

## 10. Important Trade-offs

### Multi-agent complexity

Using multiple agents introduces additional LLM calls and therefore increases latency compared with a single-agent implementation.

However, the separation provides clearer responsibilities and demonstrates structured agent collaboration.

### JSON persistence

JSON is simple and transparent but does not provide the concurrency, querying, indexing, or transactional guarantees expected from a production database.

### Web research quality

External web-search results can vary in relevance and quality.

The system therefore treats web research as supporting evidence rather than assuming every search result is authoritative.

### LLM-generated sentiment

Sentiment classification is probabilistic and depends on the supplied headlines and model behavior.

The system validates the output structure but does not claim that the sentiment score is a ground-truth financial signal.

---

## 11. Limitations

The current implementation is an assessment-scale prototype rather than a production financial research platform.

Some limitations include:

- Local JSON persistence
- Dependence on external APIs
- Dependence on the quality of web-search results
- No comprehensive market-data validation layer
- No portfolio-level risk model
- No real-time streaming market data
- No automated backtesting of hedge strategies
- No authentication or multi-user persistence layer

The hedge recommendation is therefore an analytical output of the workflow rather than an automated trading instruction.

---

## 12. What I Would Improve for Production

If developing this system beyond the assessment, I would consider:

### Persistent Storage

Replace the JSON cache with a database or document store.

### Better Retrieval

Use a dedicated financial-news/search pipeline with source ranking, deduplication, and relevance scoring.

### Evaluation

Introduce automated evaluation for:

- Tool selection
- Sentiment accuracy
- Risk identification
- Citation quality
- Structured-output validity

### Observability

Add centralized tracing and metrics for:

- Token usage
- Latency
- Tool failures
- Agent iterations
- Cost
- Model performance

### Risk Analysis

Extend the financial analysis with:

- Portfolio exposure
- Beta
- Value-at-Risk
- Historical drawdown
- Options pricing
- Scenario analysis
- Hedge backtesting

### Model Routing

Use different models based on task complexity.

For example:

- Smaller/cheaper model for classification
- More capable model for final synthesis
- Specialized financial models for quantitative tasks

---

## 13. Key Engineering Lesson

The main lesson from this task was that an agentic system is not simply an LLM connected to several tools.

The quality of the system depends heavily on:

- Tool boundaries
- State management
- Structured outputs
- Agent responsibilities
- Validation
- Observability
- Failure handling
- Memory design

The multi-agent workflow demonstrates these concepts while keeping the implementation small enough to inspect and reproduce.

---

## 14. Final Assessment Scope

The implementation focuses on Task 3 — Agentic Workflows.

The final implementation includes:

- Single-agent tool-use workflow
- Multi-agent workflow
- Specialized Agent A and Agent B roles
- Structured Pydantic handoffs
- Agent clarification loop
- Persistent JSON research memory
- Cache-based follow-up questions
- Execution tracing
- Configurable OpenAI/Groq providers
- Groq end-to-end validation
```

