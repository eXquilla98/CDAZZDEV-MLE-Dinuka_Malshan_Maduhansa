```markdown
# Reflection — Task 3: Agentic Workflows

## 1. Why I Chose an Agentic Workflow

I chose to implement Task 3 as an agentic financial research workflow because the problem requires multiple sources of information, tool selection, iterative reasoning, and coordination between different responsibilities.

A traditional sequential pipeline could call every available tool in a fixed order. However, the assessment specifically requires the agent to decide which tools to use based on observations and demonstrate a tool-call → observation → decision cycle.

The implementation therefore uses LangChain-based agents with explicit tool boundaries and structured outputs.

The workflow supports:

- Financial market data analysis
- Technical indicators
- Historical volatility
- News retrieval
- LLM-based sentiment analysis
- Web research
- Multi-agent collaboration
- Structured agent-to-agent handoffs
- Clarification between agents
- Persistent research memory
- Follow-up questions without re-running external tools
- Execution tracing and observability

## 2. Task 3A — Single Research Agent

The first part of the implementation uses a single research agent with access to five tools:

- `get_price_data(ticker, period)`
- `get_news(ticker, n)`
- `calculate_volatility(ticker, window)`
- `llm_sentiment(headlines)`
- `web_search(query)`

The agent is given the financial research objective and can decide which tools are required.

The research loop follows the general pattern:

```text
User Query
    ↓
Agent decides what information is required
    ↓
Tool call
    ↓
Observation
    ↓
Agent evaluates the observation
    ↓
Additional tool call if required
    ↓
Final structured report
```

This was important because the task was not simply about calling several APIs. The agent needed to use observations to determine what additional information was necessary.

The final report contains:

- Financial Health Summary
- Market Sentiment
- Top Three Risks
- Evidence supporting the risks
- Hedge Strategy Recommendation

The implementation also includes tool failure handling so that an individual tool failure does not necessarily terminate the entire research process.

## 3. Task 3B — Multi-Agent Architecture

For Task 3B, I separated the responsibilities into two agents.

### Agent A — Data Analyst

Agent A is responsible only for quantitative and sentiment analysis.

Its allowed tools are:

- `get_price_data`
- `calculate_volatility`
- `llm_sentiment`

Agent A produces a structured `DataAnalystOutput` object using Pydantic.

The output contains information such as:

- Latest price
- Technical indicators
- Annualized volatility
- Market sentiment
- Sentiment score
- Sentiment confidence
- Quantitative findings

Agent A does not perform web searches or retrieve external research.

### Agent B — Research Writer

Agent B is responsible for external research and report composition.

Its allowed tools are:

- `get_news`
- `web_search`

Agent B therefore handles qualitative information while Agent A owns price, volatility, and sentiment analysis.

This separation was intentional because it reduces overlap between agents and makes the responsibilities easier to reason about and test.

## 4. Structured Agent Handoff

I used Pydantic models for communication between the agents rather than passing arbitrary dictionaries or unstructured text.

The main models include:

- `DataAnalystOutput`
- `ClarificationRequest`
- `ClarificationResponse`
- `ResearchReport`

For example, the data analyst output is represented as a typed object containing the ticker, price information, volatility, sentiment, and quantitative findings.

This provides several benefits:

1. The receiving agent knows exactly what information is available.
2. Required fields are validated automatically.
3. The workflow becomes easier to debug.
4. The final output has a predictable schema.
5. The interface between agents is explicit rather than dependent on prompt formatting.

## 5. Clarification Loop

One of the most important parts of the multi-agent workflow is the clarification cycle.

The workflow is:

```text
Agent B
   ↓
External research
   ↓
Agent A
   ↓
Quantitative analysis
   ↓
Agent B
   ↓
Clarification request
   ↓
Agent A
   ↓
Clarification response
   ↓
Agent B
   ↓
Final report
```

The clarification is generated automatically by Agent B based on the information received from Agent A.

During testing, Agent B asked:

> What time window does the sentiment_score and sentiment_confidence represent?

Agent A responded that the sentiment tool provided the metrics but the exact time window was not specified in the available data.

This was useful because the agent did not invent a time period that was not present in the data.

Agent B then incorporated the clarification into the final report.

This demonstrates that the agents are not simply executing independently; they can exchange information and request additional clarification before producing the final output.

## 6. Task 3C — Persistent Memory

For Task 3C, I implemented a persistent JSON research cache.

The cache is stored at:

```text
memory/research_cache.json
```

Research is stored using the ticker and research date:

```text
ticker
    └── date
          ├── analyst_output
          ├── news_results
          ├── web_results
          └── final_report
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

I chose JSON for this assessment because the memory requirement is relatively small and a JSON-based store is simple to inspect, version, and commit alongside the project.

For a production system, I would consider using a database or dedicated vector/document store depending on the scale and retrieval requirements.

## 7. Follow-Up Agent

The follow-up agent demonstrates that previously collected research can be reused without calling the external research tools again.

For example, after completing the original AAPL research, a follow-up question such as:

```text
What was the main downside risk identified for AAPL?
```

can be answered from the cached research.

The follow-up workflow is:

```text
Follow-Up Question
       ↓
Load cached research
       ↓
LLM interprets cached information
       ↓
Answer
```

No new price, news, volatility, sentiment, or web-search calls are required.

This satisfies the short-term follow-up requirement while avoiding unnecessary external API calls.

## 8. Observability and Agent Trace

I implemented execution tracing using:

```text
agent_trace.jsonl
```

Each trace entry records information including:

- Timestamp
- Agent
- Event type
- Tool name
- Tool inputs
- Truncated output
- Execution duration
- Success status

The tool output is truncated to a maximum of 200 characters to keep the trace manageable.

The trace allows the execution sequence to be inspected after the workflow completes.

For example, the multi-agent execution produces events corresponding to:

```text
Agent B → news
Agent B → web search
Agent A → price data
Agent A → volatility
Agent A → sentiment
Agent B → clarification request
Agent A → clarification response
Agent B → final report
```

The follow-up agent also records a cache-read event.

This was particularly useful during development because it made it possible to verify which agent performed each action and how long individual operations took.

## 9. LLM Provider Abstraction

I implemented a shared LLM provider configuration so that the application can use different LLM providers without changing the agent implementation.

The provider is selected using:

```text
LLM_PROVIDER
```

Supported providers are:

```text
openai
groq
```

The provider factory is responsible for constructing the appropriate LangChain chat model.

This means the agents do not need to contain provider-specific logic.

The configuration can therefore be changed through environment variables rather than modifying the agent code.

I also applied the provider selection to the sentiment analysis component, which uses the provider's API directly.

This allowed me to validate the workflow using Groq while retaining OpenAI compatibility.

## 10. Design Trade-Offs

### JSON Cache vs Database

I chose a JSON cache because:

- The assessment dataset is small.
- The cache is easy to inspect.
- It is easy to reproduce locally.
- It can be committed as assessment evidence.
- It introduces minimal infrastructure.

For a production application with many users and frequent updates, a database would be more appropriate.

### Two Agents vs One Larger Agent

I intentionally separated the data analyst and research writer.

A single agent would be simpler, but it would have access to every tool and responsibility.

The two-agent design provides:

- Clear responsibility boundaries
- Smaller tool sets
- Structured communication
- Easier testing
- Better observability
- A natural clarification mechanism

The trade-off is increased workflow complexity and additional LLM calls.

### JSONL Trace vs Full Logging Framework

A JSONL trace was selected because it is:

- Simple
- Append-only
- Human-readable
- Easy to inspect
- Easy to process programmatically

For production use, I would consider a centralized observability system with structured metrics, tracing, error aggregation, and monitoring.

## 11. Error Handling

External tools can fail because of:

- Network errors
- API limitations
- Rate limits
- Invalid tickers
- Missing data
- Search failures

The implementation therefore attempts to prevent an individual tool failure from immediately terminating the entire research process.

The trace also records the success state of tool calls.

This makes failed operations visible during debugging.

A production implementation would additionally include retry policies, exponential backoff, circuit breakers, and more explicit fallback strategies.

## 12. Limitations

There are several limitations in the current implementation.

### Market Data

The financial analysis is based primarily on historical market data and technical indicators.

Technical indicators cannot reliably predict future prices on their own.

### News Data

News availability depends on external news sources and search results.

Search results can change over time and may contain incomplete or duplicated information.

### LLM Sentiment

LLM-based sentiment is inherently probabilistic.

The sentiment score should therefore be treated as an analytical signal rather than an objective measurement.

### Hedge Recommendation

The hedge strategy is generated from the available market information and should not be considered personalized financial advice.

A production financial system would require stronger validation, risk controls, and potentially human review.

### Persistent Memory

The JSON cache is suitable for the assessment but is not designed for concurrent multi-user workloads.

## 13. What I Would Improve for Production

If this system were developed beyond the assessment, I would improve it in several areas.

### Data Layer

I would introduce:

- A dedicated market-data service
- Database-backed research storage
- Data freshness validation
- Historical data versioning

### Agent Reliability

I would add:

- Retry policies
- Tool timeouts
- Structured error recovery
- Agent execution limits
- More deterministic routing where appropriate

### Observability

I would introduce:

- Centralized tracing
- Token/cost tracking
- Latency monitoring
- Tool success-rate metrics
- Agent evaluation metrics

### Evaluation

I would create an automated evaluation framework measuring:

- Tool-selection accuracy
- Factual consistency
- Risk identification quality
- Citation quality
- Structured-output validity
- Follow-up answer correctness

### Security

Production deployment would also require:

- Secret management
- API authentication
- Access controls
- Input validation
- Rate limiting
- Audit logging

## 14. Key Engineering Lesson

The main lesson from this task was that an agentic workflow is not simply an LLM connected to several tools.

The important engineering challenge is controlling how the system reasons and how information moves between components.

The implementation therefore focuses on:

```text
Tool boundaries
      +
Structured outputs
      +
Explicit agent responsibilities
      +
Observable execution
      +
Persistent memory
      +
Controlled follow-up
```

The structured handoff between Agent A and Agent B was particularly useful because it made the multi-agent workflow more predictable than relying entirely on free-form natural-language communication.

## 15. Final Assessment Scope

The implementation intentionally focuses on Task 3 of the assessment.

The completed functionality includes:

- [x] Task 3A single research agent
- [x] Price data tool
- [x] News retrieval tool
- [x] Historical volatility tool
- [x] LLM sentiment tool
- [x] Web search tool
- [x] Tool-call → observation → decision cycle
- [x] Structured financial research report
- [x] Task 3B multi-agent workflow
- [x] Data Analyst agent
- [x] Research Writer agent
- [x] Pydantic agent handoff
- [x] Automatic clarification loop
- [x] Task 3C persistent JSON memory
- [x] Follow-up without re-running external tools
- [x] Agent execution tracing
- [x] OpenAI/Groq provider abstraction
- [x] Error handling and graceful degradation

The implementation was developed and tested incrementally, with the execution trace and persistent cache retained as evidence of the workflow execution.
```