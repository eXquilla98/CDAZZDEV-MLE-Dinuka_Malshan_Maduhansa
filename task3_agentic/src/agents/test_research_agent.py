from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.research_agent import (
    llm_with_tools,
    RESEARCH_TOOLS,
)


def main() -> None:
    query = (
        "Analyse the current financial health and market sentiment "
        "of AAPL. Identify the top risks to its share price."
    )

    messages = [HumanMessage(content=query)]

    # ---------------------------------------------------------
    # STEP 1: Ask the LLM what it wants to do
    # ---------------------------------------------------------
    response = llm_with_tools.invoke(messages)

    print("=== FIRST LLM DECISION ===")

    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Arguments: {tool_call['args']}")

    # Add the LLM's response to the conversation
    messages.append(response)

    # ---------------------------------------------------------
    # STEP 2: Execute the selected tools
    # ---------------------------------------------------------
    for tool_call in response.tool_calls:

        selected_tool = next(
            tool
            for tool in RESEARCH_TOOLS
            if tool.name == tool_call["name"]
        )

        result = selected_tool.invoke(tool_call["args"])

        print("\n=== TOOL OBSERVATION ===")
        print(f"Tool: {tool_call['name']}")
        print(result)

        # Give the observation back to the LLM
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    # ---------------------------------------------------------
    # STEP 3: Ask the LLM what it wants to do next
    # ---------------------------------------------------------
    next_response = llm_with_tools.invoke(messages)

    print("\n=== SECOND LLM DECISION ===")

    if next_response.tool_calls:
        for tool_call in next_response.tool_calls:
            print(f"Tool: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")
    else:
        print("LLM decided it has enough information.")
        print("\nResponse:")
        print(next_response.content)


if __name__ == "__main__":
    main()