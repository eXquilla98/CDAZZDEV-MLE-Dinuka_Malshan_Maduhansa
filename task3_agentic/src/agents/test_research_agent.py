from src.agents.research_agent import RESEARCH_TOOLS


def main() -> None:
    print("=== RESEARCH AGENT TOOLS ===")

    for tool in RESEARCH_TOOLS:
        print(f"- {tool.name}")
        print(f"  Description: {tool.description}")
        print(f"  Arguments: {tool.args}")


if __name__ == "__main__":
    main()