import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRACE_FILE = Path("agent_trace.jsonl")


def _truncate_output(output: Any, max_length: int = 200) -> str:
    """Convert output to a string and truncate it."""
    text = json.dumps(
        output,
        ensure_ascii=False,
        default=str,
    )

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def write_trace(
    tool_name: str,
    inputs: dict[str, Any],
    output: Any,
    duration_ms: float,
    success: bool,
    agent: str | None = None,
    event_type: str = "tool_call",
) -> None:
    """
    Append one execution/event record to agent_trace.jsonl.

    The agent and event_type fields are optional so existing
    Task 3A tracing remains backward compatible.
    """

    trace_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "event_type": event_type,
        "tool": tool_name,
        "inputs": inputs,
        "output": _truncate_output(output),
        "duration_ms": round(duration_ms, 2),
        "success": success,
    }

    with TRACE_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                trace_record,
                ensure_ascii=False,
            )
            + "\n"
        )