from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from core.config import Settings

logger = logging.getLogger(__name__)


def _make_trim_middleware(max_messages: int):
    """Return a before_model middleware that caps conversation history."""

    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        if len(messages) <= max_messages:
            return None

        trimmed = []
        count = 0
        i = len(messages) - 1

        while i >= 0 and count < max_messages:
            msg = messages[i]
            trimmed.insert(0, msg)
            count += 1
            # Keep assistant tool-call paired with its tool result
            if msg.type == "tool" and i - 1 >= 0 and messages[i - 1].type == "assistant":
                trimmed.insert(0, messages[i - 1])
                count += 1
                i -= 1
            i -= 1

        # Drop orphaned ToolMessages at the front (no preceding AIMessage tool_call)
        while trimmed and trimmed[0].type == "tool":
            trimmed.pop(0)

        # Ensure window starts with a user message, not an assistant one
        if trimmed and trimmed[0].type == "assistant":
            for j in range(len(messages)):
                if messages[j].type == "user":
                    trimmed.insert(0, messages[j])
                    break

        if not trimmed:
            return None

        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]}

    return trim_messages


def build_agent(settings: Settings, llm, tools: list, system_prompt: str):
    checkpointer = InMemorySaver()
    trim_mw = _make_trim_middleware(settings.memory.max_messages)

    agent = create_agent(
        llm,
        tools,
        checkpointer=checkpointer,
        system_prompt=system_prompt,
        middleware=[trim_mw],
        debug=False,
    )
    logger.info("Agent ready — %d tools: %s", len(tools), [t.name for t in tools])
    return agent
