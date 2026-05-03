"""Plugin tools for extending the RAG agent.

To add a custom tool — only TWO steps:
1. Create a new file here (e.g. plugins/my_tool.py)
2. Define a function whose name starts with build_ that returns a list of @tool-decorated functions

The system auto-discovers and loads every plugin in this directory on startup.
To disable a plugin without deleting it, add its filename (without .py) to
disabled_plugins in config/config.yaml.

Example
-------
# plugins/my_tool.py
from langchain_core.tools import tool

def build_my_tool() -> list:
    @tool
    def my_tool(query: str) -> str:
        "Describe when the agent should call this tool and what it returns."
        return "result"
    return [my_tool]
"""
