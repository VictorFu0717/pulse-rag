"""Plugin tools for extending the RAG agent.

To add a custom tool:
1. Create a new file here  (e.g. plugins/my_tool.py)
2. Implement a build function that returns a list of @tool-decorated functions
3. Import and call it inside core/server.py (or wire it through config)

Example
-------
# plugins/my_tool.py
from langchain_core.tools import tool

def build_my_tool() -> list:
    @tool
    def my_tool(query: str) -> str:
        "My custom tool description"
        return "result"
    return [my_tool]
"""
