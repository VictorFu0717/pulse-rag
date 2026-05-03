"""Web search tool via Tavily.

Loaded automatically — place this file in plugins/ to enable.
Requires env: TAVILY_API_KEY  (plugin is skipped with a warning if not set)
"""
import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


def build_web_search_tool() -> list:
    if not os.getenv("TAVILY_API_KEY"):
        raise EnvironmentError("TAVILY_API_KEY not set — add it to .env or remove plugins/web_search.py")

    @tool("net_search")
    def net_search(query: str) -> str:
        """Search the web to answer questions outside the knowledge base.

        Only call this when the customer explicitly asks to search online,
        or when asking about current events unrelated to company services.
        Prefer Taiwan-focused sources — append 'site:tw' or '台灣' to queries.
        """
        search = TavilySearchResults(max_results=2)
        return search.invoke(query)

    return [net_search]
