"""
Web Search Tool - Search the web using DuckDuckGo
"""

import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchTool:
    """Search the web using DuckDuckGo."""

    name = "web_search"
    description = "Search the web for information. Use this when you need current information, documentation, or answers to questions."

    def __init__(self):
        self.ddg = None

    def _ensure_client(self):
        """Lazy-load DuckDuckGo client."""
        if self.ddg is None:
            try:
                from duckduckgo_search import DDGS
                self.ddg = DDGS()
            except ImportError:
                raise ImportError("Please install duckduckgo-search: pip install duckduckgo-search")

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """
        Search the web for a query.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects
        """
        self._ensure_client()

        try:
            results = []
            for r in self.ddg.text(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get('title', ''),
                    url=r.get('href', ''),
                    snippet=r.get('body', '')
                ))
            return results
        except Exception as e:
            return [SearchResult(
                title="Error",
                url="",
                snippet=f"Search failed: {str(e)}"
            )]

    def run(self, query: str, max_results: int = 5) -> str:
        """
        Run the search and return formatted results.

        Args:
            query: The search query
            max_results: Maximum number of results

        Returns:
            Formatted string of search results
        """
        results = self.search(query, max_results)

        if not results:
            return "No results found."

        output = []
        for i, r in enumerate(results, 1):
            output.append(f"{i}. **{r.title}**")
            output.append(f"   URL: {r.url}")
            output.append(f"   {r.snippet}")
            output.append("")

        return "\n".join(output)

    def to_schema(self) -> dict:
        """Return tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }


if __name__ == "__main__":
    # Test the tool
    tool = WebSearchTool()
    print(tool.run("Laravel 11 new features"))
