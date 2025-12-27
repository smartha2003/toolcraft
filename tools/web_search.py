from typing import Any, Optional
from smolagents.tools import Tool
import warnings

class DuckDuckGoSearchTool(Tool):
    name = "web_search"
    description = "Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."
    inputs = {'query': {'type': 'string', 'description': 'The search query to perform.'}}
    output_type = "string"

    def __init__(self, max_results=10, **kwargs):
        super().__init__()
        self.max_results = max_results
        try:
            # Try new package name first, fall back to old one
            try:
                from ddgs import DDGS
            except ImportError:
                # Suppress the deprecation warning for now
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    from duckduckgo_search import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `ddgs` (or `duckduckgo_search`) to run this tool: for instance run `pip install ddgs`."
            ) from e
        self.ddgs = DDGS(**kwargs)

    def forward(self, query: str) -> str:
        try:
            results = self.ddgs.text(query, max_results=self.max_results)
            if len(results) == 0:
                return "## Search Results\n\nNo results found. Try a different search query or use visit_webpage to access specific URLs directly."
            postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
            return "## Search Results\n\n" + "\n\n".join(postprocessed_results)
        except Exception as e:
            # Instead of crashing, return a helpful message
            return f"## Search Results\n\nSearch encountered an error: {str(e)}\n\nTry using visit_webpage with a specific URL instead, or rephrase your search query."
