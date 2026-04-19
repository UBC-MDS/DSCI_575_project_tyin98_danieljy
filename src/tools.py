import os
from tavily import TavilyClient
from langchain_core.tools import tool

@tool
def web_search(query: str, max_results: int = 3):
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY not found in environment variables. Web search is disabled."
    client = TavilyClient(api_key=api_key)

    try:
        results = client.search(query, max_results=max_results)
        snippets = [r["content"] for r in results.get("results", [])]
        if not snippets:
            return "No web search results found."
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Error during web search: {str(e)}"
