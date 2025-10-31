"""Search tools for web search and URL fetching."""
import requests
from typing import Optional
from bs4 import BeautifulSoup
import sys
from pathlib import Path
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


def _extract_text_from_html(html: str, max_length: int = 5000) -> str:
    """Extract and clean text from HTML content."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        return text
    except Exception as e:
        return f"Error parsing HTML: {str(e)}"


@tool
def google_search_tool(query: str) -> str:
    """Search the web using Google Search API or DuckDuckGo fallback.
    
    Use this tool when you need to find current information, news, or general
    web content. If the user provides a search query, use this tool.
    
    Args:
        query: The search query string
    
    Returns:
        Search results as a formatted string
    """
    try:
        # Try Google Custom Search API first if available
        if Config.GOOGLE_SEARCH_API_KEY and Config.GOOGLE_SEARCH_ENGINE_ID:
            search = GoogleSearchAPIWrapper(
                google_api_key=Config.GOOGLE_SEARCH_API_KEY,
                google_cse_id=Config.GOOGLE_SEARCH_ENGINE_ID
            )
            results = search.results(query, num_results=5)
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"URL: {result.get('link', 'N/A')}\n"
                    f"Snippet: {result.get('snippet', 'N/A')}\n"
                )
            return "\n---\n".join(formatted_results)
        else:
            # Fallback to DuckDuckGo
            search = DuckDuckGoSearchRun()
            return search.run(query)
    except Exception as e:
        return f"Error performing search: {str(e)}"


@tool
def fetch_url_tool(url: str) -> str:
    """Fetch and extract text content from a URL.
    
    Use this tool when the user provides a specific URL or link that you need
    to read and extract information from.
    
    Args:
        url: The URL to fetch content from
    
    Returns:
        Extracted text content from the URL
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        content = _extract_text_from_html(response.text)
        return f"Content from {url}:\n\n{content}"
    except requests.exceptions.Timeout:
        return f"Error: Timeout while fetching {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

