from langchain_community.tools import DuckDuckGoSearchRun
import logging

logger = logging.getLogger(__name__)

def perform_web_search(query: str) -> str:
    """
    Performs a live web search when the RAG notes do not contain the answer.
    """
    try:
        logger.info(f"Initiating Web Search for: {query}")
        search_tool = DuckDuckGoSearchRun()
        results = search_tool.invoke(query)
        
        if not results:
            return "No relevant information found on the web."
            
        return results
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return "Search currently unavailable."