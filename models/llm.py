import google.generativeai as genai
from config.config import GEMINI_API_KEY
import logging

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self):
        if GEMINI_API_KEY:
            # FIX 1: We use the variable here, NOT the actual key string!
            genai.configure(api_key=GEMINI_API_KEY)
            # FIX 2: We use gemini-pro to bypass the 404 error you got earlier
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None

    def evaluate_rag_coverage(self, query: str, rag_context: str) -> bool:
        """Determines if the RAG context is enough, or if we need a web search."""
        if not self.model or not rag_context:
            return True # If no context, we definitely need web search
            
        prompt = f"""
        User Query: "{query}"
        Local Notes Context: "{rag_context}"
        
        Can you fully and accurately answer the user query using ONLY the Local Notes Context?
        If the query asks for the 'latest', 'recent', or current trends, reply NO.
        Reply with ONLY "YES" or "NO".
        """
        try:
            response = self.model.generate_content(prompt)
            return "NO" in response.text.strip().upper()
        except Exception:
            return True # Default to searching if check fails

    def get_response(self, query: str, mode: str, rag_context: str, web_context: str) -> str:
        """Generates the final answer based on concise/detailed modes."""
        if not self.model:
            return "Please configure GEMINI_API_KEY in your .env file."

        persona = (
            "You are an expert Placement Preparation Assistant helping CS students with DSA, "
            "Java, DBMS, OS, and interviews. "
        )
        
        if mode == "Concise":
            persona += "Provide a short, direct, and summarized answer. Get straight to the point."
        else:
            persona += "Provide a highly detailed, comprehensive answer. Use bullet points, code snippets if relevant, and explain concepts thoroughly."

        prompt = f"{persona}\n\nUser Question: {query}\n"
        
        if rag_context:
            prompt += f"\n--- Context from Local Notes ---\n{rag_context}\n"
        if web_context:
            prompt += f"\n--- Context from Live Web Search ---\n{web_context}\n"
            
        prompt += "\nAnswer the question using the provided context. If no context is relevant, use your general knowledge."

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error generating the response."