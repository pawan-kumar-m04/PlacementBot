from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

    def create_embeddings(self, chunks):
        """Creates or updates the vector embeddings."""
        if not chunks:
            return False
            
        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vector_store.add_documents(chunks)
            return True
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return False

    def search_embeddings(self, query: str, k: int = 3) -> str:
        """Searches the local notes for the answer."""
        if not self.vector_store:
            return ""
            
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return ""
