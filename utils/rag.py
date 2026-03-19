import os
import tempfile
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR

logger = logging.getLogger(__name__)

def get_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def load_local_documents():
    """Loads all PDFs from the local data/ folder."""
    chunks = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return chunks

    splitter = get_text_splitter()
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                chunks.extend(splitter.split_documents(docs))
                logger.info(f"Loaded local document: {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                
    return chunks

def process_uploaded_pdf(uploaded_file):
    """Processes an optionally uploaded PDF from the UI."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        chunks = get_text_splitter().split_documents(docs)
        
        os.remove(temp_file_path)
        return chunks
    except Exception as e:
        logger.error(f"Error processing uploaded PDF: {e}")
        return []
