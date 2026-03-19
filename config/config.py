import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# RAG Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DATA_DIR = "data" # Directory for pre-loaded notes
