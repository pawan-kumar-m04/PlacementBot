# AI Placement Preparation Chatbot

An intelligent assistant designed to help Computer Science students prepare for technical interviews, coding rounds, and campus placements. This project was built to demonstrate proficiency in connecting static knowledge bases with real-time internet data.

## Features Implemented
* **RAG (Retrieval-Augmented Generation):** Users can query local PDF notes (like DSA cheatsheets or OS concepts). The app uses HuggingFace embeddings and a local FAISS vector database for fast, offline retrieval.
* **Autonomous Web Search:** If the AI determines the answer isn't in the local notes (e.g., "What are the latest 2026 interview questions?"), it automatically triggers a DuckDuckGo live web search to fetch real-time data.
* **Dynamic Response Modes:** A UI toggle allows users to switch between **Concise** (for quick revision) and **Detailed** (for in-depth concept learning) response styles.

## Tech Stack
* **Frontend:** Streamlit
* **LLM Engine:** Google Gemini (`gemini-2.5-flash`)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Orchestration:** LangChain

##  How to Run Locally
1. Clone the repository.
2. Install the dependencies: `pip install -r requirements.txt`
3. Add your Google Gemini API key to a `.env` file (`GEMINI_API_KEY=your_key`).
4. Place any reference PDFs inside the `data/` directory.
5. Run the application: `streamlit run app.py`
