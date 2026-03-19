import streamlit as st
import os
from models.llm import LLMManager
from models.embeddings import RAGManager
from utils.rag import load_local_documents, process_uploaded_pdf
from utils.websearch import perform_web_search

# --- UI Configuration ---
st.set_page_config(page_title="Placement Prep Bot", page_icon="🎓", layout="wide")
st.title("🎓 AI Placement Preparation Chatbot")
st.markdown("Your ultimate assistant for DSA, OS, DBMS, and interview prep. Powered by local notes and live web search.")

# --- Initialization ---
@st.cache_resource
def initialize_systems():
    llm = LLMManager()
    rag = RAGManager()
    
    # Load default data/ folder notes on startup
    with st.spinner("Loading local placement notes..."):
        local_chunks = load_local_documents()
        if local_chunks:
            rag.create_embeddings(local_chunks)
            
    return llm, rag

llm_manager, rag_manager = initialize_systems()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I've loaded your placement notes. Ask me about DSA, interviews, or the latest tech trends!"}]

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Chat Settings")
    response_mode = st.radio("Response Mode", ["Concise", "Detailed"])
    
    st.divider()
    
    st.header("📄 Add More Notes")
    st.caption("Upload additional PDFs (e.g., your resume, extra cheatsheets)")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file and st.button("Add to Knowledge Base"):
        with st.spinner("Processing..."):
            new_chunks = process_uploaded_pdf(uploaded_file)
            if new_chunks and rag_manager.create_embeddings(new_chunks):
                st.success("Notes added successfully!")
            else:
                st.error("Failed to process PDF.")

# --- Main Chat UI ---
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("E.g., What is Binary Search? or What are the latest Java interview questions?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        status_text = st.empty()
        rag_context = ""
        web_context = ""

        # Step 1: Check Notes (RAG)
        if rag_manager.vector_store:
            status_text.text("📚 Checking placement notes...")
            rag_context = rag_manager.search_embeddings(prompt)

        # Step 2: Check if Web Search is needed (If not found in notes)
        status_text.text("🧠 Evaluating if web search is needed...")
        needs_search = llm_manager.evaluate_rag_coverage(prompt, rag_context)

        if needs_search:
            status_text.text("🌐 Searching the internet for latest info...")
            web_context = perform_web_search(prompt)
            st.toast("Used Live Web Search for up-to-date info!")
            
        # Step 3: Generate Answer
        status_text.text(f"✍️ Generating {response_mode.lower()} answer...")
        response = llm_manager.get_response(prompt, response_mode, rag_context, web_context)
        
        status_text.empty()
        st.write(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})