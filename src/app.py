"""
RAG Assistant - Streamlit Web Interface

A professional UI for asking questions about PDF documents
using the RAG (Retrieval-Augmented Generation) pipeline.
"""

import streamlit as st
import os
from pathlib import Path
from rag_pipeline import ragpipeline
from retriever import retrieve_relevant_chunks
from embeddings import is_cache_valid


# ===============================================================
# Page Configuration
# ===============================================================
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================
# Custom CSS Styling
# ===============================================================
st.markdown("""
<style>
    /* Gradient background */
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container */
    .main {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        margin: 1rem;
    }
    
    /* All text default to dark */
    * {
        color: #1a1a1a !important;
    }
    
    /* Header styling */
    h1 {
        text-align: center;
        color: #2c2c2c !important;
        margin-bottom: 0.5rem;
        font-weight: 900;
        font-size: 2.5rem;
    }
    
    /* Subheader */
    h2 {
        color: #1a1a1a !important;
        margin-top: 1.5rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        font-weight: 700;
    }
    
    /* Search input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 0.8rem;
        font-size: 1rem;
        color: #000 !important;
        background-color: #fff !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #764ba2;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Result box styling */
    .result-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-top: 1rem;
        color: #000 !important;
    }
    
    /* Source citation */
    .source-badge {
        display: inline-block;
        background-color: #e7f3ff;
        color: #667eea;
        padding: 0.4rem 0.8rem;
        border-radius: 5px;
        margin: 0.3rem;
        font-size: 0.85rem;
        border: 1px solid #667eea;
    }
    
    /* Status indicator */
    .status-good {
        color: #1a7c00 !important;
        font-weight: bold;
    }
    
    .status-warning {
        color: #cc6600 !important;
        font-weight: bold;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #ddd;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ===============================================================
# Sidebar - System Status
# ===============================================================
with st.sidebar:
    st.markdown("## 📊 System Status")
    st.divider()
    
    # Check if FAISS index exists
    index_exists = os.path.exists("data/faiss_index.bin")
    
    if index_exists:
        st.markdown("<p class='status-good'>✅ FAISS Index: Ready</p>", unsafe_allow_html=True)
        cache_valid = is_cache_valid()
        if cache_valid:
            st.markdown("<p class='status-good'>✅ Cache: Valid</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='status-warning'>⚠️ Cache: Outdated</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='status-warning'>⚠️ FAISS Index: Not Found</p>", unsafe_allow_html=True)
        st.info("📌 Run `python src/ingestion.py` to build the index")
    
    # PDF files count
    pdf_files = list(Path("data").glob("*.pdf"))
    st.metric("📄 PDF Files", len(pdf_files))
    
    st.divider()
    st.markdown("## ⚙️ Settings")
    
    # Configurable search parameters
    k_results = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="How many document chunks to use as context"
    )
    
    temperature = st.slider(
        "Response creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="0 = Deterministic, 1 = Creative"
    )
    
    st.divider()
    st.markdown("## 📚 About")
    st.markdown("""
    **RAG Assistant** combines document retrieval with AI 
    to answer questions based on your PDFs.
    
    - 🔍 Powered by FAISS
    - 🧠 LLM: Mistral AI
    - 📄 Parser: PyPDF2
    """)


# ===============================================================
# Main Content
# ===============================================================

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1>🤖 RAG Assistant</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    <p><i>Ask questions about your documents. Get intelligent answers.</i></p>
</div>
""", unsafe_allow_html=True)

# Check if system is ready
if not index_exists:
    st.error("❌ FAISS index not found. Please run: `python src/ingestion.py`")
else:
    # ===============================================================
    # Question Input
    # ===============================================================
    st.markdown("## 💬 Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="What information would you like to know?",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # ===============================================================
    # Processing and Display Results
    # ===============================================================
    if search_button and question.strip():
        with st.spinner("🔄 Searching documents and generating answer..."):
            try:
                # Get answer from RAG pipeline
                answer = ragpipeline(question)
                
                # Get relevant chunks for display
                relevant_chunks = retrieve_relevant_chunks(question, k=k_results)
                
                # Display answer
                st.markdown("## 🧠 Answer")
                st.markdown(f"""
                <div class='result-box'>
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources
                if relevant_chunks:
                    st.markdown("## 📚 Sources")
                    st.markdown("*Documents used to generate this answer:*")
                    
                    for idx, (score, metadata) in enumerate(relevant_chunks, 1):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"""
                            **{idx}. {metadata}**  
                            Relevance score: {score:.3f}
                            """)
                        with col2:
                            relevance_percent = min(int(score * 100), 100)
                            st.metric("Match", f"{relevance_percent}%")
                
                st.success("✅ Answer generated successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure your Mistral API key is set correctly in `src/rag_pipeline.py`")
    
    elif search_button and not question.strip():
        st.warning("⚠️ Please enter a question to search")
    
    # ===============================================================
    # Quick Tips
    # ===============================================================
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        - **Be specific**: "What are the main findings?" works better than "Tell me about the document"
        - **Use keywords**: Include specific terms from your documents
        - **Ask one thing**: Focus on a single question at a time
        - **Check sources**: Review the sources to verify the answer quality
        """)


# ===============================================================
# Footer
# ===============================================================
st.markdown("""
<div class='footer'>
    <p>🚀 RAG Assistant | Powered by <b>FAISS</b> + <b>Mistral AI</b> + <b>SentenceTransformers</b></p>
    <p>📖 <a href='https://github.com/elmehdi03/rag-assistant' target='_blank'>View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
