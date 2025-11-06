##🤖 RAG Assistant
A Retrieval-Augmented Generation (RAG) assistant that answers questions in natural language using the content of your own documents (PDFs, reports, technical manuals…).
This project implements a complete RAG pipeline from scratch, without relying on LangChain or LlamaIndex — giving full control over ingestion, embedding, retrieval, and response generation.

##🎯 Objectives


Create an intelligent assistant able to retrieve and summarize knowledge from local documents.


Demonstrate a modular and transparent RAG pipeline using FAISS and SentenceTransformers.


Build an interactive Streamlit interface to query documents in real time.



##💡 Typical Use Cases


🏦 Regulatory assistant – Basel IV / banking compliance documentation


⚙️ DevOps / technical assistant – internal configuration or process manuals


🧑‍💼 Corporate knowledge base – company procedures or internal memos


🎓 Academic or research helper – paper summarization or literature search



##⚙️ Architecture
The pipeline consists of five main components:


Document Ingestion → ingestion.py


Extracts and cleans text from PDF files using PyPDF2


Splits documents into context-preserving chunks




Embeddings Generation → embeddings.py


Uses SentenceTransformers (all-MiniLM-L6-v2) to convert text chunks into dense vectors


Saves embeddings and metadata locally for fast reuse




Vector Search (Retrieval) → faiss.IndexFlatIP


Performs high-speed similarity search using FAISS


Returns the top-k most relevant document chunks




RAG Pipeline → rag_pipeline.py


Combines retrieved context with the user’s query


Generates a contextual response (with a local fallback or future LLM integration)




Web Interface → app.py


Streamlit-based UI with GPU detection, cache validation, and live querying





##🛠️ Tech Stack
CategoryToolsLanguagePython 3.xVector IndexingFAISSEmbeddingsSentenceTransformers (Hugging Face)ParsingPyPDF2InterfaceStreamlitUtilitiesNumPy, Pickle, Torch (CUDA support)

##📂 Project Structure
rag-assistant/
├── data/                    # Document storage
│   ├── *.pdf                # Source PDF files
│   ├── faiss_index.bin      # FAISS vector index
│   └── metadata.pkl         # Embedding metadata
│
├── src/                     # Core source code
│   ├── ingestion.py         # PDF parsing & cleaning
│   ├── embeddings.py        # Embedding generation & FAISS operations
│   ├── retriever.py         # Vector search logic
│   ├── rag_pipeline.py      # RAG orchestration
│   └── app.py               # Streamlit web interface
│
├── requirements.txt         # Python dependencies
├── LICENSE
└── README.md


🚀 Getting Started
1️⃣ Prerequisites
python --version   # Python 3.8+
pip install -r requirements.txt

If FAISS fails to install:
# CPU
pip install faiss-cpu
# or GPU (for RTX 4070 and similar)
pip install faiss-gpu

2️⃣ Build the FAISS Index
Place your PDFs inside the data/ folder, then run:
python src/ingestion.py

This will:


Load all documents


Extract and chunk text


Generate embeddings


Build and save the FAISS index (faiss_index.bin)


3️⃣ Launch the Web Interface
streamlit run src/app.py

The app will open in your browser at http://localhost:8501
Type a question such as:

“Who won the World Cup?” or “What does Basel IV say about credit risk?”


##⚙️ Configuration
Embedding Model
Default model:
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

You can replace it in embeddings.py with any SentenceTransformer model.
LLM Integration (Optional)
In rag_pipeline.py, replace the placeholder fake_llm() function with your own LLM API call (e.g. OpenAI GPT-4, Mistral, Claude, or a local model via Ollama).

##📈 Roadmap


 PDF ingestion and cleaning


 SentenceTransformer embeddings


 FAISS indexing and retrieval


 Streamlit interface


 Integration with production LLMs (Mistral / GPT-4)


 Improved semantic chunking


 Source citation display in UI


 Document upload from interface


 Conversation memory


 Dockerization & cloud deployment



##🧠 Example Workflow
from ingestion import load_documents_from_folder
from embeddings import build_faiss_index, search
from rag_pipeline import ragpipeline

# 1. Load and embed documents
texts, metadata = load_documents_from_folder("data")
build_faiss_index(texts, metadata)

# 2. Ask a question
query = "What does Basel IV say about credit risk?"
results = search(query, k=3)
answer = ragpipeline(query)

print(answer)


##📜 License
This project is released under the MIT License.
