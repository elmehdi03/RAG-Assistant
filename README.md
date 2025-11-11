# 🤖 RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) assistant that answers natural-language questions using the content of your own documents (PDFs, reports, manuals…). This project implements a transparent RAG pipeline from ingestion → embeddings → FAISS retrieval → response generation, without requiring LangChain or LlamaIndex.

✨ Highlights
- � Production-Ready: Real LLM integration (Mistral AI), GPU acceleration, professional UI  
- 📦 Local-first: documents, embeddings, and index are stored locally for privacy and offline use.  
- 🧩 Modular: replace ingestion, embedding model, or LLM easily.  
- 🖥️ Professional UI: Streamlit-based interface with gradient design, status indicators, and source citations.
- ⚡ GPU-Accelerated: Optimized for NVIDIA GPUs (CUDA 12.1) with CPU fallback
- 👁️ File Monitoring: Automatic PDF detection and re-indexing via watchdog

Status: **Production-Ready** — suitable for deployment and real-world use.

---

Table of Contents
- 🚀 Features
- ⚡ Quick Start
- 🗂️ Project Layout
- 🧠 How it works (architecture)
- ⚙️ Configuration
- 🧪 Examples
- 🛠️ Troubleshooting & Tips
- 🛣️ Roadmap
- 🤝 Contributing
- 📜 License
- ✉️ Contact

---

## 🚀 Features
- 📄 PDF ingestion and text extraction (PyPDF2)  
- ✂️ Context-preserving chunking with metadata tracking  
- 🧠 GPU-accelerated embeddings using SentenceTransformers (all-MiniLM-L6-v2)  
- 🔎 FAISS vector store with similarity search (CPU index, GPU-accelerated embeddings)  
- � Real LLM integration with Mistral AI (mistral-small model)  
- 🌐 Professional Streamlit UI with gradient design, adjustable parameters, and source citations
- 📊 System status monitoring (GPU, cache validation, PDF count)
- 👁️ Automatic file monitoring and re-indexing (watchdog)
- ⚡ Full GPU support with CUDA 12.1 and CPU fallback

---

## ⚡ Quick Start (5 minutes)

1️⃣ Clone and install
```bash
git clone https://github.com/elmehdi03/rag-assistant.git
cd rag-assistant
python -m venv .venv
source .venv/bin/activate    # on Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

2️⃣ Add documents
- Put PDF files in the `data/` directory:
  - `data/your_manual.pdf`
  - `data/other_docs.pdf`

3️⃣ Build the FAISS index (ingest, embed, index)
```bash
python src/ingestion.py
```
This will:
- 📥 Load PDFs from `data/`
- 🧼 Extract and clean text
- 🧩 Split into chunks (configurable)
- ⚙️ Create embeddings and store FAISS index + metadata to `data/`

4️⃣ Run the Streamlit app
```bash
streamlit run src/app.py
```
Open http://localhost:8501 in your browser and ask a question like:
- “Who won the World Cup?” ⚽  
- “What does Basel IV say about credit risk?” 📚

---

## � Security & API Configuration

### Setting up Mistral API Key (Secure Method)

**Never commit API keys to git!** This project uses environment variables for secure configuration.

1. **Copy the template:**
   ```bash
   cp .env.example .env
   ```

2. **Get your API key:**
   - Visit [Mistral AI Console](https://console.mistral.ai/)
   - Create an account or sign in
   - Generate a new API key

3. **Add to `.env` file:**
   ```bash
   MISTRAL_API_KEY=your-actual-api-key-here
   ```

4. **Verify it's protected:**
   - The `.env` file is automatically git-ignored
   - Never share or commit this file
   - Each team member should have their own `.env` file

---

## �🗂️ Project layout
rag-assistant/
- data/                    # Document storage & generated index
  - *.pdf                  # Source PDF files
  - faiss_index.bin        # FAISS binary index (generated)
  - metadata.pkl           # Chunk metadata (generated)
- src/
  - ingestion.py           # PDF parsing, cleaning, chunking
  - embeddings.py          # Embedding generation, FAISS operations
  - retriever.py           # Retrieval logic
  - rag_pipeline.py        # Combines context + query and calls an LLM
  - app.py                 # Streamlit UI
- requirements.txt
- LICENSE
- README.md

---

## 🧠 How it works (high level)
1. 📥 Ingestion: PDF → text → cleaned paragraphs → chunks (context-preserving)  
2. 🧠 Embeddings: text chunks → vector embeddings (SentenceTransformers)  
3. 🗃️ Indexing: FAISS index built from vectors, metadata stored separately  
4. 🔎 Retrieval: nearest-neighbor search (top-k) returns best chunks  
5. 📝 RAG: retrieved chunks + user prompt are fed to an LLM function (replaceable) to produce an answer, optionally with citations

---

## ⚙️ Configuration

Embedding model (default)
- `src/embeddings.py` uses:
  model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
- Swap to any SentenceTransformer model by changing the name or path. 🔁

FAISS installation
- CPU:
  ```bash
  pip install faiss-cpu
  ```
- GPU:
  ```bash
  pip install faiss-gpu
  ```
If installation fails, see Troubleshooting below. 🧰

LLM integration
- The repo ships with a simple placeholder (`fake_llm`) for demo responses.
- To use a production LLM:
  - Replace `fake_llm` in `src/rag_pipeline.py` with a function that calls OpenAI, Ollama, Mistral, Claude, etc. ☁️
  - Ensure you handle token limits and truncate or summarize retrieved chunks if needed.

Example: minimal OpenAI integration (conceptual)
```python
# in src/rag_pipeline.py
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def call_openai(prompt: str, max_tokens=512, temperature=0.0):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # choose the model you have access to
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp["choices"][0]["message"]["content"].strip()
```
🔐 Keep secrets out of the code; use environment variables.

---

## 🧪 Usage examples (Python)
Programmatic search + answer:
```python
from src.ingestion import load_documents_from_folder
from src.embeddings import build_faiss_index, search
from src.rag_pipeline import ragpipeline

# Build index (one-time)
texts, metadata = load_documents_from_folder("data")
build_faiss_index(texts, metadata)

# Ask a question
query = "What does Basel IV say about credit risk?"
results = search(query, k=3)      # returns nearest chunks
answer = ragpipeline(query)       # contextualized answer
print(answer)
```

Streamlit UI
- The UI shows GPU detection, cache validation, and lets you query the loaded index interactively. 🖱️

---

## 🛠️ Troubleshooting & tips
- FAISS install errors:
  - Use `faiss-cpu` if you don't have an NVidia GPU: `pip install faiss-cpu` 🧾
  - On Linux, ensure `gcc` and `python-dev` headers are installed.
- CUDA / GPU:
  - If using `faiss-gpu`, CUDA drivers and toolkit must match your GPU. 🔌
- Large PDFs:
  - Consider increasing chunk size or using an initial text-cleaning pass to remove boilerplate. 🧹
- Embedding reuse:
  - The embedding step saves metadata and vectors. Re-run ingestion only when documents change. 🔁
- Reducing index size:
  - Remove stopwords or apply light normalization before embedding (experimental). 🔬

---

## 🛣️ Roadmap (planned)
- ✨ Improved semantic chunking and adaptive chunk size  
- 📎 Source citation display in UI (show chunk origins)  
- ♻️ Conversation memory (context across turns)  
- 📤 Document upload from the UI  
- 🐳 Dockerfile and containerized deployment  
- 🔗 Integration examples for OpenAI, Ollama, Mistral, and local LLMs  
- ✅ CI automation and tests

---

## 🤝 Contributing
- Contributions welcome! Open an issue or a PR.  
- Suggestion flow:
  1. Create an issue describing the change 📝  
  2. Add tests where relevant ✅  
  3. Keep changes modular (ingestion, embeddings, retriever, UI) 🛠️

Code style / linting
- Prefer small, well-tested changes. Use `black` / `flake8` if adding more code.

---

## 📜 License
MIT — see LICENSE file. 🧾

---

## 🙏 Acknowledgements
- SentenceTransformers (UKPLab / Hugging Face) ❤️  
- FAISS (Facebook AI Research) ⚡  
- Streamlit 🌊

---

## ✉️ Contact
Maintainer: @elmehdi03  
Report issues at: https://github.com/elmehdi03/rag-assistant/issues
