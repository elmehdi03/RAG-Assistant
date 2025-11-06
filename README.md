# RAG Assistant# 📚



A Retrieval-Augmented Generation (RAG) assistant that answers questions based on document knowledge base.



## Overview

A Retrieval-Augmented Generation (RAG) assistant that answers questions in natural language based on a document knowledge base. This project implements a complete RAG pipeline from scratch without using LangChain or LlamaIndex frameworks.



This assistant processes PDF documents, creates a searchable vector database, and answers questions by retrieving relevant context from your documents.



## Use Cases## 🎯 OverviewA Retrieval-Augmented Generation (RAG) assistant that answers questions in natural language based on a document knowledge base. This project implements a complete RAG pipeline from scratch without using LangChain or LlamaIndex frameworks.## 🎯 Objectif



- Financial regulatory assistant

- Technical documentation assistant

- Internal knowledge base Q&AThis assistant can process PDF documents, create a searchable vector database, and answer questions by retrieving relevant context from your documents. It's designed to be simple, modular, and easy to understand.Développer un assistant intelligent capable de répondre à des questions en langage naturel à partir d’une base documentaire métier (PDF, rapports financiers, docs internes).

- Research paper analysis



## Architecture

**Use cases:**## 🎯 Overview

1. **Document Ingestion** - Parses PDF files using PyPDF2

2. **Embeddings Generation** - Uses SentenceTransformers for text embeddings- Financial regulatory assistant (e.g., Basel IV compliance)

3. **Vector Search** - FAISS for similarity search

4. **RAG Pipeline** - Combines retrieval and generation- Technical documentation assistantExemples de cas d’usage :

5. **Web Interface** - Streamlit application

- Internal knowledge base Q&A

## Tech Stack

- Research paper analysisThis assistant can process PDF documents, create a searchable vector database, and answer questions by retrieving relevant context from your documents. It's designed to be simple, modular, and easy to understand.- Assistant réglementaire Bâle IV

- Python 3.x

- FAISS - Vector search and indexing

- SentenceTransformers - Embeddings

- PyPDF2 - PDF parsing## ⚙️ Architecture- Assistant technique DevOps

- Streamlit - Web interface

- NumPy - Numerical operations

- Pickle - Data persistence

The RAG pipeline consists of:**Use cases:**

## Project Structure



```

rag-assistant/1. **Document Ingestion** (`ingestion.py`)- Financial regulatory assistant (e.g., Basel IV compliance)## ⚙️ Architecture

├── data/

│   ├── *.pdf   - Parses PDF files using PyPDF2

│   ├── faiss_index.bin

│   └── metadata.pkl   - Extracts text page-by-page with metadata tracking- Technical documentation assistantPipeline RAG :

├── src/

│   ├── app.py

│   ├── ingestion.py

│   ├── embeddings.py2. **Embeddings Generation** (`embeddings.py`)- Internal knowledge base Q&A- Ingestion et parsing des documents (PDF/DOCX/HTML)

│   ├── retriever.py

│   └── rag_pipeline.py   - Uses SentenceTransformers (`all-MiniLM-L6-v2` model)

├── LICENSE

└── README.md   - Creates dense vector representations of text chunks- Research paper analysis- Génération d’embeddings (SentenceTransformers)

```

   - Builds and manages FAISS vector index

## Getting Started

- Indexation dans une base vectorielle (FAISS ou Pinecone)

### Installation

3. **Vector Search** (`retriever.py`)

1. Clone the repository

2. Create virtual environment: `python -m venv .venv`   - Performs similarity search using FAISS## ⚙️ Architecture- Retrieval des passages pertinents

3. Activate: `.venv\Scripts\Activate.ps1`

4. Install: `pip install sentence-transformers faiss-cpu numpy PyPDF2 streamlit`   - Returns top-k relevant document chunks with scores



### Usage- Réponse contextuelle générée par un LLM (Mistral / GPT-4)



1. Build FAISS index: `python src/ingestion.py`4. **RAG Pipeline** (`rag_pipeline.py`)

2. Test retrieval: `python src/retriever.py`

3. Launch web interface: `streamlit run src/app.py`   - Orchestrates retrieval and generationThe RAG pipeline consists of:- Interface utilisateur (Streamlit)



## Features   - Combines retrieved context with user questions



- PDF document parsing   - Currently uses a placeholder LLM (ready for integration)

- Vector embeddings using SentenceTransformers

- FAISS indexing and search

- RAG pipeline orchestration

- Streamlit web interface5. **Web Interface** (`app.py`)1. **Document Ingestion** (`ingestion.py`)## 🛠️ Stack technique



## Roadmap   - Interactive Streamlit application



- [x] PDF ingestion and parsing   - Simple question-answer interface   - Parses PDF files using PyPDF2- Python

- [x] Vector embedding generation

- [x] FAISS indexing and search

- [x] Basic RAG pipeline

- [x] Streamlit web interface## 🛠️ Tech Stack   - Extracts text page-by-page with metadata tracking- LangChain ou LlamaIndex

- [ ] Integrate production LLM

- [ ] Improved chunking strategy

- [ ] Source citations in UI

- [ ] Document upload functionality- **Python 3.x**- FAISS / Pinecone

- [ ] Conversation memory

- [ ] Evaluation metrics- **FAISS** - Vector similarity search and indexing

- [ ] Docker containerization

- [ ] Cloud deployment- **SentenceTransformers** - Embedding generation (HuggingFace)2. **Embeddings Generation** (`embeddings.py`)- HuggingFace SentenceTransformers



## License- **PyPDF2** - PDF document parsing



See LICENSE file- **Streamlit** - Web UI framework   - Uses SentenceTransformers (`all-MiniLM-L6-v2` model)- Streamlit



## Contributing- **NumPy** - Numerical operations



Contributions welcome! Feel free to open issues or submit pull requests.- **Pickle** - Metadata persistence   - Creates dense vector representations of text chunks- Docker




## 📂 Project Structure   - Builds and manages FAISS vector index



```## 📂 Structure du projet

rag-assistant/

├── data/                    # Document storage3. **Vector Search** (`retriever.py`)rag-assistant/

│   ├── *.pdf               # Source PDF documents

│   ├── faiss_index.bin     # FAISS vector index (generated)   - Performs similarity search using FAISS│── data/ # Datasets (PDF, rapports, docs techniques)

│   └── metadata.pkl        # Document metadata (generated)

├── src/   - Returns top-k relevant document chunks with scores│── notebooks/ # POC & explorations

│   ├── app.py              # Streamlit web interface

│   ├── ingestion.py        # PDF parsing and loading│── src/ # Code source principal

│   ├── embeddings.py       # Embedding generation & FAISS operations

│   ├── retriever.py        # Vector search wrapper4. **RAG Pipeline** (`rag_pipeline.py`)│ ├── ingestion.py # Parsing & nettoyage des documents

│   └── rag_pipeline.py     # RAG orchestration

├── LICENSE   - Orchestrates retrieval and generation│ ├── embeddings.py # Génération des embeddings

└── README.md

```   - Combines retrieved context with user questions│ ├── retriever.py # Recherche vectorielle



## 🚀 Getting Started   - Currently uses a placeholder LLM (ready for integration)│ ├── rag_pipeline.py # Orchestration RAG



### Prerequisites│ ├── app.py # Interface Streamlit



```powershell5. **Web Interface** (`app.py`)│── requirements.txt # Dépendances Python

python --version  # Python 3.8+

```   - Interactive Streamlit application│── Dockerfile # Conteneurisation



### Installation   - Simple question-answer interface│── README.md # Documentation projet



1. Clone the repository:│── .gitignore # Exclusions Git

```powershell

git clone https://github.com/elmehdi03/rag-assistant.git## 🛠️ Tech Stack

cd rag-assistant

```## 🚀 Roadmap



2. Create and activate a virtual environment:- **Python 3.x**1. POC local avec quelques PDF + FAISS + GPT-4.

```powershell

python -m venv .venv- **FAISS** - Vector similarity search and indexing2. Amélioration des embeddings et de la segmentation.

.venv\Scripts\Activate.ps1

```- **SentenceTransformers** - Embedding generation (HuggingFace)3. Interface Streamlit avec affichage des sources.



3. Install dependencies:- **PyPDF2** - PDF document parsing4. Déploiement Docker + Cloud.

```powershell

pip install sentence-transformers faiss-cpu numpy PyPDF2 streamlit- **Streamlit** - Web UI framework5. Extension (fine-tuning, feedback utilisateur).

```

- **NumPy** - Numerical operations

### Usage- **Pickle** - Metadata persistence



#### 1. Build the FAISS Index## 📂 Project Structure



First, place your PDF documents in the `data/` folder, then run:```

rag-assistant/

```powershell├── data/                    # Document storage

python src/ingestion.py│   ├── *.pdf               # Source PDF documents

```│   ├── faiss_index.bin     # FAISS vector index (generated)

│   └── metadata.pkl        # Document metadata (generated)

This will:├── src/

- Load all PDFs from the `data/` folder│   ├── app.py              # Streamlit web interface

- Extract text page-by-page│   ├── ingestion.py        # PDF parsing and loading

- Generate embeddings│   ├── embeddings.py       # Embedding generation & FAISS operations

- Build and save the FAISS index│   ├── retriever.py        # Vector search wrapper

│   └── rag_pipeline.py     # RAG orchestration

#### 2. Test Retrieval├── LICENSE

└── README.md

```powershell```

python src/retriever.py

```## 🚀 Getting Started



#### 3. Launch the Web Interface### Prerequisites



```powershell```powershell

streamlit run src/app.pypython --version  # Python 3.8+

``````



The app will open in your browser at `http://localhost:8501`### Installation



## 🔧 Configuration1. Clone the repository:

```powershell

### Embedding Modelgit clone https://github.com/elmehdi03/rag-assistant.git

cd rag-assistant

The default model is `sentence-transformers/all-MiniLM-L6-v2` (efficient and fast). To use a different model, edit `embeddings.py`:```



```python2. Create and activate a virtual environment:

model = SentenceTransformer("your-model-name")```powershell

```python -m venv .venv

.venv\Scripts\Activate.ps1

### Index Storage Paths```



Modify paths in `embeddings.py`:3. Install dependencies:

```powershell

```pythonpip install sentence-transformers faiss-cpu numpy PyPDF2 streamlit

INDEX_PATH = "data/faiss_index.bin"```

META_PATH = "data/metadata.pkl"

```### Usage



### LLM Integration#### 1. Build the FAISS Index



The current implementation uses a placeholder LLM in `rag_pipeline.py`. To integrate a real LLM:First, place your PDF documents in the `data/` folder, then run:



```python```powershell

# Replace fake_llm() with your LLM of choice:python src/ingestion.py

# - OpenAI GPT-4```

# - Anthropic Claude

# - Mistral AIThis will:

# - Local models (Ollama, llama.cpp)- Load all PDFs from the `data/` folder

```- Extract text page-by-page

- Generate embeddings

## 📝 Example Workflow- Build and save the FAISS index



```python#### 2. Test Retrieval

from embeddings import build_faiss_index

from ingestion import load_documents_from_pdf```powershell

from rag_pipeline import ragpipelinepython src/retriever.py

```

# 1. Load and index documents

texts, metadata = load_documents_from_pdf("data")#### 3. Launch the Web Interface

build_faiss_index(texts, metadata)

```powershell

# 2. Ask questionsstreamlit run src/app.py

question = "What does Basel IV say about credit risk?"```

answer = ragpipeline(question)

print(answer)The app will open in your browser at `http://localhost:8501`

```

## 🔧 Configuration

## 🎯 Roadmap

### Embedding Model

- [x] PDF ingestion and parsing

- [x] Vector embedding generationThe default model is `sentence-transformers/all-MiniLM-L6-v2` (efficient and fast). To use a different model, edit `embeddings.py`:

- [x] FAISS indexing and search

- [x] Basic RAG pipeline```python

- [x] Streamlit web interfacemodel = SentenceTransformer("your-model-name")

- [ ] Integrate production LLM (GPT-4, Claude, Mistral)```

- [ ] Improve chunking strategy (semantic splitting)

- [ ] Display source citations in UI### Index Storage Paths

- [ ] Add document upload functionality

- [ ] Implement conversation memoryModify paths in `embeddings.py`:

- [ ] Add evaluation metrics (RAGAS)

- [ ] Docker containerization```python

- [ ] Cloud deploymentINDEX_PATH = "data/faiss_index.bin"

META_PATH = "data/metadata.pkl"

## 📄 License```



See [LICENSE](LICENSE) file for details.### LLM Integration



## 🤝 ContributingThe current implementation uses a placeholder LLM in `rag_pipeline.py`. To integrate a real LLM:



Contributions are welcome! Feel free to open issues or submit pull requests.```python

# Replace fake_llm() with your LLM of choice:
# - OpenAI GPT-4
# - Anthropic Claude
# - Mistral AI
# - Local models (Ollama, llama.cpp)
```

## 📝 Example Workflow

```python
from embeddings import build_faiss_index
from ingestion import load_documents_from_pdf
from rag_pipeline import ragpipeline

# 1. Load and index documents
texts, metadata = load_documents_from_pdf("data")
build_faiss_index(texts, metadata)

# 2. Ask questions
question = "What does Basel IV say about credit risk?"
answer = ragpipeline(question)
print(answer)
```

## 🎯 Roadmap

- [x] PDF ingestion and parsing
- [x] Vector embedding generation
- [x] FAISS indexing and search
- [x] Basic RAG pipeline
- [x] Streamlit web interface
- [ ] Integrate production LLM (GPT-4, Claude, Mistral)
- [ ] Improve chunking strategy (semantic splitting)
- [ ] Display source citations in UI
- [ ] Add document upload functionality
- [ ] Implement conversation memory
- [ ] Add evaluation metrics (RAGAS)
- [ ] Docker containerization
- [ ] Cloud deployment

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
