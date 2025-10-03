# 📚 RAG Assistant

## 🎯 Objectif
Développer un assistant intelligent capable de répondre à des questions en langage naturel à partir d’une base documentaire métier (PDF, rapports financiers, docs internes).

Exemples de cas d’usage :
- Assistant réglementaire Bâle IV
- Assistant technique DevOps

## ⚙️ Architecture
Pipeline RAG :
- Ingestion et parsing des documents (PDF/DOCX/HTML)
- Génération d’embeddings (SentenceTransformers)
- Indexation dans une base vectorielle (FAISS ou Pinecone)
- Retrieval des passages pertinents
- Réponse contextuelle générée par un LLM (Mistral / GPT-4)
- Interface utilisateur (Streamlit)

## 🛠️ Stack technique
- Python
- LangChain ou LlamaIndex
- FAISS / Pinecone
- HuggingFace SentenceTransformers
- Streamlit
- Docker

## 📂 Structure du projet
rag-assistant/
│── data/ # Datasets (PDF, rapports, docs techniques)
│── notebooks/ # POC & explorations
│── src/ # Code source principal
│ ├── ingestion.py # Parsing & nettoyage des documents
│ ├── embeddings.py # Génération des embeddings
│ ├── retriever.py # Recherche vectorielle
│ ├── rag_pipeline.py # Orchestration RAG
│ ├── app.py # Interface Streamlit
│── requirements.txt # Dépendances Python
│── Dockerfile # Conteneurisation
│── README.md # Documentation projet
│── .gitignore # Exclusions Git

## 🚀 Roadmap
1. POC local avec quelques PDF + FAISS + GPT-4.
2. Amélioration des embeddings et de la segmentation.
3. Interface Streamlit avec affichage des sources.
4. Déploiement Docker + Cloud.
5. Extension (fine-tuning, feedback utilisateur).
