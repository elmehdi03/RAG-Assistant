from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np
import os
import pickle

# Path to store FAISS index and metadata
INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/metadata.pkl"

# Load a small, fast embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts using SentenceTransformers.
    Args:
        texts (list of str): List of sentences or document chunks.
    Returns:
        np.ndarray: Embedding matrix.
    """
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(texts, metadata=None):
    """
    Build a FAISS index for given texts and save it.
    Args:
        texts (list): Chunks of documents.
        metadata (list): Optional metadata (e.g., filenames, page numbers).
    """
    print("Generating embeddings...")
    embeddings = generate_embeddings(texts)

    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatIP(d)  # Inner Product (cosine similarity if normalized)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_PATH)

    # Save metadata if provided
    if metadata:
        with open(META_PATH, "wb") as f:
            pickle.dump(metadata, f)

    print(f"FAISS index saved with {len(texts)} entries.")

def load_faiss_index():
    """
    Load FAISS index and metadata.
    Returns:
        index (faiss.Index)
        metadata (list)
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("No FAISS index found. Run build_faiss_index() first.")

    index = faiss.read_index(INDEX_PATH)
    metadata = None
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
    return index, metadata

def search(query, k=5):
    """
    Search top-k similar documents for a query.
    Args:
        query (str): The input question.
        k (int): Number of results to return.
    Returns:
        results (list of tuples): (score, metadata)
    """
    index, metadata = load_faiss_index()
    q_emb = generate_embeddings([query])
    scores, ids = index.search(q_emb, k)

    results = []
    for i, idx in enumerate(ids[0]):
        meta = metadata[idx] if metadata else None
        results.append((scores[0][i], meta))
    return results


# Demo usage
if __name__ == "__main__":
    texts = [
        "Basel IV requires banks to hold more capital for credit risk.",
        "Streamlit is a Python library for building web apps.",
        "The Eiffel Tower is located in Paris, France.",
    ]
    metadata = ["finance_doc", "tech_doc", "geo_doc"]

    # Build FAISS index
    build_faiss_index(texts, metadata)

    # Search example
    query = "What does Basel IV say about credit?"
    results = search(query, k=2)
    print("Search results:", results)
