"""
Embeddings and FAISS Index Management

This module handles:
- Vector embedding generation using SentenceTransformers
- FAISS index creation and persistence
- GPU-accelerated embedding computation (CUDA support)
- Similarity search functionality
- Cache validation for efficient re-indexing
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import hashlib
import json
import torch

# ===============================================================
# Configuration
# ===============================================================
INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/metadata.pkl"
CACHE_HASH_PATH = "data/.cache_hash.json"

# Detect and use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"🔧 Using device: {DEVICE}")
else:
    print("💻 Using CPU (no GPU available)")

# Load embedding model once at module load
# Model will use GPU if available through PyTorch backend
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)


# ===============================================================
# GPU/Device Utilities
# ===============================================================

def get_device_info():
    """
    Get information about available compute device.
    
    Returns:
        dict: Device info including type, name, and memory
    """
    info = {
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info


def print_device_status():
    """Print current device status and capabilities."""
    info = get_device_info()
    print("\n" + "="*60)
    print("🖥️  DEVICE STATUS")
    print("="*60)
    print(f"Device: {info['device'].upper()}")
    if info['cuda_available']:
        print(f"GPU: {info['gpu_name']}")
        print(f"CUDA: {info['cuda_version']}")
        print(f"Memory: {info['gpu_memory_gb']:.2f} GB")
    print("="*60 + "\n")


# ===============================================================
# Embedding Generation
# ===============================================================

def generate_embeddings(texts):
    """
    Generate vector embeddings for a list of texts using SentenceTransformers.
    Automatically uses GPU if available for faster computation.
    
    Args:
        texts (list of str): List of document chunks or sentences
        
    Returns:
        np.ndarray: Embedding matrix of shape (n_texts, embedding_dim)
    """
    print(f"🧠 Generating embeddings for {len(texts)} chunks ({DEVICE.upper()})...")
    embeddings = model.encode(
        texts, 
        convert_to_numpy=True, 
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embeddings


# ===============================================================
# FAISS Index Management
# ===============================================================

def build_faiss_index(texts, metadata=None):
    """
    Build and save a FAISS index from document chunks.
    
    Args:
        texts (list): Document chunks to index
        metadata (list, optional): Corresponding metadata (e.g., filenames, page numbers)
    """
    if not texts:
        print("⚠️ No texts provided for indexing")
        return
    
    embeddings = generate_embeddings(texts)

    # Create FAISS index using Inner Product (cosine similarity for normalized vectors)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, INDEX_PATH)
    print(f"💾 FAISS index saved to {INDEX_PATH}")

    # Save metadata
    if metadata:
        with open(META_PATH, "wb") as f:
            pickle.dump(metadata, f)
        print(f"📋 Metadata saved to {META_PATH}")

    print(f"✅ Index complete: {len(texts)} entries")


def load_faiss_index():
    """
    Load FAISS index and metadata from disk.
    
    Returns:
        tuple: (faiss.Index, list of metadata or None)
        
    Raises:
        FileNotFoundError: If index files don't exist
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"❌ No FAISS index found at {INDEX_PATH}\n"
            "Run ingestion first: python src/ingestion.py"
        )

    index = faiss.read_index(INDEX_PATH)
    metadata = None
    
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
    
    return index, metadata


# ===============================================================
# Search Functionality
# ===============================================================

def search(query, k=5):
    """
    Search for the top-k most similar documents.
    Query embedding is generated on GPU if available.
    
    Args:
        query (str): The search query
        k (int): Number of results to return
        
    Returns:
        list: List of tuples (similarity_score, metadata)
    """
    index, metadata = load_faiss_index()
    
    # Generate embedding for query (uses GPU if available)
    q_emb = generate_embeddings([query])
    
    # Search in FAISS
    scores, ids = index.search(q_emb, k)

    # Format results
    results = []
    for i, idx in enumerate(ids[0]):
        meta = metadata[idx] if metadata else None
        results.append((scores[0][i], meta))
    
    return results


# ===============================================================
# Cache Management
# ===============================================================

def compute_data_hash(folder_path="data") -> str:
    """
    Compute SHA256 hash of all files in data folder.
    Used to detect changes in source documents.
    
    Args:
        folder_path (str): Path to data folder
        
    Returns:
        str: SHA256 hash of all files
    """
    sha = hashlib.sha256()
    
    if not os.path.exists(folder_path):
        return sha.hexdigest()
    
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            # Skip hidden files and cache files
            if f.startswith("."):
                continue
            
            file_path = os.path.join(root, f)
            try:
                with open(file_path, "rb") as file:
                    while chunk := file.read(8192):
                        sha.update(chunk)
            except (IOError, OSError):
                continue
    
    return sha.hexdigest()


def is_cache_valid(folder_path="data") -> bool:
    """
    Check if FAISS index is still valid (data hasn't changed).
    
    Args:
        folder_path (str): Path to data folder
        
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not os.path.exists(CACHE_HASH_PATH):
        return False
    
    if not os.path.exists(INDEX_PATH):
        return False
    
    try:
        with open(CACHE_HASH_PATH, "r") as f:
            saved = json.load(f)
        return saved.get("hash") == compute_data_hash(folder_path)
    except (json.JSONDecodeError, IOError):
        return False


def save_cache_hash(folder_path="data"):
    """
    Save current data hash to cache file.
    Called after successful re-indexing.
    
    Args:
        folder_path (str): Path to data folder
    """
    os.makedirs(folder_path, exist_ok=True)
    
    with open(CACHE_HASH_PATH, "w") as f:
        json.dump({"hash": compute_data_hash(folder_path)}, f)


def ensure_embeddings_up_to_date():
    """
    Check if embeddings need rebuilding and rebuild if necessary.
    Useful for ensuring index is fresh before search operations.
    """
    from ingestion import load_documents_from_pdf
    
    if is_cache_valid():
        print("✅ Cache valid — index is up to date")
        return
    
    print("🔄 Data changed — rebuilding embeddings...")
    texts, metadata = load_documents_from_pdf("data")
    
    if texts:
        build_faiss_index(texts, metadata)
        save_cache_hash()
        print("✅ Embeddings rebuilt and cached")
    else:
        print("⚠️ No PDFs found in data folder")


# Print device info on module load
print_device_status()



# ===============================================================
# Embedding Generation
# ===============================================================

def generate_embeddings(texts):
    """
    Generate vector embeddings for a list of texts.
    
    Args:
        texts (list of str): List of document chunks or sentences
        
    Returns:
        np.ndarray: Embedding matrix of shape (n_texts, embedding_dim)
    """
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# ===============================================================
# FAISS Index Management
# ===============================================================

def build_faiss_index(texts, metadata=None):
    """
    Build and save a FAISS index from document chunks.
    
    Args:
        texts (list): Document chunks to index
        metadata (list, optional): Corresponding metadata (e.g., filenames, page numbers)
    """
    if not texts:
        print("⚠️ No texts provided for indexing")
        return
    
    print(f"🧠 Generating embeddings for {len(texts)} chunks...")
    embeddings = generate_embeddings(texts)

    # Create FAISS index using Inner Product (cosine similarity for normalized vectors)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, INDEX_PATH)
    print(f"💾 FAISS index saved to {INDEX_PATH}")

    # Save metadata
    if metadata:
        with open(META_PATH, "wb") as f:
            pickle.dump(metadata, f)
        print(f"📋 Metadata saved to {META_PATH}")

    print(f"✅ Index complete: {len(texts)} entries")


def load_faiss_index():
    """
    Load FAISS index and metadata from disk.
    
    Returns:
        tuple: (faiss.Index, list of metadata or None)
        
    Raises:
        FileNotFoundError: If index files don't exist
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"❌ No FAISS index found at {INDEX_PATH}\n"
            "Run ingestion first: python src/ingestion.py"
        )

    index = faiss.read_index(INDEX_PATH)
    metadata = None
    
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
    
    return index, metadata


# ===============================================================
# Search Functionality
# ===============================================================

def search(query, k=5):
    """
    Search for the top-k most similar documents.
    
    Args:
        query (str): The search query
        k (int): Number of results to return
        
    Returns:
        list: List of tuples (similarity_score, metadata)
    """
    index, metadata = load_faiss_index()
    
    # Generate embedding for query
    q_emb = generate_embeddings([query])
    
    # Search in FAISS
    scores, ids = index.search(q_emb, k)

    # Format results
    results = []
    for i, idx in enumerate(ids[0]):
        meta = metadata[idx] if metadata else None
        results.append((scores[0][i], meta))
    
    return results


# ===============================================================
# Cache Management
# ===============================================================

def compute_data_hash(folder_path="data") -> str:
    """
    Compute SHA256 hash of all files in data folder.
    Used to detect changes in source documents.
    
    Args:
        folder_path (str): Path to data folder
        
    Returns:
        str: SHA256 hash of all files
    """
    sha = hashlib.sha256()
    
    if not os.path.exists(folder_path):
        return sha.hexdigest()
    
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            # Skip hidden files and cache files
            if f.startswith("."):
                continue
            
            file_path = os.path.join(root, f)
            try:
                with open(file_path, "rb") as file:
                    while chunk := file.read(8192):
                        sha.update(chunk)
            except (IOError, OSError):
                continue
    
    return sha.hexdigest()


def is_cache_valid(folder_path="data") -> bool:
    """
    Check if FAISS index is still valid (data hasn't changed).
    
    Args:
        folder_path (str): Path to data folder
        
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not os.path.exists(CACHE_HASH_PATH):
        return False
    
    if not os.path.exists(INDEX_PATH):
        return False
    
    try:
        with open(CACHE_HASH_PATH, "r") as f:
            saved = json.load(f)
        return saved.get("hash") == compute_data_hash(folder_path)
    except (json.JSONDecodeError, IOError):
        return False


def save_cache_hash(folder_path="data"):
    """
    Save current data hash to cache file.
    Called after successful re-indexing.
    
    Args:
        folder_path (str): Path to data folder
    """
    os.makedirs(folder_path, exist_ok=True)
    
    with open(CACHE_HASH_PATH, "w") as f:
        json.dump({"hash": compute_data_hash(folder_path)}, f)


def ensure_embeddings_up_to_date():
    """
    Check if embeddings need rebuilding and rebuild if necessary.
    Useful for ensuring index is fresh before search operations.
    """
    from ingestion import load_documents_from_pdf
    
    if is_cache_valid():
        print("✅ Cache valid — index is up to date")
        return
    
    print("🔄 Data changed — rebuilding embeddings...")
    texts, metadata = load_documents_from_pdf("data")
    
    if texts:
        build_faiss_index(texts, metadata)
        save_cache_hash()
        print("✅ Embeddings rebuilt and cached")
    else:
        print("⚠️ No PDFs found in data folder")

