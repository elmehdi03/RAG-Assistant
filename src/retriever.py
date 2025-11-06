"""
Vector Search Retrieval

This module provides search functionality for finding
relevant document chunks from the FAISS index.
"""

from embeddings import search


def retrieve_relevant_chunks(query, k=5):
    """
    Retrieve the top-k most relevant document chunks for a query.
    
    Args:
        query (str): The search query
        k (int): Number of results to return
        
    Returns:
        list: List of tuples (similarity_score, metadata)
    """
    results = search(query, k=k)
    return results

