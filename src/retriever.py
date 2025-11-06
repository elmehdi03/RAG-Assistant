from embeddings import search

def retrieve_relevant_chunks(query, k=5):
    """
    Récupère les k chunks les plus pertinents à partir de FAISS.
    """
    results = search(query, k=k)
    return results

if __name__ == "__main__":
    q = "What does Basel IV say about credit risk?"
    chunks = retrieve_relevant_chunks(q, k=3)
    print("Résultats de recherche :", chunks)
from embeddings import search

def retrieve_relevant_chunks(query, k=5):
    """
    Récupère les k chunks les plus pertinents à partir de FAISS.
    """
    results = search(query, k=k)
    return results

if __name__ == "__main__":
    q = "What does Basel IV say about credit risk?"
    chunks = retrieve_relevant_chunks(q, k=3)
    print("Résultats de recherche :", chunks)
