from retriever import retrieve_relevant_chunks


# ⚠️ Ici on met un LLM factice pour le test (remplacé plus tard par GPT/Mistral)
def fake_llm(prompt: str) -> str:
    return f"[LLM simulated answer] {prompt[:100]}..."

def ragpipeline(question: str):
    """
    Orchestration complète :
    - Retrieve les passages pertinents
    - Concatène contexte + question
    - Envoie au LLM (placeholder ici)
    """
    results = retrieve_relevant_chunks(question, k=3)

    # Construire le contexte
    context_texts = []
    for score, meta in results:
        context_texts.append(f"[{meta}] (score={round(score,2)})")

    context = "\n".join(context_texts)

    # Générer réponse
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = fake_llm(prompt)

    return answer
