"""
RAG Pipeline: Combines retrieval and LLM-based generation.
Uses Mistral AI for high-quality contextual answers.
"""

import os
from dotenv import load_dotenv
from embeddings import search
from mistralai import Mistral

# Load environment variables from .env file
load_dotenv()

# Initialize Mistral client with API key from environment
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None


def call_mistral(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Call Mistral AI API to generate a response.
    
    Args:
        prompt (str): The input prompt
        max_tokens (int): Maximum tokens in the response
        temperature (float): Sampling temperature (0-1)
    
    Returns:
        str: Generated response from Mistral
    """
    if not client:
        return (
            "⚠️ Mistral API key not configured. "
            "Please set MISTRAL_API_KEY environment variable.\n\n"
            "To use Mistral:\n"
            "1. Get an API key from https://console.mistral.ai\n"
            "2. Set environment variable: $env:MISTRAL_API_KEY = 'your-key'\n"
            f"3. Prompt was: {prompt[:100]}..."
        )
    
    try:
        message = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return message.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling Mistral: {str(e)}"


def ragpipeline(question: str) -> str:
    """
    Complete RAG pipeline:
    - Retrieves relevant document passages
    - Combines context with user question
    - Sends to Mistral AI for intelligent response generation
    
    Args:
        question (str): User's question
    
    Returns:
        str: Generated answer based on retrieved context
    """
    # Retrieve relevant chunks from FAISS
    results = search(question, k=3)

    # Build context from retrieved chunks
    context_texts = []
    for score, meta in results:
        context_texts.append(f"[{meta}] (confidence: {round(score, 2)})")

    context = "\n".join(context_texts)

    # Create the prompt for Mistral
    prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {question}

Answer:"""

    # Generate response using Mistral
    answer = call_mistral(prompt)

    return answer

