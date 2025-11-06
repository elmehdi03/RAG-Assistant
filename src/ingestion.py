"""
Document Ingestion

This module handles loading and parsing PDF files
from the data folder and preparing them for embedding.
"""

import os
from PyPDF2 import PdfReader


def load_documents_from_pdf(folder="data"):
    """
    Load all PDF files from a folder and extract text.
    
    Args:
        folder (str): Path to folder containing PDFs
        
    Returns:
        tuple: (list of texts, list of metadata)
               Each page becomes one text chunk with metadata indicating source and page
    """
    texts = []
    metadata = []

    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        return texts, metadata

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            try:
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        texts.append(text)
                        metadata.append(f"{file} - page {i+1}")
                print(f"✅ Loaded: {file} ({len(reader.pages)} pages)")
            except Exception as e:
                print(f"❌ Error reading {file}: {str(e)}")

    return texts, metadata


if __name__ == "__main__":
    docs, meta = load_documents_from_pdf()
    print(f"\n📊 Documents loaded: {len(docs)} chunks")
    if meta:
        print(f"📌 First document: {meta[0]}")
    else:
        print("⚠️ No PDFs found in data/ folder")

