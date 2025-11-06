import os
from PyPDF2 import PdfReader

def load_documents_from_pdf(folder="data"):
    """
    Charge tous les PDF dans le dossier et renvoie une liste de textes.
    Chaque document est segmenté en pages.
    """
    texts = []
    metadata = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    texts.append(text)
                    metadata.append(f"{file} - page {i+1}")

    return texts, metadata

if __name__ == "__main__":
    docs, meta = load_documents_from_pdf()
    print(f"Documents chargés : {len(docs)}")
    print("Exemple :", meta[0] if meta else "Aucun PDF trouvé")
