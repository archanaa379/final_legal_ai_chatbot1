"""
📄 UNIVERSAL PDF DOCUMENT LOADER (FOR LEGAL AI – RAG SYSTEMS)

Paste any number of PDF files into the `data/` folder and run:
    python loader.py

This script will:
1️⃣ Read all PDFs inside /data  
2️⃣ Extract text using PyPDF2 + fallback OCR (pytesseract)  
3️⃣ Clean legal text (remove headers/footers, line breaks, citations noise)  
4️⃣ Split into chunks for embeddings / retrieval  
5️⃣ Save processed chunks into /processed/legal_chunks.json

Perfect for:
- AI-Based Legal Reference Systems
- Case Retrieval
- RAG Pipelines (FAISS, ChromaDB, Elasticsearch)
- Summarization & Precedent Search Models
"""

import os
import json
import re
import PyPDF2
from pathlib import Path

# Optional OCR fallback
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_ENABLED = True
except:
    OCR_ENABLED = False


# -----------------------------
# PDF → TEXT Extractor
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except:
        extracted = None

    # Fallback OCR (for scanned PDFs)
    if not text.strip() and OCR_ENABLED:
        print(f"⚠️ Scanned PDF detected → Running OCR on: {pdf_path}")
        pages = convert_from_path(pdf_path)
        for page in pages:
            text += pytesseract.image_to_string(page)

    return text


# -----------------------------
# Legal Text Cleaner
# -----------------------------
def clean_legal_text(text):
    # Remove line breaks, multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove case headers like: "SUPREME COURT OF INDIA",-date lines etc.
    text = re.sub(r"(SUPREME COURT OF INDIA|HIGH COURT.*?|DATED:.*? )", " ", text, flags=re.I)

    # Remove page numbers
    text = re.sub(r"Page \d+ of \d+", " ", text, flags=re.I)

    # Remove typical legal formatting noise
    text = re.sub(r"[\n\r\t]+", " ", text)

    return text.strip()


# -----------------------------
# Chunk the text
# -----------------------------
def chunk_text(text, chunk_size=1000, overlap=100):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -----------------------------
# Main Loader
# -----------------------------
def load_all_documents():
    data_dir = Path("data")
    output = []

    if not data_dir.exists():
        print("❌ No 'data' folder found! Create one and add PDF files.")
        return

    for file in data_dir.glob("*.pdf"):
        print(f"📄 Loading: {file.name}")
        raw_text = extract_text_from_pdf(file)
        cleaned = clean_legal_text(raw_text)
        chunks = chunk_text(cleaned)

        for idx, ch in enumerate(chunks):
            output.append({
                "file_name": file.name,
                "chunk_id": idx,
                "content": ch
            })

    # Save results
    Path("processed").mkdir(exist_ok=True)
    with open("processed/legal_chunks.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"\n✅ Extraction complete! {len(output)} text chunks saved to:")
    print("➡ processed/legal_chunks.json")


if __name__ == "__main__":
    load_all_documents()