import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "legal_chunks.json"
OUTPUT_INDEX = "legal_index.faiss"

print("🔄 Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("📥 Loading chunks...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract text from the correct key: "content"
texts = [chunk["content"] for chunk in chunks]
print(f"📝 Total chunks loaded: {len(texts)}")

print("🧠 Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

print("📦 Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, OUTPUT_INDEX)

print("✅ Embedding completed!")
print(f"📌 Saved index to: {OUTPUT_INDEX}")