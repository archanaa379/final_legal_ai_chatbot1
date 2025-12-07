import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- FILE PATHS ----
INDEX_FILE = "legal_index.faiss"
VECTORS_FILE = "legal_vectors.npy"
CHUNKS_FILE = "legal_chunks.json"

# ---- LOAD MODEL ----
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- LOAD DATA ----
print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("Loading chunk metadata...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ---- SEARCH FUNCTION ----
def search_legal_database(query, top_k=5):
    """
    Returns top_k most relevant chunks for the given query.
    """

    # 1. Convert query to vector
    query_vec = model.encode([query])

    # 2. Search FAISS
    distances, indices = index.search(query_vec, top_k)

    # 3. Gather matching chunks
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if int(idx) == -1:
            continue

        chunk = chunks[int(idx)]

        results.append({
            "file_name": chunk["file_name"],
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "score": float(dist)
        })

    return results


# ---- TEST SCRIPT ----
if __name__ == "__main__":
    print("\nEnter a legal query to search your database.")
    user_query = input("Query: ")

    print("\nSearching...\n")
    results = search_legal_database(user_query, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"File: {r['file_name']}")
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"Score: {r['score']}")
        print(f"Content:\n{r['content'][:400]}...")
        print()