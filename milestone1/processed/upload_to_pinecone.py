import json
import numpy as np
from pinecone import Pinecone, ServerlessSpec

API_KEY = "pcsk_48xcoJ_SN8qhbSXSFK1e8kSoDTxJw6PhrKJDjaLwGGTnyKPzpprxRCJpnVwfNKK6w4kTo1"
INDEX_NAME = "legal_ai"

pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

# ---- LOAD EMBEDDINGS ----
embeddings = np.load("legal_embeddings.npy")
with open("legal_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Convert to vector objects
vectors = []
for i, emb in enumerate(embeddings):
    vectors.append({
        "id": f"vec_{i}",
        "values": emb.tolist(),
        "metadata": {"text": chunks[i]["content"]}
    })

# ---- UPLOAD IN BATCHES ----
BATCH_SIZE = 100   # safe size for Pinecone

print("🚀 Uploading in batches...")

for i in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[i:i+BATCH_SIZE]
    index.upsert(vectors=batch)
    print(f"Uploaded batch {i//BATCH_SIZE + 1}")

print("✅ Upload completed successfully!")