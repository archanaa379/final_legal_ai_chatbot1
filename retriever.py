from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Setup Pinecone
# -----------------------------
PINECONE_API_KEY = "pcsk_4vY34V_2CwxtYPNqp1qLh86TxQAQiqZD1Bt3HHLaTrVWYHEnC1RDw7Y8okvwfkGdYeAuGm"
INDEX_NAME = "legal-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# -----------------------------
# 2. Load embedding model
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# 3. Retriever (k=4)
# -----------------------------
def retrieve(query: str, k: int = 4):
    # embed query
    q_emb = model.encode(query).tolist()

    # query Pinecone
    result = index.query(
        vector=q_emb,
        top_k=k,
        include_metadata=True
    )

    # format results
    docs = []
    for m in result.matches:
        docs.append({
            "id": m.id,
            "score": m.score,
            "text": m.metadata.get("text"),
            "page_number": m.metadata.get("page_number"),
            "file_hash": m.metadata.get("file_hash"),
            "source_pdf": m.metadata.get("source_pdf"),
            "source_act": m.metadata.get("source_act"),
        })

    return docs


# -----------------------------
# 4. Test the retriever
# -----------------------------
if _name_ == "_main_":
    query = "What is the Indian Contract Act?"
    results = retrieve(query)

    for r in results:
        print("\n-------------------------")
        print("ID:", r["id"])
        print("Score:", r["score"])
        print("Text:", r["text"][:300], "...")
        print("Page:", r["page_number"])
        print("PDF:", r["source_pdf"])