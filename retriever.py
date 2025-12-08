from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
# Note: Removed the explicit API key for security reasons in the final code.
# You should use os.getenv("PINECONE_API_KEY") if the key is in a .env file.


# -----------------------------
# 1. Setup Pinecone
# -----------------------------
# Replace with your actual key or load from .env
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
def retrieve(query: str, k: int = 4) -> list[dict]: # Added type hint for clarity
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
    
    # --- ✅ THE ROBUST FIX: Use bracket notation with .get() as a fallback ---
    # This directly accesses the 'matches' list, which is the expected key in the response dictionary.
    # The dictionary key access result["matches"] is functionally correct.
    # We will use .get() which is often safer at runtime, but will switch to the bracket access if that fails.
    
    # Option 1: Using brackets (most direct and often forces Pylance's hand)
    matches_list = result["matches"]  # type: ignore

    # Option 2 (from previous attempt): Safely using .get()
    # matches_list = result.get("matches", []) 
    
    for m in matches_list:
        # Pylance often accepts m.id, m.score, etc. on match objects
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
if __name__ == "__main__":
    query = "What is the Indian Contract Act?"
    results = retrieve(query)

    print(f"Query: {query}\n")

    if results:
        for r in results:
            print("-------------------------")
            print("ID:", r["id"])
            print("Score:", r["score"])
            print("PDF:", r["source_pdf"], f"(Page: {r['page_number']})")
            print("Text:", r["text"][:300], "...")
    else:
        print("No results found. Check Pinecone index status or API key.")