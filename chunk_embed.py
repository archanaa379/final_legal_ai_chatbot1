import os
import hashlib
import logging
import torch
from dotenv import load_dotenv

# --- LangChain Components for Loading and Splitting ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Pinecone Components ---
# NOTE: Ensure you have 'pinecone-client' installed (pip install pinecone-client)
from pinecone import Pinecone, ServerlessSpec 

# --- Embedding Model ---
# NOTE: Ensure you have 'sentence-transformers' installed (pip install sentence-transformers)
from sentence_transformers import SentenceTransformer

# ---------- config ----------
# Load environment variables from a .env file (for PINECONE_API_KEY)
load_dotenv() 
PDF_FOLDER = "data"
INDEX_NAME = "legal-index"
BATCH_SIZE = 100

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- Pinecone Initialization ----------
# Fetch API key from environment variables
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check and create the index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    logging.info(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, # Corresponds to the embedding model's dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    logging.info("Index created successfully.")

index = pc.Index(INDEX_NAME)

# ---------- Embedding Model Initialization ----------
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# ---------- helpers ----------

def file_md5(path: str) -> str:
    """Calculates the MD5 hash of a file for indexing idempotency."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def infer_act_from_filename(fname: str) -> str:
    """Infers the legal act from the filename (e.g., 'ipc' -> 'Indian Penal Code')."""
    # NOTE: Since your uploaded files are general law topics 
    # (Administrative Law, Constitutional Law, Business Law),
    # this function might return "Unknown Act" for them, which is fine.
    f = fname.lower()
    if "ipc" in f or "penal" in f: return "Indian Penal Code"
    if "crpc" in f or "criminal" in f: return "Code of Criminal Procedure"
    if "evidence" in f: return "Indian Evidence Act"
    if "pocso" in f: return "POCSO Act"
    if "contract" in f: return "Indian Contract Act"
    if "domestic" in f: return "Domestic Violence Act"
    if "motor" in f: return "Motor Vehicles Act"
    if "negotiable" in f or "ni act" in f: return "Negotiable Instruments Act"
    if "juvenile" in f: return "Juvenile Justice Act"
    if "ndps" in f: return "NDPS Act"
    if "it" in f or "information technology" in f: return "Information Technology Act"
    if "constitution" in f: return "Constitution of India"
    if "administrative" in f: return "Administrative Law"
    if "business" in f: return "Business Law"
    return "Unknown Act"

def sanitize_metadata(m: dict) -> dict:
    """Ensures metadata values are correctly typed for Pinecone."""
    md = {}
    for k, v in m.items():
        if k in ("page_number", "chunk_index"):
            try:
                md[k] = int(v) # Ensure integers
            except Exception:
                md[k] = -1
            continue

        if k in ("file_hash", "source_pdf", "source_act"):
            md[k] = str(v) if v is not None else "" # Ensure strings
            continue

        if k == "text":
            txt = str(v) if v is not None else ""
            md[k] = txt[:2000] # Truncate text content for metadata limit
            continue

        # Standard check for Pinecone compatible types
        if isinstance(v, (str, int, float, bool)):
            md[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            md[k] = v
        else:
            md[k] = str(v) if v is not None else ""

    return md

# ---------- CHECK IF FILE IS ALREADY INDEXED ----------
def is_already_indexed(file_hash: str) -> bool:
    """Checks if a file (based on its hash) has already been indexed."""
    try:
        # Query with a zero vector and filter on the file_hash metadata
        res = index.query(
            vector=[0] * 384, # type: ignore
            filter={"file_hash": {"$eq": file_hash}}, # type: ignore
            top_k=1,
            include_metadata=False
        )
        return len(res.get("matches", [])) > 0  # type: ignore
    except Exception as e:
        # Logging exceptions is crucial for debugging
        logging.exception("Error while checking existing file: %s", e)
        return False

# ---------- INDEX A SINGLE PDF (ONLY IF NEW) ----------
def index_pdf(file_path: str):
    """Loads, chunks, embeds, and upserts a single PDF file."""
    file_name = os.path.basename(file_path)
    file_hash = file_md5(file_path)

    if is_already_indexed(file_hash):
        logging.info("⏭️ SKIPPING (Already Indexed): %s", file_name)
        return

    logging.info("✅ Indexing NEW PDF: %s", file_name)

    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    act_name = infer_act_from_filename(file_name)

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = splitter.split_documents(pages)
    logging.info(" - %s generated chunks", len(chunks))

    # 3. Embed and Upsert in Batches
    batch = []
    for i, chunk in enumerate(chunks):
        raw_page = chunk.metadata.get("page_number") or chunk.metadata.get("page") or None

        # Prepare all metadata
        metadata = {
            "text": chunk.page_content,
            "source_pdf": file_name,
            "source_act": act_name,
            "page_number": raw_page,
            "file_hash": file_hash,
            "chunk_index": i
        }

        safe_md = sanitize_metadata(metadata)

        # Generate embedding vector
        vector = embedding_model.encode(chunk.page_content).tolist()

        vec_obj = {
            "id": f"{file_name}-chunk-{i}",
            "values": vector,
            "metadata": safe_md
        }

        batch.append(vec_obj)

        # Upsert batch if it reaches BATCH_SIZE
        if len(batch) >= BATCH_SIZE:
            index.upsert(batch)
            logging.info("Upserted %d vectors", len(batch))
            batch = []

    # Upsert final remaining batch
    if batch:
        index.upsert(batch)
        logging.info("Upserted final %d vectors", len(batch))

    logging.info("✅ Finished indexing %s", file_name)

# ---------- INDEX ALL PDFs (ONLY NEW ONES) ----------
def index_all_pdfs():
    """Scans the PDF folder and indexes all new PDFs."""
    logging.info("📂 Scanning folder: %s", PDF_FOLDER)

    if not os.path.exists(PDF_FOLDER):
        logging.error("PDF folder not found: %s. Please create it and place your PDFs inside.", PDF_FOLDER)
        return

    for fname in os.listdir(PDF_FOLDER):
        if not fname.lower().endswith(".pdf"):
            continue

        fp = os.path.join(PDF_FOLDER, fname)
        index_pdf(fp)

    logging.info("🎉 All NEW PDFs processed successfully!")

# ---------- run ----------
if __name__ == "__main__": # Corrected the double underscore
    index_all_pdfs()