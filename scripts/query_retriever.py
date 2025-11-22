import os
import faiss
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INDEX_PATH = Path("app/faiss_index/index.faiss")
CHUNKS_PATH = Path("app/faiss_index/chunks.txt")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5  # number of chunks to retrieve

# ------------------------------------------------------------------
# LOAD INDEX AND MODEL
# ------------------------------------------------------------------
def load_index_and_chunks():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "[ERROR] FAISS index or chunks.txt not found. Run ingest_data.py first."
        )

    print("[INFO] Loading FAISS index and chunk metadata...")
    index = faiss.read_index(str(INDEX_PATH))

    chunks = []
    sources = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                src, text = parts
                sources.append(src)
                chunks.append(text)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return index, model, chunks, sources

# ------------------------------------------------------------------
# RETRIEVE TOP-K CHUNKS
# ------------------------------------------------------------------
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K):
    """Return top-k chunks most relevant to a query."""
    index, model, chunks, sources = load_index_and_chunks()

    print(f"[INFO] Embedding query: {query}")
    query_vector = model.encode([query], convert_to_numpy=True)

    distances, indices = index.search(query_vector, top_k)
    indices = indices[0]
    results = []

    for rank, i in enumerate(indices):
        if 0 <= i < len(chunks):
            results.append(
                {
                    "rank": rank + 1,
                    "source": sources[i],
                    "text": chunks[i],
                    "distance": float(distances[0][rank])
                }
            )
    return results

# ------------------------------------------------------------------
# INTERACTIVE DEMO
# ------------------------------------------------------------------
if __name__ == "__main__":
    query = input("Enter your question: ").strip()
    top_results = retrieve_relevant_chunks(query)

    print("\n--- Top Retrieved Chunks ---\n")
    for r in top_results:
        print(f"[{r['rank']}] Source: {r['source']} (distance={r['distance']:.2f})")
        print(r["text"][:500].strip())  # show first 500 chars
        print("\n--------------------------------\n")
