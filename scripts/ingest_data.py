import os
from pathlib import Path
from typing import List, Dict

import faiss
import torch
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
DATA_DIR = Path("data")
OUT_DIR = Path("app/faiss_index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ------------------------------------------------------------------
# TEXT LOADING AND CHUNKING
# ------------------------------------------------------------------
def load_texts(data_dir: Path) -> Dict[str, str]:
    """Load all .txt files in the data directory."""
    docs = {}
    for file in data_dir.glob("*.txt"):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
            docs[str(file)] = text
        except Exception as e:
            print(f"[WARN] Could not read {file.name}: {e}")
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split long text into overlapping chunks for embedding."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ------------------------------------------------------------------
# MAIN INGEST PIPELINE
# ------------------------------------------------------------------
def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"[ERROR] Data folder '{DATA_DIR}' not found. Add .txt files first.")

    print("[INFO] Loading .txt documents...")
    raw_docs = load_texts(DATA_DIR)
    if not raw_docs:
        raise RuntimeError("No .txt files found in data/. Add text files and re-run.")

    print(f"[INFO] Loaded {len(raw_docs)} text files.")

    # Chunk all documents
    all_chunks, chunk_meta = [], []
    for path_str, text in raw_docs.items():
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for c in chunks:
            all_chunks.append(c)
            chunk_meta.append(path_str)

    print(f"[INFO] Total text chunks created: {len(all_chunks)}")

    # GPU device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading embedding model ({EMBEDDING_MODEL_NAME}) on {device}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    # Generate embeddings
    print("[INFO] Generating embeddings...")
    embeddings = model.encode(
        all_chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"[INFO] FAISS index built with {index.ntotal} vectors.")

    # Save results
    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    with open(OUT_DIR / "chunks.txt", "w", encoding="utf-8") as f:
        for chunk, src in zip(all_chunks, chunk_meta):
            f.write(src + "\t" + chunk.replace("\n", " ") + "\n")

    print("[INFO] Ingestion complete. Files saved in 'app/faiss_index/'.")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
