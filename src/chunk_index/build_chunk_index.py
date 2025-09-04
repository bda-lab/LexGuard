# build_chunk_index.py
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from PyPDF2 import PdfReader    # pip install PyPDF2

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # or your chosen SBERT
model = SentenceTransformer(EMBED_MODEL)

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
        return "\n".join(texts)
    except Exception as e:
        print(f"[WARN] PDF parse failed for {path}: {e}")
        return ""

def read_text_file(path):
    # try UTF-8, fall back to cp1252, and finally ignore errors
    for enc in ("utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    # last resort
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text, chunk_size=256, stride=50):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += (chunk_size - stride)
    return chunks

def build_index_from_dir(business_dir, out_index="faiss.index", metadata_out="chunks.json"):
    docs = []
    filenames = sorted(os.listdir(business_dir))
    for fname in filenames:
        full = os.path.join(business_dir, fname)
        if os.path.isdir(full):
            continue
        lower = fname.lower()
        text = ""
        if lower.endswith(".pdf"):
            text = extract_text_from_pdf(full)
        elif lower.endswith((".txt", ".md", ".text")):
            text = read_text_file(full)
        else:
            # try to read as text but be defensive
            try:
                text = read_text_file(full)
            except Exception:
                print(f"[SKIP] Unsupported or binary file: {full}")
                continue

        if not text.strip():
            print(f"[INFO] No text extracted from {fname}, skipping.")
            continue
        docs.append({"filename": fname, "text": text})

    # Now chunk and index
    chunks = []
    for doc_id, doc in enumerate(docs):
        cs = chunk_text(doc["text"], chunk_size=200, stride=50)
        for c in cs:
            chunks.append({"doc_id": doc_id, "filename": docs[doc_id]["filename"], "text": c})

    texts = [c["text"] for c in chunks]
    if not texts:
        raise RuntimeError("No chunks to index. Check input files and text extraction.")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)  # cosine via normalized vectors
    index.add(embeddings)
    faiss.write_index(index, out_index)
    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks and index to {out_index}")

if __name__ == "__main__":
    business_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "business")
    business_dir = os.path.normpath(business_dir)
    print("Reading business files from:", business_dir)
    build_index_from_dir(business_dir)
