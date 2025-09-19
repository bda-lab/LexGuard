# implement_legal_chunking.py
"""
Implements the 3 chunking methods from:
  Ferraris et al., Legal Chunking: Evaluating Methods for Effective Legal Text Retrieval
(uses uploaded paper for reference/optional question extraction). :contentReference[oaicite:1]{index=1}

Usage:
  python implement_legal_chunking.py --docs_dir ./data/business --queries_file ./queries.txt --out_dir ./out

If --queries_file is omitted, the script tries to extract question-like lines from:
  /mnt/data/Legal_Chunking2.pdf
If none found, it falls back to a short example query set (you should provide proper queries).
"""

import os
import re
import csv
import json
import argparse
import math
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

# ---------- Config / defaults ----------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
sbert = SentenceTransformer(EMBED_MODEL)

# chunk params per paper
SIMPLE_CHUNK_SIZES = [128, 256, 512]           # words (paper used tokens; approximated by words)
SIMPLE_OVERLAPS = [8, 16]

RECURSIVE_CHUNK_SIZES = [128, 256, 512]
RECURSIVE_OVERLAPS = [8, 16]

# semantic chunking heuristics (adaptable)
SEMANTIC_MIN_WORDS = 60     # try to avoid ridiculous tiny chunks
SEMANTIC_MAX_WORDS = 600
SEMANTIC_SIM_THRESH = 0.74  # tuneable (0.7-0.8 region)

TOP_K = 3   # K most relevant chunks to retrieve (paper used K=3)

# ---------- Utilities ----------
def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception as e:
        print(f"[WARN] failed to read pdf {path}: {e}")
        return ""

def tokenize_words(text):
    # simple whitespace tokenizer; preserves punctuation as attached tokens
    return text.split()

def detokenize_words(words):
    return " ".join(words)

def simple_split(text, chunk_words=256, overlap=16):
    """
    Simple sliding window word-splitting with overlap.
    Returns list of chunk texts and (start_word_idx, end_word_idx).
    """
    words = tokenize_words(text)
    if not words:
        return []
    chunks = []
    i = 0
    step = chunk_words - overlap if chunk_words > overlap else max(1, chunk_words//2)
    while i < len(words):
        chunk = detokenize_words(words[i:i+chunk_words])
        start = i
        end = min(len(words), i+chunk_words)
        chunks.append({"text": chunk, "start_word": start, "end_word": end})
        i += step
    return chunks

def recursive_split_regex(text, chunk_words=256, overlap=16):
    """
    Recursive splitting heuristic:
      - First split on common legal section markers (Article X, Section X, 'Chapter', 'Annex', etc.)
      - For each section, if too large, fall back to simple_split inside that section.
    """
    # patterns that often mark legal sections
    split_pattern = re.compile(r'(?i)(\n\s*(?:article|section|chapter|annex|schedule)\s+\d+\b)|\n\n+')  # case-insensitive
    parts = []
    last = 0
    for m in split_pattern.finditer(text):
        idx = m.start()
        seg = text[last:idx].strip()
        if seg:
            parts.append(seg)
        last = idx
    tail = text[last:].strip()
    if tail:
        parts.append(tail)

    # if split produced nothing useful, fallback to paragraph splits
    if not parts:
        parts = re.split(r'\n\s*\n', text)

    # now ensure each part is not too big; if big => simple_split inside
    chunks = []
    for seg in parts:
        wcount = len(tokenize_words(seg))
        if wcount <= chunk_words:
            chunks.append({"text": seg})
        else:
            subchunks = simple_split(seg, chunk_words=chunk_words, overlap=overlap)
            for sc in subchunks:
                chunks.append({"text": sc["text"]})
    return chunks

def sentence_split(text):
    # naive sentence split - split on period/question/exclamation + whitespace
    # keeps abbreviations imperfectly, but OK for chunking use
    sents = re.split(r'(?<=[\.\?\!;:])\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def semantic_chunking(text, sim_threshold=SEMANTIC_SIM_THRESH, min_words=SEMANTIC_MIN_WORDS, max_words=SEMANTIC_MAX_WORDS):
    """
    Semantic sliding aggregator:
      - Split into sentences, embed sentences
      - Build chunk by aggregating consecutive sentences while chunk centroid similarity
        to next sentence >= sim_threshold (cosine). Also respect max_words.
    """
    sents = sentence_split(text)
    if not sents:
        return []
    sent_embs = sbert.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    # normalize to compute dot product as cosine
    norms = np.linalg.norm(sent_embs, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    sent_embs = sent_embs / norms

    chunks = []
    current_inds = [0]
    # centroid of current chunk embeddings
    centroid = np.copy(sent_embs[0])
    words_in_chunk = len(tokenize_words(sents[0]))
    for i in range(1, len(sents)):
        sent_vec = sent_embs[i]
        sim = float(np.dot(centroid, sent_vec))
        sent_words = len(tokenize_words(sents[i]))
        # merge if sim >= threshold and won't exceed max_words, OR if chunk smaller than min_words
        if (sim >= sim_threshold and words_in_chunk + sent_words <= max_words) or (words_in_chunk < min_words):
            # add to current chunk
            current_inds.append(i)
            # update centroid (mean)
            centroid = (centroid * (len(current_inds)-1) + sent_vec) / len(current_inds)
            words_in_chunk += sent_words
        else:
            # finalize current chunk
            chunk_text = " ".join(sents[idx] for idx in current_inds)
            chunks.append({"text": chunk_text})
            # start new chunk
            current_inds = [i]
            centroid = np.copy(sent_vec)
            words_in_chunk = sent_words
    # finalize last
    if current_inds:
        chunk_text = " ".join(sents[idx] for idx in current_inds)
        chunks.append({"text": chunk_text})
    return chunks

# ---------- Embedding and retrieval ----------
def build_faiss_index(texts, embed_model=sbert):
    embs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize to use IndexFlatIP for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    return index, embs

def retrieve_topk(index, embs, query_emb, k=3):
    # query_emb should be normalized
    D, I = index.search(query_emb.reshape(1, -1), k)
    scores = D[0].tolist()
    inds = I[0].tolist()
    return inds, scores

def cosine(a, b):
    a = np.array(a); b = np.array(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na*nb))

# ---------- Query extraction helper (tries to read uploaded paper) ----------
def extract_question_like_lines_from_pdf(pdf_path="/mnt/data/Legal_Chunking2.pdf"):
    txt = ""
    if os.path.exists(pdf_path):
        try:
            txt = extract_text_from_pdf(pdf_path)
        except Exception:
            txt = ""
    qs = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        # heuristics: lines that end with ? or start with "Q:" or start with a number + ')'
        if line.endswith('?') or line.startswith('Q:') or re.match(r'^\d+\.', line) or re.match(r'^[A-Z]\.', line):
            qs.append(line)
    # postprocess: keep unique, short
    qs = [q for q in qs if len(q.split()) >= 3 and len(q.split()) <= 40]  # reasonable question lengths
    # de-dup
    seen = set()
    out = []
    for q in qs:
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
    # limit to 50
    return out[:50]

# ---------- Experiment runner ----------
def run_experiment(docs_dir, queries, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # read all docs in directory and concatenate into single big text per file
    doc_texts = []
    filenames = sorted(os.listdir(docs_dir))
    for fname in filenames:
        full = os.path.join(docs_dir, fname)
        if os.path.isdir(full):
            continue
        lower = fname.lower()
        txt = ""
        if lower.endswith(".pdf"):
            txt = extract_text_from_pdf(full)
        elif lower.endswith(('.txt', '.md', '.text')):
            with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
        else:
            # try read
            try:
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
            except Exception:
                continue
        if txt.strip():
            doc_texts.append({"filename": fname, "text": txt})

    if not doc_texts:
        raise RuntimeError("No documents found in directory.")

    # run methods and collect chunks
    experiments = []

    # 1) Simple splitting grid
    for chunk_words in SIMPLE_CHUNK_SIZES:
        for overlap in SIMPLE_OVERLAPS:
            method_name = f"simple_{chunk_words}_ov{overlap}"
            chunks = []
            for doc in doc_texts:
                cs = simple_split(doc["text"], chunk_words=chunk_words, overlap=overlap)
                for c in cs:
                    chunks.append({"doc": doc["filename"], "method": method_name, "text": c["text"]})
            experiments.append({"name": method_name, "chunks": chunks, "params": {"chunk_words":chunk_words, "overlap":overlap}})

    # 2) Recursive regex splitting grid
    for chunk_words in RECURSIVE_CHUNK_SIZES:
        for overlap in RECURSIVE_OVERLAPS:
            method_name = f"recursive_{chunk_words}_ov{overlap}"
            chunks = []
            for doc in doc_texts:
                cs = recursive_split_regex(doc["text"], chunk_words=chunk_words, overlap=overlap)
                for c in cs:
                    chunks.append({"doc": doc["filename"], "method": method_name, "text": c["text"]})
            experiments.append({"name": method_name, "chunks": chunks, "params": {"chunk_words":chunk_words, "overlap":overlap}})

    # 3) Semantic chunking
    method_name = f"semantic_sim{int(SEMANTIC_SIM_THRESH*100)}"
    chunks = []
    for doc in doc_texts:
        cs = semantic_chunking(doc["text"], sim_threshold=SEMANTIC_SIM_THRESH)
        for c in cs:
            chunks.append({"doc": doc["filename"], "method": method_name, "text": c["text"]})
    experiments.append({"name": method_name, "chunks": chunks, "params": {"sim_threshold":SEMANTIC_SIM_THRESH}})

    # For each experiment: build FAISS index and run retrieval for queries
    results = []  # accumulate per query top-k scores
    for exp in experiments:
        name = exp["name"]
        chunk_texts = [c["text"] for c in exp["chunks"]]
        if not chunk_texts:
            print(f"[WARN] experiment {name} has 0 chunks; skipping")
            continue
        print(f"[INFO] Building index for experiment {name} with {len(chunk_texts)} chunks ...")
        index, chunk_embs = build_faiss_index(chunk_texts)
        # map indices -> metadata
        meta = exp["chunks"]

        # embed queries
        q_embs = sbert.encode(queries, convert_to_numpy=True, show_progress_bar=False)
        # normalize
        q_norms = np.linalg.norm(q_embs, axis=1, keepdims=True)
        q_norms[q_norms==0]=1.0
        q_embs = q_embs / q_norms

        for qi, q in enumerate(queries):
            inds, scores = retrieve_topk(index, chunk_embs, q_embs[qi], k=min(TOP_K, len(chunk_texts)))
            # collect metadata for topk
            for rank, (idx, sc) in enumerate(zip(inds, scores), start=1):
                md = meta[idx]
                results.append({
                    "experiment": name,
                    "query": q,
                    "rank": rank,
                    "score": float(sc),
                    "chunk_text_snippet": md["text"][:300].replace("\n"," "),
                    "chunk_doc": md["doc"]
                })
        # export per-experiment chunks optionally
        with open(os.path.join(out_dir, f"chunks_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(exp["chunks"], f, ensure_ascii=False, indent=2)

    # write results CSV
    csv_path = os.path.join(out_dir, "retrieval_results.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=["experiment","query","rank","score","chunk_doc","chunk_text_snippet"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "experiment": r["experiment"],
                "query": r["query"],
                "rank": r["rank"],
                "score": r["score"],
                "chunk_doc": r["chunk_doc"],
                "chunk_text_snippet": r["chunk_text_snippet"]
            })
    # compute summary per experiment (avg top1, avg top3)
    summary = {}
    for r in results:
        name = r["experiment"]
        summary.setdefault(name, []).append(r["score"])
    rows = []
    for name, scores in summary.items():
        avg = float(np.mean(scores))
        rows.append({"experiment": name, "avg_score": avg, "n": len(scores)})
    rows = sorted(rows, key=lambda x: x["avg_score"], reverse=True)
    print("\nExperiment summary (avg retrieval score across returned items):")
    for row in rows:
        print(f"  {row['experiment']:<30}  avg_score={row['avg_score']:.4f}  items={row['n']}")
    # save summary
    with open(os.path.join(out_dir, "summary.json"), "w", encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Results written to {out_dir}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", default=os.path.join(os.getcwd(), "data", "business"), help="Directory containing docs to chunk (pdf/txt)")
    ap.add_argument("--queries_file", default=None, help="Text file with one query per line. If omitted, script will try to extract questions from uploaded paper or use example queries.")
    ap.add_argument("--out_dir", default="./out", help="Output directory")
    args = ap.parse_args()

    # load queries
    queries = []
    if args.queries_file and os.path.exists(args.queries_file):
        with open(args.queries_file, "r", encoding="utf-8", errors='ignore') as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        # try extracting from uploaded paper
        print("[INFO] No queries_file provided; attempting to extract question-like lines from /mnt/data/Legal_Chunking2.pdf ...")
        qcand = extract_question_like_lines_from_pdf("/mnt/data/Legal_Chunking2.pdf")
        if qcand:
            print(f"[INFO] Found {len(qcand)} candidate queries from paper; using them.")
            queries = qcand
        else:
            print("[WARN] No queries found in PDF. Using fallback demo queries. For proper evaluation, provide --queries_file with domain questions.")
            queries = [
                "What are the data subject rights about access and portability?",
                "Under what grounds is processing of personal data lawful?",
                "When can personal data be shared with third parties?",
            ]

    print(f"[INFO] Running experiment on docs_dir={args.docs_dir} with {len(queries)} queries ...")
    run_experiment(args.docs_dir, queries, args.out_dir)

if __name__ == "__main__":
    main()
