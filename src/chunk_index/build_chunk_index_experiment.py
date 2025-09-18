# mistral_hipporag_semantic_fallback_build_chunk_index.py
# Hybrid chunking pipeline:
# - Uses Mistral-7B-Instruct LLM for boundary detection (with refined prompt + few-shot)
# - Falls back to HippoRAG structural + Samia semantic refinement for large docs or missing LLM
# - Generates SBERT embeddings and FAISS index

import os
import re
import json
import traceback
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import faiss
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------
# Few-shot examples & SBERT
# -------------------
few_shot_examples = """
---
Example Document 1:
<<<BEGIN_DOCUMENT>>>
SECTION 1. Introduction. This policy outlines our commitment to data privacy. We collect personal data to provide our services and for internal analytics.
SECTION 2. Data Usage. Your data is primarily used for service improvement and personalization. We do not share your personal data with third parties without explicit consent, except as required by law.
<<<END_DOCUMENT>>>
JSON Output:
[
  {"id": 1, "start": 0, "end": 158, "title": "Introduction"},
  {"id": 2, "start": 159, "end": 348, "title": "Data Usage"}
]
---
Example Document 2:
<<<BEGIN_DOCUMENT>>>
1. Scope. This agreement applies to all users accessing our online platform. By using the platform, you agree to these terms.
2. User Responsibilities. Users are responsible for maintaining the confidentiality of their account information and for all activities that occur under their account.
3. Termination. We reserve the right to terminate access to our services at our sole discretion, with or without notice, for any violation of these terms.
<<<END_DOCUMENT>>>
JSON Output:
[
  {"id": 1, "start": 0, "end": 113, "title": "Scope"},
  {"id": 2, "start": 114, "end": 286, "title": "User Responsibilities"},
  {"id": 3, "start": 287, "end": 472, "title": "Termination"}
]
---
"""

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
sbert = SentenceTransformer(EMBED_MODEL)

# -------------------
# LLM config
# -------------------
LLM_CHAR_LIMIT = 40000
LLM_MAX_NEW_TOKENS = 2048
LLM_TEMPERATURE = 0.0
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# File utils
# -------------------
def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        texts = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n".join(texts)
    except Exception as e:
        print(f"[WARN] PDF parse failed for {path}: {e}")
        return ""


def read_text_file(path):
    for enc in ("utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# -------------------
# Fallback hybrid chunking
# -------------------
def structural_split(text):
    """HippoRAG-inspired structural segmentation: sections → clauses → sentences."""
    lines = text.splitlines()
    segments, curr = [], []
    for line in lines:
        if re.match(r'^\d+(\.\d+)*[\).]?\s+', line):
            if curr:
                segments.append(" ".join(curr).strip())
                curr = []
            segments.append(line.strip())
        else:
            curr.append(line.strip())
    if curr:
        segments.append(" ".join(curr).strip())
    return [s for s in segments if s]


def semantic_refine(chunks, threshold=0.85):
    """Samia-style merge small chunks if semantically similar."""
    if not chunks:
        return chunks
    refined = [chunks[0]]
    embeddings = sbert.encode([c["text"] for c in chunks], convert_to_tensor=True)
    for i in range(1, len(chunks)):
        sim = util.cos_sim(embeddings[i], embeddings[i-1]).item()
        if sim > threshold:
            refined[-1]["text"] += " " + chunks[i]["text"]
        else:
            refined.append(chunks[i])
    return refined


def adaptive_size(chunks, min_words=100, max_words=600):
    """Merge tiny chunks, split very large ones."""
    new_chunks = []
    for c in chunks:
        words = c["text"].split()
        if len(words) < min_words and new_chunks:
            new_chunks[-1]["text"] += " " + c["text"]
        elif len(words) > max_words:
            mid = len(words) // 2
            new_chunks.append({"text": " ".join(words[:mid])})
            new_chunks.append({"text": " ".join(words[mid:])})
        else:
            new_chunks.append(c)
    return new_chunks


def add_overlap(chunks, stride=50):
    """Add overlapping context to preserve continuity."""
    overlapped = []
    for i, c in enumerate(chunks):
        words = c["text"].split()
        if i > 0:
            prev_words = chunks[i-1]["text"].split()
            prefix = " ".join(prev_words[-stride:]) if len(prev_words) > stride else chunks[i-1]["text"]
            words = prefix.split() + words
        overlapped.append({"text": " ".join(words)})
    return overlapped


def hybrid_chunk(text):
    """Full fallback pipeline: structural → semantic refine → adaptive → overlap."""
    chunks = [{"text": s} for s in structural_split(text)]
    chunks = semantic_refine(chunks)
    chunks = adaptive_size(chunks)
    chunks = add_overlap(chunks, stride=50)
    return chunks

# -------------------
# LLM prompt
# -------------------
def build_llm_prompt_for_boundaries(text, doc_name="<document>"):
    parts = [
        "You are given the full text of a legal or business document.",
        "Your task is to identify each distinct policy, section, or clause such that each returned chunk contains exactly one policy (do NOT split a policy).",
        "",
        "Return a JSON array (and ONLY the JSON array) where each item is an object with the following keys:",
        "  - id: integer (1-based ordinal in reading order)",
        "  - start: integer (inclusive, 0-based character index into the provided text)",
        "  - end: integer (exclusive, 0-based character index into the provided text)",
        "  - title: short string title if present, or null",
        "",
        "Important constraints:",
        "- Use start/end indices into the exact text provided below.",
        "- Do NOT rewrite or modify the document text.",
        "- Try not to split a single policy across multiple items.",
        "- Return JSON only. No explanation.",
        "",
        "Here are examples:",
        f"{few_shot_examples}",
        "",
        f"Document name: {doc_name}",
        "<<<BEGIN_DOCUMENT>>>",
        text,
        "<<<END_DOCUMENT>>>",
    ]
    return "\n".join(parts)


def tolerant_json_extract(output_text):
    start = output_text.find('[')
    end = output_text.rfind(']')
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in model output.")
    return json.loads(output_text[start:end+1])


def call_transformers_boundaries(text, tokenizer, model, max_new_tokens=LLM_MAX_NEW_TOKENS, device=LLM_DEVICE):
    prompt = build_llm_prompt_for_boundaries(text)
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   tokenize=True, return_dict=True, return_tensors="pt")
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
    except Exception:
        inputs = tokenizer(prompt, return_tensors="pt")

    device_obj = torch.device(device)
    model.to(device_obj)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device_obj)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=LLM_TEMPERATURE)

    if "input_ids" in inputs:
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    else:
        generated_tokens = outputs[0]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    parsed = tolerant_json_extract(generated_text)
    normalized = []
    for i, item in enumerate(parsed):
        normalized.append({
            "id": int(item.get("id", i+1)),
            "start": int(item["start"]),
            "end": int(item["end"]),
            "title": item.get("title") if item.get("title") is not None else None
        })
    return normalized


def llm_chunk_document_with_transformers(text, doc_name, tokenizer=None, model=None, char_limit=LLM_CHAR_LIMIT):
    if not tokenizer or not model:
        raise RuntimeError("tokenizer and model must be provided.")
    if len(text) > char_limit:
        return None
    try:
        boundaries = call_transformers_boundaries(text, tokenizer, model)
    except Exception as e:
        print(f"[WARN] LLM chunking failed: {e}")
        return None

    boundaries = sorted(boundaries, key=lambda x: x["start"])
    valid = []
    last_end = 0
    for b in boundaries:
        if b["start"] < 0 or b["end"] <= b["start"] or b["end"] > len(text):
            continue
        if b["start"] < last_end:
            b["start"] = last_end
        valid.append(b)
        last_end = b["end"]
    return valid

# -------------------
# Main indexing flow
# -------------------
def build_index_from_dir(business_dir,
                         out_index="faiss.index",
                         metadata_out="chunks.json",
                         max_words=500,
                         use_llm=True,
                         llm_tokenizer=None,
                         llm_model=None,
                         llm_char_limit=LLM_CHAR_LIMIT):
    docs = []
    for fname in sorted(os.listdir(business_dir)):
        full = os.path.join(business_dir, fname)
        if os.path.isdir(full):
            continue
        lower = fname.lower()
        text = ""
        if lower.endswith('.pdf'):
            text = extract_text_from_pdf(full)
        elif lower.endswith(('.txt', '.md', '.text')):
            text = read_text_file(full)
        else:
            continue
        if not text.strip():
            continue
        docs.append({"filename": fname, "text": text})

    chunks = []
    for doc_id, doc in enumerate(docs):
        text, doc_name = doc["text"], doc["filename"]
        boundaries = None
        if use_llm and llm_tokenizer and llm_model:
            boundaries = llm_chunk_document_with_transformers(text, doc_name, llm_tokenizer, llm_model, llm_char_limit)

        if boundaries:
            for b in boundaries:
                chunk_text = text[b["start"]:b["end"]].strip()
                if chunk_text:
                    chunks.append({"doc_id": doc_id, "filename": doc_name, "text": chunk_text, "meta": {"title": b.get("title")}})
        else:
            fallback_chunks = hybrid_chunk(text)
            for c in fallback_chunks:
                chunks.append({"doc_id": doc_id, "filename": doc_name, "text": c["text"], "meta": {}})

    texts = [c["text"] for c in chunks]
    if not texts:
        raise RuntimeError("No chunks to index.")

    print(f"[INFO] Encoding {len(texts)} chunks with SBERT {EMBED_MODEL}...")
    embeddings = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, out_index)

    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(chunks)} chunks and index to {out_index}")

# -------------------
# Model loader with fallback
# -------------------
def load_transformers_model_with_fallback(model_id, local_folder=None, device_map="auto"):
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if local_folder and os.path.isdir(local_folder):
        try:
            tok = AutoTokenizer.from_pretrained(local_folder, use_fast=True)
            m = AutoModelForCausalLM.from_pretrained(local_folder, device_map=device_map)
            return tok, m
        except Exception:
            pass
    if hf_token:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, use_auth_token=hf_token)
            m = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, use_auth_token=hf_token)
            return tok, m
        except Exception:
            pass
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        m = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)
        return tok, m
    except Exception:
        return None, None

# -------------------
# Main execution
# -------------------
if __name__ == "__main__":
    business_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "business"))
    print("Reading business files from:", business_dir)

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    local_model_folder = None

    tokenizer, hf_model = load_transformers_model_with_fallback(model_id, local_model_folder)
    if tokenizer is None or hf_model is None:
        print("[WARN] LLM unavailable — using fallback only.")
        llm_tokenizer = None
        llm_model = None
    else:
        llm_tokenizer = tokenizer
        llm_model = hf_model
        print("[INFO] Loaded Mistral LLM successfully.")

    build_index_from_dir(business_dir,
                         out_index="faiss.index",
                         metadata_out="chunks.json",
                         max_words=500,
                         use_llm=True,
                         llm_tokenizer=llm_tokenizer,
                         llm_model=llm_model,
                         llm_char_limit=LLM_CHAR_LIMIT)
