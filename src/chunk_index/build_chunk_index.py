# build_chunk_index_mistral.py
# Use a local transformers-based Mistral model (mistralai/Mistral-7B-Instruct-v0.3)
# to detect policy boundaries (start/end char offsets). Falls back to structural
# segmentation if the LLM isn't available.

import os
import re
import json
import traceback
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM


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

# LLM config
LLM_CHAR_LIMIT = 40000        # only call LLM for documents <= this many chars (tune)
LLM_MAX_NEW_TOKENS = 2048
LLM_TEMPERATURE = 0.0
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    for enc in ("utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def segment_text(text):
    """Simple paragraph/bullet/heading segmentation fallback."""
    lines = text.splitlines()
    segments = []
    curr = []

    bullet_re = re.compile(r'^\s*(?:[-\u2022*•]|\d+\.|\([a-zA-Z0-9]+\))\s+')

    def flush_curr():
        nonlocal curr
        if curr:
            seg = "\n".join(curr).strip()
            if seg:
                segments.append(seg)
            curr = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_curr()
            i += 1
            continue

        if bullet_re.match(line):
            flush_curr()
            segments.append(stripped)
            i += 1
            while i < len(lines) and bullet_re.match(lines[i]):
                if lines[i].strip():
                    segments.append(lines[i].strip())
                i += 1
            continue

        if len(stripped) <= 120 and (stripped.isupper() or stripped.endswith(':')):
            flush_curr()
            segments.append(stripped)
            i += 1
            continue

        curr.append(stripped)
        i += 1

    flush_curr()
    return segments


# -------------------
# Prompt / JSON parse
# -------------------
def build_llm_prompt_for_boundaries(text, doc_name="<document>"):
    parts = [
        "You are given the full text of a legal document. Identify each distinct policy/section/clause",
        "such that each returned chunk contains exactly one policy (do NOT split a policy).",
        "Return a JSON array (and ONLY the JSON array) where each item is an object with:",
        "  - id: integer (1-based ordinal in reading order)",
        "  - start: integer (inclusive, 0-based char index into the provided text)",
        "  - end: integer (exclusive char index into the provided text)",
        "  - title: short string title if present, or null",
        "",
        "Important constraints:",
        "- Use start/end indices into the exact text provided below (we will slice the original text).",
        "- Do NOT rewrite or modify the document text; return indices only.",
        "- Try not to split a single policy across multiple items. If two paragraphs form one policy, return a single item covering both.",
        "- Return JSON only. Examples: f{few_shot_examples}",
        "Do NOT return any explanation or text other than the JSON array.",
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
    json_text = output_text[start:end+1]
    return json.loads(json_text)


# -------------------
# Transformers call
# -------------------
def call_transformers_boundaries(text, tokenizer, model, max_new_tokens=LLM_MAX_NEW_TOKENS, device=LLM_DEVICE):
    prompt = build_llm_prompt_for_boundaries(text)

    # prefer apply_chat_template if the tokenizer provides it (some Mistral tokenizers may not)
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
    except Exception:
        inputs = tokenizer(prompt, return_tensors="pt")

    device_obj = torch.device(device)
    model.to(device_obj)
    # move inputs to device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device_obj)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=LLM_TEMPERATURE)

    # decode only generated portion
    if "input_ids" in inputs:
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    else:
        generated_tokens = outputs[0]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # strip prompt echo
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    parsed = tolerant_json_extract(generated_text)
    normalized = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid item {i} from model: expected object.")
        if "start" not in item or "end" not in item:
            raise RuntimeError(f"Invalid item {i}: missing start/end.")
        normalized.append({
            "id": int(item.get("id", i+1)),
            "start": int(item["start"]),
            "end": int(item["end"]),
            "title": item.get("title") if item.get("title") is not None else None
        })
    return normalized


def llm_chunk_document_with_transformers(text, doc_name, tokenizer=None, model=None, char_limit=LLM_CHAR_LIMIT):
    if not tokenizer or not model:
        raise RuntimeError("tokenizer and model instances must be provided for transformers LLM chunking.")

    if len(text) > char_limit:
        print(f"[INFO] Document length {len(text)} > {char_limit} chars: skipping LLM chunking.")
        return None

    print(f"[INFO] Calling local transformers model for boundaries for '{doc_name}' ({len(text)} chars)...")
    try:
        boundaries = call_transformers_boundaries(text, tokenizer, model)
    except Exception as e:
        print(f"[WARN] transformers LLM failed to return boundaries: {e}")
        return None

    boundaries = sorted(boundaries, key=lambda x: x["start"])
    valid = []
    last_end = 0
    for b in boundaries:
        if b["start"] < 0 or b["end"] <= b["start"] or b["end"] > len(text):
            print(f"[WARN] invalid LLM boundary {b} - skipping")
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
                         max_words=200,
                         use_llm=True,
                         llm_tokenizer=None,
                         llm_model=None,
                         llm_char_limit=LLM_CHAR_LIMIT):
    docs = []
    filenames = sorted(os.listdir(business_dir))
    for fname in filenames:
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
            try:
                text = read_text_file(full)
            except Exception:
                print(f"[SKIP] Unsupported or binary file: {full}")
                continue

        if not text.strip():
            print(f"[INFO] No text extracted from {fname}, skipping.")
            continue
        docs.append({"filename": fname, "text": text})

    chunks = []
    for doc_id, doc in enumerate(docs):
        text = doc["text"]
        doc_name = doc["filename"]

        boundaries = None
        if use_llm and llm_tokenizer and llm_model:
            try:
                boundaries = llm_chunk_document_with_transformers(
                    text, doc_name, tokenizer=llm_tokenizer, model=llm_model, char_limit=llm_char_limit
                )
            except Exception as e:
                print(f"[WARN] LLM chunking failed for {doc_name}: {e}")
                boundaries = None

        if boundaries:
            for b in boundaries:
                chunk_text = text[b["start"]:b["end"]].strip()
                if chunk_text:
                    chunks.append({
                        "doc_id": doc_id,
                        "filename": doc_name,
                        "text": chunk_text,
                        "meta": {"title": b.get("title")}
                    })
        else:
            segments = segment_text(text)
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                if max_words and len(seg.split()) > max_words:
                    subs = re.split(r'(?<=[\.!?;:\n])\s+', seg)
                    for s in subs:
                        s = s.strip()
                        if s:
                            chunks.append({
                                "doc_id": doc_id,
                                "filename": doc_name,
                                "text": s,
                                "meta": {}
                            })
                else:
                    chunks.append({
                        "doc_id": doc_id,
                        "filename": doc_name,
                        "text": seg,
                        "meta": {}
                    })

    texts = [c["text"] for c in chunks]
    if not texts:
        raise RuntimeError("No chunks to index. Check input files and text extraction.")

    print(f"[INFO] Encoding {len(texts)} chunks with SBERT model {EMBED_MODEL} ...")
    embeddings = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, out_index)

    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunks)} chunks and index to {out_index}")


# -------------------
# Helpers to load model robustly
# -------------------
def load_transformers_model_with_fallback(model_id, local_folder=None, device_map="auto"):
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # 1) try local folder
    if local_folder and os.path.isdir(local_folder):
        try:
            print(f"[INFO] Loading tokenizer/model from local folder: {local_folder}")
            tok = AutoTokenizer.from_pretrained(local_folder, use_fast=True)
            m = AutoModelForCausalLM.from_pretrained(local_folder, device_map=device_map)
            return tok, m
        except Exception as e:
            print(f"[WARN] failed to load from local folder {local_folder}: {e}")
            print(traceback.format_exc())

    # 2) try HF with token
    if hf_token:
        try:
            print("[INFO] Attempting to load model from Hugging Face using HUGGINGFACE_HUB_TOKEN...")
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, use_auth_token=hf_token)
            m = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, use_auth_token=hf_token)
            return tok, m
        except Exception as e:
            print(f"[WARN] Failed to load '{model_id}' with HUGGINGFACE_HUB_TOKEN: {e}")
            print(traceback.format_exc())

    # 3) try HF without token
    try:
        print(f"[INFO] Attempting to load model '{model_id}' from Hugging Face (no token)...") 
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        m = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)
        return tok, m
    except Exception as e:
        print(f"[ERROR] Could not load model '{model_id}': {e}")
        print(traceback.format_exc())
        print("\nPossible fixes:")
        print("  1) If the model is gated you must authenticate or provide a local copy.")
        print("     - Install huggingface_hub and run: huggingface-cli login")
        print("     - Or set env var HUGGINGFACE_HUB_TOKEN to a valid token.")
        print("  2) Or download the model manually and set local_folder to its path.")
        return None, None


# -------------------
# __main__ (load Mistral and run)
# -------------------
if __name__ == "__main__":
    business_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "business")
    business_dir = os.path.normpath(business_dir)
    print("Reading business files from:", business_dir)

    # set Mistral model id or point local_model_folder to a downloaded copy
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    local_model_folder = None  # e.g. r\"C:\\models\\mistral-7b-instruct\"

    print(f"[INFO] Loading tokenizer and model {model_id} (device={LLM_DEVICE}) ...")
    tokenizer, hf_model = load_transformers_model_with_fallback(model_id, local_folder=local_model_folder, device_map="auto")

    if tokenizer is None or hf_model is None:
        print("[WARN] transformer LLM unavailable — continuing without LLM. Structural segmentation will be used.")
        llm_tokenizer = None
        llm_model = None
    else:
        llm_tokenizer = tokenizer
        llm_model = hf_model
        print("[INFO] Loaded tokenizer and model successfully.")

    build_index_from_dir(business_dir,
                         out_index="faiss.index",
                         metadata_out="chunks.json",
                         max_words=500,
                         use_llm=True,
                         llm_tokenizer=llm_tokenizer,
                         llm_model=llm_model,
                         llm_char_limit=LLM_CHAR_LIMIT)
