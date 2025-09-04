# eventic_extractor.py
import os
import time
import json
import re
from typing import List, Tuple
import networkx as nx
import pickle

# --- optional: sentence-transformers included earlier in file, kept for compatibility ---
from sentence_transformers import SentenceTransformer

# --- OpenAI client (new v1) setup (used if available) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
_client = None
try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
        print("[INFO] OpenAI client initialized (will attempt to use it first).")
    else:
        print("[INFO] OPENAI_API_KEY not set — skipping OpenAI client.")
except Exception as e:
    print("[WARN] Could not initialize OpenAI client:", e)
    _client = None

# ---------------------------
# Prompt templates (unchanged)
# ---------------------------
Tempt1 = """Please read the following regulatory text (a short paragraph). Return a JSON array of all agents/subjects (organizations, companies, government departments, individuals) that are explicitly targeted by deontic expressions like "must", "shall", "is required to", "is obliged to", "should", "is prohibited from". Output only a JSON array of strings.

Paragraph:
----
{doc}
----
"""

Tempt2 = """Given the paragraph below and an agent name, identify whether there is a deontic word applying to that agent in the paragraph (words like must/shall/should/forbidden/required). If yes, return the moral/deontic word and the action/state phrase that follows it (as short text). Return JSON like {{ "deontic": "...", "action": "..." }}. If not, return null.

Paragraph:
----
{para}
----
Agent: {agent}
"""

# ---------------------------
# call_openai wrapper (returns None on failure)
# ---------------------------
def call_openai(prompt, model="gpt-3.5-turbo", temperature=0.0, max_tokens=512):
    """
    Tries to call OpenAI using the v1 OpenAI client. Returns text or None on failure.
    """
    if _client is None:
        return None
    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # v1 response shape: resp.choices[0].message.content
        return resp.choices[0].message.content
    except Exception as e:
        # Log and return None, so caller will fallback to local model
        print(f"[WARN] OpenAI call failed: {e}")
        return None

# ---------------------------
# Local Flan-T5 fallback
# ---------------------------
_local_model = None
_local_tokenizer = None

def init_local_model(model_name="google/flan-t5-small"):
    global _local_model, _local_tokenizer
    if _local_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print(f"[INFO] Loading local model {model_name} (this may take time)...")
            _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            # keep model on CPU (default). If you have GPU and want it, move with .to("cuda")
            print("[INFO] Local model loaded.")
        except Exception as e:
            print("[ERROR] Could not load local model:", e)
            _local_model = None
            _local_tokenizer = None
    return _local_model, _local_tokenizer

def call_local_model(prompt, max_input_length=1024, max_new_tokens=256):
    """
    Calls local Flan-T5. Truncates prompt to tokenizer size. Returns string or None on failure.
    """
    model, tokenizer = init_local_model()
    if model is None or tokenizer is None:
        return None
    try:
        # tokenizer will truncate if we set max_length
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print("[WARN] Local model generation failed:", e)
        return None

# ---------------------------
# Heuristic fallback (very simple)
# ---------------------------
DEONTIC_WORDS_RE = r"\b(must|shall|should|required to|is required to|is obliged to|is prohibited from|prohibited|forbidden|may not)\b"

def heuristic_extract_agents_and_triples(document_text: str):
    agent_candidates = set(re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", document_text))
    stopwords = {"The", "This", "If", "When", "Where", "In", "On"}
    agents = [a for a in agent_candidates if a.split()[0] not in stopwords][:50]

    triples = []
    sentences = re.split(r'(?<=[.!?])\s+', document_text)
    for sent in sentences:
        m = re.search(DEONTIC_WORDS_RE, sent, flags=re.IGNORECASE)
        if m:
            deon = m.group(0)
            action = sent[m.end():].strip()
            matched_agent = None
            for ag in agents:
                if re.search(r"\b" + re.escape(ag) + r"\b", sent):
                    matched_agent = ag
                    break
            if not matched_agent:
                matched_agent = agents[0] if agents else "Unknown"
            triples.append((matched_agent, deon, action))
    return triples

# ---------------------------
# unified call that tries OpenAI -> local -> heuristic
# ---------------------------
def call_any_model(prompt, prefer_local_if_no_openai=False):
    """
    Return text reply or None.
    Preference:
     - If OpenAI client available and prefer_local_if_no_openai=False: try OpenAI first, then local.
     - Otherwise try local.
    """
    # Try OpenAI if available and not explicitly preferring local
    if _client is not None and not prefer_local_if_no_openai:
        out = call_openai(prompt)
        if out is not None:
            return out
        # else fall through to local

    # Try local model
    out_local = call_local_model(prompt)
    if out_local is not None:
        return out_local

    # As last resort, return None to indicate fallback needed
    return None

# ---------------------------
# Text extraction utilities (PDF/TXT). Copy of earlier robust helpers.
# ---------------------------
def extract_text_from_pdf(path: str) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        print("[WARN] PyPDF2 not installed:", e)
        return ""
    try:
        reader = PdfReader(path)
        pages_text = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages_text.append(txt)
        return "\n\n".join(pages_text)
    except Exception as e:
        print(f"[WARN] PyPDF2 failed for {path}: {e}")
        return ""

def read_text_file(path: str) -> str:
    for enc in ("utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ---------------------------
# Main extraction logic (uses call_any_model + fallback heuristics)
# ---------------------------
def _chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def extract_eventic_triples(document_text: str,
                                 deontic_regex=DEONTIC_WORDS_RE,
                                 batch_para_count=8,
                                 max_local_tokens_for_agents=512,
                                 max_local_tokens_for_tempt2=128) -> List[Tuple[str,str,str]]:
    """
    Faster extraction:
      - only process paragraphs that contain deontic words
      - batch paragraphs for agent extraction via Tempt1
      - use regex to get deontic+action; call Tempt2 only if regex fails for a paragraph-agent
    """
    # split document into paragraphs (preserve short paras)
    paras = [p.strip() for p in re.split(r"\n{2,}", document_text) if p.strip()]
    if not paras:
        paras = [document_text]

    # 1) prefilter paragraphs that contain deontic words
    deontic_paras_idx = [i for i,p in enumerate(paras) if re.search(deontic_regex, p, flags=re.IGNORECASE)]
    if not deontic_paras_idx:
        # no likely regulatory language found -> fallback to whole-document heuristic
        print("[INFO] No deontic paragraphs detected (regex). Falling back to heuristic extractor.")
        return heuristic_extract_agents_and_triples(document_text)

    deontic_paras = [paras[i] for i in deontic_paras_idx]

    # 2) batch agent extraction: group paragraphs into batches and call Tempt1 once per batch
    agents = set()
    # limit paragraphs per batch so prompt length stays reasonable
    for chunk in _chunk_list(deontic_paras, batch_para_count):
        joined = "\n\n".join(chunk)
        prompt_agents = Tempt1.format(doc=joined)
        # prefer local with small token usage if OpenAI absent; call_any_model uses your fallback logic
        resp = call_any_model(prompt_agents)
        if resp is None:
            # model unavailable -> use heuristic fallback to extract agents from the chunk
            # naive agent extraction from chunk:
            found = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", joined)
            for f in found:
                if len(f) > 1:
                    agents.add(f.strip())
            continue
        # parse JSON or fall back to splitting
        try:
            parsed = json.loads(resp)
            if isinstance(parsed, list):
                for x in parsed:
                    if isinstance(x, str) and x.strip():
                        agents.add(x.strip())
        except Exception:
            # tolerant parse (commas/newlines)
            for a in re.split(r"[,;\n]+", resp):
                a = a.strip().strip('[] "')
                if a:
                    agents.add(a)

    agents = list(agents)
    if not agents:
        # fallback: use heuristic agent detection across whole doc
        print("[INFO] No agents found by LLM batches; using heuristic agent extraction.")
        agents = list(set(re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", document_text)))[:50]

    # 3) For each deontic paragraph, try quick regex extraction of deontic+action
    triples = []
    # regex to capture deontic word + following clause up to punctuation (200 chars)
    action_capture_re = re.compile(r"(?P<deon>\b(?:must|shall|should|required to|is required to|is obliged to|is prohibited from|prohibited|forbidden|may not)\b)\s+(?P<action>[^.\n]{1,250})", flags=re.IGNORECASE)
    for idx, p in zip(deontic_paras_idx, deontic_paras):
        # try to match agents mentioned in this paragraph first
        local_agents = [a for a in agents if re.search(r"\b" + re.escape(a) + r"\b", p, flags=re.IGNORECASE)]
        if not local_agents:
            # no listed agent in paragraph — try to find proper-noun phrases as local agents
            local_agents = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", p)
            local_agents = [a for a in local_agents if a.lower() not in {'the','this','that','when','if'}][:3]

        # find all deontic+action occurrences in paragraph using regex
        for m in action_capture_re.finditer(p):
            deon = m.group("deon").strip()
            action = m.group("action").strip()
            # choose agent: first local_agent that appears in paragraph, else fallback to first in agents list
            chosen_agent = None
            for a in local_agents:
                if re.search(r"\b" + re.escape(a) + r"\b", p, flags=re.IGNORECASE):
                    chosen_agent = a
                    break
            if not chosen_agent:
                chosen_agent = local_agents[0] if local_agents else (agents[0] if agents else "Unknown")
            triples.append((chosen_agent, deon, action))

        # If regex didn't find any action but paragraph contains an agent token, call Tempt2 for that agent only
        if not any(re.search(action_capture_re, p) for _ in [0]) and local_agents:
            # limit calls: only call for one agent per paragraph (first local_agent)
            agent = local_agents[0]
            prompt2 = Tempt2.format(para=p, agent=agent)
            resp2 = call_any_model(prompt2)
            if resp2:
                try:
                    j = json.loads(resp2)
                except Exception:
                    j = None
                if j and isinstance(j, dict) and j.get("deontic"):
                    triples.append((agent, j.get("deontic"), j.get("action")))
                else:
                    # last resort: try to find deontic locally and create action snippet
                    m = action_capture_re.search(p)
                    if m:
                        triples.append((agent, m.group("deon").strip(), m.group("action").strip()))

    # optionally deduplicate triples with simple normalization
    seen = set()
    out = []
    for ag,de,ac in triples:
        key = (ag.strip().lower(), de.strip().lower(), ac.strip().lower())
        if key not in seen:
            seen.add(key)
            out.append((ag.strip(), de.strip(), ac.strip()))
    return out

def build_eventic_graph(triples):
    G = nx.DiGraph()
    for agent, deontic, action in triples:
        node_action = f"ACTION::{action}"
        node_deon = f"DEONTIC::{deontic}"
        G.add_node(agent, type="agent")
        G.add_node(node_deon, type="deontic")
        G.add_node(node_action, type="action")
        G.add_edge(agent, node_deon, rel="has_deontic")
        G.add_edge(node_deon, node_action, rel="regulates")
    return G

# ---------------------------
# Script entry: read all files in data/
# ---------------------------
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    data_dir = os.path.normpath(data_dir)
    if not os.path.isdir(data_dir):
        raise RuntimeError("data/ directory not found. Put your PDFs/TXT files in the data/ folder.")

    all_texts = []
    for fname in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, fname)
        if os.path.isdir(full):
            continue
        lower = fname.lower()
        text = ""
        if lower.endswith(".pdf"):
            print(f"[INFO] Extracting PDF text from {fname} using PyPDF2...")
            text = extract_text_from_pdf(full)
            if not text.strip():
                print(f"[INFO] No selectable text in {fname}. (scanned image?)")
                # We don't automatically OCR here to avoid heavy installs — you can enable OCR if needed.
        elif lower.endswith((".txt", ".md", ".text")):
            print(f"[INFO] Reading text file {fname}...")
            text = read_text_file(full)
        else:
            # attempt to read as text defensively
            try:
                text = read_text_file(full)
            except Exception:
                print(f"[SKIP] Unsupported file type or unreadable: {fname}")
                continue

        if not text.strip():
            print(f"[WARN] No text extracted from {fname} (empty); skipping.")
            continue
        all_texts.append((fname, text))

    if not all_texts:
        raise RuntimeError("No readable documents found in data/")

    for fname, doc_text in all_texts:
        print(f"\n=== Processing document: {fname} ===")
        triples = extract_eventic_triples(doc_text)
        G = build_eventic_graph(triples)
        print("Triples:", triples)
        print("Graph nodes:", G.number_of_nodes())
        out_path = f"{os.path.splitext(fname)[0]}_eventic_graph.gpickle"
        with open(out_path, "wb") as f:
            pickle.dump(G, f)
