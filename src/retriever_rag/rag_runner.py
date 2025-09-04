#!/usr/bin/env python3
"""
rag_runner.py

Runs RAG-style fusion + LLM verdict for each chunk.

Usage:
  python src/retriever_rag/rag_runner.py \
        --static-graph data/static_graph.gpickle \
        --eventic data/eventic_graph.gpickle \
        --chunks data/chunks.json \
        --faiss data/faiss.index \
        --out results.json
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any
import numpy as np
import networkx as nx
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_runner")

# Try import faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# SentenceTransformer for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# OpenAI / local LLM fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client is available for LLM calls.")
    except Exception as e:
        logger.warning("Failed to init OpenAI client: %s", e)
        _client = None
else:
    logger.info("No OPENAI_API_KEY found — will try local model fallback.")

# Local Flan-T5 lazy loader
_local_model = None
_local_tokenizer = None
def init_local_flan(model_name="google/flan-t5-small"):
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        logger.info("Loading local model %s (this may take a while)...", model_name)
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logger.info("Local model loaded.")
    except Exception as e:
        logger.warning("Local model init failed: %s", e)
        _local_model = None
        _local_tokenizer = None
    return _local_model, _local_tokenizer

def call_llm(prompt: str, use_openai_priority: bool = True, model_openai: str = "gpt-3.5-turbo", max_tokens: int = 256) -> str:
    """
    Tries OpenAI (if available), else local Flan-T5. Returns text or raises.
    """
    if use_openai_priority and _client is not None:
        try:
            resp = _client.chat.completions.create(
                model=model_openai,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("OpenAI call failed: %s", e)

    # fallback to local model
    model, tokenizer = init_local_flan()
    if model is None or tokenizer is None:
        raise RuntimeError("No LLM available (OpenAI failed and no local model).")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text

# Prompt template (Tempt3)
Tempt3 = """You are a compliance assistant. Given the following business text chunk and a small structured knowledge fragment (triples/statements), determine whether the business text *violates* any of the regulatory rules shown.

Business chunk:
----
{chunk}
----

Structured knowledge (triples or short statements):
----
{triples_text}
----

Instructions:
- If you find no conflict, return exactly:
<Compliance Check Passed>

- If you find a conflict, return exactly:
<Compliance Check Failed>
Then provide a short JSON array of conflicting evidence entries, each with keys: "fragment" (which triple or statement conflicts), "reason" (one-sentence explanation).
Example:
<Compliance Check Failed>
[{{"fragment":"DataController shall ...","reason":"..."}}, ...]
Be concise and cite the conflicting fragments.

Only output either <Compliance Check Passed> or <Compliance Check Failed> followed by the JSON array (if failed).
"""

def load_graph(path: str) -> nx.DiGraph:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    logger.info("Loaded graph from %s (nodes=%d, edges=%d)", path, G.number_of_nodes(), G.number_of_edges())
    return G

def load_node_embeddings(npz_path: str) -> (List[str], np.ndarray):
    if not os.path.exists(npz_path):
        logger.warning("Node embeddings file not found: %s", npz_path)
        return [], None
    arr = np.load(npz_path, allow_pickle=True)
    node_order = list(arr["node_order"])
    embeddings = arr["embeddings"]
    # ensure normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return node_order, embeddings

def compute_eventic_node_embeddings(G_eventic: nx.Graph, model: SentenceTransformer, cache_path: str = None):
    nodes = list(G_eventic.nodes())
    texts = []
    for n in nodes:
        attrs = G_eventic.nodes[n]
        label = attrs.get("label") or attrs.get("name") or n
        texts.append(str(label))
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return nodes, embs

def build_fused_subgraph(chunk_text: str, chunk_vec: np.ndarray, eventic_nodes: List[str], eventic_embs: np.ndarray,
                         eventic_graph: nx.Graph, static_graph: nx.Graph, lambda_thresh: float = 0.75, hop_k: int = 1):
    """
    1. Get eventic nodes whose cosine with chunk >= lambda_thresh
    2. P = intersection of those hits with static_graph nodes (exact string match)
    3. N = neighbors of P up to hop_k in static_graph
    4. Gfus = union subgraph of eventic_graph + static_graph restricted to nodes in hits ∪ P ∪ N
    """
    # dot product with pre-normalized vectors: chunk_vec shape (d,), eventic_embs shape (N,d)
    sims = floatable = None
    sims = np.dot(eventic_embs, chunk_vec.reshape(-1))
    hit_idx = np.where(sims >= lambda_thresh)[0]
    hits = [eventic_nodes[i] for i in hit_idx]
    P = [h for h in hits if h in static_graph.nodes]
    N = set()
    for p in P:
        # neighbors (in + out)
        for nbr in static_graph.predecessors(p) if hasattr(static_graph, "predecessors") else []:
            N.add(nbr)
        for nbr in static_graph.successors(p) if hasattr(static_graph, "successors") else []:
            N.add(nbr)
        # undirected neighbors for safety
        for nbr in static_graph.neighbors(p):
            N.add(nbr)
    # If hop_k > 1 expand (simple BFS)
    if hop_k > 1 and N:
        frontier = set(N)
        for _ in range(hop_k - 1):
            newf = set()
            for n in frontier:
                for nb in static_graph.neighbors(n):
                    if nb not in N:
                        newf.add(nb)
            if not newf:
                break
            N.update(newf)
            frontier = newf

    fused_nodes = set(hits) | set(P) | set(N)
    # build new graph
    Gfus = nx.DiGraph()
    # add nodes and node attributes (from either graph if available)
    for n in fused_nodes:
        if eventic_graph.has_node(n):
            Gfus.add_node(n, **dict(eventic_graph.nodes[n]))
        elif static_graph.has_node(n):
            Gfus.add_node(n, **dict(static_graph.nodes[n]))
        else:
            Gfus.add_node(n, label=n)

    # add edges from eventic_graph among fused nodes
    for u, v, data in eventic_graph.edges(data=True):
        if u in fused_nodes and v in fused_nodes:
            Gfus.add_edge(u, v, **data)

    # add edges from static_graph among fused nodes
    for u, v, data in static_graph.edges(data=True):
        if u in fused_nodes and v in fused_nodes:
            # keep predicate attribute if present
            Gfus.add_edge(u, v, **data)

    return Gfus, hits, P, list(N)


def triples_text_from_graph(G: nx.Graph, max_items: int = 50) -> str:
    """
    Convert graph edges into short triple statements for prompt.
    """
    lines = []
    count = 0
    for u, v, data in G.edges(data=True):
        pred = data.get("predicate") or data.get("rel") or data.get("label") or "relatedTo"
        u_label = G.nodes[u].get("label") or u
        v_label = G.nodes[v].get("label") or v
        lines.append(f"- {u_label} {pred} {v_label}")
        count += 1
        if count >= max_items:
            break
    if not lines:
        # fallback to listing nodes
        for n, data in G.nodes(data=True):
            lines.append(f"- {data.get('label') or n}")
    return "\n".join(lines)


def main(args):
    if not SBERT_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")

    # load static graph
    static_graph = load_graph(args.static_graph) if args.static_graph else nx.DiGraph()
    static_node_order, static_embs = ([], None)
    npz_path = os.path.join(os.path.dirname(args.static_graph), "node_embeddings.npz") if args.static_graph else None
    if npz_path and os.path.exists(npz_path):
        static_node_order, static_embs = load_node_embeddings(npz_path)

    # load eventic graph
    if not args.eventic_graph or not os.path.exists(args.eventic_graph):
        raise FileNotFoundError("Eventic graph not provided or not found. Generate it using your eventic_extractor and save with nx.write_gpickle(...).")
    eventic_graph = load_graph(args.eventic_graph)

    # load chunks metadata
    if not os.path.exists(args.chunks):
        raise FileNotFoundError(f"chunks.json not found: {args.chunks}")
    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks", len(chunks))

    # load faiss index if provided (optional)
    index = None
    if args.faiss:
        if not FAISS_AVAILABLE:
            logger.warning("faiss not available; skipping index loading. Install faiss-cpu.")
        else:
            if not os.path.exists(args.faiss):
                logger.warning("faiss index file not found: %s", args.faiss)
            else:
                index = faiss.read_index(args.faiss)
                logger.info("Faiss index loaded from %s", args.faiss)

    # sentence-transformer model
    model = SentenceTransformer(args.embed_model)

    # precompute eventic node embeddings
    eventic_nodes, eventic_embs = compute_eventic_node_embeddings(eventic_graph, model)
    logger.info("Computed eventic embeddings: %d nodes", len(eventic_nodes))

    # if user provided a saved eventic node embeddings file, could use that here (not implemented)
    # ensure eventic_embs normalized (already normalized in function)
    results = []

    # iterate chunks
    for idx, c in enumerate(chunks):
        fname = c.get("filename", f"doc_{c.get('doc_id', 0)}")
        text = c.get("text", "")
        if not text.strip():
            continue
        logger.info("Processing chunk %d/%d (doc: %s) ...", idx + 1, len(chunks), fname)
        # embed chunk
        cvec = model.encode([text], convert_to_numpy=True)[0]
        cvec = cvec / (np.linalg.norm(cvec) + 1e-12)

        # build fused graph
        Gfus, hits, P, N = build_fused_subgraph(text, cvec, eventic_nodes, eventic_embs, eventic_graph, static_graph,
                                                lambda_thresh=args.lambda_thresh, hop_k=args.hop_k)
        triples_text = triples_text_from_graph(Gfus, max_items=args.max_triples)
        prompt = Tempt3.format(chunk=text[:2000] + ("\n\n[TRUNCATED]" if len(text) > 2000 else ""), triples_text=triples_text)
        try:
            reply = call_llm(prompt, use_openai_priority=not args.prefer_local, model_openai=args.openai_model, max_tokens=256)
        except Exception as e:
            logger.warning("LLM call failed for chunk %d: %s", idx, e)
            reply = None

        verdict = None
        evidence = None
        if reply:
            # simplify parsing: look for the exact tags
            if "<Compliance Check Passed>" in reply:
                verdict = "pass"
                evidence = []
            elif "<Compliance Check Failed>" in reply:
                verdict = "fail"
                # attempt to parse JSON after the tag
                try:
                    suffix = reply.split("<Compliance Check Failed>")[-1].strip()
                    parsed = json.loads(suffix)
                    evidence = parsed
                except Exception:
                    # fallback: put the textual suffix as explanation
                    evidence = [ {"raw": reply.split("<Compliance Check Failed>")[-1].strip()} ]
            else:
                # heuristic classification
                lower = reply.lower()
                if "failed" in lower or "violate" in lower or "violation" in lower:
                    verdict = "fail"
                else:
                    verdict = "pass"
                evidence = [ {"raw": reply[:400]} ]
        else:
            verdict = "unknown"
            evidence = []

        results.append({
            "chunk_id": idx,
            "filename": fname,
            "hits": hits,
            "P": P,
            "N": N,
            "verdict": verdict,
            "evidence": evidence,
            "llm_reply": reply,
            "triples_text": triples_text
        })

        # optional small checkpoint save every few chunks
        # if (idx + 1) % 10 == 0:
        #     tmp_out = args.out + ".partial.json"
        #     with open(tmp_out, "w", encoding="utf-8") as f:
        #         json.dump(results, f, ensure_ascii=False, indent=2)
        #     logger.info("Saved partial results to %s", tmp_out)

    # final save
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d results to %s", len(results), args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--static-graph", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "static_graph.gpickle"), help="Path to static graph gpickle")
    p.add_argument("--eventic", "--eventic-graph", dest="eventic_graph", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "eventic_graph.gpickle"), help="Path to eventic graph gpickle (generated from eventic_extractor)")
    p.add_argument("--chunks", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "chunks.json"), help="chunks metadata JSON (from build_chunk_index.py)")
    p.add_argument("--faiss", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "faiss.index"), help="Faiss index path (optional)")
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "preds.json"), help="Output JSON path")
    p.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model for embeddings")
    p.add_argument("--lambda_thresh", type=float, default=0.75, help="Cosine similarity threshold for matching eventic nodes")
    p.add_argument("--hop_k", type=int, default=1, help="Number of hops to expand neighbors in static graph")
    p.add_argument("--max_triples", type=int, default=60, help="Max number of triples to include in prompt")
    p.add_argument("--prefer_local", action="store_true", help="Prefer local model fallback instead of OpenAI")
    p.add_argument("--openai_model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use when available")
    args = p.parse_args()
    main(args)
