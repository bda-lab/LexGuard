#!/usr/bin/env python3
"""
build_static_kg.py

Builds a static knowledge graph from RDF/TTL/NT/XML/JSON/CSV files placed in data/static/
Saves a NetworkX graph to data/static_graph.gpickle and node embeddings to data/static/node_embeddings.npz

Usage:
    python src/static/build_static_kg.py
"""

import os
import sys
import json
import argparse
import glob
import logging
from typing import Dict, Any, List, Tuple
import pickle
import networkx as nx
import numpy as np

try:
    from rdflib import Graph as RDFGraph
    RDFlib_AVAILABLE = True
except Exception:
    RDFlib_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_static_kg")

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
STATIC_DIR = os.path.join(DATA_DIR, "static")
OUT_GRAPH_PATH = os.path.join(DATA_DIR, "static_graph.gpickle")
OUT_EMB_PATH = os.path.join(DATA_DIR, "static", "node_embeddings.npz")


def parse_rdf_file(path: str) -> List[Tuple[str, str, str]]:
    if not RDFlib_AVAILABLE:
        raise RuntimeError("rdflib not installed. pip install rdflib")
    g = RDFGraph()
    logger.info("Parsing RDF file: %s", path)
    try:
        g.parse(path)
    except Exception as e:
        # try to guess format by extension
        ext = os.path.splitext(path)[1].lower()
        fmt = {"ttl": "turtle", "nt": "nt", "rdf": "xml", "xml": "xml", "jsonld": "json-ld"}.get(ext.strip("."), None)
        if fmt:
            g.parse(path, format=fmt)
        else:
            raise e
    triples = []
    for s, p, o in g:
        triples.append((str(s), str(p), str(o)))
    return triples


def parse_json_triples(path: str) -> List[Tuple[str, str, str]]:
    """
    Expect JSON as either:
      - list of {"subject": "...", "predicate":"...", "object":"..."}
      - dict adjacency mapping: { "A": {"rel": ["B","C"] }, ... }
    """
    logger.info("Parsing JSON/JSON-LD file: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    triples = []
    if isinstance(data, list):
        for entry in data:
            s = entry.get("subject") or entry.get("s") or entry.get("sub")
            p = entry.get("predicate") or entry.get("p") or entry.get("pred")
            o = entry.get("object") or entry.get("o") or entry.get("obj")
            if s and p and o:
                triples.append((str(s), str(p), str(o)))
    elif isinstance(data, dict):
        # adjacency mapping
        for s, val in data.items():
            if isinstance(val, dict):
                for p, objs in val.items():
                    if isinstance(objs, list):
                        for o in objs:
                            triples.append((str(s), str(p), str(o)))
                    else:
                        triples.append((str(s), str(p), str(objs)))
            elif isinstance(val, list):
                for o in val:
                    triples.append((str(s), "relatedTo", str(o)))
            else:
                triples.append((str(s), "hasValue", str(val)))
    return triples


def parse_csv_triples(path: str) -> List[Tuple[str, str, str]]:
    """
    CSV with columns: subject,predicate,object (header optional)
    """
    import csv
    triples = []
    logger.info("Parsing CSV file: %s", path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # try detect header
    if rows and len(rows[0]) >= 3 and "subject" in rows[0][0].lower():
        rows = rows[1:]
    for r in rows:
        if len(r) >= 3:
            s, p, o = r[0].strip(), r[1].strip(), r[2].strip()
            if s:
                triples.append((s, p or "relatedTo", o or ""))
    return triples


def build_graph_from_triples(triples: List[Tuple[str, str, str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for s, p, o in triples:
        # add nodes with optional labels
        if not G.has_node(s):
            G.add_node(s, label=s)
        if not G.has_node(o):
            G.add_node(o, label=o)
        # store predicate as edge attribute (may be duplicate edges)
        G.add_edge(s, o, predicate=p)
    return G


def collect_triples_from_static_dir(static_dir: str) -> List[Tuple[str, str, str]]:
    triples = []
    if not os.path.isdir(static_dir):
        logger.warning("Static dir %s not found. Creating empty graph instead.", static_dir)
        return triples

    # find rdf/json/csv files
    files = sorted(glob.glob(os.path.join(static_dir, "*")))
    for f in files:
        if os.path.isdir(f):
            continue
        ext = os.path.splitext(f)[1].lower()
        try:
            if ext in (".ttl", ".nt", ".rdf", ".xml", ".jsonld", ".json"):
                if ext in (".ttl", ".nt", ".rdf", ".xml", ".jsonld") and RDFlib_AVAILABLE:
                    triples.extend(parse_rdf_file(f))
                else:
                    # try json parse
                    triples.extend(parse_json_triples(f))
            elif ext in (".csv", ".tsv"):
                triples.extend(parse_csv_triples(f))
            else:
                logger.debug("Skipping file (unknown format): %s", f)
        except Exception as e:
            logger.exception("Failed parsing %s: %s", f, e)
    return triples


def compute_node_embeddings(G: nx.Graph, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, Any]:
    if not SBERT_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
    texts = []
    nodes = list(G.nodes(data=True))
    node_keys = [n for n, attrs in nodes]
    # Use node attribute 'label' if available
    for n, attrs in nodes:
        label = attrs.get("label") or attrs.get("definition") or n
        texts.append(str(label))
    logger.info("Computing embeddings for %d nodes using %s ...", len(texts), model_name)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    # return mapping node->embedding and node order
    emb_map = {node_keys[i]: embeddings[i] for i in range(len(node_keys))}
    return emb_map, node_keys


def save_graph_and_embeddings(G: nx.DiGraph, emb_map: Dict[str, Any], node_order: List[str], out_graph: str, out_emb: str):
    logger.info("Saving graph to %s", out_graph)
    with open(out_graph, "wb") as f:
        pickle.dump(G, f)
    if emb_map:
        logger.info("Saving node embeddings to %s", out_emb)
        embs = np.stack([emb_map[n] for n in node_order], axis=0)
        np.savez_compressed(out_emb, node_order=node_order, embeddings=embs)
    logger.info("Done.")


def create_toy_graph() -> nx.DiGraph:
    logger.info("Creating a small toy static KG (no input found).")
    G = nx.DiGraph()
    G.add_node("DataController", label="Data Controller", definition="Entity that determines purposes and means of processing personal data")
    G.add_node("DataProcessor", label="Data Processor", definition="Entity that processes personal data on behalf of the controller")
    G.add_node("PersonalData", label="Personal Data", definition="Any information relating to an identified or identifiable natural person")
    G.add_edge("DataController", "PersonalData", predicate="controls")
    G.add_edge("DataProcessor", "PersonalData", predicate="processes")
    return G


def main(args):
    # collect triples from data/static
    triples = collect_triples_from_static_dir(STATIC_DIR)
    if not triples:
        logger.warning("No triples found in data/static. Will create a toy KG.")
        G = create_toy_graph()
    else:
        logger.info("Collected %d triples from static dir", len(triples))
        G = build_graph_from_triples(triples)

    # optionally fill node 'label' and 'definition' attributes if present in triples (heuristic)
    for n, attrs in list(G.nodes(data=True)):
        if 'label' not in attrs:
            G.nodes[n]['label'] = n

    # compute embeddings
    emb_map = None
    node_order = None
    if SBERT_AVAILABLE:
        try:
            emb_map, node_order = compute_node_embeddings(G, model_name=args.embed_model)
        except Exception as e:
            logger.warning("Failed to compute node embeddings: %s", e)
            emb_map = None

    # ensure output dir exists
    os.makedirs(os.path.dirname(OUT_GRAPH_PATH), exist_ok=True)
    if not os.path.isdir(os.path.dirname(OUT_EMB_PATH)):
        os.makedirs(os.path.dirname(OUT_EMB_PATH), exist_ok=True)

    save_graph_and_embeddings(G, emb_map or {}, node_order or [], OUT_GRAPH_PATH, OUT_EMB_PATH)
    logger.info("Static KG saved to %s", OUT_GRAPH_PATH)
    if emb_map:
        logger.info("Static node embeddings saved to %s", OUT_EMB_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model for node embeddings")
    parsed = parser.parse_args()
    main(parsed)
