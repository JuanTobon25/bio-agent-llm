# tools.py
from typing import List, Dict, Tuple
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return model, embs

class VectorIndex:
    def __init__(self, embeddings: np.ndarray):
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)  # cos sim con vectores normalizados
        self.index.add(embeddings)

    def search(self, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        D, I = self.index.search(query_vec, k)
        return D, I

# --- Concept KB (RAG) ---
def prepare_concept_kb(concepts_path: str):
    docs = load_jsonl(concepts_path)
    corpus = [f"{d['title']}. {d['text']}" for d in docs]
    model, embs = build_embeddings(corpus)
    index = VectorIndex(embs)
    return docs, corpus, model, index

def search_concepts(query: str, docs, corpus, model, index, k: int = 4):
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, idxs = index.search(q, k=k)
    results = []
    for rank, i in enumerate(idxs[0]):
        d = docs[i]
        results.append({
            "id": d["id"],
            "title": d["title"],
            "text": d["text"],
            "score": float(scores[0][rank])
        })
    return results

# --- Species KB ---
def prepare_species_kb(species_path: str):
    sp = load_jsonl(species_path)
    corpus = [f"{s['scientific_name']} | {', '.join(s['common_names'])}. Rasgos: {s['traits']}. Taxonomía: {s['taxonomy']}"
              for s in sp]
    model, embs = build_embeddings(corpus)
    index = VectorIndex(embs)
    return sp, corpus, model, index

def identify_species(description: str, sp, corpus, model, index, k: int = 5):
    q = model.encode([description], normalize_embeddings=True, convert_to_numpy=True)
    scores, idxs = index.search(q, k=k)
    results = []
    for rank, i in enumerate(idxs[0]):
        s = sp[i]
        results.append({
            "rank": rank+1,
            "scientific_name": s["scientific_name"],
            "common_names": s["common_names"],
            "taxonomy": s["taxonomy"],
            "match_explanation": s["traits"],
            "similarity": float(scores[0][rank])
        })
    return results
