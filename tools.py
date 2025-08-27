# tools.py (Groq-only, sin FAISS; búsqueda con NumPy)
from typing import List, Dict, Tuple
import json
import numpy as np
from fastembed import TextEmbedding

# Modelo multilingüe recomendado (ES/EN) para buenas búsquedas semánticas
MULTILINGUAL_EMB_MODEL = "intfloat/multilingual-e5-small"

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms

def build_embeddings(texts: List[str], model_name: str | None = None):
    use_model = model_name or MULTILINGUAL_EMB_MODEL
    embedder = TextEmbedding(model_name=use_model)
    vecs = list(embedder.embed(texts))                      # lista de np.ndarray float32
    embs = np.vstack(vecs).astype(np.float32)
    embs = l2_normalize(embs)                               # para coseno
    return embedder, embs

class VectorIndex:
    """Índice simple con NumPy: similitud coseno = producto interno tras normalizar."""
    def __init__(self, embeddings: np.ndarray):
        self.embs = embeddings  # [N, d], normalizados

    def search(self, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        # query_vec: [1, d] normalizado
        scores = query_vec @ self.embs.T                    # [1, N]
        scores = scores.astype(np.float32)
        # top-k
        idxs = np.argpartition(-scores[0], kth=min(k, scores.shape[1]-1))[:k]
        # ordenar esos top-k
        order = np.argsort(-scores[0, idxs])
        top_idxs = idxs[order]
        top_scores = scores[0, top_idxs]
        return top_scores[None, :], top_idxs[None, :]

def prepare_concept_kb(concepts_path: str):
    docs = load_jsonl(concepts_path)
    corpus = [f"{d['title']}. {d['text']}" for d in docs]
    embedder, embs = build_embeddings(corpus)
    index = VectorIndex(embs)
    return docs, corpus, embedder, index

def encode_queries(embedder: TextEmbedding, queries: List[str]) -> np.ndarray:
    qvecs = list(embedder.embed(queries))
    q = np.vstack(qvecs).astype(np.float32)
    q = l2_normalize(q)
    return q

def search_concepts(query: str, docs, corpus, embedder: TextEmbedding, index: VectorIndex, k: int = 4):
    q = encode_queries(embedder, [query])
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

def prepare_species_kb(species_path: str):
    sp = load_jsonl(species_path)
    corpus = [
        f"{s['scientific_name']} | {', '.join(s['common_names'])}. "
        f"Rasgos: {s['traits']}. Taxonomía: {s['taxonomy']}"
        for s in sp
    ]
    embedder, embs = build_embeddings(corpus)
    index = VectorIndex(embs)
    return sp, corpus, embedder, index

def identify_species(description: str, sp, corpus, embedder: TextEmbedding, index: VectorIndex, k: int = 5):
    q = encode_queries(embedder, [description])
    scores, idxs = index.search(q, k=k)
    results = []
    for rank, i in enumerate(idxs[0]):
        s = sp[i]
        results.append({
            "rank": rank + 1,
            "scientific_name": s["scientific_name"],
            "common_names": s["common_names"],
            "taxonomy": s["taxonomy"],
            "match_explanation": s["traits"],
            "similarity": float(scores[0][rank])
        })
    return results

