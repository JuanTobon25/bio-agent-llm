# tools.py (Groq-only build: fastembed + faiss, sin torch/transformers)
from typing import List, Dict, Tuple
import json
import numpy as np
import faiss

# fastembed es liviano y no depende de torch
from fastembed import TextEmbedding


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


def build_embeddings(
    texts: List[str],
    model_name: str | None = None,
):
    """
    Crea embeddings con fastembed. Por defecto usa un modelo pequeño por defecto.
    Si quieres fijar un modelo (p. ej., uno multilingüe soportado por fastembed),
    pásalo en model_name.
    """
    embedder = TextEmbedding(model_name=model_name) if model_name else TextEmbedding()
    # fastembed.embed() devuelve un generador de vectores (np.ndarray float32)
    vecs = list(embedder.embed(texts))
    embs = np.vstack(vecs).astype(np.float32)
    embs = l2_normalize(embs)  # para usar coseno con FAISS IP
    return embedder, embs


class VectorIndex:
    def __init__(self, embeddings: np.ndarray):
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)  # inner product (con vectores normalizados = coseno)
        self.index.add(embeddings)

    def search(self, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        D, I = self.index.search(query_vec, k)
        return D, I


# --- Concept KB (RAG) ---
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


# --- Species KB ---
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
