# tools.py — FastEmbed + NumPy + BM25 (sin FAISS / sin torch)
from typing import List, Dict, Tuple
import json
import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

# Intentamos primero un modelo más potente; si falla, caemos a MiniLM
EMB_MODEL_CANDIDATES = [
    "intfloat/multilingual-e5-large",                    # mejor calidad (más pesado)
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # ligero y multilingüe
]

# -------------------------------
# Utilidades de carga y vectores
# -------------------------------
def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms

def _create_embedder_with_fallback() -> Tuple[TextEmbedding, str]:
    last_err = None
    for name in EMB_MODEL_CANDIDATES:
        try:
            return TextEmbedding(model_name=name), name
        except Exception as e:
            last_err = e
            continue
    # Fallback al default de fastembed si nada funcionó
    return TextEmbedding(), "fastembed-default"

def build_embeddings(texts: List[str]) -> Tuple[TextEmbedding, np.ndarray, str]:
    """
    Genera embeddings normalizados (float32) para 'texts' y devuelve (embedder, embs, model_name_usado)
    """
    embedder, used = _create_embedder_with_fallback()
    vecs = list(embedder.embed(texts))          # lista de np.ndarray float32
    embs = np.vstack(vecs).astype(np.float32)
    embs = l2_normalize(embs)                   # para similitud coseno con producto interno
    return embedder, embs, used

# -------------------------------
# Índice híbrido (denso + BM25)
# -------------------------------
class VectorIndex:
    """
    Índice con:
      - Embeddings normalizados (coseno)
      - BM25 clásico (palabras)
      - Corpus original (texto)
    Permite búsqueda híbrida: score = alpha * denso + (1-alpha) * bm25
    """
    def __init__(self, embeddings: np.ndarray, corpus_texts: List[str]):
        self.embs = embeddings                              # [N, d] float32 normalizados
        self.corpus = corpus_texts                          # lista[str]
        # Construcción perezosa de BM25 (se arma cuando se necesita)
        self._bm25 = None
        self._tokenized = None

    def _ensure_bm25(self):
        if self._bm25 is None or self._tokenized is None:
            self._tokenized = [c.lower().split() for c in self.corpus]
            self._bm25 = BM25Okapi(self._tokenized)

    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Devuelve los scores densos (coseno) contra TODO el corpus: [N]."""
        return (query_vec @ self.embs.T).astype(np.float32)[0]  # [N]

    def bm25_scores(self, query: str) -> np.ndarray:
        """Scores BM25 para TODO el corpus: [N]."""
        self._ensure_bm25()
        return np.asarray(self._bm25.get_scores(query.lower().split()), dtype=np.float32)

    def hybrid_search(self, query: str, embedder: TextEmbedding, k: int = 8, alpha: float = 0.6):
        """
        Mezcla scores densos y BM25 en [0,1] (normalizados min-max) y devuelve top-k:
        Lista de tuplas: [(idx, mix_score, dense_score, bm25_score), ...]
        """
        # 1) denso
        qvec = np.vstack(list(embedder.embed([query]))).astype(np.float32)
        qvec = l2_normalize(qvec)
        dense = self.dense_scores(qvec)  # [N]

        # 2) BM25
        bm25 = self.bm25_scores(query)   # [N]

        # 3) normalización min-max (numéricamente estable)
        def _minmax(x: np.ndarray) -> np.ndarray:
            x_min, x_max = float(x.min()), float(x.max())
            if x_max - x_min < 1e-8:
                return np.zeros_like(x, dtype=np.float32)
            return (x - x_min) / (x_max - x_min)

        dense_n = _minmax(dense)
        bm25_n  = _minmax(bm25)

        # 4) mezcla
        mix = alpha * dense_n + (1.0 - alpha) * bm25_n

        # 5) top-k
        N = len(self.corpus)
        k = max(1, min(k, N))
        top = np.argsort(-mix)[:k]
        return [(int(i), float(mix[i]), float(dense_n[i]), float(bm25_n[i])) for i in top]

# -------------------------------
# KB de conceptos (RAG para Q&A)
# -------------------------------
def prepare_concept_kb(concepts_path: str):
    docs = load_jsonl(concepts_path)
    corpus = [f"{d['title']}. {d['text']}" for d in docs]
    embedder, embs, _used = build_embeddings(corpus)
    index = VectorIndex(embs, corpus)
    return docs, corpus, embedder, index

def search_concepts(query: str, docs, corpus, embedder: TextEmbedding, index: VectorIndex, k: int = 8, alpha: float = 0.6):
    """
    Búsqueda híbrida para conceptos. Devuelve lista de dicts:
      {id, title, text, score, dense, bm25}
    """
    hits = index.hybrid_search(query, embedder, k=k, alpha=alpha)
    results = []
    for i, mix_s, d_s, b_s in hits:
        d = docs[i]
        results.append({
            "id": d.get("id", f"doc_{i}"),
            "title": d.get("title", "Sin título"),
            "text": d.get("text", ""),
            "score": float(mix_s),
            "dense": float(d_s),
            "bm25": float(b_s),
        })
    return results

# -------------------------------
# KB de especies (identificador)
# -------------------------------
def _species_record_to_text(s: Dict) -> str:
    common = s.get("common_names", [])
    if isinstance(common, str):
        common = [common]
    return (
        f"{s.get('scientific_name','')} | {', '.join(common)}. "
        f"Rasgos: {s.get('traits','')}. Taxonomía: {s.get('taxonomy','')}"
    )

def prepare_species_kb(species_path: str):
    sp = load_jsonl(species_path)
    corpus = [_species_record_to_text(s) for s in sp]
    embedder, embs, _used = build_embeddings(corpus)
    index = VectorIndex(embs, corpus)
    return sp, corpus, embedder, index

def identify_species(description: str, sp, corpus, embedder: TextEmbedding, index: VectorIndex, k: int = 8, alpha: float = 0.6):
    """
    Búsqueda híbrida para especies. Devuelve lista de dicts ordenados:
      {rank, scientific_name, common_names, taxonomy, match_explanation, similarity, dense, bm25}
    - similarity = score mezclado (0..1)
    """
    hits = index.hybrid_search(description, embedder, k=k, alpha=alpha)
    results = []
    for rank, (i, mix_s, d_s, b_s) in enumerate(hits, start=1):
        s = sp[i]
        results.append({
            "rank": rank,
            "scientific_name": s.get("scientific_name",""),
            "common_names": s.get("common_names", []),
            "taxonomy": s.get("taxonomy",""),
            "match_explanation": s.get("traits",""),
            "similarity": float(mix_s),
            "dense": float(d_s),
            "bm25": float(b_s),
        })
    return results

