# tools.py  —  Groq-only, FastEmbed + NumPy (sin FAISS / sin torch)
from typing import List, Dict, Tuple
import json
import numpy as np
from fastembed import TextEmbedding

# ✅ Modelo multilingüe (ES/EN) soportado por FastEmbed
#   Alternativa de mayor calidad (más pesada): "intfloat/multilingual-e5-large"
MULTILINGUAL_EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# -------------------------------
# Utilidades de carga y vectores
# -------------------------------
def load_jsonl(path: str) -> List[Dict]:
    """Carga un archivo JSONL (una línea = un objeto JSON)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Normaliza filas a norma L2 = 1 (evita divisiones por cero)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


def _create_embedder(model_name: str | None) -> TextEmbedding:
    """Crea el embedder de FastEmbed con fallback elegante si el modelo no está soportado."""
    try:
        return TextEmbedding(model_name=model_name) if model_name else TextEmbedding()
    except ValueError:
        # Fallback al default de FastEmbed (asegura que la app no se caiga)
        return TextEmbedding()


def build_embeddings(
    texts: List[str],
    model_name: str | None = None,
) -> Tuple[TextEmbedding, np.ndarray]:
    """
    Genera embeddings para una lista de textos con FastEmbed.
    Devuelve (embedder, embeddings_normalizados[np.float32]).
    """
    use_model = model_name or MULTILINGUAL_EMB_MODEL
    embedder = _create_embedder(use_model)
    vecs = list(embedder.embed(texts))          # lista de vectores np.ndarray float32
    embs = np.vstack(vecs).astype(np.float32)
    embs = l2_normalize(embs)                   # para similitud coseno via producto interno
    return embedder, embs


class VectorIndex:
    """
    Índice simple con NumPy:
    - Asume embeddings normalizados.
    - Similitud coseno = producto interno.
    - top-k vía argpartition (rápido y sin dependencias nativas).
    """
    def __init__(self, embeddings: np.ndarray):
        self.embs = embeddings  # shape [N, d], normalizados (float32)

    def search(self, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        query_vec: shape [1, d] normalizado.
        Retorna (scores[1, k], indices[1, k]) ordenados de mayor a menor.
        """
        if self.embs.size == 0:
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)

        k = max(1, min(k, self.embs.shape[0]))
        scores = (query_vec @ self.embs.T).astype(np.float32)   # [1, N]
        idxs = np.argpartition(-scores[0], kth=k-1)[:k]         # top-k desordenado
        order = np.argsort(-scores[0, idxs])                    # ordena esos k
        top_idxs = idxs[order]
        top_scores = scores[0, top_idxs]
        return top_scores[None, :], top_idxs[None, :]


# -------------------------------
# KB de conceptos (RAG para Q&A)
# -------------------------------
def prepare_concept_kb(concepts_path: str):
    """
    Carga la KB de conceptos y construye:
      - corpus (title + text)
      - embedder y embeddings normalizados
      - índice VectorIndex
    Retorna: (docs, corpus, embedder, index)
    """
    docs = load_jsonl(concepts_path)
    corpus = [f"{d['title']}. {d['text']}" for d in docs]
    embedder, embs = build_embeddings(corpus, model_name=MULTILINGUAL_EMB_MODEL)
    index = VectorIndex(embs)
    return docs, corpus, embedder, index


def encode_queries(embedder: TextEmbedding, queries: List[str]) -> np.ndarray:
    """Embeddea y normaliza una o más consultas; retorna array [n, d]."""
    qvecs = list(embedder.embed(queries))
    q = np.vstack(qvecs).astype(np.float32)
    q = l2_normalize(q)
    return q


def search_concepts(query: str, docs, corpus, embedder: TextEmbedding, index: VectorIndex, k: int = 4):
    """Busca los k conceptos más similares a la consulta (por coseno)."""
    q = encode_queries(embedder, [query])
    scores, idxs = index.search(q, k=k)
    results = []
    for rank, i in enumerate(idxs[0]):
        d = docs[int(i)]
        results.append({
            "id": d.get("id", f"doc_{i}"),
            "title": d.get("title", "Sin título"),
            "text": d.get("text", ""),
            "score": float(scores[0][rank])
        })
    return results


# -------------------------------
# KB de especies (identificador)
# -------------------------------
def prepare_species_kb(species_path: str):
    """
    Carga fichas de especies y construye corpus + embeddings + índice.
    Retorna: (sp, corpus, embedder, index)
    """
    sp = load_jsonl(species_path)
    corpus = [
        f"{s['scientific_name']} | {', '.join(s['common_names'])}. "
        f"Rasgos: {s['traits']}. Taxonomía: {s['taxonomy']}"
        for s in sp
    ]
    embedder, embs = build_embeddings(corpus, model_name=MULTILINGUAL_EMB_MODEL)
    index = VectorIndex(embs)
    return sp, corpus, embedder, index


def identify_species(description: str, sp, corpus, embedder: TextEmbedding, index: VectorIndex, k: int = 5):
    """Rankea especies desde una descripción libre; retorna lista de candidatos ordenados."""
    q = encode_queries(embedder, [description])
    scores, idxs = index.search(q, k=k)
    results = []
    for rank, i in enumerate(idxs[0]):
        s = sp[int(i)]
        results.append({
            "rank": rank + 1,
            "scientific_name": s["scientific_name"],
            "common_names": s["common_names"],
            "taxonomy": s["taxonomy"],
            "match_explanation": s["traits"],
            "similarity": float(scores[0][rank])
        })
    return results


