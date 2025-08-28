 # tools.py — FastEmbed + NumPy + BM25 (ligero, robusto y con embedder compartido)
from typing import List, Dict, Tuple
import json
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

# Intentamos modelos en orden: primero ligero (rápido), luego pesado (mejor calidad)
EMB_MODEL_CANDIDATES = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",   # ligero y multilingüe
    "intfloat/multilingual-e5-large",                                 # más pesado (calidad >, costo >>)
]

# =========================
# Carga de JSON (robusta)
# =========================
def load_jsonl(path: str) -> List[Dict]:
    """
    Carga:
      - JSONL (un JSON por línea), o
      - un JSON array completo [ {...}, {...} ].
    Tolera BOM, líneas vacías y comentarios que empiezan con // o #.
    Si hay error, reporta línea + fragmento.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = raw.lstrip("\ufeff")              # quita BOM si existe
    s = raw.lstrip()

    # Caso 1: el archivo es un JSON array completo
    if s.startswith("["):
        try:
            data = json.loads(s)
            if not isinstance(data, list):
                raise ValueError("El archivo parece un array JSON pero no es una lista.")
            return data
        except Exception as e:
            raise ValueError(f"{path}: JSON array inválido: {e}")

    # Caso 2: JSONL (un objeto por línea)
    out = []
    for ln, line in enumerate(raw.splitlines(), start=1):
        cur = line.strip()
        if not cur:
            continue
        if cur.startswith("//") or cur.startswith("#"):
            continue
        try:
            obj = json.loads(cur)
            out.append(obj)
        except json.JSONDecodeError as e:
            frag = (cur[:140] + "…") if len(cur) > 140 else cur
            raise ValueError(f"Error parseando {path} en línea {ln}: {e}. Fragmento: {frag}")
    return out

# =========================
# Utilidades de vectores
# =========================
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms

# =========================
# Embedder compartido
# =========================
_SHARED_EMBEDDER: TextEmbedding | None = None
_SHARED_MODEL_NAME: str | None = None

def get_shared_embedder() -> Tuple[TextEmbedding, str]:
    """Crea una sola instancia de TextEmbedding y la reutiliza para conceptos y especies."""
    global _SHARED_EMBEDDER, _SHARED_MODEL_NAME
    if _SHARED_EMBEDDER is not None:
        return _SHARED_EMBEDDER, (_SHARED_MODEL_NAME or "unknown")

    last_err = None
    for name in EMB_MODEL_CANDIDATES:
        try:
            _SHARED_EMBEDDER = TextEmbedding(model_name=name)
            _SHARED_MODEL_NAME = name
            break
        except Exception as e:
            last_err = e
            continue

    if _SHARED_EMBEDDER is None:
        # Fallback al default interno de fastembed
        _SHARED_EMBEDDER = TextEmbedding()
        _SHARED_MODEL_NAME = "fastembed-default"
    return _SHARED_EMBEDDER, _SHARED_MODEL_NAME

def embed_texts(texts: List[str]) -> Tuple[np.ndarray, str]:
    """Embeddings normalizados (float32) para una lista de textos usando el embedder compartido."""
    embedder, used = get_shared_embedder()
    vecs = list(embedder.embed(texts))          # lista de np.ndarray float32
    embs = np.vstack(vecs).astype(np.float32)
    embs = l2_normalize(embs)
    return embs, used

# =========================
# Índice híbrido (denso + BM25)
# =========================
class VectorIndex:
    """
    Índice con:
      - Embeddings normalizados (coseno)
      - BM25 clásico (palabras)
      - Corpus original (texto)
    Permite búsqueda híbrida: score = alpha * denso + (1-alpha) * bm25
    """
    def __init__(self, embeddings: np.ndarray, corpus_texts: List[str]):
        self.embs = embeddings                    # [N, d] float32 normalizados
        self.corpus = corpus_texts                # lista[str]
        self._bm25: BM25Okapi | None = None
        self._tokenized: List[List[str]] | None = None

    def _ensure_bm25(self):
        if self._bm25 is None or self._tokenized is None:
            self._tokenized = [c.lower().split() for c in self.corpus]
            self._bm25 = BM25Okapi(self._tokenized)

    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        return (query_vec @ self.embs.T).astype(np.float32)[0]  # [N]

    def bm25_scores(self, query: str) -> np.ndarray:
        self._ensure_bm25()
        return np.asarray(self._bm25.get_scores(query.lower().split()), dtype=np.float32)

    def hybrid_search(self, query: str, k: int = 8, alpha: float = 0.6):
        """
        Devuelve lista de (idx, mix_score, dense_norm, bm25_norm).
        alpha pondera embeddings (0..1).
        """
        # 1) denso
        qvec, _ = embed_texts([query])            # [1, d] normalizado
        dense = self.dense_scores(qvec)           # [N]

        # 2) BM25
        bm25 = self.bm25_scores(query)            # [N]

        # 3) min-max estable
        def mm(x: np.ndarray) -> np.ndarray:
            x_min, x_max = float(x.min()), float(x.max())
            if x_max - x_min < 1e-8:
                return np.zeros_like(x, dtype=np.float32)
            return (x - x_min) / (x_max - x_min)

        d_n = mm(dense)
        b_n = mm(bm25)

        # 4) mezcla
        mix = alpha * d_n + (1.0 - alpha) * b_n

        # 5) top-k
        N = len(self.corpus)
        k = max(1, min(k, N))
        top = np.argsort(-mix)[:k]
        return [(int(i), float(mix[i]), float(d_n[i]), float(b_n[i])) for i in top]

# =========================
# KB de conceptos (RAG)
# =========================
def prepare_concept_kb(concepts_path: str):
    docs = load_jsonl(concepts_path)
    corpus = [f"{d['title']}. {d['text']}" for d in docs]
    embs, _used = embed_texts(corpus)
    index = VectorIndex(embs, corpus)
    # devolvemos el embedder por compatibilidad con app.py
    return docs, corpus, get_shared_embedder()[0], index

def search_concepts(query: str, docs, corpus, embedder: TextEmbedding, index: VectorIndex, k: int = 8, alpha: float = 0.6):
    hits = index.hybrid_search(query, k=k, alpha=alpha)
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

# =========================
# KB de especies (identificador)
# =========================
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
    embs, _used = embed_texts(corpus)
    index = VectorIndex(embs, corpus)
    return sp, corpus, get_shared_embedder()[0], index

def identify_species(description: str, sp, corpus, embedder: TextEmbedding, index: VectorIndex, k: int = 8, alpha: float = 0.6):
    hits = index.hybrid_search(description, k=k, alpha=alpha)
    results = []
    for rank, (i, mix_s, d_s, b_s) in enumerate(hits, start=1):
        s = sp[i]
        results.append({
            "rank": rank,
            "scientific_name": s.get("scientific_name",""),
            "common_names": s.get("common_names", []),
            "taxonomy": s.get("taxonomy",""),
            "match_explanation": s.get("traits",""),
            "similarity": float(mix_s),  # score mezclado 0..1
            "dense": float(d_s),
            "bm25": float(b_s),
        })
    return results
