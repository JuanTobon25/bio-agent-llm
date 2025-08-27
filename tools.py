# tools.py — FastEmbed + NumPy; sin FAISS / sin torch
from typing import List, Dict, Tuple
import json, io
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

# Modelo multilingüe soportado por FastEmbed (ES/EN)
MULTILINGUAL_EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms

def _create_embedder(model_name: str | None) -> TextEmbedding:
    try:
        return TextEmbedding(model_name=model_name) if model_name else TextEmbedding()
    except ValueError:
        return TextEmbedding()

def build_embeddings(texts: List[str], model_name: str | None = None) -> Tuple[TextEmbedding, np.ndarray]:
    use_model = model_name or MULTILINGUAL_EMB_MODEL
    embedder = _create_embedder(use_model)
    vecs = list(embedder.embed(texts))          # lista de np.ndarray float32
    embs = np.vstack(vecs).astype(np.float32)
    embs = l2_normalize(embs)
    return embedder, embs

class VectorIndex:
    """Índice simple con NumPy: coseno = producto interno tras normalizar."""
    def __init__(self, embeddings: np.ndarray):
        self.embs = embeddings  # [N, d] float32 normalizado

    def search(self, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.embs.size == 0:
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)
        k = max(1, min(k, self.embs.shape[0]))
        scores = (query_vec @ self.embs.T).astype(np.float32)   # [1, N]
        idxs = np.argpartition(-scores[0], kth=k-1)[:k]
        order = np.argsort(-scores[0, idxs])
        top_idxs = idxs[order]
        top_scores = scores[0, top_idxs]
        return top_scores[None, :], top_idxs[None, :]

# ---------- Concept KB (RAG) ----------
def prepare_concept_kb(concepts_path: str):
    docs = load_jsonl(concepts_path)
    corpus = [f"{d['title']}. {d['text']}" for d in docs]
    embedder, embs = build_embeddings(corpus, model_name=MULTILINGUAL_EMB_MODEL)
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
        d = docs[int(i)]
        results.append({
            "id": d.get("id", f"doc_{i}"),
            "title": d.get("title", "Sin título"),
            "text": d.get("text", ""),
            "score": float(scores[0][rank])
        })
    return results

# ---------- Species KB ----------
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
    embedder, embs = build_embeddings(corpus, model_name=MULTILINGUAL_EMB_MODEL)
    index = VectorIndex(embs)
    return sp, corpus, embedder, index

def identify_species(description: str, sp, corpus, embedder: TextEmbedding, index: VectorIndex, k: int = 5):
    q = encode_queries(embedder, [description])
    scores, idxs = index.search(q, k=k)
    results = []
    for rank, i in enumerate(idxs[0]):
        s = sp[int(i)]
        results.append({
            "rank": rank + 1,
            "scientific_name": s.get("scientific_name",""),
            "common_names": s.get("common_names", []),
            "taxonomy": s.get("taxonomy",""),
            "match_explanation": s.get("traits",""),
            "similarity": float(scores[0][rank])
        })
    return results

# === NUEVO: cargar KB adicional desde CSV/JSONL (sidebar) ===
def parse_species_upload(uploaded_file) -> List[Dict]:
    """Convierte un UploadedFile (CSV o JSONL) a lista de dicts con las claves requeridas."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
        # columnas mínimas: scientific_name, common_names, traits, taxonomy
        df = df.fillna("")
        records = []
        for _, row in df.iterrows():
            common = row.get("common_names", "")
            if isinstance(common, str):
                common = [x.strip() for x in common.split("|") if x.strip()]  # "jaguar|yaguareté"
            records.append({
                "scientific_name": str(row.get("scientific_name","")),
                "common_names": common,
                "traits": str(row.get("traits","")),
                "taxonomy": str(row.get("taxonomy","")),
            })
        return records

    if name.endswith(".jsonl"):
        text = data.decode("utf-8")
        recs = [json.loads(line) for line in text.splitlines() if line.strip()]
        # normaliza common_names si viene como string
        for r in recs:
            if isinstance(r.get("common_names"), str):
                r["common_names"] = [x.strip() for x in r["common_names"].split("|") if x.strip()]
        return recs

    raise ValueError("Formato no soportado. Usa CSV o JSONL.")

def rebuild_species_kb(base_sp: List[Dict], extra_sp: List[Dict]):
    """Une base + extra y reconstruye corpus/embeddings/índice."""
    merged = list(base_sp) + list(extra_sp)
    corpus = [_species_record_to_text(s) for s in merged]
    embedder, embs = build_embeddings(corpus, model_name=MULTILINGUAL_EMB_MODEL)
    index = VectorIndex(embs)
    return merged, corpus, embedder, index

