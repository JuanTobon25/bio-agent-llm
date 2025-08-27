# app.py — Groq-only, sidebar con foto en tarjeta (borde + sombra + esquinas redondeadas)
import streamlit as st
import pandas as pd
from pathlib import Path
import base64

from tools import (
    prepare_concept_kb, search_concepts,
    prepare_species_kb, identify_species
)
from agent import BioLLM

st.set_page_config(page_title="Agente LLM de Biología", page_icon="🧬", layout="wide")

# =========================
# Parámetros por defecto
# =========================
K_DEFAULT = 8          # Top-K para recuperación semántica (más recall para re-ranking)
CONF_THRESHOLD = 0.45  # Umbral para avisos de baja confianza

# Imagen de la sidebar (tu archivo local)
SIDEBAR_IMAGE_LOCAL = Path("assets/Mono.jpg")  # Asegúrate de tener assets/mono.jpg en el repo
SIDEBAR_IMAGE_URL   = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg"

def _listify_str(x):
    """Convierte x en lista de strings para ', '.join(...) sin romper si es str/None."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    try:
        return [str(v) for v in x]
    except TypeError:
        return [str(x)]

# -------- Patch: tarjeta con borde + sombra para la foto de la sidebar --------
def sidebar_photo_card(image_path: Path, url_fallback: str, caption: str = ""):
    """Muestra una imagen en la sidebar con borde, sombra y esquinas redondeadas."""
    if image_path.exists():
        data = image_path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        ext = image_path.suffix.lstrip(".").lower() or "png"
        src = f"data:image/{ext};base64,{b64}"
    else:
        src = url_fallback

    st.sidebar.markdown(
        f"""
        <style>
        .photo-card {{
            border: 2px solid #e9ecef;
            border-radius: 18px;
            padding: 8px;
            background: #ffffff;
            box-shadow: 0 6px 16px rgba(0,0,0,.08);
            margin-bottom: 10px;
        }}
        .photo-card img {{
            width: 100%;
            border-radius: 12px;
            display: block;
        }}
        .photo-card .caption {{
            text-align: center;
            font-size: 0.85rem;
            color: #6c757d;
            margin-top: 6px;
        }}
        </style>
        <div class="photo-card">
            <img src="{src}" alt="{caption}">
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Sidebar mínima
# =========================
st.sidebar.header("⚙️ Configuración (Groq)")

# 1) Key desde Secrets (prioridad) o barra lateral
groq_key_secret = None
try:
    groq_key_secret = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    groq_key_secret = None

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
effective_groq_key = groq_key_secret if groq_key_secret else (groq_api_key or None)

if not effective_groq_key:
    st.sidebar.warning("Agrega tu `GROQ_API_KEY` para usar la app.")
    st.error("Esta app está en modo **Groq-only**. Agrega tu `GROQ_API_KEY` en *Secrets* o escríbela en la barra lateral.")
    st.stop()

groq_model = st.sidebar.selectbox("Modelo Groq", ["llama3-70b-8192", "llama3-8b-8192"], index=0)
st.sidebar.caption("🔌 Motor activo: **Groq**")

st.sidebar.markdown("---")
# 👉 usa la tarjeta con marco y sombra para la foto del mono
sidebar_photo_card(SIDEBAR_IMAGE_LOCAL, SIDEBAR_IMAGE_URL, caption="Mono")

# =========================
# Carga y cachés
# =========================
@st.cache_resource(show_spinner=False)
def load_kb():
    docs, corpus_c, embedder_c, index_c = prepare_concept_kb("kb/concepts.jsonl")
    sp, corpus_s, embedder_s, index_s    = prepare_species_kb("kb/species.jsonl")
    return (docs, corpus_c, embedder_c, index_c), (sp, corpus_s, embedder_s, index_s)

(concepts_pack, species_pack) = load_kb()
docs, corpus_c, embedder_c, index_c = concepts_pack
sp, corpus_s, embedder_s, index_s   = species_pack

@st.cache_resource(show_spinner=False)
def make_llm_cached(groq_model_name: str, api_key: str):
    return BioLLM(groq_api_key=api_key, groq_model=groq_model_name)

llm = make_llm_cached(groq_model, effective_groq_key)

# =========================
# UI
# =========================
st.title("🧬 Agente LLM de Biología")
st.caption("Identificador de especies por descripción • Conceptos y procesos (RAG + explicación)")

tabs = st.tabs(["Identificar especie", "Conceptos y procesos"])

# --- Tab 1: Identificar especie (sin filtros) ---
with tabs[0]:
    st.subheader("🦋 Identificador de especies por descripción")
    desc = st.text_area(
        "Describe rasgos, color, hábitat, comportamiento…",
        "Mamífero muy grande con orejas grandes y trompa; habita sabanas africanas."
    )
    run_id = st.button("Identificar", type="primary", key="id_btn")

    if run_id and desc.strip():
        # Recuperación semántica base
        results = identify_species(desc, sp, corpus_s, embedder_s, index_s, k=K_DEFAULT)

        st.write("📊 Candidatos (mayor similitud primero):")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        # Re-ranking por LLM sobre top-K
        rr = llm.rerank_species(desc, results, top_n=min(K_DEFAULT, len(results)))
        best = rr.get("best", {}) or {}
        confidence = float(rr.get("confidence", 0.0) or 0.0)
        notes = rr.get("notes", "") or ""

        if results and results[0].get("similarity", 0.0) < CONF_THRESHOLD:
            st.warning("La confianza del índice es baja. Añade rasgos distintivos (patrones, medidas, región exacta).")

        common_names_str = ", ".join(_listify_str(best.get("common_names")))
        sci_name = best.get("scientific_name", "—")
        verdict_md = f"""**🔎 Veredicto del LLM (re-ranking)**  
**{sci_name}** ({common_names_str}) — confianza LLM: **{confidence:.2f}**"""
        if best.get("reason"):
            verdict_md += f"\n\n**Motivo:** {best['reason']}"
        if notes:
            verdict_md += f"\n\n*{notes}*"
        st.success(verdict_md)

# --- Tab 2: Conceptos y procesos (unificados) ---
with tabs[1]:
    st.subheader("📚 Conceptos y procesos (con RAG)")
    mode = st.radio("Modo", ["Pregunta (Q&A)", "Explicar proceso"], horizontal=True)
    text = st.text_input("Escribe tu pregunta o el proceso a explicar", "¿Qué ocurre en la fase luminosa de la fotosíntesis?")
    run_cp = st.button("Generar", type="primary", key="cp_btn")

    if run_cp and text.strip():
        hits = search_concepts(text, docs, corpus_c, embedder_c, index_c, k=K_DEFAULT)
        with st.expander("🔎 Contexto recuperado"):
            for h in hits:
                st.markdown(f"**{h['title']}** — score: `{h['score']:.3f}`")
                st.write(h["text"])
                st.markdown("---")

        if mode.startswith("Explicar"):
            ans = llm.answer_concepts_or_process(text, hits, mode="process")
        else:
            ans = llm.answer_concepts_or_process(text, hits, mode="qa")

        st.success(ans)

