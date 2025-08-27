# app.py — Groq-only, sidebar simplificada con imagen
import streamlit as st
import pandas as pd
from pathlib import Path

from tools import (
    prepare_concept_kb, search_concepts,
    prepare_species_kb, identify_species
)
from agent import BioLLM

st.set_page_config(page_title="Agente LLM de Biología", page_icon="🧬", layout="wide")

# =========================
# Parámetros por defecto
# =========================
K_DEFAULT = 5          # Top-K para recuperación semántica
CONF_THRESHOLD = 0.45  # Umbral de confianza (identificador)

# =========================
# Sidebar: Configuración mínima
# =========================
st.sidebar.header("⚙️ Configuración (Groq)")

# 1) Leer key desde Secrets (prioridad) o desde la sidebar
groq_key_secret = None
try:
    groq_key_secret = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    groq_key_secret = None

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Genera tu key en https://console.groq.com/keys (recomendado guardarla en Settings → Secrets).",
)

# Determinar key efectiva
effective_groq_key = groq_key_secret if groq_key_secret else (groq_api_key or None)

# 🚫 Groq-only: si no hay key, detenemos
if not effective_groq_key:
    st.sidebar.warning("Agrega tu `GROQ_API_KEY` para usar la app.")
    st.error("Esta app está en modo **Groq-only**. Agrega tu `GROQ_API_KEY` en *Manage app → Settings → Secrets* o escríbela en la barra lateral.")
    st.stop()

groq_model = st.sidebar.selectbox(
    "Modelo Groq",
    ["llama3-70b-8192", "llama3-8b-8192"],
    index=0
)
st.sidebar.caption("🔌 Motor activo: **Groq**")

# — Imagen en la sidebar —
# Coloca un archivo en tu repo: assets/mascot.png
# Si no existe, usa una URL pública como respaldo.
st.sidebar.markdown("---")
sidebar_img_path = Path("assets/mascot.png")
if sidebar_img_path.exists():
    st.sidebar.image(str(sidebar_img_path), caption="Identificador de especies", use_container_width=True)
else:
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg",
        caption="Identificador de especies",
        use_container_width=True,
    )

# =========================
# Recursos cacheados
# =========================
@st.cache_resource(show_spinner=False)
def load_kb():
    docs, corpus_c, embedder_c, index_c = prepare_concept_kb("kb/concepts.jsonl")
    sp, corpus_s, embedder_s, index_s = prepare_species_kb("kb/species.jsonl")
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

# --- Tab 1: Identificar especie ---
with tabs[0]:
    st.subheader("🦋 Identificador de especies por descripción")
    desc = st.text_area(
        "Describe rasgos, color, hábitat, comportamiento…",
        "Mamífero muy grande con orejas grandes y trompa; habita sabanas africanas."
    )
    extra = st.text_input("Filtros opcionales (región/hábitat/palabras clave — p. ej., 'Andes, bosque nublado')", "")
    run_id = st.button("Identificar", type="primary", key="id_btn")

    if run_id and desc.strip():
        # Recuperación semántica base
        results = identify_species(desc, sp, corpus_s, embedder_s, index_s, k=K_DEFAULT)

        # Boost simple por palabras clave (si aparecen en traits/taxonomía/nombres)
        if extra.strip():
            kw = extra.lower()
            for r in results:
                blob = " ".join([
                    r["scientific_name"],
                    " ".join(r["common_names"]),
                    r["taxonomy"],
                    r["match_explanation"],
                ]).lower()
                if any(token.strip() and token.strip() in blob for token in kw.split(",")):
                    r["similarity"] = float(min(1.0, r["similarity"] + 0.05))
            results = sorted(results, key=lambda x: x["similarity"], reverse=True)

        df = pd.DataFrame(results)
        st.write("📊 Candidatos (mayor similitud primero):")
        st.dataframe(df, use_container_width=True)

        # Re-ranking por LLM
        rr = llm.rerank_species(desc, results, top_n=min(K_DEFAULT, len(results)))
        best = rr.get("best", {})
        confidence = rr.get("confidence", 0.0)
        notes = rr.get("notes", "")

        # Mensaje si la confianza del índice es baja
        if results and results[0]["similarity"] < CONF_THRESHOLD:
            st.warning(
                "La confianza del índice es baja. Añade rasgos distintivos (patrones, medidas, región exacta) "
                "o más descriptores."
            )

        st.success("🔎 Veredicto del LLM (re-ranking):")
        st.write(
            f"**{best.get('scientific_name', '—')}** "
            f"({', '.join(best.get('common_names', []))}) — "
            f"confianza LLM: **{confidence:.2f}**"
        )
        if best.get("reason"):
            st.info(best["reason"])
        if notes:
            st.caption(notes)

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

        st.success("Respuesta:")
        st.write(ans)



