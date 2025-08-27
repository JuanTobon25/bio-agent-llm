# app.py  — Groq-only, sin pestaña OCR
import streamlit as st
import pandas as pd

from tools import (
    prepare_concept_kb, search_concepts,
    prepare_species_kb, identify_species
)
from agent import BioLLM

st.set_page_config(page_title="Agente LLM de Biología", page_icon="🧬", layout="wide")

# =========================
# Sidebar: Configuración
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
    help="Genera tu key en https://console.groq.com/keys (recomendado guardarla en Settings → Secrets)."
)

# Determinar key efectiva
effective_groq_key = groq_key_secret if groq_key_secret else (groq_api_key or None)

# 🚫 Groq-only: si no hay key, detenemos
if not effective_groq_key:
    st.error("Esta app está en modo **Groq-only**. Agrega tu `GROQ_API_KEY` en *Manage app → Settings → Secrets* o escríbela en la barra lateral.")
    st.stop()

groq_model = st.sidebar.selectbox(
    "Modelo Groq",
    ["llama3-70b-8192", "llama3-8b-8192"],
    index=0
)

st.sidebar.caption("🔌 Motor activo: **Groq**")

st.sidebar.markdown("---")
st.sidebar.header("🔎 Recuperación")
k = st.sidebar.slider("Top-K (búsqueda semántica)", 1, 8, 5)
level = st.sidebar.selectbox("Nivel de explicación", ["secundaria", "universitario", "divulgación"], index=1)

# (Opcional) Umbral de confianza para identificar especie
conf_threshold = st.sidebar.slider("Umbral de confianza (identificador)", 0.30, 0.80, 0.45, 0.01)

# =========================
# Recursos cacheados
# =========================
@st.cache_resource(show_spinner=False)
def load_kb():
    # Con fastembed: prepare_* devuelve (docs/sp, corpus, embedder, index)
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
st.title("🧬 Agente LLM de Biología (Groq-only)")
st.caption("Identificar especies por descripción • Q&A de conceptos con RAG • Explicación de procesos")

# Solo 3 pestañas enfocadas
tabs = st.tabs(["Identificar especie", "Conceptos (Q&A)", "Explicar proceso"])

# --- Tab 1: Identificar especie ---
with tabs[0]:
    st.subheader("🦋 Identificador de especies por descripción")
    desc = st.text_area(
        "Describe rasgos, color, hábitat, comportamiento…",
        "Mamífero muy grande con orejas grandes y trompa; habita sabanas africanas."
    )
    run_id = st.button("Identificar", type="primary", key="id_btn")
    if run_id and desc.strip():
        results = identify_species(desc, sp, corpus_s, embedder_s, index_s, k=k)
        df = pd.DataFrame(results)
        st.write("📊 Candidatos (mayor similitud primero):")
        st.dataframe(df, use_container_width=True)

        top = results[0]
        # Aviso si la confianza es baja
        if top["similarity"] < conf_threshold:
            st.warning(
                "La confianza es baja. Intenta añadir rasgos distintivos (patrones de pelaje/plumas, longitud del pico, "
                "forma de hojas, hábitat preciso, comportamiento, región geográfica)."
            )

        # Explicación natural con LLM usando el top-1 como contexto
        expl_prompt = (
            "Explica brevemente por qué la descripción del usuario podría corresponder a la especie siguiente, "
            "enfocado en los rasgos coincidentes. Si hay rasgos que NO coinciden, menciónalos también.\n\n"
            f"Descripción del usuario: {desc}\n"
            f"Especie candidata: {top['scientific_name']} ({', '.join(top['common_names'])})\n"
            f"Rasgos conocidos: {top['match_explanation']}\n"
            "Explicación:"
        )
        natural_expl = llm.generate(expl_prompt, max_new_tokens=200)
        st.info(natural_expl)

# --- Tab 2: Conceptos (Q&A con RAG) ---
with tabs[1]:
    st.subheader("📚 Preguntas de Biología con contexto (RAG)")
    q = st.text_input("Ej.: ¿Qué ocurre en la fase luminosa de la fotosíntesis?")
    ask = st.button("Responder", type="primary", key="qa_btn")

    if ask and q.strip():
        hits = search_concepts(q, docs, corpus_c, embedder_c, index_c, k=k)
        with st.expander("🔎 Contexto recuperado"):
            for h in hits:
                st.markdown(f"**{h['title']}** — score: `{h['score']:.3f}`")
                st.write(h["text"])
                st.markdown("---")
        ans = llm.answer_with_context(q, hits)
        st.success("Respuesta:")
        st.write(ans)

# --- Tab 3: Explicar proceso ---
with tabs[2]:
    st.subheader("🧪 Explicación de procesos biológicos")
    topic = st.text_input("Proceso (p. ej., mitosis, ósmosis, transcripción y traducción)", "mitosis")
    run_explain = st.button("Explicar", type="primary", key="exp_btn")
    if run_explain and topic.strip():
        explanation = llm.explain_process_steps(topic, level=level)
        st.success("Explicación:")
        st.write(explanation)

