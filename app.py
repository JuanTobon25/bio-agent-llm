# app.py
import streamlit as st
import pandas as pd

from tools import (
    prepare_concept_kb, search_concepts,
    prepare_species_kb, identify_species
)
from agent import BioLLM

# OCR (opcional)
try:
    import easyocr
    from PIL import Image
except Exception:
    easyocr, Image = None, None

st.set_page_config(page_title="Agente LLM de Biología", page_icon="🧬", layout="wide")

# ---------- Sidebar: Configuración ----------
st.sidebar.header("⚙️ Configuración del LLM")
engine = st.sidebar.selectbox("Motor LLM", ["flan (local)", "groq (requiere API Key)"])
groq_api_key = None
if engine.startswith("groq"):
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Genera tu key en console.groq.com")
groq_model = st.sidebar.selectbox("Modelo Groq", ["llama3-8b-8192", "llama3-70b-8192"]) if engine.startswith("groq") else None

st.sidebar.markdown("---")
st.sidebar.header("🔎 Recuperación")
k = st.sidebar.slider("Top-K (búsqueda semántica)", 1, 8, 4)
level = st.sidebar.selectbox("Nivel de explicación", ["secundaria", "universitario", "divulgación"], index=1)

# ---------- Recursos (cacheados) ----------
@st.cache_resource(show_spinner=False)
def load_kb_and_models():
    docs, corpus_c, model_c, index_c = prepare_concept_kb("kb/concepts.jsonl")
    sp, corpus_s, model_s, index_s = prepare_species_kb("kb/species.jsonl")
    return (docs, corpus_c, model_c, index_c), (sp, corpus_s, model_s, index_s)

(concepts_pack, species_pack) = load_kb_and_models()
docs, corpus_c, model_c, index_c = concepts_pack
sp, corpus_s, model_s, index_s   = species_pack

@st.cache_resource(show_spinner=False)
def load_ocr_reader():
    if easyocr is None:
        return None
    return easyocr.Reader(['es', 'en'], gpu=False)

def make_llm():
    if engine.startswith("groq") and groq_api_key:
        return BioLLM(engine="groq", groq_api_key=groq_api_key, groq_model=groq_model)
    return BioLLM(engine="flan")

llm = make_llm()

# ---------- UI ----------
st.title("🧬 Agente LLM de Biología")
st.caption("Q&A con RAG • Identificación de especies por descripción • Explicación de procesos • OCR→Análisis (Groq o local)")

tabs = st.tabs(["Q&A (RAG)", "Identificar especie", "Explicar proceso", "OCR → Análisis"])

# --- Tab 1: Q&A (RAG) ---
with tabs[0]:
    st.subheader("❓ Preguntas de Biología con contexto (RAG local)")
    q = st.text_input("Ej.: ¿Qué ocurre en la fase luminosa de la fotosíntesis?")
    ask = st.button("Responder", type="primary", key="qa_btn")

    if ask and q.strip():
        hits = search_concepts(q, docs, corpus_c, model_c, index_c, k=k)
        with st.expander("🔎 Contexto recuperado"):
            for h in hits:
                st.markdown(f"**{h['title']}** — score: `{h['score']:.3f}`")
                st.write(h["text"])
                st.markdown("---")
        ans = llm.answer_with_context(q, hits)
        st.success("Respuesta:")
        st.write(ans)

# --- Tab 2: Identificar especie ---
with tabs[1]:
    st.subheader("🦋 Identificador de especies por descripción")
    desc = st.text_area("Describe rasgos, color, hábitat, comportamiento…",
                        "Ave tropical roja con amarillo y azul, pico fuerte, bosque húmedo.")
    run_id = st.button("Identificar", type="primary", key="id_btn")
    if run_id and desc.strip():
        results = identify_species(desc, sp, corpus_s, model_s, index_s, k=k)
        df = pd.DataFrame(results)
        st.write("📊 Candidatos (mayor similitud primero):")
        st.dataframe(df, use_container_width=True)

        # Opcional: explicación natural con LLM usando el top-1 como contexto
        top = results[0]
        expl_prompt = (
            "Explica de forma breve por qué la descripción podría corresponder a la especie siguiente, "
            "enfocándote en los rasgos coincidentes.\n\n"
            f"Descripción del usuario: {desc}\n"
            f"Especie candidata: {top['scientific_name']} ({', '.join(top['common_names'])})\n"
            f"Rasgos conocidos: {top['match_explanation']}\n"
            "Explicación:"
        )
        natural_expl = llm.generate(expl_prompt, max_new_tokens=180)
        st.info(natural_expl)

# --- Tab 3: Explicar proceso ---
with tabs[2]:
    st.subheader("🧪 Explicación de procesos biológicos")
    topic = st.text_input("Proceso (p. ej., mitosis, ósmosis, transcripción y traducción)", "mitosis")
    run_explain = st.button("Explicar", type="primary", key="exp_btn")
    if run_explain and topic.strip():
        explanation = llm.explain_process_steps(topic, level=level)
        st.success("Explicación:")
        st.write(explanation)

# --- Tab 4: OCR → Análisis (tu ejemplo integrado) ---
with tabs[3]:
    st.subheader("🖼️ OCR → Análisis con LLM")
    if easyocr is None or Image is None:
        st.warning("Instala dependencias de OCR (easyocr, Pillow) para usar esta pestaña.")
    else:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            up = st.file_uploader("Sube imagen (jpg, png, jpeg)", type=["png", "jpg", "jpeg"])
            reader = load_ocr_reader()
            if up is not None:
                image = Image.open(up)
                st.image(image, caption="Imagen cargada", use_column_width=True)
                if st.button("✨ Procesar Imagen", type="primary"):
                    try:
                        bytes_data = up.getvalue()
                        with st.spinner("🔍 Extrayendo texto (OCR)…"):
                            result = reader.readtext(bytes_data)
                            texto_ocr = " ".join([res[1] for res in result])
                            st.session_state.texto_ocr = texto_ocr
                        with st.spinner("🧠 Analizando con LLM…"):
                            prompt = (
                                "Eres un asistente experto en analizar y resumir texto. "
                                "Corrige posibles errores de OCR si son evidentes y entrega un resumen claro. "
                                "Si hay preguntas, respóndelas. Si son datos, organízalos.\n\n"
                                f"Texto:\n---\n{st.session_state.texto_ocr}\n---\n\nRespuesta:"
                            )
                            st.session_state.respuesta_ia = llm.generate(prompt, max_new_tokens=320)
                    except Exception as e:
                        st.error(f"Error procesando la imagen: {e}")

        with col2:
            st.markdown("#### Resultados")
            if "texto_ocr" in st.session_state and st.session_state.texto_ocr:
                with st.expander("Ver texto extraído", expanded=False):
                    st.text_area("Texto OCR", st.session_state.texto_ocr, height=160, disabled=True)
            if "respuesta_ia" in st.session_state and st.session_state.respuesta_ia:
                st.markdown("#### Respuesta del asistente")
                st.markdown(st.session_state.respuesta_ia)
            else:
                st.info("Aquí aparecerán los resultados después de procesar una imagen.")
