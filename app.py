# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from tools import (
    prepare_concept_kb, search_concepts,
    prepare_species_kb, identify_species
)
from agent import BioLLM

st.set_page_config(page_title="Agente LLM de Biología", page_icon="🧬", layout="wide")

# =========================
# Parámetros por defecto
# =========================
K_DEFAULT = 8
CONF_THRESHOLD = 0.45

SIDEBAR_IMAGE_LOCAL = Path("assets/Mono.jpg")
SIDEBAR_IMAGE_URL   = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg"

def sidebar_photo_card(image_path: Path, url_fallback: str, caption: str = ""):
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
sidebar_photo_card(SIDEBAR_IMAGE_LOCAL, SIDEBAR_IMAGE_URL, caption="")

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
st.caption("Identificador de especies por descripción • Conceptos y procesos • EDA y Clasificación")

tabs = st.tabs(["Identificar especie", "Conceptos y procesos", "EDA + Clasificación + Chat"])

# --- Tab 1: Identificar especie ---
with tabs[0]:
    st.subheader("🦋 Identificador de especies por descripción")
    desc = st.text_area(
        "Describe rasgos, color, hábitat, comportamiento…",
        "Mamífero muy grande con orejas grandes y trompa; habita sabanas africanas."
    )
    run_id = st.button("Identificar", type="primary", key="id_btn")
    if run_id and desc.strip():
        results = identify_species(desc, sp, corpus_s, embedder_s, index_s, k=K_DEFAULT)
        st.write("📊 Candidatos (mayor similitud primero):")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        rr = llm.rerank_species(desc, results, top_n=min(K_DEFAULT, len(results)))
        best = rr.get("best", {}) or {}
        confidence = float(rr.get("confidence", 0.0) or 0.0)
        if results and results[0].get("similarity", 0.0) < CONF_THRESHOLD:
            st.warning("Confianza baja: añade rasgos distintivos.")
        sci_name = best.get("scientific_name", "—")
        st.success(f"**🔎 Veredicto del LLM:** {sci_name} — confianza: {confidence:.2f}")

# --- Tab 2: Conceptos y procesos ---
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
        ans = llm.answer_concepts_or_process(text, hits, mode="process" if mode.startswith("Explicar") else "qa")
        st.success(ans)

# --- Tab 3: EDA + Clasificación + Chat basado en dataset ---
with tabs[2]:
    st.subheader("📊 EDA + Clasificación + Chat sobre dataset")
    uploaded = st.file_uploader("Sube un archivo CSV con tu dataset de especies", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("### Vista previa del dataset:")
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")

        # ====== EDA ======
        st.write("## 🔍 Exploratory Data Analysis (EDA)")

        # 1. Valores nulos
        st.write("### Valores nulos por columna")
        st.dataframe(df.isnull().sum())

        # 2. Estadísticas descriptivas
        st.write("### Estadísticas descriptivas")
        st.dataframe(df.describe(include="all"))

        # 3. Detección de outliers (solo numéricos)
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            st.write("### Boxplots (detección de outliers)")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)

        # 4. Histogramas de variables numéricas
        if len(numeric_cols) > 0:
            st.write("### Histogramas de variables numéricas")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribución de {col}")
                st.pyplot(fig)

        # 5. Distribución de variables categóricas
        cat_cols = df.select_dtypes(include="object").columns
        if len(cat_cols) > 0:
            st.write("### Distribución de variables categóricas")
            for col in cat_cols:
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Distribución de {col}")
                st.pyplot(fig)

        # 6. Correlaciones
        if len(numeric_cols) > 1:
            st.write("### Mapa de calor de correlaciones")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # ====== Clasificación ======
        clf, X, y, X_test, y_test = None, None, None, None, None
        if "Especie" in df.columns:
            st.write("## 🤖 Entrenamiento de clasificador (Random Forest)")
            X = df.drop("Especie", axis=1)
            y = df["Especie"]

            for col in X.select_dtypes(include="object").columns:
                X[col] = LabelEncoder().fit_transform(X[col])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            st.success(f"Precisión en test: {clf.score(X_test, y_test):.2f}")

            st.write("### Predicción de nueva muestra")
            user_input = {}
            for col in X.columns:
                val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
                user_input[col] = val
            if st.button("Predecir especie"):
                sample = pd.DataFrame([user_input])
                pred = clf.predict(sample)[0]
                st.success(f"🔮 La especie predicha es: **{pred}**")

        # ====== Resumen EDA para LLM ======
        eda_summary = f"""
        El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.
        Columnas: {list(df.columns)}.
        Columnas numéricas: {list(numeric_cols)}.
        Columnas categóricas: {list(cat_cols)}.
        Valores nulos detectados: {df.isnull().sum().to_dict()}.
        Estadísticas principales:
        {df.describe(include='all').to_dict()}.
        """

        # ====== Chat basado en dataset + modelo ======
        st.subheader("💬 Chat con el agente sobre el dataset")
        question = st.text_input("Escribe tu pregunta sobre el dataset o una especie")

        if question:
            dataset_summary = f"Resumen del dataset: {eda_summary}"
            if clf:
                dataset_summary += f" Clasificador RandomForest entrenado con precisión {clf.score(X_test, y_test):.2f}."

            # Buscar coincidencias en columna Especie
            matched_rows = None
            if "Especie" in df.columns:
                matched_rows = df[df["Especie"].str.contains(question, case=False, na=False)]

            example_text = ""
            if matched_rows is not None and not matched_rows.empty:
                st.write("### Ejemplos encontrados en el dataset:")
                st.dataframe(matched_rows.head(3))
                example_text = "\n\n".join([
                    " | ".join([f"{col}: {row[col]}" for col in df.columns])
                    for _, row in matched_rows.head(3).iterrows()
                ])

            context_text = f"""
            {dataset_summary}

            Ejemplos relevantes:
            {example_text if example_text else "No se encontraron ejemplos exactos en el dataset."}

            Pregunta del usuario: {question}
            """

            answer = llm.answer_concepts_or_process(
                question,
                [{"title": "dataset", "text": context_text, "score": 1.0}],
                mode="qa"
            )
            st.success(answer)



