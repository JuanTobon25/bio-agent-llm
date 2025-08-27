# 🧬 **Agente LLM de Biología** <small>(Streamlit · Groq)</small>

## ✨ **Qué hace**
- 🦋 **Identifica especies** por descripción  
  → Búsqueda semántica (FastEmbed) + **re-ranking** con Llama-3 (Groq), explicación y confianza.
- 📚 **Conceptos y procesos (RAG)**  
  → Q&A y explicaciones usando **solo** `kb/concepts.jsonl` (muestra *Fuentes*).

## 🧠 **Modelos & arquitectura**
- 🤖 **LLM:** Groq (`llama3-70b-8192` / `llama3-8b-8192`)
- 🔎 **Embeddings:** FastEmbed multilingüe + índice **NumPy** (coseno)
- 🎯 **Top-K por defecto:** **8**  
- 🧹 **Sin** FLAN ni OCR (build liviano)

## 🗂 **Estructura**
├── app.py # UI de Streamlit (2 pestañas)

├── agent.py # Cliente Groq + prompts + re-ranking de especies

├── tools.py # FastEmbed + índice NumPy y utilidades de KB

├── kb/

│ ├── concepts.jsonl # Conceptos (RAG) — un JSON por línea

│ └── species.jsonl # Especies — un JSON por línea

├── assets/

│ └── mono.jpg # Imagen de la sidebar (con marco/sombra)

├── requirements.txt

└── README.md

## 🚀 Cómo correr (local)

1) Instala dependencias:
```bash
pip install -r requirements.txt
