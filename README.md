🧬 Agente LLM de Biología (Streamlit • Groq-only)

Qué hace

🦋 Identificar especies por descripción: búsqueda semántica (FastEmbed) + re-ranking con Llama-3 (Groq) y explicación/confianza.

📚 Conceptos y procesos (RAG): Q&A y explicaciones usando solo el contexto de kb/concepts.jsonl (con Fuentes).

🖼️ UI con 2 pestañas y sidebar mínima (API key, modelo y foto con marco/sombra).

Modelos / Arquitectura

🤖 LLM: Groq (llama3-70b-8192 / llama3-8b-8192).

🔎 Embeddings: FastEmbed multilingüe + índice NumPy (coseno).

🎯 Top-K por defecto: 8. Sin FLAN ni OCR.

Estructura

app.py        # UI
agent.py      # cliente Groq + prompts + re-ranking
tools.py      # embeddings + índice + utilidades KB
kb/           # concepts.jsonl, species.jsonl
assets/mono.jpg
requirements.txt


Configurar y correr

🌐 Streamlit Cloud → Settings → Secrets:

GROQ_API_KEY = "tu_api_key_de_groq"


(ó escríbela en la sidebar).
2) 🖼️ Cambia la foto opcional: assets/mono.jpg.
3) ▶️ Local:

pip install -r requirements.txt
streamlit run app.py
