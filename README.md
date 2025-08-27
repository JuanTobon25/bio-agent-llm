🧬 Agente LLM de Biología (Streamlit · Groq)
✨ Qué hace

🦋 Identifica especies por descripción
→ Búsqueda semántica (FastEmbed) + re-ranking con Llama-3 (Groq), explicación y confianza.

📚 Conceptos y procesos (RAG)
→ Q&A y explicaciones usando solo kb/concepts.jsonl (muestra Fuentes).

🧠 Modelos & arquitectura

🤖 LLM: Groq (llama3-70b-8192 / llama3-8b-8192)

🔎 Embeddings: FastEmbed multilingüe + índice NumPy (coseno)

🎯 Top-K por defecto: 8

🧹 Sin FLAN ni OCR (build liviano)

🗂 Estructura
app.py        # UI (2 pestañas)
agent.py      # Groq + prompts + re-ranking
tools.py      # embeddings + índice + utilidades
kb/           # concepts.jsonl, species.jsonl
assets/mono.jpg
requirements.txt

🚀 Cómo correr

Streamlit Cloud → Settings → Secrets

GROQ_API_KEY = "tu_api_key_de_groq"


(ó escríbela en la sidebar)

(Opcional) 🖼️ Cambia la foto: coloca tu imagen en assets/mono.jpg.

Local

pip install -r requirements.txt
streamlit run app.py
