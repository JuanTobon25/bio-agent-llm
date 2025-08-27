# Agente LLM de Biología (Streamlit)

## Capacidades
1. **Q&A con RAG local**: responde usando una KB propia (`kb/concepts.jsonl`).
2. **Identificar especies por descripción**: ranking semántico sobre `kb/species.jsonl`.
3. **Explicar procesos**: pasos claros (mitosis, ósmosis, etc.).
4. **OCR → Análisis**: extrae texto de imágenes y lo resume/analiza.

## Motores LLM
- **Local (por defecto):** FLAN-T5-small (gratuito).
- **Groq (opcional):** Llama-3 (seleccionable en Sidebar). Requiere `GROQ_API_KEY`.

## Cómo correr
```bash
pip install -r requirements.txt
streamlit run app.py
