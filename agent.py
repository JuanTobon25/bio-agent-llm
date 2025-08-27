# agent.py (Groq-only, con re-ranking de especies y respuestas con fuentes)
from typing import List, Dict, Optional
import json

try:
    from groq import Groq
except Exception:
    Groq = None


class BioLLM:
    """
    Implementación liviana que usa exclusivamente Groq.
    Requiere GROQ_API_KEY. Si no está, la app se detiene en app.py.
    """

    def __init__(self, groq_api_key: str, groq_model: str = "llama3-70b-8192"):
        if Groq is None:
            raise ImportError("El paquete 'groq' no está instalado.")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY no configurada.")
        self.client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model

    def _chat(self, messages, max_tokens: int = 220, temperature: float = 0.1) -> str:
        cc = self.client.chat.completions.create(
            messages=messages,
            model=self.groq_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (cc.choices[0].message.content or "").strip()

    def generate(self, prompt: str, max_new_tokens: int = 220, temperature: float = 0.1) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a bilingual biology tutor. Detect the user's language (Spanish or English) "
                    "and answer in that language. Be concise, correct, and avoid hallucinations."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            return self._chat(messages, max_tokens=max_new_tokens, temperature=temperature)
        except Exception as e:
            return f"No pude completar con Groq: {e}"

    # ---------- RAG mejorado (con 'fuentes') ----------
    def answer_with_context(
        self,
        question: str,
        context_snippets: List[Dict],
        max_new_tokens: int = 220,
    ) -> str:
        """Responde usando SOLO el contexto y lista títulos como fuentes."""
        context = "\n\n".join([f"- {c['title']}: {c['text']}" for c in context_snippets])
        titles = ", ".join([c["title"] for c in context_snippets])
        prompt = (
            "Eres un tutor de Biología. Responde de forma breve, correcta y en el idioma de la pregunta, "
            "usando SOLO el contexto. Si no hay suficiente información, responde exactamente: "
            "'No estoy seguro con el contexto disponible.'\n\n"
            f"Contexto:\n{context}\n\nPregunta: {question}\n"
            f"Al final, añade una línea 'Fuentes: {titles}'.\n\nRespuesta:"
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    def answer_concepts_or_process(
        self,
        text: str,
        context_snippets: List[Dict],
        mode: str = "qa",
        max_new_tokens: int = 260,
    ) -> str:
        """
        Unifica Q&A (RAG) y Explicar proceso:
          - mode='qa': respuesta corta usando SOLO contexto.
          - mode='process': pasos numerados y estructuras clave, usando SOLO contexto si existe, o conocimiento general si no hay contexto.
        """
        context = "\n\n".join([f"- {c['title']}: {c['text']}" for c in context_snippets]) if context_snippets else ""
        titles = ", ".join([c["title"] for c in context_snippets]) if context_snippets else "—"
        if mode == "process":
            prompt = (
                "Eres un tutor de Biología. Explica el siguiente tema en pasos numerados, "
                "mencionando estructuras y funciones clave. Responde en el idioma de la entrada. "
                "Si hay contexto, úsalo estrictamente; si no lo hay, responde con conocimiento general.\n\n"
                f"Contexto (opcional):\n{context}\n\nTema: {text}\n"
                f"Al final, añade una línea 'Fuentes: {titles}'.\n\nExplicación:"
            )
        else:
            prompt = (
                "Eres un tutor de Biología. Responde de forma breve y correcta, "
                "usando SOLO el contexto disponible. Si no es suficiente, di: "
                "'No estoy seguro con el contexto disponible.'\n\n"
                f"Contexto:\n{context}\n\nPregunta: {text}\n"
                f"Al final, añade una línea 'Fuentes: {titles}'.\n\nRespuesta:"
            )
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    # ---------- Re-ranking por LLM para identificación de especies ----------
    def rerank_species(self, description: str, candidates: List[Dict], top_n: int = 5) -> Dict:
        """
        Reordena/valida candidatos con el LLM y devuelve:
          { "best": {scientific_name, common_names, reason}, "confidence": float, "notes": str }
        """
        subset = candidates[:top_n]
        table_lines = []
        for c in subset:
            line = (
                f"- name={c['scientific_name']}; common={', '.join(c['common_names'])}; "
                f"traits={c['match_explanation']}; taxonomy={c['taxonomy']}; similarity={c.get('similarity', 0):.3f}"
            )
            table_lines.append(line)
        table = "\n".join(table_lines)

        system = (
            "You are a careful species identifier. Compare the user description with each candidate and pick the single "
            "most plausible species. Penalize candidates with mismatching key traits (e.g., 'trunk' vs 'no trunk'). "
            "Return strictly a JSON object with keys: best (object with scientific_name, common_names, reason), "
            "confidence (0-1), notes."
        )
        user = (
            f"User description: {description}\n\nCandidates:\n{table}\n\n"
            "Respond in JSON only."
        )
        try:
            raw = self._chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=320,
                temperature=0.0,
            )
            data = json.loads(raw)
            return data
        except Exception:
            # Fallback simple si el JSON falla: elige el top-1 original
            top = subset[0]
            return {
                "best": {
                    "scientific_name": top["scientific_name"],
                    "common_names": top["common_names"],
                    "reason": "Elegido por máxima similitud del índice semántico.",
                },
                "confidence": float(top.get("similarity", 0.5)),
                "notes": "Fallo al parsear JSON del LLM; se usó el ranking semántico.",
            }


