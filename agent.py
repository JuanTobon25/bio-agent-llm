# agent.py — Groq-only, con re-ranking y calibración de confianza
from typing import List, Dict
import json

try:
    from groq import Groq
except Exception:
    Groq = None


def calibrate_confidence(llm_conf: float, top_similarity: float) -> float:
    """
    Combina la confianza del LLM con la similitud del top-1 del índice híbrido.
    Rango forzado [0,1]. Ajusta los pesos si lo deseas.
    """
    llm_conf = float(max(0.0, min(1.0, llm_conf)))
    top_similarity = float(max(0.0, min(1.0, top_similarity)))
    mixed = 0.65 * llm_conf + 0.35 * top_similarity
    return float(max(0.0, min(1.0, mixed)))


class BioLLM:
    """
    Cliente Groq. Requiere GROQ_API_KEY y un modelo (por defecto llama3-70b-8192).
    """
    def __init__(self, groq_api_key: str, groq_model: str = "llama3-70b-8192"):
        if Groq is None:
            raise ImportError("El paquete 'groq' no está instalado.")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY no configurada.")
        self.client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model

    # ---------------- Core chat ----------------
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

    # ---------------- RAG / Conceptos & Procesos ----------------
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
          - mode='qa': respuesta breve con SOLO el contexto.
          - mode='process': pasos numerados; usa contexto si lo hay, si no, conocimiento general.
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

    # ---------------- Re-ranking de especies ----------------
    def rerank_species(self, description: str, candidates: List[Dict], top_n: int = 5) -> Dict:
        """
        Reordena/valida candidatos con el LLM y devuelve:
          { "best": {scientific_name, common_names, reason}, "confidence": float, "notes": str }
        Además: ajusta 'confidence' mezclando señal del LLM con la similitud del top-1 del índice.
        """
        if not candidates:
            return {"best": {}, "confidence": 0.0, "notes": "No hay candidatos para reordenar."}

        subset = candidates[: max(1, min(top_n, len(candidates)))]
        table_lines = []
        for c in subset:
            common = ", ".join(c.get("common_names", [])) if isinstance(c.get("common_names"), list) else str(c.get("common_names", ""))
            line = (
                f"- name={c.get('scientific_name','')}; common={common}; "
                f"traits={c.get('match_explanation','')}; taxonomy={c.get('taxonomy','')}; "
                f"similarity={float(c.get('similarity', 0.0)):.3f}"
            )
            table_lines.append(line)
        table = "\n".join(table_lines)

        system = (
            "You are a careful species identifier. Compare the user description with each candidate and pick the single "
            "most plausible species. Penalize candidates with mismatching key traits. "
            "Return strictly a JSON object with keys: best (object with scientific_name, common_names, reason), "
            "confidence (0-1), notes. Reply in the same language as the user's description when writing 'reason' and 'notes'."
        )
        user = f"User description: {description}\n\nCandidates:\n{table}\n\nRespond in JSON only."

        try:
            raw = self._chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=380,
                temperature=0.0,
            )
            # Intentar extraer JSON válido
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("Respuesta sin JSON")
            data = json.loads(raw[start:end + 1])

            # Calibrar confianza con la similitud del top-1 del índice
            top_sim = float(candidates[0].get("similarity", 0.0))
            data["confidence"] = calibrate_confidence(float(data.get("confidence", 0.0)), top_sim)
            return data

        except Exception:
            # Fallback: top-1 del ranking semántico/híbrido
            top = subset[0]
            top_sim = float(top.get("similarity", 0.5))
            return {
                "best": {
                    "scientific_name": top.get("scientific_name", ""),
                    "common_names": top.get("common_names", []),
                    "reason": "Elegido por máxima similitud del índice (fallback).",
                },
                "confidence": calibrate_confidence(0.5, top_sim),
                "notes": "Fallo al parsear JSON del LLM; se usó el ranking del índice.",
            }


