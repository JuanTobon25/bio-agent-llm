# agent.py (Groq-only)
from typing import List, Dict, Optional

try:
    from groq import Groq
except Exception as e:
    Groq = None


class BioLLM:
    """
    Implementación liviana que usa exclusivamente Groq.
    Requiere GROQ_API_KEY. Si no está, debes manejar el error en app.py.
    """

    def __init__(self, groq_api_key: str, groq_model: str = "llama3-8b-8192"):
        if Groq is None:
            raise ImportError("El paquete 'groq' no está instalado.")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY no configurada.")
        self.client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model

    def generate(self, prompt: str, max_new_tokens: int = 220, temperature: float = 0.2) -> str:
        try:
            cc = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.groq_model,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return (cc.choices[0].message.content or "").strip()
        except Exception as e:
            return f"No pude completar con Groq: {e}"

    # ---------- Funciones de alto nivel ----------

    def answer_with_context(
        self,
        question: str,
        context_snippets: List[Dict],
        max_new_tokens: int = 220
    ) -> str:
        context = "\n\n".join([f"- {c['title']}: {c['text']}" for c in context_snippets])
        prompt = (
            "Eres un tutor de Biología. Responde de forma breve, correcta y en español, "
            "usando SOLO el contexto. Si no hay información suficiente, responde exactamente: "
            "'No estoy seguro con el contexto disponible.'\n\n"
            f"Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    def explain_process_steps(
        self,
        topic: str,
        level: str = "universitario",
        max_new_tokens: int = 260
    ) -> str:
        prompt = (
            f"Explica en pasos numerados el proceso biológico '{topic}' al nivel {level}. "
            "Sé conciso, menciona estructuras y funciones clave, y evita información no solicitada."
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens)

