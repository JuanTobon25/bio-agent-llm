# agent.py
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

try:
    from groq import Groq
except Exception:
    Groq = None

MODEL_NAME = "google/flan-t5-small"

class BioLLM:
    """
    engine = "groq" (si tienes API Key) o "flan" (local por defecto)
    """
    def __init__(self, engine: str = "flan", groq_api_key: str = None, groq_model: str = "llama3-8b-8192"):
        self.engine = "flan"
        self.groq_model = groq_model
        self.client = None

        if engine == "groq" and groq_api_key and Groq is not None:
            try:
                self.client = Groq(api_key=groq_api_key)
                self.engine = "groq"
            except Exception:
                self.engine = "flan"

        if self.engine == "flan":
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def _generate_groq(self, prompt: str, max_new_tokens: int = 220, temperature: float = 0.2) -> str:
        cc = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.groq_model,
            temperature=temperature,
            max_tokens=max_new_tokens
        )
        return cc.choices[0].message.content.strip()

    def _generate_flan(self, prompt: str, max_new_tokens: int = 220) -> str:
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
        return out.strip()

    def generate(self, prompt: str, max_new_tokens: int = 220, temperature: float = 0.2) -> str:
        if self.engine == "groq" and self.client is not None:
            return self._generate_groq(prompt, max_new_tokens, temperature)
        return self._generate_flan(prompt, max_new_tokens)

    def answer_with_context(self, question: str, context_snippets: List[Dict], max_new_tokens: int = 220) -> str:
        context = "\n\n".join([f"- {c['title']}: {c['text']}" for c in context_snippets])
        prompt = (
            "Eres un tutor de Biología. Responde de manera breve, correcta y en español, "
            "usando SOLO el contexto. Si no hay información suficiente, di: "
            "'No estoy seguro con el contexto disponible.'\n\n"
            f"Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    def explain_process_steps(self, topic: str, level: str = "universitario", max_new_tokens: int = 260) -> str:
        prompt = (
            f"Explica en pasos numerados el proceso biológico '{topic}' al nivel {level}. "
            "Sé conciso, menciona estructuras y funciones clave, y evita información no solicitada."
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens)
