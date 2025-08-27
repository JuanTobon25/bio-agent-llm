# agent.py
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Groq es opcional
try:
    from groq import Groq  # pip install groq
except Exception:
    Groq = None

# Modelo local (gratuito) recomendado para CPU
MODEL_NAME = "google/flan-t5-small"


class BioLLM:
    """
    BioLLM soporta dos motores:
      - 'flan' (local, por defecto): usa google/flan-t5-small en CPU
      - 'groq' (opcional): usa la API de Groq (ej. llama3-8b-8192) si proporcionas GROQ_API_KEY

    Args:
        engine: 'flan' o 'groq'
        groq_api_key: tu API Key de Groq (si usas engine='groq')
        groq_model: nombre del modelo en Groq (p. ej., 'llama3-8b-8192')
    """

    def __init__(
        self,
        engine: str = "flan",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama3-8b-8192",
    ):
        self.engine = "flan"  # por defecto
        self.client = None
        self.groq_model = groq_model

        # Intentar configurar Groq si el usuario lo pide y hay key
        if engine == "groq" and groq_api_key and Groq is not None:
            try:
                self.client = Groq(api_key=groq_api_key)
                self.engine = "groq"
            except Exception:
                # Si falla Groq, caemos a FLAN local sin romper la app
                self.engine = "flan"

        # Motor local (FLAN-T5) — forzar CPU + float32 para evitar errores de dtype
        if self.engine == "flan":
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,   # evita bf16/fp16 en CPU
                low_cpu_mem_usage=True
            )
            self.model.eval()
            # device=-1 → CPU siempre
            self.pipe = pipeline(
                task="text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1
            )

    # ---------- Rutas de generación ----------

    def _generate_groq(
        self,
        prompt: str,
        max_new_tokens: int = 220,
        temperature: float = 0.2,
    ) -> str:
        """Generación vía Groq (si está disponible)."""
        if self.client is None:
            # Fallback defensivo, no debería ocurrir si __init__ configuró bien
            return "No hay cliente Groq disponible. Cambia el motor a 'flan (local)'."

        try:
            cc = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.groq_model,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return (cc.choices[0].message.content or "").strip()
        except Exception as e:
            # Fallback suave para no romper la app si falla la API
            return f"No pude completar con Groq: {e}"

    def _generate_flan(self, prompt: str, max_new_tokens: int = 220) -> str:
        """Generación local con FLAN-T5-small en CPU."""
        out = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )[0]["generated_text"]
        return out.strip()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 220,
        temperature: float = 0.2,
    ) -> str:
        """Interfaz unificada de generación."""
        if self.engine == "groq" and self.client is not None:
            return self._generate_groq(prompt, max_new_tokens, temperature)
        return self._generate_flan(prompt, max_new_tokens)

    # ---------- Funciones de alto nivel para la app ----------

    def answer_with_context(
        self,
        question: str,
        context_snippets: List[Dict],
        max_new_tokens: int = 220
    ) -> str:
        """
        Responde usando SOLO el contexto (RAG).
        Si no hay suficiente info, debe decirlo explícitamente.
        """
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
        """
        Explica un proceso biológico en pasos numerados.
        """
        prompt = (
            f"Explica en pasos numerados el proceso biológico '{topic}' al nivel {level}. "
            "Sé conciso, menciona estructuras y funciones clave, y evita información no solicitada."
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens)

