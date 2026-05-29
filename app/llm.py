"""LLM providers behind a tiny interface.

`StubLLM` is deterministic and offline so the answer-generation step is
testable without network or cost. Real providers are imported lazily and
used only when a matching key is configured.
"""
from __future__ import annotations

from typing import Protocol

from .config import Settings

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say you "
    "don't know. Be concise."
)


class LLMProvider(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


def build_prompt(question: str, contexts: list[str]) -> str:
    """Assemble a grounded RAG prompt from retrieved context chunks."""
    joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        f"Context:\n{joined}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


class StubLLM:
    """Deterministic offline 'LLM'.

    Returns a grounded, predictable answer derived from the supplied context
    so the pipeline and API are fully testable. It surfaces the most relevant
    context sentence rather than hallucinating.
    """

    def complete(self, system: str, prompt: str) -> str:
        context = prompt.split("Question:", 1)[0]
        # Strip the leading "Context:" label and citation markers.
        body = context.replace("Context:", "", 1)
        for i in range(1, 30):
            body = body.replace(f"[{i}]", " ")
        sentences = [s.strip() for s in body.replace("\n", " ").split(".") if s.strip()]
        if not sentences:
            return "I don't know based on the provided context."
        return sentences[0] + "."


class AnthropicLLM:
    def __init__(self, api_key: str, model: str) -> None:
        import anthropic  # lazy

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text").strip()


class GeminiLLM:
    def __init__(self, api_key: str, model: str) -> None:
        import google.generativeai as genai  # lazy

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)

    def complete(self, system: str, prompt: str) -> str:
        resp = self._model.generate_content(prompt)
        return (resp.text or "").strip()


class OpenAILLM:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI  # lazy

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def get_llm(settings: Settings) -> LLMProvider:
    """Select an LLM provider, falling back to the offline stub if unavailable."""
    provider = settings.llm_provider
    try:
        if provider == "anthropic" and settings.anthropic_api_key:
            return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
        if provider == "gemini" and settings.gemini_api_key:
            return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
        if provider == "openai" and settings.openai_api_key:
            return OpenAILLM(settings.openai_api_key, settings.openai_model)
    except Exception:
        pass
    return StubLLM()
