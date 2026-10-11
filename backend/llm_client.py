"""Provider switch for OpenAI-compatible and Gemini chat models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import openai


def is_gemini_model(model: str) -> bool:
    """Return True when a model name should be served by Gemini."""
    normalized = model.strip().lower()
    if normalized.startswith("models/"):
        normalized = normalized.removeprefix("models/")
    return normalized.startswith("gemini")


class LLMClient:
    """Small text-generation wrapper used by the chatbot pipeline."""

    def __init__(
        self,
        model: str,
        openai_api_key: str = "",
        openai_base_url: str = "",
        gemini_api_key: str = "",
        gemini_api_endpoint: str = "",
    ):
        self.model = model
        self.provider = "gemini" if is_gemini_model(model) else "openai"

        if self.provider == "gemini":
            self._init_gemini(gemini_api_key or openai_api_key, gemini_api_endpoint)
        else:
            self._init_openai(openai_api_key, openai_base_url)

    def _init_openai(self, api_key: str, base_url: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI-compatible models.")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _init_gemini(self, api_key: str, api_endpoint: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY or OPENAI_API_KEY is required for Gemini models.")

        try:
            import google.generativeai as genai
            from google.api_core.client_options import ClientOptions
        except ImportError as exc:
            raise RuntimeError(
                "Gemini models require google-generativeai. "
                "Install it with: pip install -r requirements.txt"
            ) from exc

        genai.configure(
            api_key=api_key,
            transport="rest",
            client_options=ClientOptions(api_endpoint=api_endpoint),
        )
        self.client = genai.GenerativeModel(self.model)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from chat-style messages."""
        if self.provider == "gemini":
            return self._chat_gemini(messages, temperature=temperature, max_tokens=max_tokens)
        return self._chat_openai(messages, temperature=temperature, max_tokens=max_tokens)

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    def _chat_gemini(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        generation_config: Dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        response = self.client.generate_content(
            self._messages_to_prompt(messages),
            generation_config=generation_config,
        )
        return getattr(response, "text", "") or ""

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        labels = {
            "system": "System instructions",
            "user": "User",
            "assistant": "Assistant",
        }
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if content:
                parts.append(f"{labels.get(role, role.title())}:\n{content}")
        return "\n\n".join(parts)
