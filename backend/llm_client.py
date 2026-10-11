"""Provider switch for OpenAI-compatible and Gemini chat models."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

import httpx
import openai

DEFAULT_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com"


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
        provider: str = "auto",
    ):
        self.model = model
        if provider in ("openai", "gemini"):
            self.provider = provider
        else:
            self.provider = "gemini" if is_gemini_model(model) else "openai"

        if self.provider == "gemini":
            self._init_gemini(gemini_api_key or openai_api_key, gemini_api_endpoint)
        else:
            self._init_openai(openai_api_key, openai_base_url)

    def _init_openai(self, api_key: str, base_url: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI-compatible models.")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url or None, timeout=60.0, max_retries=1)

    def _init_gemini(self, api_key: str, api_endpoint: str) -> None:
        # Gemini REST API called directly; the google-generativeai SDK is deprecated.
        if not api_key:
            raise ValueError("GEMINI_API_KEY or OPENAI_API_KEY is required for Gemini models.")
        endpoint = (api_endpoint or DEFAULT_GEMINI_ENDPOINT).rstrip("/")
        model = self.model.strip().removeprefix("models/")
        self._gemini_url = f"{endpoint}/v1beta/models/{model}"
        self.client = httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0), headers={"x-goog-api-key": api_key})

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

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield the generated text in pieces as the model produces it."""
        if self.provider == "gemini":
            body = self._gemini_body(messages, temperature, max_tokens)
            with self.client.stream("POST", f"{self._gemini_url}:streamGenerateContent", params={"alt": "sse"}, json=body) as response:
                if response.is_error:
                    response.read()
                    _raise_gemini_error(response)
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        text = _gemini_text(json.loads(line[6:]))
                        if text:
                            yield text
            return

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        for chunk in self.client.chat.completions.create(**kwargs):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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
        response = self.client.post(f"{self._gemini_url}:generateContent", json=self._gemini_body(messages, temperature, max_tokens))
        if response.is_error:
            _raise_gemini_error(response)
        return _gemini_text(response.json())

    def _gemini_body(self, messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int]) -> Dict[str, Any]:
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system" and m.get("content"))
        contents = [
            {"role": "model" if m.get("role") == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
            if m.get("role") != "system" and m.get("content")
        ]
        generation_config: Dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        body: Dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body


def _gemini_text(data: Dict[str, Any]) -> str:
    """Answer text from a Gemini response, skipping 'thought' parts of thinking models."""
    parts = []
    for candidate in data.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if part.get("text") and not part.get("thought"):
                parts.append(part["text"])
    return "".join(parts)


def _raise_gemini_error(response: httpx.Response) -> None:
    try:
        message = response.json().get("error", {}).get("message") or response.text
    except ValueError:
        message = response.text
    raise RuntimeError(f"Gemini API error {response.status_code}: {message[:300]}")
