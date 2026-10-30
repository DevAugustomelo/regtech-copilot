"""Ollama LLM provider implementation."""

from typing import Any

import httpx

from regtech_copilot.llm.base import (
    GenerationOptions,
    LLMResponse,
)


class OllamaProviderError(Exception):
    """Base exception for Ollama provider errors."""


class OllamaConnectionError(OllamaProviderError):
    """Raised when the Ollama service cannot be reached."""


class OllamaResponseError(OllamaProviderError):
    """Raised when Ollama returns an unexpected response."""


class OllamaClient:
    """HTTP client for generating text with Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Ollama base URL cannot be empty.")

        if not model.strip():
            raise ValueError("Ollama model cannot be empty.")

        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self._owns_client = http_client is None

        self._http_client = http_client or httpx.AsyncClient(
            timeout=timeout,
        )

    async def generate(
        self,
        prompt: str,
        *,
        options: GenerationOptions,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response using the Ollama chat API."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages: list[dict[str, str]] = []

        if system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": options.temperature,
                "top_p": options.top_p,
                "num_predict": options.max_tokens,
            },
        }

        try:
            response = await self._http_client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            raise OllamaProviderError(
                f"Ollama returned HTTP status {exc.response.status_code}."
            ) from exc

        except httpx.RequestError as exc:
            raise OllamaConnectionError(
                "Could not communicate with the Ollama service."
            ) from exc

        try:
            data: Any = response.json()

            if not isinstance(data, dict):
                raise ValueError("Response must be a JSON object.")

            message = data.get("message")

            if not isinstance(message, dict):
                raise ValueError("Response does not contain a valid message.")

            content = message.get("content")

            if not isinstance(content, str):
                raise ValueError("Response message does not contain valid content.")

            input_tokens = data.get("prompt_eval_count")
            output_tokens = data.get("eval_count")

            if input_tokens is not None and not isinstance(input_tokens, int):
                raise ValueError("Invalid input token count.")

            if output_tokens is not None and not isinstance(output_tokens, int):
                raise ValueError("Invalid output token count.")

        except (ValueError, TypeError) as exc:
            raise OllamaResponseError("Ollama returned an invalid response.") from exc

        return LLMResponse(
            content=content,
            provider="ollama",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def aclose(self) -> None:
        """Close the HTTP client if this instance owns it."""

        if self._owns_client:
            await self._http_client.aclose()
