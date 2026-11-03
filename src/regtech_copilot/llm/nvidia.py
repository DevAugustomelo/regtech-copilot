"""NVIDIA NIM LLM provider implementation."""

from typing import Any

import httpx

from regtech_copilot.llm.base import (
    GenerationOptions,
    LLMResponse,
)


class NvidiaProviderError(Exception):
    """Base exception for NVIDIA provider errors."""


class NvidiaAuthenticationError(NvidiaProviderError):
    """Raised when NVIDIA authentication fails."""


class NvidiaConnectionError(NvidiaProviderError):
    """Raised when the NVIDIA service cannot be reached."""


class NvidiaResponseError(NvidiaProviderError):
    """Raised when NVIDIA returns an unexpected response."""


class NvidiaClient:
    """HTTP client for generating text with NVIDIA NIM."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("NVIDIA API key cannot be empty.")

        if not base_url.strip():
            raise ValueError("NVIDIA base URL cannot be empty.")

        if not model.strip():
            raise ValueError("NVIDIA model cannot be empty.")

        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._owns_client = http_client is None

        self._http_client = http_client or httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def generate(
        self,
        prompt: str,
        *,
        options: GenerationOptions,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response using the NVIDIA NIM chat API."""

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
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_tokens": options.max_tokens,
            "stream": False,
        }

        try:
            response = await self._http_client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise NvidiaAuthenticationError(
                    "NVIDIA authentication failed. Check your API key."
                ) from exc

            raise NvidiaProviderError(
                f"NVIDIA returned HTTP status {exc.response.status_code}."
            ) from exc

        except httpx.RequestError as exc:
            raise NvidiaConnectionError(
                "Could not communicate with the NVIDIA service."
            ) from exc

        try:
            data: Any = response.json()

            if not isinstance(data, dict):
                raise ValueError("Response must be a JSON object.")

            choices = data.get("choices")

            if not isinstance(choices, list) or not choices:
                raise ValueError("Response does not contain valid choices.")

            first_choice = choices[0]

            if not isinstance(first_choice, dict):
                raise ValueError("Invalid response choice.")

            message = first_choice.get("message")

            if not isinstance(message, dict):
                raise ValueError("Response does not contain a valid message.")

            content = message.get("content")

            if not isinstance(content, str):
                raise ValueError("Response message does not contain valid content.")

            usage = data.get("usage", {})

            if not isinstance(usage, dict):
                raise ValueError("Invalid token usage information.")

            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")

            if input_tokens is not None and type(input_tokens) is not int:
                raise ValueError("Invalid input token count.")

            if output_tokens is not None and type(output_tokens) is not int:
                raise ValueError("Invalid output token count.")

            response_model = data.get("model", self.model)

            if not isinstance(response_model, str):
                raise ValueError("Invalid model information.")

        except (ValueError, TypeError) as exc:
            raise NvidiaResponseError("NVIDIA returned an invalid response.") from exc

        return LLMResponse(
            content=content,
            provider="nvidia",
            model=response_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def aclose(self) -> None:
        """Close the HTTP client if this instance owns it."""

        if self._owns_client:
            await self._http_client.aclose()
