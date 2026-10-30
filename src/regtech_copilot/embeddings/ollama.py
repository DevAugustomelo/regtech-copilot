"""Ollama embedding provider implementation."""

import math
from typing import Any

import httpx

from regtech_copilot.embeddings.base import EmbeddingResponse


class OllamaEmbeddingError(Exception):
    """Base exception for Ollama embedding errors."""


class OllamaEmbeddingConnectionError(OllamaEmbeddingError):
    """Raised when the Ollama embedding service cannot be reached."""


class OllamaEmbeddingResponseError(OllamaEmbeddingError):
    """Raised when Ollama returns an unexpected embedding response."""


class OllamaEmbeddingClient:
    """HTTP client for generating embeddings with Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        expected_dimensions: int | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Ollama base URL cannot be empty.")

        if not model.strip():
            raise ValueError("Ollama embedding model cannot be empty.")

        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")

        if expected_dimensions is not None and expected_dimensions <= 0:
            raise ValueError("Expected dimensions must be greater than zero.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.expected_dimensions = expected_dimensions
        self._owns_client = http_client is None

        self._http_client = http_client or httpx.AsyncClient(
            timeout=timeout,
        )

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Generate embeddings for a collection of texts."""

        if not texts:
            raise ValueError("At least one text must be provided.")

        if any(not text.strip() for text in texts):
            raise ValueError("Texts cannot contain empty values.")

        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            response = await self._http_client.post(
                f"{self.base_url}/api/embed",
                json=payload,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            raise OllamaEmbeddingError(
                f"Ollama returned HTTP status {exc.response.status_code}."
            ) from exc

        except httpx.RequestError as exc:
            raise OllamaEmbeddingConnectionError(
                "Could not communicate with the Ollama embedding service."
            ) from exc

        try:
            data: Any = response.json()

            if not isinstance(data, dict):
                raise ValueError("Response must be a JSON object.")

            raw_embeddings = data.get("embeddings")

            if not isinstance(raw_embeddings, list) or not raw_embeddings:
                raise ValueError("Response does not contain valid embeddings.")

            if len(raw_embeddings) != len(texts):
                raise ValueError(
                    "Number of embeddings does not match number of input texts."
                )

            embeddings: list[list[float]] = []

            for vector in raw_embeddings:
                if not isinstance(vector, list) or not vector:
                    raise ValueError("Embedding vector cannot be empty.")

                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in vector
                ):
                    raise ValueError("Embedding vector contains invalid values.")

                embeddings.append([float(value) for value in vector])

            dimensions = len(embeddings[0])

            if any(len(vector) != dimensions for vector in embeddings):
                raise ValueError("Embedding vectors have inconsistent dimensions.")

        except (ValueError, TypeError) as exc:
            raise OllamaEmbeddingResponseError(
                "Ollama returned an invalid embedding response."
            ) from exc

        if (
            self.expected_dimensions is not None
            and dimensions != self.expected_dimensions
        ):
            raise OllamaEmbeddingResponseError(
                f"Expected {self.expected_dimensions} dimensions, "
                f"received {dimensions}."
            )
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model,
            dimensions=dimensions,
        )

    async def aclose(self) -> None:
        """Close the HTTP client if this instance owns it."""

        if self._owns_client:
            await self._http_client.aclose()
