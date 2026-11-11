"""HTTP client used by the Streamlit interface to communicate with the API."""

from typing import Any

import httpx


class APIClientError(Exception):
    """Raised when the backend API cannot complete a request."""


class APIClient:
    """Synchronous HTTP client for the RegTech Copilot API."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
        upload_timeout: float = 900.0,
    ) -> None:
        """Initialize the API client."""
        self._upload_timeout = upload_timeout

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an HTTP request and normalize API errors."""
        try:
            response = self._client.request(
                method,
                endpoint,
                **kwargs,
            )
            response.raise_for_status()

            data = response.json()

            if not isinstance(data, dict):
                raise APIClientError(
                    "A API retornou uma resposta em formato inesperado."
                )

            return data

        except httpx.TimeoutException as exc:
            raise APIClientError(
                "A API demorou demais para responder. "
                "Verifique o processamento e tente novamente."
            ) from exc

        except httpx.HTTPStatusError as exc:
            try:
                error_data = exc.response.json()
                detail = error_data.get("detail", "Erro na API.")
            except (ValueError, AttributeError):
                detail = "Erro na API."

            raise APIClientError(
                f"Erro HTTP {exc.response.status_code}: {detail}"
            ) from exc

        except httpx.RequestError as exc:
            raise APIClientError(
                "Não foi possível conectar à API. "
                "Verifique se o servidor FastAPI está em execução."
            ) from exc

    def health_check(self) -> dict[str, Any]:
        """Check whether the backend API is available."""
        return self._request("GET", "/health")

    def upload_document(
        self,
        filename: str,
        file_content: bytes,
    ) -> dict[str, Any]:
        """Upload a PDF and execute the document ingestion pipeline."""
        return self._request(
            "POST",
            "/documents",
            files={
                "file": (
                    filename,
                    file_content,
                    "application/pdf",
                )
            },
            timeout=httpx.Timeout(self._upload_timeout, connect=10.0),
        )

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Submit a question to the RAG pipeline."""
        payload: dict[str, Any] = {"query": question}

        if top_k is not None:
            payload["top_k"] = top_k

        return self._request(
            "POST",
            "/query",
            json=payload,
        )

    def list_documents(self) -> dict[str, Any]:
        """List all indexed documents."""
        return self._request("GET", "/documents")

    def get_document(self, document_id: str) -> dict[str, Any]:
        """Retrieve details for one indexed document."""
        return self._request("GET", f"/documents/{document_id}")

    def delete_document(self, document_id: str) -> dict[str, Any]:
        """Delete a document and its associated indexes."""
        return self._request("DELETE", f"/documents/{document_id}")
