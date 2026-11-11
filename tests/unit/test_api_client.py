import httpx
import pytest

from regtech_copilot.ui.api_client import APIClient


@pytest.fixture
def client():
    api_client = APIClient("http://testserver")
    api_client.close()
    yield api_client
    api_client.close()


def test_list_documents(client):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/documents"

        return httpx.Response(
            200,
            json={"total": 1, "documents": [{"filename": "lgpd.pdf"}]},
        )

    client._client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )

    result = client.list_documents()

    assert result["total"] == 1
    assert result["documents"][0]["filename"] == "lgpd.pdf"


def test_get_document(client):
    document_id = "035a58cb-536b-40b0-b01b-facba17d0144"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"/documents/{document_id}"

        return httpx.Response(
            200,
            json={"document_id": document_id, "filename": "lgpd.pdf"},
        )

    client._client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )

    result = client.get_document(document_id)

    assert result["document_id"] == document_id


def test_delete_document(client):
    document_id = "035a58cb-536b-40b0-b01b-facba17d0144"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == f"/documents/{document_id}"

        return httpx.Response(
            200,
            json={"document_id": document_id, "status": "deleted"},
        )

    client._client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )

    result = client.delete_document(document_id)

    assert result["status"] == "deleted"
