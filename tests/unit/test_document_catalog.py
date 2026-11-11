"""Essential integrity tests for the document catalog."""

from pathlib import Path

import pytest

from regtech_copilot.documents.catalog import (
    DocumentCatalog,
    DocumentCatalogError,
    DuplicateDocumentError,
)


@pytest.fixture
def catalog(tmp_path: Path) -> DocumentCatalog:
    """Create an isolated catalog for each test."""
    return DocumentCatalog(tmp_path / "documents.sqlite3")


@pytest.fixture
def document_data() -> dict[str, object]:
    """Provide valid metadata for a document."""
    return {
        "document_id": "document-001",
        "filename": "lgpd.pdf",
        "source_path": "/documents/lgpd.pdf",
        "sha256": "a" * 64,
        "page_count": 49,
        "chunks_created": 130,
        "embedding_model": "nomic-embed-text",
        "embedding_dimensions": 768,
    }


def test_register_and_retrieve_document(
    catalog: DocumentCatalog,
    document_data: dict[str, object],
) -> None:
    """Registration must support retrieval by ID and hash."""
    registered = catalog.register(**document_data)

    assert catalog.get(registered.document_id) == registered
    assert catalog.get_by_sha256("a" * 64) == registered
    assert catalog.list_all() == [registered]
    assert catalog.count() == 1


def test_register_rejects_duplicate_sha256(
    catalog: DocumentCatalog,
    document_data: dict[str, object],
) -> None:
    """The same file must not be registered under another ID."""
    catalog.register(**document_data)

    duplicate = {
        **document_data,
        "document_id": "document-002",
        "filename": "copy.pdf",
    }

    with pytest.raises(DuplicateDocumentError):
        catalog.register(**duplicate)

    assert catalog.count() == 1


def test_register_rejects_duplicate_document_id(
    catalog: DocumentCatalog,
    document_data: dict[str, object],
) -> None:
    """An existing document ID must not be overwritten."""
    catalog.register(**document_data)

    duplicate = {
        **document_data,
        "filename": "another-name.pdf",
        "sha256": "b" * 64,
    }

    with pytest.raises(DocumentCatalogError):
        catalog.register(**duplicate)

    assert catalog.count() == 1
    assert catalog.get("document-001").filename == "lgpd.pdf"


def test_delete_document_and_handle_unknown_id(
    catalog: DocumentCatalog,
    document_data: dict[str, object],
) -> None:
    """Deletion must report whether a record existed."""
    catalog.register(**document_data)

    assert catalog.delete("document-001") is True
    assert catalog.get("document-001") is None
    assert catalog.delete("document-001") is False
    assert catalog.count() == 0


def test_catalog_persists_after_reinitialization(
    tmp_path: Path,
    document_data: dict[str, object],
) -> None:
    """Registered documents must survive catalog reinitialization."""
    database_path = tmp_path / "documents.sqlite3"

    first_catalog = DocumentCatalog(database_path)
    registered = first_catalog.register(**document_data)

    second_catalog = DocumentCatalog(database_path)

    assert second_catalog.get("document-001") == registered
    assert second_catalog.count() == 1
