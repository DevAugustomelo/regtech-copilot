"""Tests for the PDF document loader."""

import pymupdf
import pytest

from regtech_copilot.ingestion.pdf_loader import (
    PDFEmptyError,
    PDFEncryptedError,
    PDFFileNotFoundError,
    PDFLoader,
    PDFLoaderError,
)


@pytest.fixture
def loader() -> PDFLoader:
    """Return a PDF loader instance."""
    return PDFLoader()


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a sample PDF with two pages."""

    file_path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        first_page = pdf.new_page()
        first_page.insert_text(
            (72, 72),
            "First page regulatory content.",
        )

        second_page = pdf.new_page()
        second_page.insert_text(
            (72, 72),
            "Second page regulatory content.",
        )

        pdf.save(file_path)

    return file_path


def test_load_extracts_pdf_content(
    loader: PDFLoader,
    sample_pdf,
) -> None:
    document = loader.load(sample_pdf)

    assert document.filename == "sample.pdf"
    assert "First page regulatory content." in document.content
    assert "Second page regulatory content." in document.content
    assert document.page_count == 2


def test_load_preserves_page_numbers(
    loader: PDFLoader,
    sample_pdf,
) -> None:
    document = loader.load(sample_pdf)

    assert len(document.pages) == 2
    assert document.pages[0].page_number == 1
    assert document.pages[1].page_number == 2


def test_load_preserves_page_content(
    loader: PDFLoader,
    sample_pdf,
) -> None:
    document = loader.load(sample_pdf)

    assert "First page" in document.pages[0].content
    assert "Second page" in document.pages[1].content


def test_load_preserves_document_metadata(
    loader: PDFLoader,
    sample_pdf,
) -> None:
    document = loader.load(sample_pdf)

    assert document.metadata["format"] == "pdf"
    assert document.metadata["source"]


def test_load_raises_when_file_does_not_exist(
    loader: PDFLoader,
    tmp_path,
) -> None:
    file_path = tmp_path / "missing.pdf"

    with pytest.raises(PDFFileNotFoundError):
        loader.load(file_path)


def test_load_rejects_directory(
    loader: PDFLoader,
    tmp_path,
) -> None:
    with pytest.raises(PDFLoaderError, match="not a file"):
        loader.load(tmp_path)


def test_load_rejects_empty_pdf(
    loader: PDFLoader,
    tmp_path,
) -> None:
    file_path = tmp_path / "empty.pdf"

    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(file_path)

    with pytest.raises(PDFEmptyError):
        loader.load(file_path)


def test_load_rejects_encrypted_pdf(
    loader: PDFLoader,
    tmp_path,
) -> None:
    file_path = tmp_path / "encrypted.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Protected content.")

        pdf.save(
            file_path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )

    with pytest.raises(PDFEncryptedError):
        loader.load(file_path)
