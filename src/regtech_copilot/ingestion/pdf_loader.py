"""PDF document loader using PyMuPDF."""

from pathlib import Path

import pymupdf

from regtech_copilot.schemas.documents import Document, DocumentPage


class PDFLoaderError(Exception):
    """Base exception for PDF loading errors."""


class PDFFileNotFoundError(PDFLoaderError):
    """Raised when the PDF file does not exist."""


class PDFEncryptedError(PDFLoaderError):
    """Raised when the PDF requires a password."""


class PDFEmptyError(PDFLoaderError):
    """Raised when the PDF contains no extractable text."""


class PDFLoader:
    """Load PDF files and extract their text page by page."""

    def load(self, file_path: str | Path) -> Document:
        """Extract text and metadata from a PDF file."""

        path = Path(file_path)

        if not path.exists():
            raise PDFFileNotFoundError(f"PDF file not found: {path}")

        if not path.is_file():
            raise PDFLoaderError(f"Path is not a file: {path}")

        try:
            with pymupdf.open(path) as pdf:
                if pdf.needs_pass:
                    raise PDFEncryptedError(f"PDF requires a password: {path.name}")

                pages = tuple(
                    DocumentPage(
                        page_number=page_number,
                        content=page.get_text("text").strip(),
                    )
                    for page_number, page in enumerate(pdf, start=1)
                )

                content = "\n\n".join(page.content for page in pages if page.content)

                if not content.strip():
                    raise PDFEmptyError(
                        f"PDF contains no extractable text: {path.name}"
                    )

                return Document(
                    filename=path.name,
                    content=content,
                    page_count=len(pdf),
                    pages=pages,
                    metadata={
                        "source": str(path.resolve()),
                        "format": "pdf",
                    },
                )

        except PDFLoaderError:
            raise
        except Exception as exc:
            raise PDFLoaderError(f"Failed to load PDF '{path.name}': {exc}") from exc
