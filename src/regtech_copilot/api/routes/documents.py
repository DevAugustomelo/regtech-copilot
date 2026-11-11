"""Document management and ingestion endpoints."""

import asyncio
import hashlib
import logging
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from regtech_copilot.api.dependencies import IngestionServiceDependency
from regtech_copilot.documents.catalog import (
    DocumentRecord,
    DuplicateDocumentError,
)
from regtech_copilot.ingestion.pdf_loader import PDFLoaderError
from regtech_copilot.ingestion.service import IngestionServiceError

router = APIRouter(tags=["Documents"])

logger = logging.getLogger(__name__)

UPLOAD_CHUNK_SIZE = 1024 * 1024
PDF_SIGNATURE = b"%PDF-"


class DocumentUploadResponse(BaseModel):
    """Response schema for a successfully ingested document."""

    document_id: str
    filename: str
    page_count: int
    chunks_created: int
    embedding_model: str
    embedding_dimensions: int


class DocumentSummaryResponse(BaseModel):
    """Public metadata for an indexed document."""

    document_id: str
    filename: str
    page_count: int | None
    chunks_created: int
    created_at: str
    file_exists: bool


class DocumentListResponse(BaseModel):
    """Response containing all indexed documents."""

    total: int
    documents: list[DocumentSummaryResponse]


class DocumentDeleteResponse(BaseModel):
    """Confirmation returned after document deletion."""

    document_id: str
    status: str


def _sanitize_filename(filename: str | None) -> str:
    """Extract a safe filename without accepting directory components."""

    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )

    safe_filename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]

    if not safe_filename or not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    return safe_filename


async def _save_upload(
    upload: UploadFile,
    destination: Path,
    max_size_bytes: int,
) -> None:
    """Stream an uploaded PDF to disk while enforcing its size limit."""

    total_size = 0
    first_chunk = True

    with destination.open("wb") as file:
        while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
            if first_chunk:
                first_chunk = False

                if not chunk.startswith(PDF_SIGNATURE):
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail="The uploaded file is not a valid PDF.",
                    )

            total_size += len(chunk)

            if total_size > max_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="The uploaded file exceeds the maximum allowed size.",
                )

            file.write(chunk)

    if total_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )


def _calculate_sha256(path: Path) -> str:
    """Calculate a file SHA-256 without loading it entirely into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(UPLOAD_CHUNK_SIZE), b""):
            digest.update(block)

    return digest.hexdigest()


def _document_summary(record: DocumentRecord) -> DocumentSummaryResponse:
    """Convert a catalog record into a public API response."""

    return DocumentSummaryResponse(
        document_id=record.document_id,
        filename=record.filename,
        page_count=record.page_count,
        chunks_created=record.chunks_created,
        created_at=record.created_at,
        file_exists=Path(record.source_path).is_file(),
    )


async def _rollback_ingestion(request: Request, document_id: str) -> None:
    """Remove partially registered vector and lexical indexes."""

    identifier = UUID(document_id)

    await asyncio.to_thread(
        request.app.state.vector_store.delete,
        identifier,
    )
    await asyncio.to_thread(
        request.app.state.lexical_store.delete,
        identifier,
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List indexed documents",
)
async def list_documents(request: Request) -> DocumentListResponse:
    """Return all documents registered in the catalog."""

    records = await asyncio.to_thread(request.app.state.document_catalog.list_all)

    documents = [_document_summary(record) for record in records]

    return DocumentListResponse(
        total=len(documents),
        documents=documents,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentSummaryResponse,
    summary="Get document details",
)
async def get_document(
    document_id: UUID,
    request: Request,
) -> DocumentSummaryResponse:
    """Return metadata for a single indexed document."""

    record = await asyncio.to_thread(
        request.app.state.document_catalog.get,
        str(document_id),
    )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return _document_summary(record)


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a regulatory PDF",
)
async def upload_document(
    request: Request,
    ingestion_service: IngestionServiceDependency,
    file: Annotated[UploadFile, File()],
) -> DocumentUploadResponse:
    """Upload a PDF and execute the complete ingestion pipeline."""

    filename = _sanitize_filename(file.filename)

    settings = request.app.state.settings
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    upload_directory = Path(settings.documents_path) / str(uuid4())
    upload_directory.mkdir(parents=True, exist_ok=False)

    destination = upload_directory / filename
    ingestion_result = None
    registered = False

    try:
        await _save_upload(
            upload=file,
            destination=destination,
            max_size_bytes=max_size_bytes,
        )

        async with request.app.state.document_ingestion_lock:
            sha256 = await asyncio.to_thread(_calculate_sha256, destination)

            existing = await asyncio.to_thread(
                request.app.state.document_catalog.get_by_sha256,
                sha256,
            )

            if existing is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "message": "This document is already indexed.",
                        "document_id": existing.document_id,
                        "filename": existing.filename,
                    },
                )

            ingestion_result = await ingestion_service.ingest(destination)

            await asyncio.to_thread(
                request.app.state.document_catalog.register,
                document_id=ingestion_result.document_id,
                filename=filename,
                source_path=str(destination.resolve()),
                sha256=sha256,
                page_count=ingestion_result.page_count,
                chunks_created=ingestion_result.chunks_created,
                embedding_model=ingestion_result.embedding_model,
                embedding_dimensions=ingestion_result.embedding_dimensions,
            )
            registered = True

        return DocumentUploadResponse(
            document_id=ingestion_result.document_id,
            filename=filename,
            page_count=ingestion_result.page_count,
            chunks_created=ingestion_result.chunks_created,
            embedding_model=ingestion_result.embedding_model,
            embedding_dimensions=ingestion_result.embedding_dimensions,
        )

    except DuplicateDocumentError as exc:
        if ingestion_result is not None:
            await _rollback_ingestion(request, ingestion_result.document_id)

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This document is already indexed.",
        ) from exc

    except PDFLoaderError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The PDF could not be processed. Check that it is valid, "
            "unencrypted, and contains extractable text.",
        ) from exc

    except IngestionServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document ingestion failed.",
        ) from exc

    except Exception:
        if ingestion_result is not None:
            try:
                await _rollback_ingestion(
                    request,
                    ingestion_result.document_id,
                )
            except Exception:
                logger.exception(
                    "Failed to roll back document %s after upload failure.",
                    ingestion_result.document_id,
                )

        raise

    finally:
        if not registered:
            shutil.rmtree(upload_directory, ignore_errors=True)

        await file.close()


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete an indexed document",
)
async def delete_document(
    document_id: UUID,
    request: Request,
) -> DocumentDeleteResponse:
    """Delete a document from all indexes and its managed PDF file."""

    catalog = request.app.state.document_catalog

    record = await asyncio.to_thread(
        catalog.get,
        str(document_id),
    )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    documents_root = Path(request.app.state.settings.documents_path).resolve()
    source_path = Path(record.source_path).resolve()

    if not source_path.is_relative_to(documents_root):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document file is outside the managed documents directory.",
        )

    await asyncio.to_thread(
        request.app.state.vector_store.delete,
        document_id,
    )
    await asyncio.to_thread(
        request.app.state.lexical_store.delete,
        document_id,
    )

    if source_path.is_file():
        await asyncio.to_thread(source_path.unlink)

    parent_directory = source_path.parent

    if parent_directory != documents_root:
        with suppress(OSError):
            await asyncio.to_thread(parent_directory.rmdir)

    await asyncio.to_thread(catalog.delete, str(document_id))

    return DocumentDeleteResponse(
        document_id=str(document_id),
        status="deleted",
    )
