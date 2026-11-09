"""Document upload and ingestion endpoints."""

import shutil
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from regtech_copilot.api.dependencies import IngestionServiceDependency
from regtech_copilot.ingestion.pdf_loader import PDFLoaderError
from regtech_copilot.ingestion.service import IngestionServiceError

router = APIRouter(tags=["Documents"])

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

    try:
        await _save_upload(
            upload=file,
            destination=destination,
            max_size_bytes=max_size_bytes,
        )

        result = await ingestion_service.ingest(destination)

        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=filename,
            page_count=result.page_count,
            chunks_created=result.chunks_created,
            embedding_model=result.embedding_model,
            embedding_dimensions=result.embedding_dimensions,
        )

    except PDFLoaderError as exc:
        shutil.rmtree(upload_directory, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The PDF could not be processed. Check that it is valid, "
            "unencrypted, and contains extractable text.",
        ) from exc

    except IngestionServiceError as exc:
        shutil.rmtree(upload_directory, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document ingestion failed.",
        ) from exc

    except Exception:
        shutil.rmtree(upload_directory, ignore_errors=True)
        raise

    finally:
        await file.close()
