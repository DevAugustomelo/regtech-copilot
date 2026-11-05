"""RAG query endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from regtech_copilot.api.dependencies import GenerationServiceDependency
from regtech_copilot.generation.service import GenerationResult

router = APIRouter(tags=["Query"])


class QueryRequest(BaseModel):
    """Request schema for a RAG query."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        min_length=1,
        max_length=4_000,
        description="Question about the regulatory documents.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of document chunks to retrieve.",
    )


class QuerySource(BaseModel):
    """A source chunk used to generate the answer."""

    reference: str
    filename: str
    document_id: str
    chunk_id: str
    page_number: int | None
    distance: float
    excerpt: str


class QueryResponse(BaseModel):
    """Response schema for a RAG query."""

    answer: str
    sources: list[QuerySource]
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None


def _build_sources(result: GenerationResult) -> list[QuerySource]:
    """Convert retrieved chunks into API source references."""

    sources: list[QuerySource] = []

    for index, source in enumerate(result.sources, start=1):
        chunk = source.chunk
        filename = chunk.metadata.get("filename", "Documento sem nome")

        sources.append(
            QuerySource(
                reference=f"F{index}",
                filename=str(filename),
                document_id=str(chunk.document_id),
                chunk_id=str(chunk.chunk_id),
                page_number=chunk.page_number,
                distance=source.distance,
                excerpt=chunk.content,
            )
        )

    return sources


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about the indexed regulatory documents",
)
async def query_documents(
    payload: QueryRequest,
    generation_service: GenerationServiceDependency,
) -> QueryResponse:
    """Generate an evidence-grounded answer using RAG."""

    result = await generation_service.generate(
        payload.query,
        top_k=payload.top_k,
    )

    return QueryResponse(
        answer=result.answer,
        sources=_build_sources(result),
        provider=result.provider,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
