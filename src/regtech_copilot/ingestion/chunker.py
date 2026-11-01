"""Token-based text chunking with page-level source tracking."""

import tiktoken

from regtech_copilot.core.config import Settings, get_settings
from regtech_copilot.schemas.documents import Document, DocumentChunk


class TextChunkerError(Exception):
    """Base exception for text chunking errors."""


class TextChunker:
    """Split documents into overlapping token-based chunks."""

    def __init__(
        self,
        settings: Settings | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        config = settings or get_settings()

        self.chunk_size = config.chunk_size_tokens
        self.chunk_overlap = config.chunk_overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")

        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

    def _get_token_offsets(self, text: str) -> list[int]:
        """Map token boundaries to character offsets in the original text."""

        tokens = self.encoding.encode(text)
        offsets = [0]
        byte_offset = 0
        text_bytes = text.encode("utf-8")

        for token in tokens:
            byte_offset += len(self.encoding.decode_single_token_bytes(token))

            try:
                character_offset = len(text_bytes[:byte_offset].decode("utf-8"))
            except UnicodeDecodeError:
                character_offset = len(
                    text_bytes[:byte_offset].decode("utf-8", errors="ignore")
                )

            offsets.append(character_offset)

        return offsets

    def chunk(self, document: Document) -> list[DocumentChunk]:
        """Split a document into chunks while preserving page references."""

        if document.pages:
            page_contents = [
                (page.page_number, page.content) for page in document.pages
            ]
        else:
            page_contents = [(1, document.content)]

        chunks: list[DocumentChunk] = []

        for page_number, content in page_contents:
            if not content.strip():
                continue

            tokens = self.encoding.encode(content)
            offsets = self._get_token_offsets(content)
            start = 0

            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))

                while True:
                    char_start = offsets[start]
                    char_end = offsets[end]
                    chunk_content = content[char_start:char_end].strip()

                    if len(self.encoding.encode(chunk_content)) <= self.chunk_size:
                        break

                    end -= 1

                if not chunk_content:
                    raise TextChunkerError(
                        "Unable to generate a valid non-empty chunk."
                    )

                chunks.append(
                    DocumentChunk(
                        document_id=document.document_id,
                        content=chunk_content,
                        chunk_index=len(chunks),
                        page_number=page_number,
                        metadata={
                            **document.metadata,
                            "filename": document.filename,
                            "token_count": len(self.encoding.encode(chunk_content)),
                        },
                    )
                )

                if end == len(tokens):
                    break

                next_start = end - self.chunk_overlap

                if next_start <= start:
                    next_start = start + 1

                start = next_start

        return chunks
