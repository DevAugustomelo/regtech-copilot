"""Token-based text chunking with page and legal article tracking."""

import re

import tiktoken

from regtech_copilot.core.config import Settings, get_settings
from regtech_copilot.schemas.documents import Document, DocumentChunk


class TextChunkerError(Exception):
    """Base exception for text chunking errors."""


class TextChunker:
    """Split documents into overlapping token-based chunks."""

    ARTICLE_PATTERN = re.compile(
        r"(?m)^[^\S\r\n]*Art\.[^\S\r\n]*"
        r"([0-9]+(?:-[A-Z])?)(?:[º°o]\.?|\.)[^\S\r\n]+"
    )

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

    def _split_into_article_sections(
        self,
        content: str,
        active_article: str | None,
    ) -> tuple[list[tuple[str, str | None, bool]], str | None]:
        """Split a page at article headings and track article continuations."""

        matches = list(self.ARTICLE_PATTERN.finditer(content))

        if not matches:
            return [
                (content, active_article, active_article is not None)
            ], active_article

        sections: list[tuple[str, str | None, bool]] = []
        cursor = 0

        for index, match in enumerate(matches):
            if match.start() > cursor:
                preceding_content = content[cursor : match.start()]

                if preceding_content.strip():
                    sections.append(
                        (
                            preceding_content,
                            active_article,
                            active_article is not None,
                        )
                    )

            article_number = match.group(1)
            next_match = matches[index + 1] if index + 1 < len(matches) else None

            section_end = next_match.start() if next_match else len(content)

            sections.append(
                (
                    content[match.start() : section_end],
                    article_number,
                    False,
                )
            )

            active_article = article_number
            cursor = section_end

        return sections, active_article

    def _chunk_content(
        self,
        document: Document,
        page_number: int,
        content: str,
        article_reference: str | None = None,
        is_continuation: bool = False,
    ) -> list[DocumentChunk]:
        """Split one page section into token-limited chunks."""

        if article_reference is not None:
            section_type = "continuação" if is_continuation else "trecho"
            context_prefix = (
                "Contexto estrutural gerado pelo sistema "
                f"(não integra o texto legal): {section_type} do "
                f"Art. {article_reference}{'º' if article_reference.isdigit() else ''}."
            )
            content = f"{context_prefix}\n\n{content}"

        tokens = self.encoding.encode(content)

        if not tokens:
            return []

        offsets = self._get_token_offsets(content)
        chunks: list[DocumentChunk] = []
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
                raise TextChunkerError("Unable to generate a valid non-empty chunk.")

            metadata = {
                **document.metadata,
                "filename": document.filename,
                "token_count": len(self.encoding.encode(chunk_content)),
            }

            if article_reference is not None:
                metadata["article_reference"] = article_reference
                metadata["article_continuation"] = is_continuation

            chunks.append(
                DocumentChunk(
                    document_id=document.document_id,
                    content=chunk_content,
                    chunk_index=0,
                    page_number=page_number,
                    metadata=metadata,
                )
            )

            if end == len(tokens):
                break

            next_start = end - self.chunk_overlap

            if next_start <= start:
                next_start = start + 1

            start = next_start

        return chunks

    def chunk(self, document: Document) -> list[DocumentChunk]:
        """Split a document while preserving page and article references."""

        if document.pages:
            page_contents = [
                (page.page_number, page.content) for page in document.pages
            ]
        else:
            page_contents = [(1, document.content)]

        has_article_markers = any(
            self.ARTICLE_PATTERN.search(content) for _, content in page_contents
        )

        chunks: list[DocumentChunk] = []
        active_article: str | None = None

        for page_number, content in page_contents:
            if not content.strip():
                continue

            if has_article_markers:
                sections, active_article = self._split_into_article_sections(
                    content,
                    active_article,
                )

                for section_content, article_reference, is_continuation in sections:
                    chunks.extend(
                        self._chunk_content(
                            document=document,
                            page_number=page_number,
                            content=section_content,
                            article_reference=article_reference,
                            is_continuation=is_continuation,
                        )
                    )
            else:
                chunks.extend(
                    self._chunk_content(
                        document=document,
                        page_number=page_number,
                        content=content,
                    )
                )

        return [
            DocumentChunk(
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=index,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                metadata=chunk.metadata,
            )
            for index, chunk in enumerate(chunks)
        ]
