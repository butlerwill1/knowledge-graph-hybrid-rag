"""Split parsed pages into paragraph-aware, page-scoped chunk records."""

from __future__ import annotations

from app.models.schemas import ChunkRecord, ParsedDocument


class SectionAwareChunker:
    """Build page-scoped chunks while preserving paragraph boundaries."""

    def __init__(self, target_chars: int = 1200) -> None:
        """Set the preferred maximum size for combined paragraphs."""

        self.target_chars = target_chars

    def chunk(self, parsed: ParsedDocument) -> list[ChunkRecord]:
        """Split parsed pages into chunk records without crossing page boundaries."""

        chunks: list[ChunkRecord] = []
        for page in parsed.pages:
            # Blank lines are the only heading/paragraph signal currently used.
            paragraphs = [part.strip() for part in page.text.split("\n\n") if part.strip()]
            current = ""
            span_start = 0
            for paragraph in paragraphs:
                candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
                if len(candidate) <= self.target_chars:
                    current = candidate
                    continue
                # A single oversized paragraph remains intact; this branch only
                # flushes text that was accumulated before it.
                if current:
                    chunks.append(
                        ChunkRecord(
                            document_id=parsed.document.id,
                            page=page.page_number,
                            text=current,
                            span_start=span_start,
                            span_end=span_start + len(current),
                        )
                    )
                    # Spans describe the chunk sequence on this parsed page. They
                    # are provenance hints rather than offsets into the PDF bytes.
                    span_start += len(current)
                current = paragraph
            if current:
                chunks.append(
                    ChunkRecord(
                        document_id=parsed.document.id,
                        page=page.page_number,
                        text=current,
                        span_start=span_start,
                        span_end=span_start + len(current),
                    )
                )
        return chunks
