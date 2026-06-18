from __future__ import annotations

from app.models.schemas import ChunkRecord, ParsedDocument


class SectionAwareChunker:
    def __init__(self, target_chars: int = 1200) -> None:
        self.target_chars = target_chars

    def chunk(self, parsed: ParsedDocument) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for page in parsed.pages:
            paragraphs = [part.strip() for part in page.text.split("\n\n") if part.strip()]
            current = ""
            span_start = 0
            for paragraph in paragraphs:
                candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
                if len(candidate) <= self.target_chars:
                    current = candidate
                    continue
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

