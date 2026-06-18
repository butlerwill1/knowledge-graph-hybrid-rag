from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.models.schemas import ChunkRecord, ClaimRecord, EntityRecord, GraphEdge


class ExtractedEntity(BaseModel):
    name: str
    label: str = "Entity"
    aliases: list[str] = Field(default_factory=list)
    evidence: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractedClaim(BaseModel):
    text: str
    evidence: str = ""
    entity_names: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    source_entity: str
    target_entity: str
    relation: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ChunkExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class LLMExtractor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.llm_extraction_ready
        self.model = settings.llm_model
        self.max_chunks = settings.llm_extraction_max_chunks
        self.client: Any | None = None

        if not self.enabled:
            return

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI SDK to enable LLM extraction.") from exc

        default_headers = {}
        if settings.openrouter_http_referer:
            default_headers["HTTP-Referer"] = settings.openrouter_http_referer
        if settings.openrouter_title:
            default_headers["X-OpenRouter-Title"] = settings.openrouter_title

        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            default_headers=default_headers or None,
        )

    def extract(self, chunks: list[ChunkRecord]) -> tuple[list[EntityRecord], list[ClaimRecord], list[GraphEdge]]:
        if not self.enabled or self.client is None:
            return [], [], []

        entities: list[EntityRecord] = []
        claims: list[ClaimRecord] = []
        edges: list[GraphEdge] = []

        for chunk in chunks[: self.max_chunks]:
            parsed = self._extract_chunk(chunk)
            chunk_entities, chunk_claims, chunk_edges = self._to_records(chunk, parsed)
            entities.extend(chunk_entities)
            claims.extend(chunk_claims)
            edges.extend(chunk_edges)

        return entities, claims, edges

    def _extract_chunk(self, chunk: ChunkRecord) -> ChunkExtraction:
        response = self.client.responses.parse(
            model=self.model,
            store=False,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You extract only explicitly supported facts from a PDF chunk. "
                        "Return entities, claims, and relations grounded in the text. "
                        "Do not infer missing facts. Prefer empty arrays over guesses. "
                        "Use short evidence spans copied from the chunk when possible."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Page: {chunk.page}\n"
                        f"Section: {chunk.section}\n"
                        f"Chunk ID: {chunk.id}\n"
                        "Extract entities, standalone factual claims, and relations from this chunk.\n\n"
                        f"{chunk.text}"
                    ),
                },
            ],
            text_format=ChunkExtraction,
        )
        parsed = response.output_parsed
        if parsed is None:
            return ChunkExtraction()
        return parsed

    def _to_records(
        self,
        chunk: ChunkRecord,
        parsed: ChunkExtraction,
    ) -> tuple[list[EntityRecord], list[ClaimRecord], list[GraphEdge]]:
        entities: list[EntityRecord] = []
        claims: list[ClaimRecord] = []
        edges: list[GraphEdge] = []
        entities_by_name: dict[str, EntityRecord] = {}

        def get_or_create_entity(
            name: str,
            label: str = "Entity",
            aliases: list[str] | None = None,
            evidence: str = "",
            confidence: float = 0.35,
        ) -> EntityRecord:
            key = name.strip().lower()
            if not key:
                raise ValueError("Entity name cannot be empty.")
            if key in entities_by_name:
                return entities_by_name[key]

            entity = EntityRecord(
                canonical_name=name.strip(),
                label=label,
                aliases=aliases or [],
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                page=chunk.page,
                confidence=confidence,
                extractor="openai",
                source_text=evidence or chunk.text,
            )
            entities_by_name[key] = entity
            entities.append(entity)
            edges.append(
                GraphEdge(
                    source_id=chunk.id,
                    target_id=entity.id,
                    relation="MENTIONS",
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    page=chunk.page,
                    confidence=entity.confidence,
                    extractor="openai",
                )
            )
            return entity

        for item in parsed.entities:
            get_or_create_entity(
                name=item.name,
                label=item.label or "Entity",
                aliases=item.aliases,
                evidence=item.evidence,
                confidence=item.confidence,
            )

        for item in parsed.claims:
            claim = ClaimRecord(
                text=item.text.strip(),
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                page=chunk.page,
                confidence=item.confidence,
                extractor="openai",
                source_text=item.evidence or chunk.text,
            )
            claims.append(claim)
            edges.append(
                GraphEdge(
                    source_id=chunk.id,
                    target_id=claim.id,
                    relation="MAKES_CLAIM",
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    page=chunk.page,
                    confidence=claim.confidence,
                    extractor="openai",
                )
            )
            for entity_name in item.entity_names:
                entity = get_or_create_entity(name=entity_name, evidence=item.evidence, confidence=item.confidence)
                edges.append(
                    GraphEdge(
                        source_id=claim.id,
                        target_id=entity.id,
                        relation="ABOUT",
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        page=chunk.page,
                        confidence=min(claim.confidence, entity.confidence),
                        extractor="openai",
                    )
                )

        for item in parsed.relations:
            source = get_or_create_entity(name=item.source_entity, confidence=item.confidence)
            target = get_or_create_entity(name=item.target_entity, confidence=item.confidence)
            edges.append(
                GraphEdge(
                    source_id=source.id,
                    target_id=target.id,
                    relation=item.relation.strip().upper().replace(" ", "_"),
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    page=chunk.page,
                    confidence=item.confidence,
                    extractor="openai",
                )
            )

        return entities, claims, edges
