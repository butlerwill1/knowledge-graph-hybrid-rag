from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings
from app.models.schemas import ChunkRecord, ClaimRecord, EntityRecord, GraphEdge


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    label: str = "Entity"
    aliases: list[str] = Field(default_factory=list)
    evidence: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence: str = ""
    subject_entity_names: list[str] = Field(default_factory=list)
    mentioned_entity_names: list[str] = Field(default_factory=list)
    source_work_names: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_entity: str
    target_entity: str
    relation: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ChunkExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


def _make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenRouter strict structured outputs work best with explicit required fields."""
    schema.pop("default", None)
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        schema["additionalProperties"] = False
        schema["required"] = list(properties)
        for value in properties.values():
            if isinstance(value, dict):
                _make_strict_schema(value)
    if schema.get("type") == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            _make_strict_schema(items)
    for value in schema.get("$defs", {}).values():
        if isinstance(value, dict):
            _make_strict_schema(value)
    return schema


class LLMExtractor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.llm_extraction_ready
        self.model = settings.llm_model
        self.max_chunks = settings.llm_extraction_max_chunks
        self.extractor_name = "openrouter" if "openrouter.ai" in settings.llm_base_url else "openai"
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
        schema = _make_strict_schema(ChunkExtraction.model_json_schema())
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract only explicitly supported facts from a PDF chunk. "
                        "Return entities, claims, and relations grounded in the text. "
                        "Do not infer missing facts. Prefer empty arrays over guesses. "
                        "Use short evidence spans copied from the chunk when possible. "
                        "For each claim, distinguish direct subject entities from entities that are merely mentioned. "
                        "Do not mark the paper title or source work as a claim subject unless the claim is directly "
                        "about the paper itself, such as its authors, venue, title, contribution, or evaluation. "
                        "Use source_work_names only for the paper, appendix, section, or source document context."
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
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "chunk_extraction",
                    "strict": True,
                    "schema": schema,
                },
            },
            extra_body={
                "provider": {
                    "require_parameters": True,
                }
            },
        )
        content = response.choices[0].message.content
        if not content:
            parsed = ChunkExtraction()
            self._record_usage(chunk, response, parsed)
            return parsed
        try:
            parsed = ChunkExtraction.model_validate_json(content)
        except ValidationError as exc:
            try:
                json.loads(content)
            except json.JSONDecodeError as json_exc:
                raise RuntimeError("LLM extraction returned invalid JSON.") from json_exc
            raise RuntimeError("LLM extraction returned JSON that does not match the extraction schema.") from exc
        self._record_usage(chunk, response, parsed)
        return parsed

    def _record_usage(self, chunk: ChunkRecord, response: Any, parsed: ChunkExtraction) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
        total_tokens = _usage_value(usage, "total_tokens")
        actual_cost = _usage_float(usage, "cost", "total_cost")
        cost_details = _usage_dict(usage, "cost_details")
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        input_cost = _token_cost(input_tokens, self.settings.llm_input_cost_per_million)
        output_cost = _token_cost(output_tokens, self.settings.llm_output_cost_per_million)
        total_cost = None
        if input_cost is not None or output_cost is not None:
            total_cost = (input_cost or 0.0) + (output_cost or 0.0)

        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extractor": self.extractor_name,
            "model": getattr(response, "model", self.model),
            "response_id": getattr(response, "id", None),
            "document_id": chunk.document_id,
            "chunk_id": chunk.id,
            "page": chunk.page,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "actual_cost": actual_cost,
            "actual_cost_unit": "openrouter_credits" if actual_cost is not None else None,
            "cost_details": cost_details,
            "estimated_input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "estimated_total_cost": total_cost,
            "cost_currency": self.settings.llm_cost_currency,
            "entities_extracted": len(parsed.entities),
            "claims_extracted": len(parsed.claims),
            "relations_extracted": len(parsed.relations),
        }
        with self.settings.llm_usage_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")

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
                extractor=self.extractor_name,
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
                    extractor=self.extractor_name,
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
                extractor=self.extractor_name,
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
                    extractor=self.extractor_name,
                )
            )
            mentioned_names = set(item.mentioned_entity_names)
            subject_names = set(item.subject_entity_names)
            source_work_names = set(item.source_work_names)
            subject_names -= source_work_names
            for entity_name in sorted(mentioned_names | subject_names | source_work_names):
                get_or_create_entity(name=entity_name, evidence=item.evidence, confidence=item.confidence)
            for entity_name in sorted(subject_names):
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
                        extractor=self.extractor_name,
                    )
                )
            for entity_name in sorted(source_work_names):
                entity = get_or_create_entity(name=entity_name, evidence=item.evidence, confidence=item.confidence)
                edges.append(
                    GraphEdge(
                        source_id=claim.id,
                        target_id=entity.id,
                        relation="FROM_WORK",
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        page=chunk.page,
                        confidence=min(claim.confidence, entity.confidence),
                        extractor=self.extractor_name,
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
                    extractor=self.extractor_name,
                )
            )

        return entities, claims, edges


def _usage_value(usage: Any, *names: str) -> int | None:
    if usage is None:
        return None
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


def _usage_float(usage: Any, *names: str) -> float | None:
    if usage is None:
        return None
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            return float(value)
    return None


def _usage_dict(usage: Any, name: str) -> dict[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return None


def _token_cost(tokens: int | None, cost_per_million: float | None) -> float | None:
    if tokens is None or cost_per_million is None:
        return None
    return tokens * cost_per_million / 1_000_000
