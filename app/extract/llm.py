"""Extract and validate structured graph records with an OpenAI-compatible LLM."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings
from app.models.schemas import ChunkRecord, ClaimRecord, EntityRecord, GraphEdge


class ExtractedEntity(BaseModel):
    """Describe one entity returned by the structured LLM response."""

    model_config = ConfigDict(extra="forbid")

    canonical_name: str
    mention_text: str
    label: str = "Entity"
    aliases: list[str] = Field(default_factory=list)
    evidence: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractedClaim(BaseModel):
    """Describe a standalone claim and its different entity roles."""

    model_config = ConfigDict(extra="forbid")

    text: str
    evidence: str = ""
    subject_entity_names: list[str] = Field(default_factory=list)
    mentioned_entity_names: list[str] = Field(default_factory=list)
    source_work_names: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    """Describe a directed semantic relation between two extracted entities."""

    model_config = ConfigDict(extra="forbid")

    source_entity: str
    target_entity: str
    relation: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ChunkExtraction(BaseModel):
    """Contain the complete structured response for one current chunk."""

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


_PRONOUN_ENTITY_NAMES = {
    "he",
    "her",
    "hers",
    "him",
    "his",
    "it",
    "its",
    "she",
    "that",
    "them",
    "these",
    "they",
    "this",
    "those",
    "we",
    "you",
}
_GENERIC_REFERENCE_NOUNS = {
    "agent",
    "agents",
    "approach",
    "approaches",
    "architecture",
    "architectures",
    "component",
    "components",
    "concept",
    "concepts",
    "entity",
    "entities",
    "finding",
    "findings",
    "framework",
    "frameworks",
    "idea",
    "ideas",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "papers",
    "process",
    "processes",
    "result",
    "results",
    "section",
    "sections",
    "study",
    "studies",
    "system",
    "systems",
    "technique",
    "techniques",
    "template",
    "templates",
    "work",
    "works",
}
_LEADING_UNRESOLVED_CLAIM = re.compile(
    r"^(?:it|they|them|he|she|this|that|these|those)\s+"
    r"(?:is|are|was|were|has|have|had|can|could|will|would|may|might|must|should|shows?|suggests?)\b",
    re.IGNORECASE,
)
_POSSESSIVE_UNRESOLVED_CLAIM = re.compile(r"^(?:its|their|his|her)\s+", re.IGNORECASE)
_FORMER_LATTER_REFERENCE = re.compile(
    r"\b(?:the\s+)?(?:former|latter|above(?:mentioned)?|aforementioned)\b",
    re.IGNORECASE,
)


class LLMExtractor:
    """Call an OpenAI-compatible model and translate validated output to graph records."""

    def __init__(self, settings: Settings) -> None:
        """Configure the provider client when LLM extraction is ready."""

        self.settings = settings
        self.enabled = settings.llm_extraction_ready
        self.model = settings.llm_model
        self.max_chunks = settings.llm_extraction_max_chunks
        self.extractor_name = "openrouter" if "openrouter.ai" in settings.llm_base_url else "openai"
        self.client: Any | None = None

        # A missing key disables paid extraction without preventing deterministic ingestion.
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
        """Extract up to the configured chunk limit and combine graph records."""

        if not self.enabled or self.client is None:
            return [], [], []

        entities: list[EntityRecord] = []
        claims: list[ClaimRecord] = []
        edges: list[GraphEdge] = []

        for index, chunk in enumerate(chunks[: self.max_chunks]):
            # Adjacent chunks help resolve references at boundaries but are never
            # independently extracted during the current call.
            previous_chunk = chunks[index - 1] if index > 0 else None
            next_chunk = chunks[index + 1] if index + 1 < len(chunks) else None
            if previous_chunk is not None and previous_chunk.document_id != chunk.document_id:
                previous_chunk = None
            if next_chunk is not None and next_chunk.document_id != chunk.document_id:
                next_chunk = None

            parsed = self._extract_chunk(chunk, previous_chunk=previous_chunk, next_chunk=next_chunk)
            chunk_entities, chunk_claims, chunk_edges = self._to_records(chunk, parsed)
            entities.extend(chunk_entities)
            claims.extend(chunk_claims)
            edges.extend(chunk_edges)

        return entities, claims, edges

    def _extract_chunk(
        self,
        chunk: ChunkRecord,
        previous_chunk: ChunkRecord | None = None,
        next_chunk: ChunkRecord | None = None,
    ) -> ChunkExtraction:
        """Request, parse, validate, and account for one structured completion."""

        schema = _make_strict_schema(ChunkExtraction.model_json_schema())
        adjacent_context_chars = self.settings.llm_adjacent_context_chars
        previous_context = ""
        next_context = ""
        # Only the boundary nearest the current chunk is useful for coreference,
        # which also keeps the paid prompt smaller than sending three full chunks.
        if previous_chunk is not None and adjacent_context_chars:
            previous_context = previous_chunk.text[-adjacent_context_chars:]
        if next_chunk is not None and adjacent_context_chars:
            next_context = next_chunk.text[:adjacent_context_chars]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract only explicitly supported facts from a PDF chunk. "
                        "Return entities, claims, and relations grounded in the text. "
                        "Do not infer missing facts. Prefer empty arrays over guesses. "
                        "Extract only from the CURRENT CHUNK. Adjacent chunks are context only and must not be "
                        "independently extracted. Use short evidence spans copied from the current chunk. "
                        "For every entity, canonical_name must be a stable, independently meaningful name. "
                        "Resolve references such as 'these templates', 'this method', 'the model', and pronouns "
                        "to their explicit referent when the supplied context supports it. Put the exact phrase "
                        "used in the current chunk in mention_text. If the entity appears both by its explicit name "
                        "and by an ambiguous reference, prefer the ambiguous phrase in mention_text so the resolution "
                        "can be audited. If a reference cannot be resolved confidently, "
                        "omit the entity and any dependent claim or relation. Do not use vague references as "
                        "canonical names. Use a specific entity label rather than 'Entity' when the type is clear. "
                        "Claims must be standalone factual propositions: replace ambiguous references with resolved "
                        "entity names, require at least one direct subject_entity_name, and omit vague, introductory, "
                        "navigational, or non-substantive statements. Every entity name used by a claim or relation "
                        "must exactly match an entity canonical_name in the entities array. "
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
                        "Extract entities, standalone factual claims, and relations from the CURRENT CHUNK only.\n\n"
                        "--- PREVIOUS CHUNK CONTEXT (REFERENCE RESOLUTION ONLY) ---\n"
                        f"{previous_context or '[none]'}\n\n"
                        "--- CURRENT CHUNK (EXTRACT FROM THIS TEXT) ---\n"
                        f"{chunk.text}\n\n"
                        "--- NEXT CHUNK CONTEXT (REFERENCE RESOLUTION ONLY) ---\n"
                        f"{next_context or '[none]'}"
                    ),
                },
            ],
            # Strict structured output prevents free-form prose from reaching the
            # graph translation layer.
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
            self._record_usage(chunk, response, parsed, _validation_counts(parsed, parsed))
            return parsed
        try:
            parsed = ChunkExtraction.model_validate_json(content)
        except ValidationError as exc:
            try:
                json.loads(content)
            except json.JSONDecodeError as json_exc:
                raise RuntimeError("LLM extraction returned invalid JSON.") from json_exc
            raise RuntimeError("LLM extraction returned JSON that does not match the extraction schema.") from exc
        validated, validation_counts = self._validate_extraction(parsed)
        self._record_usage(chunk, response, validated, validation_counts)
        return validated

    def _validate_extraction(
        self,
        parsed: ChunkExtraction,
    ) -> tuple[ChunkExtraction, dict[str, int]]:
        """Remove vague or internally inconsistent items from model output."""

        entities: list[ExtractedEntity] = []
        entities_by_name: dict[str, ExtractedEntity] = {}

        # This lookup becomes the authority for every claim and relation endpoint.
        # Unknown names are discarded rather than materialised as implicit entities.
        for item in parsed.entities:
            canonical_name = _normalise_text(item.canonical_name)
            mention_text = _normalise_text(item.mention_text)
            if not canonical_name or not mention_text or _is_vague_entity_name(canonical_name):
                continue

            key = canonical_name.casefold()
            if key in entities_by_name:
                continue

            aliases = list(
                dict.fromkeys(
                    alias
                    for alias in (_normalise_text(alias) for alias in item.aliases)
                    if alias and alias.casefold() != key
                )
            )
            validated_entity = item.model_copy(
                update={
                    "canonical_name": canonical_name,
                    "mention_text": mention_text,
                    "aliases": aliases,
                    "evidence": item.evidence.strip(),
                }
            )
            entities.append(validated_entity)
            entities_by_name[key] = validated_entity

        def resolve_names(names: list[str]) -> list[str]:
            """Resolve response names to accepted canonical spelling and order."""

            resolved: list[str] = []
            seen: set[str] = set()
            for name in names:
                entity = entities_by_name.get(_normalise_text(name).casefold())
                if entity is None or entity.canonical_name.casefold() in seen:
                    continue
                seen.add(entity.canonical_name.casefold())
                resolved.append(entity.canonical_name)
            return resolved

        claims: list[ExtractedClaim] = []
        for item in parsed.claims:
            text = _normalise_text(item.text)
            if not text or _has_unresolved_reference(text):
                continue

            subject_names = resolve_names(item.subject_entity_names)
            source_work_names = resolve_names(item.source_work_names)
            # Source works provide provenance. Treating them as direct claim
            # subjects created broad and misleading ABOUT traversals.
            source_work_keys = {name.casefold() for name in source_work_names}
            subject_names = [name for name in subject_names if name.casefold() not in source_work_keys]
            if not subject_names:
                continue

            mentioned_names = resolve_names(item.mentioned_entity_names)
            claims.append(
                item.model_copy(
                    update={
                        "text": text,
                        "evidence": item.evidence.strip(),
                        "subject_entity_names": subject_names,
                        "mentioned_entity_names": mentioned_names,
                        "source_work_names": source_work_names,
                    }
                )
            )

        relations: list[ExtractedRelation] = []
        for item in parsed.relations:
            source = entities_by_name.get(_normalise_text(item.source_entity).casefold())
            target = entities_by_name.get(_normalise_text(item.target_entity).casefold())
            relation = _normalise_text(item.relation)
            if source is None or target is None or source is target or not relation:
                continue
            relations.append(
                item.model_copy(
                    update={
                        "source_entity": source.canonical_name,
                        "target_entity": target.canonical_name,
                        "relation": relation,
                    }
                )
            )

        validated = ChunkExtraction(entities=entities, claims=claims, relations=relations)
        return validated, _validation_counts(parsed, validated)

    def _record_usage(
        self,
        chunk: ChunkRecord,
        response: Any,
        parsed: ChunkExtraction,
        validation_counts: dict[str, int],
    ) -> None:
        """Append provider usage, cost, and validation counts as one JSONL row."""

        usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
        total_tokens = _usage_value(usage, "total_tokens")
        # OpenRouter's response cost is authoritative when present. Configured
        # per-token rates are retained only as optional local estimates.
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
            **validation_counts,
        }
        with self.settings.llm_usage_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")

    def _to_records(
        self,
        chunk: ChunkRecord,
        parsed: ChunkExtraction,
    ) -> tuple[list[EntityRecord], list[ClaimRecord], list[GraphEdge]]:
        """Translate a validated chunk extraction into application graph records."""

        # Validate again so direct callers cannot bypass graph-quality guards.
        parsed, _ = self._validate_extraction(parsed)
        entities: list[EntityRecord] = []
        claims: list[ClaimRecord] = []
        edges: list[GraphEdge] = []
        entities_by_name: dict[str, EntityRecord] = {}

        def get_or_create_entity(
            name: str,
            mention_text: str = "",
            label: str = "Entity",
            aliases: list[str] | None = None,
            evidence: str = "",
            confidence: float = 0.35,
        ) -> EntityRecord:
            """Create one entity and MENTIONS edge per canonical name in this chunk."""

            key = name.strip().lower()
            if not key:
                raise ValueError("Entity name cannot be empty.")
            if key in entities_by_name:
                return entities_by_name[key]

            entity = EntityRecord(
                canonical_name=name.strip(),
                mention_text=mention_text.strip() or name.strip(),
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
                name=item.canonical_name,
                mention_text=item.mention_text,
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
            # Mentioned entities are materialised for provenance, while only direct
            # subjects receive ABOUT and source works receive FROM_WORK.
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


def _normalise_text(value: str) -> str:
    """Collapse response whitespace for stable names and claim text."""

    return re.sub(r"\s+", " ", value).strip()


def _is_vague_entity_name(name: str) -> bool:
    """Identify narrow classes of unresolved or demonstrative entity names."""

    words = re.findall(r"[a-z0-9]+", name.casefold())
    if not words:
        return True
    if " ".join(words) in _PRONOUN_ENTITY_NAMES:
        return True
    if words[0] in {"former", "latter"}:
        return True
    if len(words) >= 2 and words[0] == "the" and words[1] in {"former", "latter"}:
        return True
    if len(words) <= 5 and words[-1] in _GENERIC_REFERENCE_NOUNS:
        if words[0] in {"this", "that", "these", "those", "such", "the"}:
            return True
    return False


def _has_unresolved_reference(text: str) -> bool:
    """Return whether a claim still contains a recognised vague reference."""

    if (
        _LEADING_UNRESOLVED_CLAIM.search(text)
        or _POSSESSIVE_UNRESOLVED_CLAIM.search(text)
        or _FORMER_LATTER_REFERENCE.search(text)
    ):
        return True

    words = re.findall(r"[a-z0-9]+", text.casefold())
    if words and words[0] in {"this", "that", "these", "those", "such"}:
        if any(word in _GENERIC_REFERENCE_NOUNS for word in words[1:4]):
            return True
    if len(words) >= 2 and words[0] == "the" and words[1] in _GENERIC_REFERENCE_NOUNS:
        return True
    return False


def _validation_counts(raw: ChunkExtraction, validated: ChunkExtraction) -> dict[str, int]:
    """Summarise raw extraction size and records rejected by validation."""

    return {
        "raw_entities_extracted": len(raw.entities),
        "raw_claims_extracted": len(raw.claims),
        "raw_relations_extracted": len(raw.relations),
        "entities_rejected": len(raw.entities) - len(validated.entities),
        "claims_rejected": len(raw.claims) - len(validated.claims),
        "relations_rejected": len(raw.relations) - len(validated.relations),
    }


def _usage_value(usage: Any, *names: str) -> int | None:
    """Read the first available integer field from object- or dict-like usage."""

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
    """Read the first available floating-point usage field."""

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
    """Convert a nested provider usage object into a serialisable dictionary."""

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
    """Estimate cost from token count and a per-million-token rate."""

    if tokens is None or cost_per_million is None:
        return None
    return tokens * cost_per_million / 1_000_000
