"""Define shared Pydantic records for ingestion, graph storage, and querying."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for record factories."""

    return datetime.now(timezone.utc)


class DocumentRecord(BaseModel):
    """Identify an ingested source document and its local provenance."""

    id: str = Field(default_factory=lambda: f"doc-{uuid4().hex}")
    title: str
    source_path: str
    topic: str = "general"
    created_at: datetime = Field(default_factory=utc_now)


class ParsedPage(BaseModel):
    """Represent text extracted from one source page."""

    page_number: int
    text: str


class ParsedDocument(BaseModel):
    """Combine document metadata with all parsed pages."""

    document: DocumentRecord
    pages: list[ParsedPage]


class ChunkRecord(BaseModel):
    """Represent a page-scoped text unit used by extraction and retrieval."""

    id: str = Field(default_factory=lambda: f"chunk-{uuid4().hex}")
    document_id: str
    page: int
    text: str
    section: str = "Body"
    span_start: int = 0
    span_end: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class EntityRecord(BaseModel):
    """Represent an extracted entity mention with canonical and source wording."""

    id: str = Field(default_factory=lambda: f"entity-{uuid4().hex}")
    canonical_name: str
    mention_text: str = ""
    label: str = "Entity"
    aliases: list[str] = Field(default_factory=list)
    document_id: str
    chunk_id: str
    page: int
    confidence: float = 0.5
    extractor: str = "deterministic"
    source_text: str


class ClaimRecord(BaseModel):
    """Represent a factual statement and the source span supporting it."""

    id: str = Field(default_factory=lambda: f"claim-{uuid4().hex}")
    text: str
    document_id: str
    chunk_id: str
    page: int
    confidence: float = 0.5
    extractor: str = "deterministic"
    source_text: str


class GraphEdge(BaseModel):
    """Represent a typed, provenance-bearing relationship between record IDs."""

    id: str = Field(default_factory=lambda: f"edge-{uuid4().hex}")
    source_id: str
    target_id: str
    relation: str
    document_id: str
    chunk_id: str
    page: int
    confidence: float = 0.5
    extractor: str = "deterministic"


class IngestionResult(BaseModel):
    """Summarise records created during one ingestion request."""

    document: DocumentRecord
    chunks_created: int
    entities_created: int
    claims_created: int
    edges_created: int


class EvidenceSnippet(BaseModel):
    """Normalise graph and vector retrieval results for answer assembly."""

    source_type: Literal["chunk", "claim", "entity"]
    source_id: str
    document_id: str
    page: int
    text: str
    score: float


class QueryRequest(BaseModel):
    """Define a retrieval question and result limit."""

    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Return a grounded answer with its evidence and graph facts."""

    answer: str
    evidence: list[EvidenceSnippet]
    graph_facts: list[str]
