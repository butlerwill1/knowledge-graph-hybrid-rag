from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"doc-{uuid4().hex}")
    title: str
    source_path: str
    topic: str = "general"
    created_at: datetime = Field(default_factory=utc_now)


class ParsedPage(BaseModel):
    page_number: int
    text: str


class ParsedDocument(BaseModel):
    document: DocumentRecord
    pages: list[ParsedPage]


class ChunkRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"chunk-{uuid4().hex}")
    document_id: str
    page: int
    text: str
    section: str = "Body"
    span_start: int = 0
    span_end: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class EntityRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"entity-{uuid4().hex}")
    canonical_name: str
    label: str = "Entity"
    aliases: list[str] = Field(default_factory=list)
    document_id: str
    chunk_id: str
    page: int
    confidence: float = 0.5
    extractor: str = "deterministic"
    source_text: str


class ClaimRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"claim-{uuid4().hex}")
    text: str
    document_id: str
    chunk_id: str
    page: int
    confidence: float = 0.5
    extractor: str = "deterministic"
    source_text: str


class GraphEdge(BaseModel):
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
    document: DocumentRecord
    chunks_created: int
    entities_created: int
    claims_created: int
    edges_created: int


class EvidenceSnippet(BaseModel):
    source_type: Literal["chunk", "claim", "entity"]
    source_id: str
    document_id: str
    page: int
    text: str
    score: float


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    evidence: list[EvidenceSnippet]
    graph_facts: list[str]

