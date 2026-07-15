"""Define the graph persistence protocol used by ingestion and retrieval."""

from __future__ import annotations

from typing import Protocol

from app.models.schemas import ChunkRecord, ClaimRecord, DocumentRecord, EntityRecord, GraphEdge


class GraphStore(Protocol):
    """Structural interface required by ingestion and retrieval services."""

    def ensure_schema(self) -> None:
        """Create any constraints or indexes required by the backend."""
        ...

    def upsert_document(self, document: DocumentRecord) -> None:
        """Insert or update one document by ID."""
        ...

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Insert or update chunk records by ID."""
        ...

    def upsert_entities(self, entities: list[EntityRecord]) -> None:
        """Insert or update entity records by ID."""
        ...

    def upsert_claims(self, claims: list[ClaimRecord]) -> None:
        """Insert or update claim records by ID."""
        ...

    def upsert_edges(self, edges: list[GraphEdge]) -> None:
        """Insert or update typed graph relationships by ID."""
        ...

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """Return all existing chunks matching the requested IDs."""
        ...

    def search_entities(self, query: str, limit: int = 5) -> list[EntityRecord]:
        """Return entities whose canonical names match the query."""
        ...

    def claims_for_entities(self, entity_ids: list[str], limit: int = 10) -> list[ClaimRecord]:
        """Return claims linked to the requested entities through ABOUT."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...
