from __future__ import annotations

from collections import defaultdict

from app.models.schemas import ChunkRecord, ClaimRecord, DocumentRecord, EntityRecord, GraphEdge


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: dict[str, ChunkRecord] = {}
        self.entities: dict[str, EntityRecord] = {}
        self.claims: dict[str, ClaimRecord] = {}
        self.edges: dict[str, GraphEdge] = {}
        self.entity_claim_index: dict[str, list[str]] = defaultdict(list)

    def ensure_schema(self) -> None:
        return

    def upsert_document(self, document: DocumentRecord) -> None:
        self.documents[document.id] = document

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def upsert_entities(self, entities: list[EntityRecord]) -> None:
        for entity in entities:
            self.entities[entity.id] = entity

    def upsert_claims(self, claims: list[ClaimRecord]) -> None:
        for claim in claims:
            self.claims[claim.id] = claim

    def upsert_edges(self, edges: list[GraphEdge]) -> None:
        for edge in edges:
            self.edges[edge.id] = edge
            if edge.relation == "ABOUT" and edge.source_id.startswith("claim-"):
                self.entity_claim_index[edge.target_id].append(edge.source_id)

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]

    def search_entities(self, query: str, limit: int = 5) -> list[EntityRecord]:
        lowered = query.lower()
        matches = [
            entity
            for entity in self.entities.values()
            if lowered in entity.canonical_name.lower() or entity.canonical_name.lower() in lowered
        ]
        return matches[:limit]

    def claims_for_entities(self, entity_ids: list[str], limit: int = 10) -> list[ClaimRecord]:
        claim_ids: list[str] = []
        for entity_id in entity_ids:
            claim_ids.extend(self.entity_claim_index.get(entity_id, []))
        unique_ids = list(dict.fromkeys(claim_ids))
        return [self.claims[claim_id] for claim_id in unique_ids[:limit] if claim_id in self.claims]

    def close(self) -> None:
        return
