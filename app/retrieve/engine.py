from __future__ import annotations

from app.graph.base import GraphStore
from app.models.schemas import EvidenceSnippet, QueryRequest
from app.retrieve.vector_store import FileVectorStore


class RetrievalEngine:
    def __init__(self, graph_store: GraphStore, vector_store: FileVectorStore) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store

    def retrieve(self, request: QueryRequest) -> tuple[list[EvidenceSnippet], list[str]]:
        vector_hits = self.vector_store.search(request.question, limit=request.top_k)
        evidence: list[EvidenceSnippet] = [
            EvidenceSnippet(
                source_type="chunk",
                source_id=chunk.id,
                document_id=chunk.document_id,
                page=chunk.page,
                text=chunk.text,
                score=score,
            )
            for chunk, score in vector_hits
        ]

        graph_facts: list[str] = []
        entity_hits = self.graph_store.search_entities(request.question, limit=3)
        related_claims = self.graph_store.claims_for_entities([entity.id for entity in entity_hits], limit=request.top_k)
        for claim in related_claims:
            evidence.append(
                EvidenceSnippet(
                    source_type="claim",
                    source_id=claim.id,
                    document_id=claim.document_id,
                    page=claim.page,
                    text=claim.text,
                    score=claim.confidence,
                )
            )
            graph_facts.append(f"{claim.id}: {claim.text}")

        deduped = list({snippet.source_id: snippet for snippet in evidence}.values())
        deduped.sort(key=lambda item: item.score, reverse=True)
        return deduped[: request.top_k], graph_facts

