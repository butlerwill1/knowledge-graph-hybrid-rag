from __future__ import annotations

import re

from app.models.schemas import ChunkRecord, ClaimRecord, EntityRecord, GraphEdge


class DeterministicExtractor:
    entity_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
    sentence_splitter = re.compile(r"(?<=[.!?])\s+")
    claim_verbs = ("is", "are", "was", "were", "has", "have", "can", "should", "must")

    def extract(self, chunks: list[ChunkRecord]) -> tuple[list[EntityRecord], list[ClaimRecord], list[GraphEdge]]:
        entities: list[EntityRecord] = []
        claims: list[ClaimRecord] = []
        edges: list[GraphEdge] = []

        for chunk in chunks:
            chunk_entities = self._extract_entities(chunk)
            chunk_claims = self._extract_claims(chunk)
            entities.extend(chunk_entities)
            claims.extend(chunk_claims)

            for entity in chunk_entities:
                edges.append(
                    GraphEdge(
                        source_id=chunk.id,
                        target_id=entity.id,
                        relation="MENTIONS",
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        page=chunk.page,
                        confidence=entity.confidence,
                    )
                )

            for claim in chunk_claims:
                edges.append(
                    GraphEdge(
                        source_id=chunk.id,
                        target_id=claim.id,
                        relation="MAKES_CLAIM",
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        page=chunk.page,
                        confidence=claim.confidence,
                    )
                )
                for entity in chunk_entities[:3]:
                    edges.append(
                        GraphEdge(
                            source_id=claim.id,
                            target_id=entity.id,
                            relation="ABOUT",
                            document_id=chunk.document_id,
                            chunk_id=chunk.id,
                            page=chunk.page,
                            confidence=min(claim.confidence, entity.confidence),
                        )
                    )
        return entities, claims, edges

    def _extract_entities(self, chunk: ChunkRecord) -> list[EntityRecord]:
        seen: set[str] = set()
        results: list[EntityRecord] = []
        for match in self.entity_pattern.findall(chunk.text):
            name = match.strip()
            if len(name) < 3 or name.lower() in seen:
                continue
            seen.add(name.lower())
            results.append(
                EntityRecord(
                    canonical_name=name,
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    page=chunk.page,
                    confidence=0.45,
                    source_text=chunk.text,
                )
            )
        return results

    def _extract_claims(self, chunk: ChunkRecord) -> list[ClaimRecord]:
        results: list[ClaimRecord] = []
        for sentence in self.sentence_splitter.split(chunk.text):
            normalized = sentence.strip()
            lowered = normalized.lower()
            if len(normalized) < 40 or not any(f" {verb} " in f" {lowered} " for verb in self.claim_verbs):
                continue
            results.append(
                ClaimRecord(
                    text=normalized,
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    page=chunk.page,
                    confidence=0.35,
                    source_text=chunk.text,
                )
            )
        return results

