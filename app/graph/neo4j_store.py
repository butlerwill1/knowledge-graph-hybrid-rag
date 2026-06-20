from __future__ import annotations

from app.config import Settings
from app.models.schemas import ChunkRecord, ClaimRecord, DocumentRecord, EntityRecord, GraphEdge


class Neo4jGraphStore:
    _SCHEMA_STATEMENTS = (
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE",
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install the 'neo4j' extra to use the Neo4j graph backend.") from exc
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._driver.session() as session:
            for statement in self._SCHEMA_STATEMENTS:
                session.run(statement).consume()

    def upsert_document(self, document: DocumentRecord) -> None:
        payload = document.model_dump(mode="json")
        with self._driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $row.id})
                SET d += $row
                """,
                row=payload,
            ).consume()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        rows = [chunk.model_dump(mode="json") for chunk in chunks]
        if not rows:
            return
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {id: row.id})
                SET c += row
                """,
                rows=rows,
            ).consume()

    def upsert_entities(self, entities: list[EntityRecord]) -> None:
        rows = [entity.model_dump(mode="json") for entity in entities]
        if not rows:
            return
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {id: row.id})
                SET e += row
                """,
                rows=rows,
            ).consume()

    def upsert_claims(self, claims: list[ClaimRecord]) -> None:
        rows = [claim.model_dump(mode="json") for claim in claims]
        if not rows:
            return
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:Claim {id: row.id})
                SET c += row
                """,
                rows=rows,
            ).consume()

    def upsert_edges(self, edges: list[GraphEdge]) -> None:
        if not edges:
            return
        # Cypher relationship types can't be parameterised, so group by relation type
        # and issue one UNWIND statement per type.
        by_type: dict[str, list[dict]] = {}
        for edge in edges:
            row = edge.model_dump(mode="json")
            by_type.setdefault(edge.relation, []).append(row)

        with self._driver.session() as session:
            for rel_type, rows in by_type.items():
                import re as _re
                safe_type = _re.sub(r"[^A-Z0-9_]", "_", rel_type.upper().replace(" ", "_").replace("-", "_"))
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (source {{id: row.source_id}})
                    MATCH (target {{id: row.target_id}})
                    MERGE (source)-[rel:`{safe_type}` {{id: row.id}}]->(target)
                    SET rel += row
                    """,
                    rows=rows,
                ).consume()

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        if not chunk_ids:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                UNWIND $chunk_ids AS chunk_id
                MATCH (c:Chunk {id: chunk_id})
                RETURN c
                """,
                chunk_ids=chunk_ids,
            )
            return [ChunkRecord.model_validate(dict(record["c"])) for record in result]

    def search_entities(self, query: str, limit: int = 5) -> list[EntityRecord]:
        lowered_query = query.lower()
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.canonical_name) CONTAINS $query
                   OR $query CONTAINS toLower(e.canonical_name)
                RETURN e
                ORDER BY e.confidence DESC, e.canonical_name ASC
                LIMIT $limit
                """,
                query=lowered_query,
                limit=limit,
            )
            return [EntityRecord.model_validate(dict(record["e"])) for record in result]

    def claims_for_entities(self, entity_ids: list[str], limit: int = 10) -> list[ClaimRecord]:
        if not entity_ids:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                UNWIND $entity_ids AS entity_id
                MATCH (c:Claim)-[rel:ABOUT]->(e:Entity {id: entity_id})
                RETURN DISTINCT c
                ORDER BY c.confidence DESC, c.id ASC
                LIMIT $limit
                """,
                entity_ids=entity_ids,
                limit=limit,
            )
            return [ClaimRecord.model_validate(dict(record["c"])) for record in result]

    def close(self) -> None:
        self._driver.close()
