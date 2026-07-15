# Graph Persistence

This package defines the graph-store boundary and provides Neo4j and in-memory implementations.

## Interface

`base.py` defines the `GraphStore` protocol. Implementations must support:

- Schema initialisation.
- Document, chunk, entity, claim, and edge upserts.
- Chunk lookup by ID.
- Entity search by query text.
- Claim expansion from entity IDs.
- Resource cleanup.

The ingestion and retrieval packages depend on this protocol rather than a specific database.

## Neo4j

`neo4j_store.py` uses the official Neo4j Python driver. It creates unique constraints for IDs on `Document`, `Chunk`, `Entity`, and `Claim` nodes.

Node writes use `MERGE` on the generated record ID followed by `SET +=` for all serialised properties. Relationship writes are grouped by type because Cypher relationship types cannot be passed as ordinary parameters. Types are normalised to uppercase letters, digits, and underscores before the Cypher statement is built.

Entity search currently performs case-insensitive substring matching against `canonical_name`. Claim expansion follows incoming `ABOUT` relationships:

```cypher
MATCH (c:Claim)-[:ABOUT]->(e:Entity)
```

Neo4j persistence lives in the configured DBMS data directory. When `docker-compose.yml` is used, it lives in the Docker volume named `neo4j_data`, not in this repository.

## In-Memory Store

`in_memory.py` stores records in dictionaries keyed by ID. It maintains a small index from entity IDs to claim IDs when `ABOUT` edges are inserted.

This backend is useful for tests and short development sessions. All data disappears when the process exits.

## Current Graph Model

The graph contains four main node labels:

- `Document`
- `Chunk`
- `Entity`
- `Claim`

Chunks link to entities through `MENTIONS` and claims through `MAKES_CLAIM`. Claims link to direct subjects through `ABOUT` and source works through `FROM_WORK`. Extractors can also create dynamic entity-to-entity relationships.

Documents, chunks, entities, claims, and relationships all carry provenance properties. There are currently no explicit `Document` to `Chunk` relationships; ownership is represented through `document_id`.

## Known Limitations

- Entity uniqueness is based on generated `id`, not `canonical_name`, so equivalent entities from different chunks remain separate nodes.
- Search does not currently use aliases, `mention_text`, full-text indexes, or vector indexes.
- There is no extraction-run node or transaction boundary spanning the complete ingestion.
- Rerunning ingestion creates new IDs and can therefore add duplicate records.
