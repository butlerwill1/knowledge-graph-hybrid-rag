# Models

This package defines the shared Pydantic contracts used by ingestion, extraction, graph persistence, retrieval, and the API.

## Main Records

| Model | Purpose |
| --- | --- |
| `DocumentRecord` | Source title, path, topic, ID, and creation time. |
| `ParsedPage` | Page number and extracted text. |
| `ParsedDocument` | Document metadata plus parsed pages. |
| `ChunkRecord` | A page-scoped text chunk with source spans and metadata. |
| `EntityRecord` | Canonical entity name, original mention, type, aliases, provenance, confidence, and extractor. |
| `ClaimRecord` | Standalone or deterministic source claim with provenance. |
| `GraphEdge` | Typed relationship between record IDs with provenance. |
| `IngestionResult` | Counts returned after an ingestion request. |
| `EvidenceSnippet` | Normalised retrieval evidence. |
| `QueryRequest` | Question and `top_k`. |
| `QueryResponse` | Answer, supporting evidence, and graph facts. |

IDs are UUID-based strings prefixed by record type, such as `doc-`, `chunk-`, `entity-`, `claim-`, and `edge-`.

`EntityRecord.canonical_name` is the reusable graph name selected by an extractor. `mention_text` preserves the local source wording that led to that entity. These fields support local reference resolution, but they do not yet provide graph-wide canonicalisation.

Changes here affect JSON files, graph persistence, and API schemas. New required fields therefore need compatibility defaults or an explicit data migration plan.
