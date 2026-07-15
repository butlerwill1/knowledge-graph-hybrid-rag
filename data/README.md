# Local Data

This directory contains source documents and generated runtime data. JSON, JSONL, and PDF files are ignored by Git, while these README files document the expected layout.

| Path | Contents |
| --- | --- |
| [`raw_pdfs`](raw_pdfs/README.md) | Uploaded PDF and text source files. |
| [`parsed`](parsed/README.md) | Parsed document metadata and page text. |
| [`extracted`](extracted/README.md) | Combined chunks, entities, claims, and edges produced by ingestion. |
| [`llm_test_runs`](llm_test_runs/README.md) | Manual LLM extraction experiments and checkpointed test runs. |
| `vector_store.json` | Chunk records and local hashed token vectors. |
| `llm_usage.jsonl` | One usage and cost record per successful LLM extraction call. |

## Data Ownership

The contents can include complete source text, personal information from documents, provider response identifiers, and usage costs. Treat this directory as local working data rather than public fixtures.

Deleting generated files does not delete records already imported into Neo4j. Similarly, deleting Neo4j data does not remove these local audit files.

## Relationship to Neo4j

Neo4j database files are not kept here. Docker Compose stores them in the `neo4j_data` volume; Neo4j Desktop stores them inside its own DBMS directory.

Files under `llm_test_runs` are not automatically imported. Files under `extracted` are written from the same records passed to the graph store during normal ingestion.
