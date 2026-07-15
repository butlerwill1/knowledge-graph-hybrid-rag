# Extracted Graph Records

Normal ingestion writes one JSON file per document here after deterministic and optional LLM extraction have been combined.

Each file contains four arrays:

- `chunks`: the source `ChunkRecord` objects.
- `entities`: deterministic and LLM `EntityRecord` objects.
- `claims`: deterministic and LLM `ClaimRecord` objects.
- `edges`: provenance, claim, source-work, and entity-relation edges.

Use the `extractor` property to separate spaCy records from OpenRouter records. LLM entities also populate `mention_text` with the source phrase used to identify the canonical entity.

During normal ingestion these are the same records passed to the configured graph store. The JSON is therefore an audit/export representation, but editing it later does not update Neo4j.

Files are ignored by Git. Rerunning ingestion generates new record IDs and can create a separate output file and duplicate graph records.
