# Retrieval

This package combines local vector retrieval with graph entity and claim retrieval.

## File Vector Store

`vector_store.py` persists a dictionary keyed by chunk ID. Each row contains the serialised `ChunkRecord` and a 256-dimensional vector.

The vector is a normalised hashed token-frequency representation:

1. Lowercase word tokens are extracted with a regular expression.
2. Each token is assigned to a dimension using Python's built-in `hash()`.
3. Dimension counts are L2-normalised.
4. Queries are represented in the same way and ranked by cosine similarity.

This is a cheap retrieval baseline, not a semantic embedding model. Python hash randomisation also means vectors written in one Python process may not be compatible with query vectors produced after a restart.

## Hybrid Retrieval Engine

`engine.py` executes these paths for each `QueryRequest`:

1. Search the vector store for the top chunk matches.
2. Search graph entities by substring match against the question.
3. Expand the matched entities to claims through `ABOUT` relationships.
4. Convert chunks and claims into a common `EvidenceSnippet` structure.
5. Deduplicate by source ID, sort by score, and return the top `top_k` records.

Claim confidence is used as its retrieval score, while chunk evidence uses cosine similarity. Those scores are not calibrated to the same scale.

## Known Limitations

- Search is lexical rather than genuinely semantic.
- Graph entity search uses the whole question as a substring comparison.
- Aliases and `mention_text` are not searched.
- There is no reranking model.
- Vector and claim scores are mixed without calibration.
- No document or topic filters are available.
