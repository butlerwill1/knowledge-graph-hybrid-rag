# Extraction

This package converts chunks into `EntityRecord`, `ClaimRecord`, and `GraphEdge` objects. Both extractors run during normal ingestion, but the LLM extractor returns no records unless it is enabled and has an API key.

## Deterministic Extractor

`deterministic.py` uses the spaCy `en_core_web_sm` pipeline.

It performs three operations:

1. Selected NER labels become entities, such as people, organisations, locations, works, events, and products.
2. Sentences longer than the configured minimum become claims when they contain one of the recognised assertion verbs.
3. Pairs of sentence entities receive a dependency-based verb relation where possible, otherwise `CO_OCCURS_WITH`.

Deterministic records use `extractor="spacy"`. Confidence values are rule-based constants rather than calibrated probabilities.

## LLM Extractor

`llm.py` uses the OpenAI Python SDK against an OpenAI-compatible endpoint, currently OpenRouter by default. It requests a strict `ChunkExtraction` JSON schema containing:

- Entities with `canonical_name`, `mention_text`, label, aliases, evidence, and confidence.
- Claims with direct subjects, mentioned entities, source works, evidence, and confidence.
- Entity-to-entity semantic relations.

Only the current chunk may be extracted. A bounded tail from the previous chunk and head from the next chunk are included as context so the model can resolve phrases such as `these templates` or `the model` without extracting adjacent text twice.

## Validation

The LLM response is validated in two stages:

1. Pydantic rejects JSON that does not match the strict schema.
2. Extraction validation normalises names and removes vague or inconsistent graph records.

The second stage currently:

- Rejects pronouns and generic demonstrative phrases as canonical entity names.
- Keeps the original source wording in `mention_text`.
- Resolves claim entity lists only against accepted entities from the response.
- Requires each accepted claim to have at least one direct subject.
- Rejects claims with recognised unresolved references.
- Prevents the same entity being treated as both a source work and direct subject.
- Rejects relations with missing, identical, or invalid endpoints.

## Generated Edges

| Relationship | Meaning |
| --- | --- |
| `MENTIONS` | A chunk contains an extracted entity mention. |
| `MAKES_CLAIM` | A chunk contains an extracted claim. |
| `ABOUT` | A claim directly concerns an entity. |
| `FROM_WORK` | A claim came from a paper or source work but is not directly about that work. |
| Dynamic relation | A semantic entity-to-entity relation selected by spaCy or the LLM. |

## Usage and Cost Logging

Each successful LLM call appends one JSON object to `LLM_USAGE_LOG_PATH`. The row includes model, response ID, chunk provenance, token counts, actual OpenRouter cost when supplied, optional local estimates, accepted record counts, and validation rejection counts.

Failed calls currently do not append a usage row because there is no completion response to inspect.

## Known Limitations

- Extraction is chunk-local and does not merge equivalent entities across chunks.
- Prompt instructions reduce vague output but cannot guarantee semantic completeness.
- The hard validation patterns are intentionally narrow to avoid rejecting legitimate titles.
- The regular ingestion path has no retry, backoff, or checkpoint support for provider failures.
- A long chunk can cause the model to omit useful claims even when it avoids bad ones.
- LLM test-run JSON is experimental output and is not imported into Neo4j automatically.
