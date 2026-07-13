# knowledge-graph-hybrid-rag

This scaffold ingests PDFs into a lightweight knowledge graph, persists facts to Neo4j, extracts structured facts through the OpenAI Python SDK pointed at OpenRouter, and exposes grounded query endpoints through FastAPI.

## Architecture

- `app/ingest`: PDF parsing and chunking
- `app/extract`: deterministic extractors plus structured LLM extraction
- `app/graph`: graph persistence interfaces, in-memory backend, and Neo4j adapter
- `app/retrieve`: vector retrieval and graph-aware evidence expansion
- `app/answer`: grounded answer assembly with citations
- `app/api`: HTTP endpoints

The ingestion path now supports:

- deterministic extraction for low-cost baseline facts
- LLM extraction using OpenAI Structured Outputs for entities, claims, and relations
- Neo4j persistence for documents, chunks, entities, claims, and relation edges

The vector store is still file-backed in this revision so retrieval remains easy to run locally while the graph layer is fully persistent.

## Quick start

1. Create a virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.
3. Start Neo4j:

```powershell
docker compose up -d
```

4. Run the API:

```powershell
python -m uvicorn app.main:app --reload
```

## Endpoints

- `GET /health`
- `POST /ingest/pdf`
- `POST /query`

## Notes

- `GRAPH_BACKEND=neo4j` uses the persistent Cypher-backed graph store.
- `ENABLE_LLM_EXTRACTION=true` enables structured extraction through OpenRouter-compatible chat completions.
- The OpenAI SDK is configured against OpenRouter by `LLM_BASE_URL=https://openrouter.ai/api/v1`. The extractor sends a strict JSON schema via `response_format` and validates the returned JSON with Pydantic before writing graph records.
- LLM extraction appends per-chunk usage records to `LLM_USAGE_LOG_PATH`. Token counts and actual OpenRouter cost are captured from the provider response when present; local estimated costs are only populated when `LLM_INPUT_COST_PER_MILLION` and `LLM_OUTPUT_COST_PER_MILLION` are configured.
- The retrieval path is hybrid by design: vector retrieval for chunks, graph expansion for related claims, and source citations on the final answer.
