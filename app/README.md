# Application Package

The `app` package contains the running FastAPI service and all ingestion, extraction, persistence, retrieval, and response logic.

## Startup

`main.py` constructs the application in this order:

1. Load the shared settings from `config.py`.
2. Select either `InMemoryGraphStore` or `Neo4jGraphStore` from `GRAPH_BACKEND`.
3. Load the file-backed vector store.
4. Construct the ingestion pipeline, retrieval engine, and answer service.
5. Attach those services to `app.state` for the API routes.
6. Close the graph store when FastAPI shuts down.

Importing `app.main` creates the application-level `app` object used by Uvicorn.

## Packages

| Package | Responsibility |
| --- | --- |
| [`api`](api/README.md) | FastAPI route definitions. |
| [`ingest`](ingest/README.md) | File parsing, chunking, and ingestion orchestration. |
| [`extract`](extract/README.md) | spaCy and OpenRouter graph extraction. |
| [`graph`](graph/README.md) | Graph persistence abstraction and implementations. |
| [`retrieve`](retrieve/README.md) | Vector search and graph claim expansion. |
| [`answer`](answer/README.md) | Evidence-based response assembly. |
| [`models`](models/README.md) | Pydantic data contracts shared across packages. |
| [`eval`](eval/README.md) | Evaluation helpers. |

## Top-Level Files

### `config.py`

Defines environment-backed settings with `pydantic-settings`. It accepts OpenRouter-oriented names as well as generic OpenAI-compatible aliases. Importing the module creates the shared `settings` object and ensures local data directories exist.

### `main.py`

Composes the application. This is the Uvicorn entry point:

```powershell
python -m uvicorn app.main:app --reload
```

Keep business logic in the relevant package rather than adding it directly to `main.py` or the route functions.
