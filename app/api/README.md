# API

This package exposes the service through FastAPI. `routes.py` defines all current endpoints and accesses application services through `request.app.state`.

## Endpoints

| Method | Path | Behaviour |
| --- | --- | --- |
| `GET` | `/health` | Returns `{"status": "ok"}`. It does not probe downstream dependencies. |
| `POST` | `/ingest/pdf` | Saves the uploaded file into `RAW_DATA_DIR`, then runs the synchronous ingestion pipeline. |
| `POST` | `/query` | Validates a `QueryRequest`, retrieves evidence, and returns a `QueryResponse`. |

Although the route is called `/ingest/pdf`, the parser behind it also accepts `.txt` files. Other extensions raise a validation error from the parser.

## Request Flow

`POST /ingest/pdf` performs potentially expensive parsing, spaCy processing, LLM calls, JSON writes, graph writes, and vector writes in one request. There is no job queue or progress endpoint yet, so large documents and provider delays can keep the request open for a long time.

`POST /query` is synchronous. It combines local vector search and graph expansion before passing evidence to the answer service.

## Adding Routes

Add transport-level validation and HTTP error handling here. Put reusable domain logic in the appropriate application package and register its service in `app/main.py` when shared state is required.
