"""Define HTTP endpoints for health checks, ingestion, and querying."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.schemas import IngestionResult, QueryRequest, QueryResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight process health response."""

    return {"status": "ok"}


@router.post("/ingest/pdf", response_model=IngestionResult)
async def ingest_pdf(request: Request, file: UploadFile = File(...)) -> IngestionResult:
    """Save an uploaded source file and run the synchronous ingestion pipeline."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    # Persist the source before parsing so generated records retain a stable path.
    destination = Path(request.app.state.settings.raw_data_dir) / file.filename
    destination.write_bytes(await file.read())
    return request.app.state.ingestion_pipeline.ingest_file(destination)


@router.post("/query", response_model=QueryResponse)
def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """Retrieve graph and vector evidence, then assemble a grounded response."""

    evidence, graph_facts = request.app.state.retrieval_engine.retrieve(payload)
    return request.app.state.answer_service.build_response(payload.question, evidence, graph_facts)
