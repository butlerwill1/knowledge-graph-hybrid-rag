from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.schemas import IngestionResult, QueryRequest, QueryResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ingest/pdf", response_model=IngestionResult)
async def ingest_pdf(request: Request, file: UploadFile = File(...)) -> IngestionResult:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    destination = Path(request.app.state.settings.raw_data_dir) / file.filename
    destination.write_bytes(await file.read())
    return request.app.state.ingestion_pipeline.ingest_file(destination)


@router.post("/query", response_model=QueryResponse)
def query(request: Request, payload: QueryRequest) -> QueryResponse:
    evidence, graph_facts = request.app.state.retrieval_engine.retrieve(payload)
    return request.app.state.answer_service.build_response(payload.question, evidence, graph_facts)

