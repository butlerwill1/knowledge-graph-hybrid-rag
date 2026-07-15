"""Compose and expose the FastAPI application and its shared services."""

from __future__ import annotations

from fastapi import FastAPI

from app.answer.service import GroundedAnswerService
from app.api.routes import router
from app.config import settings
from app.graph.in_memory import InMemoryGraphStore
from app.graph.neo4j_store import Neo4jGraphStore
from app.ingest.pipeline import IngestionPipeline
from app.retrieve.engine import RetrievalEngine
from app.retrieve.vector_store import FileVectorStore


def build_graph_store():
    """Construct the graph backend selected by `GRAPH_BACKEND`."""

    if settings.graph_backend == "in_memory":
        return InMemoryGraphStore()
    if settings.graph_backend == "neo4j":
        return Neo4jGraphStore(settings)
    raise RuntimeError(f"Unsupported GRAPH_BACKEND: {settings.graph_backend}")


def create_app() -> FastAPI:
    """Build the FastAPI app and attach long-lived services to application state."""

    app = FastAPI(title=settings.app_name)
    graph_store = build_graph_store()
    vector_store = FileVectorStore(settings.vector_store_path)
    ingestion_pipeline = IngestionPipeline(settings=settings, graph_store=graph_store, vector_store=vector_store)
    retrieval_engine = RetrievalEngine(graph_store=graph_store, vector_store=vector_store)
    answer_service = GroundedAnswerService()

    # Routes retrieve these shared services from app.state instead of creating
    # new database clients or stores for every request.
    app.state.settings = settings
    app.state.graph_store = graph_store
    app.state.vector_store = vector_store
    app.state.ingestion_pipeline = ingestion_pipeline
    app.state.retrieval_engine = retrieval_engine
    app.state.answer_service = answer_service
    app.include_router(router)

    @app.on_event("shutdown")
    def close_graph_store() -> None:
        """Release graph-driver resources during application shutdown."""

        graph_store.close()

    return app


# Uvicorn imports this module-level ASGI application.
app = create_app()
