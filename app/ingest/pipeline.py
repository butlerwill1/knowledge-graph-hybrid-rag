from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.extract.deterministic import DeterministicExtractor
from app.extract.llm import LLMExtractor
from app.graph.base import GraphStore
from app.ingest.chunker import SectionAwareChunker
from app.ingest.parser import PDFParser
from app.models.schemas import IngestionResult
from app.retrieve.vector_store import FileVectorStore


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        graph_store: GraphStore,
        vector_store: FileVectorStore,
        parser: PDFParser | None = None,
        chunker: SectionAwareChunker | None = None,
        deterministic_extractor: DeterministicExtractor | None = None,
        llm_extractor: LLMExtractor | None = None,
    ) -> None:
        self.settings = settings
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.parser = parser or PDFParser()
        self.chunker = chunker or SectionAwareChunker()
        self.deterministic_extractor = deterministic_extractor or DeterministicExtractor()
        self.llm_extractor = llm_extractor or LLMExtractor(settings)

    def ingest_file(self, file_path: Path) -> IngestionResult:
        parsed = self.parser.parse(file_path)
        chunks = self.chunker.chunk(parsed)
        deterministic_entities, deterministic_claims, deterministic_edges = self.deterministic_extractor.extract(chunks)
        llm_entities, llm_claims, llm_edges = self.llm_extractor.extract(chunks)

        entities = deterministic_entities + llm_entities
        claims = deterministic_claims + llm_claims
        edges = deterministic_edges + llm_edges

        self.graph_store.upsert_document(parsed.document)
        self.graph_store.upsert_chunks(chunks)
        self.graph_store.upsert_entities(entities)
        self.graph_store.upsert_claims(claims)
        self.graph_store.upsert_edges(edges)
        self.vector_store.add_chunks(chunks)

        parsed_output = self.settings.parsed_data_dir / f"{parsed.document.id}.json"
        extracted_output = self.settings.extracted_data_dir / f"{parsed.document.id}.json"
        parsed_output.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
        extracted_output.write_text(
            json.dumps(
                {
                    "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                    "entities": [entity.model_dump(mode="json") for entity in entities],
                    "claims": [claim.model_dump(mode="json") for claim in claims],
                    "edges": [edge.model_dump(mode="json") for edge in edges],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return IngestionResult(
            document=parsed.document,
            chunks_created=len(chunks),
            entities_created=len(entities),
            claims_created=len(claims),
            edges_created=len(edges),
        )
