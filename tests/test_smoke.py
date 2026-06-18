from app.answer.service import GroundedAnswerService
from app.config import Settings
from app.extract.llm import ChunkExtraction, ExtractedClaim, ExtractedEntity, ExtractedRelation, LLMExtractor
from app.graph.in_memory import InMemoryGraphStore
from app.models.schemas import ChunkRecord, QueryRequest
from app.retrieve.engine import RetrievalEngine
from app.retrieve.vector_store import FileVectorStore


def test_retrieval_and_answer_smoke(tmp_path):
    graph_store = InMemoryGraphStore()
    vector_store = FileVectorStore(tmp_path / "vectors.json")
    chunk = ChunkRecord(document_id="doc-1", page=1, text="Alice Corp is a supplier of graph databases.")
    graph_store.upsert_chunks([chunk])
    vector_store.add_chunks([chunk])

    engine = RetrievalEngine(graph_store=graph_store, vector_store=vector_store)
    answer_service = GroundedAnswerService()

    evidence, graph_facts = engine.retrieve(QueryRequest(question="What is Alice Corp?"))
    response = answer_service.build_response("What is Alice Corp?", evidence, graph_facts)

    assert response.evidence
    assert "Alice Corp" in response.answer


def test_llm_extractor_is_disabled_without_key():
    settings = Settings(ENABLE_LLM_EXTRACTION=True, OPENROUTER_API_KEY=None)
    extractor = LLMExtractor(settings)
    entities, claims, edges = extractor.extract([])

    assert extractor.enabled is False
    assert entities == []
    assert claims == []
    assert edges == []


def test_llm_translation_creates_graph_records():
    settings = Settings(enable_llm_extraction=False)
    extractor = LLMExtractor(settings)
    chunk = ChunkRecord(document_id="doc-1", page=2, text="Alice Corp acquired Beta Labs in 2024.")
    parsed = ChunkExtraction(
        entities=[
            ExtractedEntity(name="Alice Corp", label="Organization", confidence=0.8),
            ExtractedEntity(name="Beta Labs", label="Organization", confidence=0.8),
        ],
        claims=[
            ExtractedClaim(
                text="Alice Corp acquired Beta Labs in 2024.",
                entity_names=["Alice Corp", "Beta Labs"],
                confidence=0.9,
            )
        ],
        relations=[
            ExtractedRelation(
                source_entity="Alice Corp",
                target_entity="Beta Labs",
                relation="acquired",
                confidence=0.85,
            )
        ],
    )

    entities, claims, edges = extractor._to_records(chunk, parsed)

    assert len(entities) == 2
    assert len(claims) == 1
    assert any(edge.relation == "ABOUT" for edge in edges)
    assert any(edge.relation == "ACQUIRED" for edge in edges)
