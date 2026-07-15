import json
from types import SimpleNamespace

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
            ExtractedEntity(
                canonical_name="Alice Corp",
                mention_text="Alice Corp",
                label="Organization",
                confidence=0.8,
            ),
            ExtractedEntity(
                canonical_name="Beta Labs",
                mention_text="Beta Labs",
                label="Organization",
                confidence=0.8,
            ),
            ExtractedEntity(
                canonical_name="Acquisition Report",
                mention_text="the report",
                label="Work",
                confidence=0.8,
            ),
        ],
        claims=[
            ExtractedClaim(
                text="Alice Corp acquired Beta Labs in 2024.",
                subject_entity_names=["Alice Corp", "Beta Labs"],
                mentioned_entity_names=["Alice Corp", "Beta Labs"],
                source_work_names=["Acquisition Report"],
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

    assert len(entities) == 3
    assert len(claims) == 1
    assert any(edge.relation == "ABOUT" for edge in edges)
    assert any(edge.relation == "FROM_WORK" for edge in edges)
    assert any(edge.relation == "ACQUIRED" for edge in edges)


def test_llm_extractor_uses_openrouter_structured_chat_completion(tmp_path):
    settings = Settings(
        ENABLE_LLM_EXTRACTION=True,
        OPENROUTER_API_KEY="test-key",
        LLM_MODEL="openai/gpt-5.4-mini",
        LLM_USAGE_LOG_PATH=tmp_path / "llm_usage.jsonl",
        LLM_INPUT_COST_PER_MILLION=0.75,
        LLM_OUTPUT_COST_PER_MILLION=4.5,
    )
    extractor = LLMExtractor(settings)
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            content = ChunkExtraction(
                entities=[
                    ExtractedEntity(
                        canonical_name="Alice Corp",
                        mention_text="Alice Corp",
                        label="Organization",
                        confidence=0.8,
                    )
                ],
                claims=[],
                relations=[],
            ).model_dump_json()
            return SimpleNamespace(
                id="chatcmpl-test",
                model="openai/gpt-5.4-mini",
                usage=SimpleNamespace(
                    prompt_tokens=1200,
                    completion_tokens=300,
                    total_tokens=1500,
                    cost=0.00219,
                    cost_details=SimpleNamespace(upstream_inference_cost=0.0017),
                ),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content),
                    )
                ]
            )

    extractor.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    parsed = extractor._extract_chunk(
        ChunkRecord(id="chunk-1", document_id="doc-1", page=1, text="Alice Corp builds graph retrieval systems."),
        previous_chunk=ChunkRecord(id="chunk-0", document_id="doc-1", page=1, text="Previous context."),
        next_chunk=ChunkRecord(id="chunk-2", document_id="doc-1", page=2, text="Next context."),
    )

    assert parsed.entities[0].canonical_name == "Alice Corp"
    assert parsed.entities[0].mention_text == "Alice Corp"
    assert captured["model"] == "openai/gpt-5.4-mini"
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["extra_body"]["provider"]["require_parameters"] is True
    prompt = captured["messages"][1]["content"]
    assert "Previous context." in prompt
    assert "CURRENT CHUNK" in prompt
    assert "Next context." in prompt

    usage_row = json.loads((tmp_path / "llm_usage.jsonl").read_text(encoding="utf-8"))
    assert usage_row["chunk_id"] == "chunk-1"
    assert usage_row["model"] == "openai/gpt-5.4-mini"
    assert usage_row["input_tokens"] == 1200
    assert usage_row["output_tokens"] == 300
    assert usage_row["actual_cost"] == 0.00219
    assert usage_row["actual_cost_unit"] == "openrouter_credits"
    assert usage_row["cost_details"]["upstream_inference_cost"] == 0.0017
    assert round(usage_row["estimated_total_cost"], 5) == 0.00225
    assert usage_row["entities_rejected"] == 0


def test_llm_validation_keeps_resolved_mentions_and_rejects_vague_output():
    extractor = LLMExtractor(Settings(ENABLE_LLM_EXTRACTION=False))
    parsed = ChunkExtraction(
        entities=[
            ExtractedEntity(
                canonical_name="first-order prompting templates",
                mention_text="These templates",
                label="Concept",
                confidence=0.93,
            ),
            ExtractedEntity(
                canonical_name="These templates",
                mention_text="These templates",
                label="Entity",
                confidence=0.93,
            ),
        ],
        claims=[
            ExtractedClaim(
                text=(
                    "First-order prompting templates generate behaviour conditioned only on an agent's "
                    "current environment."
                ),
                evidence=(
                    "These templates are effective in generating behavior that is conditioned solely on the "
                    "agent's current environment"
                ),
                subject_entity_names=["first-order prompting templates"],
                mentioned_entity_names=["first-order prompting templates"],
                confidence=0.93,
            ),
            ExtractedClaim(
                text="These templates are effective in generating behaviour.",
                subject_entity_names=["These templates"],
                mentioned_entity_names=["These templates"],
                confidence=0.93,
            ),
        ],
    )

    validated, counts = extractor._validate_extraction(parsed)

    assert [entity.canonical_name for entity in validated.entities] == ["first-order prompting templates"]
    assert validated.entities[0].mention_text == "These templates"
    assert len(validated.claims) == 1
    assert validated.claims[0].subject_entity_names == ["first-order prompting templates"]
    assert counts["entities_rejected"] == 1
    assert counts["claims_rejected"] == 1

    chunk = ChunkRecord(document_id="doc-1", page=4, text=parsed.claims[0].evidence)
    entities, claims, edges = extractor._to_records(chunk, validated)

    assert entities[0].canonical_name == "first-order prompting templates"
    assert entities[0].mention_text == "These templates"
    assert len(claims) == 1
    assert any(edge.relation == "ABOUT" for edge in edges)
