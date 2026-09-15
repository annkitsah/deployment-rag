from pathlib import Path

from app.agents.answerer import GenerationAnswerer
from app.agents.decision import AgentDecisionEngine, LLMDecisionEngine
from app.agents.executor import AgentExecutor
from app.agents.orchestrator import AgentOrchestrator
from app.agents.planner import AgentPlanner
from app.agents.refiner import AgentQueryRefiner, LLMQueryRefiner
from app.config.settings import Settings
from app.container import (
    ApplicationContainer,
    create_application_container,
)
from app.generation.providers.ollama import OllamaGenerationProvider
from app.generation.service import GenerationService
from app.ingestion.service import IngestionService
from app.retrieval.candidates import CandidateRetriever
from app.retrieval.context import RetrievalContextAssembler
from app.retrieval.index_lifecycle import IndexLifecycle
from app.retrieval.index_persistence import IndexSnapshotStore
from app.retrieval.inverted_index import InvertedIndex
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.page_index import PageIndex
from app.retrieval.service import RetrievalService


def make_settings(tmp_path: Path) -> Settings:
    """Build isolated test settings."""

    return Settings(
        mistral_api_key="test-api-key",
        data_dir=str(tmp_path / "data"),
        raw_data_dir=str(tmp_path / "data" / "raw"),
        processed_data_dir=str(tmp_path / "data" / "processed"),
        index_dir=str(tmp_path / "data" / "indexes"),
        metadata_dir=str(tmp_path / "data" / "metadata"),
    )


def test_create_application_container_builds_complete_graph(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    container = create_application_container(settings)

    assert isinstance(container, ApplicationContainer)

    assert isinstance(container.ingestion_service, IngestionService)

    assert isinstance(container.inverted_index, InvertedIndex)
    assert isinstance(container.page_index, PageIndex)
    assert isinstance(
        container.candidate_retriever,
        CandidateRetriever,
    )
    assert isinstance(
        container.lexical_retriever,
        LexicalRetriever,
    )
    assert isinstance(
        container.retrieval_service,
        RetrievalService,
    )
    assert isinstance(
        container.index_lifecycle,
        IndexLifecycle,
    )

    assert isinstance(
        container.generation_provider,
        OllamaGenerationProvider,
    )
    assert isinstance(
        container.generation_service,
        GenerationService,
    )
    assert isinstance(
        container.answerer,
        GenerationAnswerer,
    )

    assert isinstance(container.planner, AgentPlanner)
    assert isinstance(container.executor, AgentExecutor)
    assert isinstance(
        container.decision_engine,
        LLMDecisionEngine,
    )
    assert isinstance(
        container.decision_engine.fallback,
        AgentDecisionEngine,
    )
    assert isinstance(container.refiner, LLMQueryRefiner)
    assert isinstance(
        container.refiner.fallback,
        AgentQueryRefiner,
    )
    assert isinstance(
        container.agent_orchestrator,
        AgentOrchestrator,
    )


def test_retrieval_components_share_same_page_index(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    container = create_application_container(settings)

    assert (
        container.index_lifecycle.page_index
        is container.page_index
    )

    assert (
        container.candidate_retriever.page_index
        is container.page_index
    )

    assert (
        container.lexical_retriever.candidate_retriever
        is container.candidate_retriever
    )

def test_ingestion_service_uses_container_index_lifecycle(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    container = create_application_container(settings)

    assert (
        container.ingestion_service.index_lifecycle
        is container.index_lifecycle
    )

    assert (
        container.index_lifecycle.page_index
        is container.page_index
    )


def test_container_wires_index_snapshot_store(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    container = create_application_container(settings)

    assert isinstance(
        container.index_snapshot_store,
        IndexSnapshotStore,
    )
    assert (
        container.index_lifecycle.snapshot_store
        is container.index_snapshot_store
    )
    assert container.index_snapshot_store.snapshot_path == (
        Path(settings.index_dir) / "inverted_index_snapshot.json"
    )


def test_container_initialize_creates_index_dir(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    container = create_application_container(settings)

    container.initialize()

    assert Path(settings.index_dir).is_dir()


def test_container_initialize_creates_schema_and_loads_index(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    container = create_application_container(settings)

    indexed_page_count = container.initialize()

    assert indexed_page_count == 0
    assert container.repository.get_by_id("doc_missing") is None


def test_container_wires_retrieval_top_k_and_max_pages_from_settings(
    tmp_path: Path,
) -> None:
    # Deliberately non-default values, to prove real wiring rather than
    # an accidental match against Settings' own defaults.
    settings = Settings(
        mistral_api_key="test-api-key",
        data_dir=str(tmp_path / "data"),
        raw_data_dir=str(tmp_path / "data" / "raw"),
        processed_data_dir=str(tmp_path / "data" / "processed"),
        index_dir=str(tmp_path / "data" / "indexes"),
        metadata_dir=str(tmp_path / "data" / "metadata"),
        retrieval_top_k=3,
        retrieval_max_pages=7,
        retrieval_max_chars=50_000,
    )

    container = create_application_container(settings)

    assert container.executor.top_k == 3
    assert container.retrieval_context_assembler.max_pages == 7
    assert container.retrieval_context_assembler.max_chars == 50_000
    assert (
        container.retrieval_service.context_assembler
        is container.retrieval_context_assembler
    )

def test_executor_uses_container_retrieval_service(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    container = create_application_container(settings)

    assert (
        container.executor.retrieval_service
        is container.retrieval_service
    )


def test_generation_answerer_uses_container_generation_service(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    container = create_application_container(settings)

    assert (
        container.answerer.generation_service
        is container.generation_service
    )

    assert (
        container.answerer.model
        == settings.ollama_model
    )

    assert (
        container.answerer.temperature
        == settings.generation_temperature
    )

    assert (
        container.answerer.max_tokens
        == settings.generation_max_tokens
    )


def test_agent_orchestrator_uses_container_dependencies(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    container = create_application_container(settings)

    assert container.agent_orchestrator.planner is container.planner
    assert (
        container.agent_orchestrator.executor
        is container.executor
    )
    assert (
        container.agent_orchestrator.decision_engine
        is container.decision_engine
    )
    assert (
        container.agent_orchestrator.answerer
        is container.answerer
    )
    assert (
        container.agent_orchestrator.refiner
        is container.refiner
    )


def test_generation_provider_uses_configured_model(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    container = create_application_container(settings)

    assert (
        container.generation_provider.model_name
        == settings.ollama_model
    )


def test_decision_engine_and_refiner_use_configured_model(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    container = create_application_container(settings)

    assert container.decision_engine.model == settings.ollama_model
    assert container.refiner.model == settings.ollama_model
    assert (
        container.decision_engine.generation_service
        is container.generation_service
    )
    assert (
        container.refiner.generation_service
        is container.generation_service
    )