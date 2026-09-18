from pathlib import Path

from app.agents.answerer import GenerationAnswerer
from app.agents.decision import LLMDecisionEngine
from app.agents.executor import AgentExecutor
from app.agents.orchestrator import AgentOrchestrator
from app.agents.planner import AgentPlanner
from app.agents.refiner import LLMQueryRefiner
from app.config.settings import Settings, get_settings
from app.documents.page_store import PageStore
from app.documents.repository import DocumentRepository
from app.generation.provider import GenerationProvider
from app.generation.providers.mistral import MistralGenerationProvider
from app.generation.providers.ollama import OllamaGenerationProvider
from app.generation.service import GenerationService
from app.ingestion.service import IngestionService
from app.ocr.pipeline import OCRPipeline
from app.ocr.providers.mistral import MistralOCRProvider
from app.retrieval.candidates import CandidateRetriever
from app.retrieval.context import RetrievalContextAssembler
from app.retrieval.index_lifecycle import IndexLifecycle
from app.retrieval.index_persistence import IndexSnapshotStore
from app.retrieval.inverted_index import InvertedIndex
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.page_index import PageIndex
from app.retrieval.service import RetrievalService
from app.generation.provider import GenerationProvider
from app.generation.providers.groq import GroqGenerationProvider
from app.generation.providers.mistral import MistralGenerationProvider
from app.generation.providers.ollama import OllamaGenerationProvider


class ApplicationContainer:
    """Application dependency composition root."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.repository = DocumentRepository(
            Path(settings.metadata_dir)/"documents.db",
        )

        self.page_store = PageStore(
            Path(settings.processed_data_dir),
        )

        self.ocr_pipeline = self._build_ocr_pipeline()

        self.inverted_index = InvertedIndex()

        self.page_index = PageIndex(
            page_store=self.page_store,
            inverted_index=self.inverted_index,
        )

        self.index_snapshot_store = IndexSnapshotStore(
            Path(settings.index_dir) / "inverted_index_snapshot.json",
        )

        self.index_lifecycle = IndexLifecycle(
            page_store=self.page_store,
            page_index=self.page_index,
            snapshot_store=self.index_snapshot_store,
        )

        self.ingestion_service = IngestionService(
            repository=self.repository,
            page_store=self.page_store,
            ocr_pipeline=self.ocr_pipeline,
            processed_root=Path(settings.processed_data_dir),
            index_lifecycle=self.index_lifecycle,
        )

        self.candidate_retriever = CandidateRetriever(
            page_index=self.page_index,
        )

        self.lexical_retriever = LexicalRetriever(
            page_store=self.page_store,
            candidate_retriever=self.candidate_retriever,
        )

        self.retrieval_context_assembler = RetrievalContextAssembler(
            max_pages=settings.retrieval_max_pages,
            max_chars=settings.retrieval_max_chars,
        )

        self.retrieval_service = RetrievalService(
            page_store=self.page_store,
            retriever=self.lexical_retriever,
            context_assembler=self.retrieval_context_assembler,
        )

        self.generation_provider = self._build_generation_provider()

        self.generation_service = GenerationService(
            provider=self.generation_provider,
        )

        # Use the provider's actual model name (Ollama or Mistral)
        generation_model = self.generation_provider.model_name

        self.answerer = GenerationAnswerer(
            generation_service=self.generation_service,
            model=generation_model,
            temperature=settings.generation_temperature,
            max_tokens=settings.generation_max_tokens,
        )

        self.planner = AgentPlanner()
        self.executor = AgentExecutor(
            retrieval_service=self.retrieval_service,
            top_k=settings.retrieval_top_k,
        )

        self.decision_engine = LLMDecisionEngine(
            generation_service=self.generation_service,
            model=generation_model,
        )
        self.refiner = LLMQueryRefiner(
            generation_service=self.generation_service,
            model=generation_model,
        )

        self.agent_orchestrator = AgentOrchestrator(
            planner=self.planner,
            executor=self.executor,
            decision_engine=self.decision_engine,
            answerer=self.answerer,
            refiner=self.refiner,
        )

    def initialize(self) -> int:
        """Prepare persistent state for the application to start serving.

        Creates the SQLite schema if it does not exist yet and loads any
        already-persisted pages (from a previous run) into the in-memory
        search index, since the index itself does not survive a restart.

        Returns the number of pages loaded into the index.
        """

        Path(self.settings.processed_data_dir).mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(self.settings.metadata_dir).mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(self.settings.index_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.repository.initialize()

        return self.index_lifecycle.build()

    def _build_ocr_pipeline(self) -> OCRPipeline:
        """Build the configured OCR pipeline."""

        if not self.settings.ocr_enabled:
            raise RuntimeError(
                "OCR is disabled, but no alternative OCR pipeline "
                "implementation is configured."
            )

        if not self.settings.mistral_api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY is required when OCR is enabled."
            )

        provider = MistralOCRProvider(
            api_key=self.settings.mistral_api_key,
            model=self.settings.mistral_ocr_model,
            timeout_ms=self.settings.mistral_ocr_timeout_ms,
        )

        return OCRPipeline(
            provider=provider,
        )

    def _build_generation_provider(self) -> GenerationProvider:
        provider = (self.settings.generation_provider or "groq").strip().lower()

        if provider == "groq":
            if not self.settings.groq_api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is required when GENERATION_PROVIDER=groq."
                )
            return GroqGenerationProvider(
                api_key=self.settings.groq_api_key,
                default_model=self.settings.groq_model,
                timeout_ms=self.settings.groq_generation_timeout_ms,
            )

        if provider == "mistral":
            if not self.settings.mistral_api_key:
                raise RuntimeError(
                    "MISTRAL_API_KEY is required when GENERATION_PROVIDER=mistral."
                )
            return MistralGenerationProvider(
                api_key=self.settings.mistral_api_key,
                default_model=self.settings.mistral_generation_model,
                timeout_ms=self.settings.mistral_generation_timeout_ms,
            )

        return OllamaGenerationProvider(
            base_url=self.settings.ollama_base_url,
            default_model=self.settings.ollama_model,
            timeout_ms=self.settings.ollama_generation_timeout_ms,
        )


def create_application_container(
    settings: Settings | None = None,
) -> ApplicationContainer:
    """Build the application's dependency graph."""

    resolved_settings = settings or get_settings()

    return ApplicationContainer(
        settings=resolved_settings,
    )