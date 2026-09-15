from pathlib import Path

import pytest

from app.agents.answerer import GenerationAnswerer
from app.agents.executor import AgentExecutor
from app.agents.orchestrator import AgentOrchestrator
from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.generation.models import GenerationRequest, GenerationResponse
from app.generation.provider import GenerationProvider
from app.generation.service import GenerationService
from app.retrieval.service import RetrievalService


class StubGenerationProvider(GenerationProvider):
    """Deterministic generation provider for orchestration tests."""

    def __init__(
        self,
        response_text: str = "Generated grounded answer.",
    ) -> None:
        self.response_text = response_text
        self.requests: list[GenerationRequest] = []

    @property
    def provider_name(self) -> str:
        return "stub"

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        self.requests.append(request)

        return GenerationResponse(
            model=request.model,
            text=self.response_text,
        )


def create_page(
    document_id: str,
    page_number: int,
    text: str,
) -> PageRecord:
    return PageRecord(
        document_id=document_id,
        page_number=page_number,
        text=text,
        width=612.0,
        height=792.0,
    )


@pytest.fixture
def page_store(tmp_path: Path) -> PageStore:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                "doc-001",
                1,
                (
                    "Vectorless RAG uses lexical retrieval "
                    "instead of vector embeddings."
                ),
            ),
        ]
    )

    return store


@pytest.fixture
def generation_provider() -> StubGenerationProvider:
    return StubGenerationProvider()


@pytest.fixture
def orchestrator(
    page_store: PageStore,
    generation_provider: StubGenerationProvider,
) -> AgentOrchestrator:
    retrieval_service = RetrievalService(page_store)

    executor = AgentExecutor(
        retrieval_service=retrieval_service,
    )

    generation_service = GenerationService(
        provider=generation_provider,
    )

    answerer = GenerationAnswerer(
        generation_service=generation_service,
        model="mistral-small-latest",
        temperature=0.0,
        max_tokens=300,
    )

    return AgentOrchestrator(
        executor=executor,
        answerer=answerer,
    )


def test_orchestrator_generates_grounded_answer(
    orchestrator: AgentOrchestrator,
) -> None:
    response = orchestrator.run(
        "What does vectorless RAG use for retrieval?"
    )

    assert response.answer == "Generated grounded answer."
    assert response.query == (
        "What does vectorless RAG use for retrieval?"
    )
    assert response.iterations == 1


def test_orchestrator_passes_retrieved_context_to_generation(
    orchestrator: AgentOrchestrator,
    generation_provider: StubGenerationProvider,
) -> None:
    orchestrator.run(
        "What does vectorless RAG use for retrieval?"
    )

    assert len(generation_provider.requests) == 1

    request = generation_provider.requests[0]

    assert request.model == "mistral-small-latest"
    assert request.temperature == 0.0
    assert request.max_tokens == 300

    assert len(request.messages) == 3

    assert request.messages[0].role == "system"
    assert request.messages[1].role == "system"
    assert request.messages[2].role == "user"

    assert (
        "Vectorless RAG uses lexical retrieval"
        in request.messages[1].content
    )

    assert (
        request.messages[2].content
        == "What does vectorless RAG use for retrieval?"
    )


def test_orchestrator_preserves_original_query_after_refinement(
    page_store: PageStore,
    generation_provider: StubGenerationProvider,
) -> None:
    retrieval_service = RetrievalService(page_store)

    executor = AgentExecutor(
        retrieval_service=retrieval_service,
    )

    generation_service = GenerationService(
        provider=generation_provider,
    )

    answerer = GenerationAnswerer(
        generation_service=generation_service,
        model="mistral-small-latest",
    )

    orchestrator = AgentOrchestrator(
        executor=executor,
        answerer=answerer,
    )

    response = orchestrator.run(
        "vectorless retrieval"
    )

    assert response.query == "vectorless retrieval"
    assert response.answer == "Generated grounded answer."
    assert response.iterations == 1