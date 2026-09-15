from abc import ABC, abstractmethod

from app.generation.models import (
    GenerationRequest,
    GenerationResponse,
)


class GenerationProvider(ABC):
    """Abstraction for text generation providers."""

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate text from a validated generation request."""

        raise NotImplementedError
