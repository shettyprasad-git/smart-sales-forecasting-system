from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.app.schemas.ai_reasoning import AIReasoningResponse
from backend.app.schemas.recommendation_ai import AIRecommendationResponse


class LLMError(Exception):
    """Base exception for LLM provider operations."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM provider is misconfigured or missing credentials."""


class LLMProviderError(LLMError):
    """Raised when an external LLM call or API connection fails."""


class LLMResponseValidationError(LLMError):
    """Raised when LLM output violates required JSON schemas or consistency checks."""


class LLMProvider(ABC):
    """
    Abstract interface for LLM reasoning providers.
    Decouples the sales intelligence service from specific model vendors (e.g. Gemini, Hugging Face).
    """

    @abstractmethod
    def generate_reasoning(self, evidence_package: dict[str, Any]) -> AIReasoningResponse:
        """
        Generate structured reasoning from a compact empirical evidence package.
        """
        pass

    def generate_recommendations(
        self, recommendation_package: dict[str, Any]
    ) -> AIRecommendationResponse:
        """
        Generate structured prescriptive action recommendations from an evidence package.
        """
        raise NotImplementedError("generate_recommendations must be implemented by concrete provider")

