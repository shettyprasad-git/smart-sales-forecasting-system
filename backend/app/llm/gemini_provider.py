from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from backend.app.core.config import settings
from backend.app.llm.prompts import (
    RECOMMENDATION_SYSTEM_INSTRUCTION,
    SIMULATION_EXPLANATION_SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION,
)
from backend.app.llm.provider import (
    LLMConfigurationError,
    LLMMalformedResponseError,
    LLMProvider,
    LLMProviderError,
    LLMProviderUnavailableError,
    LLMResponseValidationError,
)
from backend.app.schemas.ai_reasoning import AIReasoningResponse
from backend.app.schemas.recommendation_ai import AIRecommendationResponse

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """
    Concrete LLM provider using the official Google GenAI Python SDK (`google-genai`).
    Leverages Gemini models (default: gemini-3.8-flash) with structured Pydantic schema enforcement.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or getattr(settings, "gemini_api_key", None)
            or os.environ.get("GEMINI_API_KEY")
        )
        self.model_name = (
            model_name
            or getattr(settings, "gemini_model", None)
            or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        )
        self._client = client

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in .env or environment variables."
            )
        try:
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except Exception as exc:
            logger.error("Failed to initialize Google GenAI Client: %s", exc)
            raise LLMConfigurationError(f"Failed to initialize Gemini Client: {exc}") from exc

    def _classify_exception(self, exc: Exception) -> Exception:
        err_msg = str(exc).lower()
        if "api_key_invalid" in err_msg or "api key not valid" in err_msg or "invalid api key" in err_msg:
            return LLMConfigurationError("Gemini API key is invalid or unauthorized.")
        if "not_found" in err_msg or "is not found" in err_msg:
            return LLMConfigurationError(f"Configured Gemini model '{self.model_name}' was not found.")
        if (
            "resource_exhausted" in err_msg
            or "unavailable" in err_msg
            or "503" in err_msg
            or "429" in err_msg
            or "rate limit" in err_msg
            or "quota" in err_msg
            or "timeout" in err_msg
            or "timed out" in err_msg
            or "deadline" in err_msg
        ):
            return LLMProviderUnavailableError(f"Gemini provider is currently unavailable or capacity constrained: {exc}")
        return LLMProviderError(f"Gemini API request failed: {exc}")

    def generate_reasoning(self, evidence_package: dict[str, Any]) -> AIReasoningResponse:
        client = self._get_client()
        anomaly_id = evidence_package.get("anomaly", {}).get("anomaly_id", "unknown")

        user_content = (
            "Analyze the following empirical sales anomaly evidence package and generate "
            f"structured executive reasoning:\n\n{json.dumps(evidence_package, indent=2)}"
        )

        start_time = time.perf_counter()
        logger.info(
            "Invoking Gemini reasoning | anomaly_id=%s | model=%s",
            anomaly_id,
            self.model_name,
        )

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=AIReasoningResponse,
                    temperature=0.2,
                ),
            )
        except LLMConfigurationError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Gemini API request failed | anomaly_id=%s | latency=%.2fms | error=%s",
                anomaly_id,
                elapsed_ms,
                exc,
            )
            raise self._classify_exception(exc) from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Gemini response received | anomaly_id=%s | latency=%.2fms",
            anomaly_id,
            elapsed_ms,
        )

        raw_text = getattr(response, "text", None)
        if not raw_text:
            # Check if parsed attribute is already available
            if hasattr(response, "parsed") and isinstance(response.parsed, AIReasoningResponse):
                return response.parsed
            raise LLMMalformedResponseError("Gemini returned an empty response.")

        try:
            payload = json.loads(raw_text)
            return AIReasoningResponse.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error(
                "Gemini response failed Pydantic validation | anomaly_id=%s | error=%s",
                anomaly_id,
                exc,
            )
            raise LLMMalformedResponseError(f"Invalid structured response from Gemini: {exc}") from exc

    def generate_recommendations(
        self, recommendation_package: dict[str, Any]
    ) -> AIRecommendationResponse:
        client = self._get_client()
        anomaly_id = recommendation_package.get("anomaly", {}).get("anomaly_id", "unknown")

        user_content = (
            "Evaluate the following empirical anomaly evidence package and deterministically eligible "
            "recommendation categories, then formulate structured prescriptive action recommendations:\n\n"
            f"{json.dumps(recommendation_package, indent=2)}"
        )

        start_time = time.perf_counter()
        logger.info(
            "Invoking Gemini recommendations | anomaly_id=%s | model=%s",
            anomaly_id,
            self.model_name,
        )

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=RECOMMENDATION_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=AIRecommendationResponse,
                    temperature=0.2,
                ),
            )
        except LLMConfigurationError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Gemini recommendation request failed | anomaly_id=%s | latency=%.2fms | error=%s",
                anomaly_id,
                elapsed_ms,
                exc,
            )
            raise self._classify_exception(exc) from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Gemini recommendation response received | anomaly_id=%s | latency=%.2fms",
            anomaly_id,
            elapsed_ms,
        )

        raw_text = getattr(response, "text", None)
        if not raw_text:
            if hasattr(response, "parsed") and isinstance(response.parsed, AIRecommendationResponse):
                return response.parsed
            raise LLMMalformedResponseError("Gemini returned an empty response for recommendations.")

        try:
            payload = json.loads(raw_text)
            return AIRecommendationResponse.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error(
                "Gemini recommendation response failed Pydantic validation | anomaly_id=%s | error=%s",
                anomaly_id,
                exc,
            )
            raise LLMMalformedResponseError(f"Invalid structured recommendation response from Gemini: {exc}") from exc

    def generate_simulation_explanation(
        self, simulation_package: dict[str, Any]
    ) -> str:
        client = self._get_client()
        sim_id = simulation_package.get("simulation_id", "unknown")

        user_content = (
            "Review the following pre-calculated what-if simulation results, assumptions, and limitations, "
            "and provide a concise executive narrative explanation (2-3 sentences):\n\n"
            f"{json.dumps(simulation_package, indent=2)}"
        )

        start_time = time.perf_counter()
        logger.info(
            "Invoking Gemini simulation explanation | simulation_id=%s | model=%s",
            sim_id,
            self.model_name,
        )

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=SIMULATION_EXPLANATION_SYSTEM_INSTRUCTION,
                    temperature=0.2,
                ),
            )
        except LLMConfigurationError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Gemini simulation explanation request failed | simulation_id=%s | latency=%.2fms | error=%s",
                sim_id,
                elapsed_ms,
                exc,
            )
            raise self._classify_exception(exc) from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Gemini simulation explanation received | simulation_id=%s | latency=%.2fms",
            sim_id,
            elapsed_ms,
        )

        raw_text = getattr(response, "text", None)
        if not raw_text or not raw_text.strip():
            raise LLMResponseValidationError("Gemini returned an empty simulation explanation.")
        return raw_text.strip()

