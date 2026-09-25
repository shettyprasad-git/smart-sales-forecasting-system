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
    Enforces a strict, hard maximum of 2 external Gemini requests per operation:
      Request #1: gemini-3.5-flash-lite (primary model)
      Request #2: gemini-3.8-flash (bounded fallback, transient errors only)
    Tokens are bounded to max_output_tokens (default: 800) across both models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        fallback_model: str | None = None,
        max_output_tokens: int | None = None,
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
            or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        )
        fb = (
            fallback_model
            if fallback_model is not None
            else (
                getattr(settings, "gemini_fallback_model", None)
                or os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash")
            )
        )
        self.fallback_model = fb.strip() if fb and fb.strip() else None
        if self.fallback_model == self.model_name:
            self.fallback_model = None

        tokens = max_output_tokens or getattr(settings, "gemini_max_output_tokens", None)
        if tokens is None:
            raw_tokens = os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "800")
            tokens = int(raw_tokens) if raw_tokens.isdigit() else 800
        self.max_output_tokens = tokens

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
            logger.error("Failed to initialize Google GenAI Client: %s", exc.__class__.__name__)
            raise LLMConfigurationError(f"Failed to initialize Gemini Client: {exc.__class__.__name__}") from exc

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        for attr in ("code", "status_code", "http_status"):
            val = getattr(exc, attr, None)
            if isinstance(val, int) and 100 <= val <= 599:
                return val
            if isinstance(val, str) and val.isdigit():
                iv = int(val)
                if 100 <= iv <= 599:
                    return iv

        resp = getattr(exc, "response", None)
        if resp is not None:
            for attr in ("status_code", "code"):
                val = getattr(resp, attr, None)
                if isinstance(val, int) and 100 <= val <= 599:
                    return val

        err_str = str(exc)
        for candidate in (503, 502, 500, 504, 429, 408, 404, 403, 401, 400):
            if f"{candidate}" in err_str:
                return candidate

        return None

    @classmethod
    def _categorize_exception(cls, exc: Exception) -> str:
        code = cls._extract_status_code(exc)
        err_str = str(exc).lower()

        if code == 503 or "503" in err_str or "unavailable" in err_str or "high demand" in err_str:
            return "capacity"
        if (
            code == 429
            or "429" in err_str
            or "resource_exhausted" in err_str
            or "rate limit" in err_str
            or "quota" in err_str
        ):
            return "rate_limited"
        if (
            code in (408, 504)
            or isinstance(exc, TimeoutError)
            or "timeout" in err_str
            or "timed out" in err_str
            or "deadline" in err_str
        ):
            return "timeout"
        if code in (500, 502) or "bad gateway" in err_str or "server error" in err_str:
            return "provider_error"
        if (
            code == 401
            or "api_key_invalid" in err_str
            or "api key not valid" in err_str
            or "invalid api key" in err_str
            or "unauthorized" in err_str
        ):
            return "authentication_error"
        if code == 403 or "permission_denied" in err_str or "forbidden" in err_str:
            return "permission_denied"
        if code == 404 or "not_found" in err_str or "not found" in err_str:
            return "not_found"
        if code == 400 or "invalid_argument" in err_str or "bad request" in err_str:
            return "invalid_argument"
        if (
            isinstance(exc, ConnectionError)
            or "connection reset" in err_str
            or "connection refused" in err_str
        ):
            return "connection_error"
        return "provider_error"

    @classmethod
    def _is_transient_error(cls, exc: Exception) -> bool:
        category = cls._categorize_exception(exc)
        if category in ("capacity", "rate_limited", "timeout", "connection_error", "provider_error"):
            return True
        if category in ("authentication_error", "permission_denied", "not_found", "invalid_argument"):
            return False
        code = cls._extract_status_code(exc)
        if code in (400, 401, 403, 404):
            return False
        if code in (408, 429, 500, 502, 503, 504):
            return True
        return False

    def _classify_exception(self, exc: Exception) -> Exception:
        if isinstance(
            exc,
            (
                LLMConfigurationError,
                LLMProviderUnavailableError,
                LLMMalformedResponseError,
                LLMResponseValidationError,
            ),
        ):
            return exc

        category = self._categorize_exception(exc)
        if category == "not_found":
            return LLMConfigurationError(f"Configured Gemini model '{self.model_name}' was not found.")
        if category in ("authentication_error", "permission_denied"):
            return LLMConfigurationError("Gemini API key is invalid or unauthorized.")
        if category in ("capacity", "rate_limited", "timeout", "connection_error", "provider_error"):
            return LLMProviderUnavailableError(
                f"Gemini provider is currently unavailable or capacity constrained ({category})."
            )
        return LLMProviderError(f"Gemini API request failed ({category}).")

    def _execute_with_retry_and_fallback(
        self,
        contents: Any,
        config: types.GenerateContentConfig,
        context_tag: str,
    ) -> Any:
        client = self._get_client()

        # Request #1: Primary Model (default: gemini-3.5-flash-lite)
        start_time = time.perf_counter()
        logger.info(
            "Invoking Gemini | provider=gemini | requested_model=%s | attempt=1/2 | context=%s",
            self.model_name,
            context_tag,
        )

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Gemini request succeeded | provider=gemini | requested_model=%s | actual_model=%s | fallback_used=false | latency_ms=%.2f | context=%s",
                self.model_name,
                self.model_name,
                elapsed_ms,
                context_tag,
            )
            return response
        except LLMConfigurationError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            status_code = self._extract_status_code(exc)
            category = self._categorize_exception(exc)
            is_transient = self._is_transient_error(exc)

            if not is_transient:
                logger.error(
                    "Gemini non-transient error | provider=gemini | requested_model=%s | actual_model=%s | fallback_used=false | status=%s | latency_ms=%.2f | error_class=%s | context=%s",
                    self.model_name,
                    self.model_name,
                    status_code,
                    elapsed_ms,
                    category,
                    context_tag,
                )
                raise self._classify_exception(exc) from exc

            # Primary model experienced transient error: log and immediately try fallback once
            logger.warning(
                "Gemini primary unavailable | provider=gemini | requested_model=%s | fallback_used=false | status=%s | latency_ms=%.2f | error_class=%s | context=%s",
                self.model_name,
                status_code,
                elapsed_ms,
                category,
                context_tag,
            )

            # Request #2: Bounded Fallback Model (default: gemini-3.8-flash)
            if self.fallback_model and self.fallback_model != self.model_name:
                fallback_start = time.perf_counter()
                logger.info(
                    "Invoking Gemini fallback | provider=gemini | requested_model=%s | fallback_model=%s | attempt=2/2 | context=%s",
                    self.model_name,
                    self.fallback_model,
                    context_tag,
                )
                try:
                    response = client.models.generate_content(
                        model=self.fallback_model,
                        contents=contents,
                        config=config,
                    )
                    elapsed_ms = (time.perf_counter() - fallback_start) * 1000
                    logger.info(
                        "Gemini fallback succeeded | provider=gemini | requested_model=%s | actual_model=%s | fallback_used=true | latency_ms=%.2f | context=%s",
                        self.model_name,
                        self.fallback_model,
                        elapsed_ms,
                        context_tag,
                    )
                    return response
                except Exception as fb_exc:
                    elapsed_ms = (time.perf_counter() - fallback_start) * 1000
                    fb_status = self._extract_status_code(fb_exc)
                    fb_category = self._categorize_exception(fb_exc)
                    logger.error(
                        "Gemini fallback failed | provider=gemini | requested_model=%s | actual_model=%s | fallback_used=true | status=%s | latency_ms=%.2f | error_class=%s | context=%s",
                        self.model_name,
                        self.fallback_model,
                        fb_status,
                        elapsed_ms,
                        fb_category,
                        context_tag,
                    )
                    raise self._classify_exception(fb_exc) from fb_exc

            # Fallback model not configured or same as primary
            raise self._classify_exception(exc) from exc

    def generate_reasoning(self, evidence_package: dict[str, Any]) -> AIReasoningResponse:
        anomaly_id = evidence_package.get("anomaly", {}).get("anomaly_id", "unknown")
        context_tag = f"reasoning:{anomaly_id}"

        user_content = (
            "Analyze the following empirical sales anomaly evidence package and generate "
            f"structured executive reasoning:\n\n{json.dumps(evidence_package, indent=2)}"
        )

        response = self._execute_with_retry_and_fallback(
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=AIReasoningResponse,
                temperature=0.2,
                max_output_tokens=self.max_output_tokens,
            ),
            context_tag=context_tag,
        )

        raw_text = getattr(response, "text", None)
        if not raw_text:
            if hasattr(response, "parsed") and isinstance(response.parsed, AIReasoningResponse):
                return response.parsed
            raise LLMMalformedResponseError("Gemini returned an empty response.")

        try:
            payload = json.loads(raw_text)
            return AIReasoningResponse.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error(
                "Gemini response failed Pydantic validation | context=%s | error=%s",
                context_tag,
                exc.__class__.__name__,
            )
            raise LLMMalformedResponseError(f"Invalid structured response from Gemini: {exc}") from exc

    def generate_recommendations(
        self, recommendation_package: dict[str, Any]
    ) -> AIRecommendationResponse:
        anomaly_id = recommendation_package.get("anomaly", {}).get("anomaly_id", "unknown")
        context_tag = f"recommendations:{anomaly_id}"

        user_content = (
            "Evaluate the following empirical anomaly evidence package and deterministically eligible "
            "recommendation categories, then formulate structured prescriptive action recommendations:\n\n"
            f"{json.dumps(recommendation_package, indent=2)}"
        )

        response = self._execute_with_retry_and_fallback(
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=RECOMMENDATION_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=AIRecommendationResponse,
                temperature=0.2,
                max_output_tokens=self.max_output_tokens,
            ),
            context_tag=context_tag,
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
                "Gemini recommendation response failed Pydantic validation | context=%s | error=%s",
                context_tag,
                exc.__class__.__name__,
            )
            raise LLMMalformedResponseError(f"Invalid structured recommendation response from Gemini: {exc}") from exc

    def generate_simulation_explanation(
        self, simulation_package: dict[str, Any]
    ) -> str:
        sim_id = simulation_package.get("simulation_id", "unknown")
        context_tag = f"simulation:{sim_id}"

        user_content = (
            "Review the following pre-calculated what-if simulation results, assumptions, and limitations, "
            "and provide a concise executive narrative explanation (2-3 sentences):\n\n"
            f"{json.dumps(simulation_package, indent=2)}"
        )

        response = self._execute_with_retry_and_fallback(
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SIMULATION_EXPLANATION_SYSTEM_INSTRUCTION,
                temperature=0.2,
                max_output_tokens=self.max_output_tokens,
            ),
            context_tag=context_tag,
        )

        raw_text = getattr(response, "text", None)
        if not raw_text or not raw_text.strip():
            raise LLMResponseValidationError("Gemini returned an empty simulation explanation.")
        return raw_text.strip()
