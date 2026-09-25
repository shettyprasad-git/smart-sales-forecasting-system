from __future__ import annotations

import copy
import json
import logging
from unittest.mock import MagicMock

import pytest

from google.genai import types
from google.genai.models import _GenerateContentConfig_to_mldev

from backend.app.core.config import settings
from backend.app.llm.gemini_provider import GeminiProvider
from backend.app.llm.provider import (
    LLMConfigurationError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMProviderUnavailableError,
)
from backend.app.schemas.ai_reasoning import AIReasoningResponse
from backend.app.schemas.recommendation_ai import AIRecommendationResponse


class MockAPIError(Exception):
    """Simulates google.genai.errors.APIError or HTTP client exceptions."""

    def __init__(self, code: int, message: str, status: str = ""):
        super().__init__(f"{code} {status}. {message}")
        self.code = code
        self.message = message
        self.status = status


MOCK_EVIDENCE_PACKAGE = {
    "anomaly": {
        "anomaly_id": "anom-20251228-sal-agg",
        "date": "2025-12-28",
        "direction": "positive",
        "actual_value": 150000.0,
        "expected_baseline": 100000.0,
        "deviation": 50000.0,
        "deviation_percent": 50.0,
    }
}

VALID_REASONING_PAYLOAD = {
    "anomaly_id": "anom-20251228-sal-agg",
    "reasoning_headline": "Sales revenue deviation was associated with category shifts and price realization.",
    "executive_interpretation": "The observed spike represents strong customer engagement coinciding with active factors.",
    "key_insights": [
        {
            "dimension": "category",
            "statement": "Clothing category was a strong associated factor in total variance.",
            "supporting_evidence": "Clothing sales exceeded baseline by 20.3%.",
            "confidence": "high",
            "associated_driver": "Category: Clothing",
            "related_driver": "Category: Clothing",
        }
    ],
    "alternative_explanations": ["Broad catalog demand surge"],
    "evidence_assessment": "Category and pricing evidence have robust sample sizes.",
    "uncertainties": ["Lack of elasticity models precludes counterfactual volume conclusions."],
    "validation_questions": ["Were there inventory shortages in competing categories?"],
    "risk_flags": ["Monitor inventory levels."],
    "causal_disclaimer": "Correlational only.",
    "cached": False,
}

VALID_RECOMMENDATION_PAYLOAD = {
    "summary": "Actionable steps to capitalize on observed volume increase.",
    "recommendations": [
        {
            "recommendation_type": "inventory_review",
            "title": "Replenish High-Velocity SKUs",
            "action": "Review safety stock levels for fast-moving items.",
            "reason": "Prevents stockouts during sustained demand surge.",
            "supporting_evidence": "Clothing sales exceeded baseline by 20.3%.",
            "confidence": "high",
            "priority": "high",
            "risk_level": "low",
        }
    ],
}


def create_mock_client():
    client = MagicMock()
    return client


def test_1_default_models_configuration():
    """Requirement: Primary is gemini-3.5-flash-lite, fallback is gemini-3.8-flash."""
    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert settings.gemini_fallback_model == "gemini-3.8-flash"
    assert settings.gemini_max_output_tokens == 800

    provider = GeminiProvider(api_key="test-api-key")
    assert provider.model_name == "gemini-3.5-flash-lite"
    assert provider.fallback_model == "gemini-3.8-flash"
    assert provider.max_output_tokens == 800


def test_2_lite_success_exactly_one_call(caplog):
    """Lite success -> exactly 1 API call, no fallback."""
    client = create_mock_client()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(VALID_REASONING_PAYLOAD)
    mock_resp.parsed = None
    client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(api_key="test-api-key", client=client)

    caplog.set_level(logging.INFO)
    result = provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    assert isinstance(result, AIReasoningResponse)
    assert client.models.generate_content.call_count == 1
    assert client.models.generate_content.call_args[1]["model"] == "gemini-3.5-flash-lite"

    log_text = caplog.text
    assert "requested_model=gemini-3.5-flash-lite" in log_text
    assert "actual_model=gemini-3.5-flash-lite" in log_text
    assert "fallback_used=false" in log_text


def test_3_lite_503_fallback_exactly_two_calls():
    """Lite 503 -> fallback -> exactly 2 API calls."""
    client = create_mock_client()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(VALID_REASONING_PAYLOAD)
    mock_resp.parsed = None

    client.models.generate_content.side_effect = [
        MockAPIError(503, "High demand on Lite", status="UNAVAILABLE"),
        mock_resp,
    ]

    provider = GeminiProvider(api_key="test-api-key", client=client)
    result = provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    assert isinstance(result, AIReasoningResponse)
    assert client.models.generate_content.call_count == 2
    assert client.models.generate_content.call_args_list[0][1]["model"] == "gemini-3.5-flash-lite"
    assert client.models.generate_content.call_args_list[1][1]["model"] == "gemini-3.8-flash"


def test_4_lite_429_fallback_exactly_two_calls():
    """Lite 429 -> fallback -> exactly 2 API calls."""
    client = create_mock_client()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(VALID_REASONING_PAYLOAD)
    mock_resp.parsed = None

    client.models.generate_content.side_effect = [
        MockAPIError(429, "Rate limited on Lite", status="RESOURCE_EXHAUSTED"),
        mock_resp,
    ]

    provider = GeminiProvider(api_key="test-api-key", client=client)
    result = provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    assert isinstance(result, AIReasoningResponse)
    assert client.models.generate_content.call_count == 2
    assert client.models.generate_content.call_args_list[0][1]["model"] == "gemini-3.5-flash-lite"
    assert client.models.generate_content.call_args_list[1][1]["model"] == "gemini-3.8-flash"


def test_5_lite_400_exactly_one_call_no_fallback():
    """Lite 400 -> exactly 1 API call, no fallback."""
    client = create_mock_client()
    client.models.generate_content.side_effect = MockAPIError(
        400, "Invalid argument: bad request payload", status="INVALID_ARGUMENT"
    )

    provider = GeminiProvider(api_key="test-api-key", client=client)

    with pytest.raises(LLMProviderError):
        provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    assert client.models.generate_content.call_count == 1


def test_6_lite_401_exactly_one_call_no_fallback():
    """Lite 401 -> exactly 1 API call, no fallback."""
    client = create_mock_client()
    client.models.generate_content.side_effect = MockAPIError(
        401, "API key not valid", status="UNAUTHENTICATED"
    )

    provider = GeminiProvider(api_key="test-api-key", client=client)

    with pytest.raises(LLMConfigurationError):
        provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    assert client.models.generate_content.call_count == 1


def test_7_lite_404_exactly_one_call_no_fallback():
    """Lite 404 -> exactly 1 API call, no fallback."""
    client = create_mock_client()
    client.models.generate_content.side_effect = MockAPIError(
        404, "models/gemini-3.5-flash-lite is not found", status="NOT_FOUND"
    )

    provider = GeminiProvider(api_key="test-api-key", client=client)

    with pytest.raises(LLMConfigurationError):
        provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    assert client.models.generate_content.call_count == 1


def test_8_lite_failure_plus_fallback_failure_exactly_two_calls():
    """Lite failure + fallback failure -> exactly 2 API calls, raises LLMProviderUnavailableError."""
    client = create_mock_client()
    client.models.generate_content.side_effect = [
        MockAPIError(503, "Lite unavailable", status="UNAVAILABLE"),
        MockAPIError(503, "Fallback 3.8 also unavailable", status="UNAVAILABLE"),
    ]

    provider = GeminiProvider(api_key="test-api-key", client=client)

    with pytest.raises(LLMProviderUnavailableError):
        provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    assert client.models.generate_content.call_count == 2


def test_9_no_path_can_produce_three_plus_requests():
    """Hard guarantee: Even with subsequent errors queued in side_effect, call count strictly stops at 2."""
    client = create_mock_client()
    client.models.generate_content.side_effect = [
        MockAPIError(503, "Error 1", status="UNAVAILABLE"),
        MockAPIError(503, "Error 2", status="UNAVAILABLE"),
        MockAPIError(503, "Error 3 should never be executed", status="UNAVAILABLE"),
        MockAPIError(503, "Error 4 should never be executed", status="UNAVAILABLE"),
    ]

    provider = GeminiProvider(api_key="test-api-key", client=client)

    with pytest.raises(LLMProviderUnavailableError):
        provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    # Strictly 2 calls maximum!
    assert client.models.generate_content.call_count == 2


def test_10_max_output_tokens_800_passed_to_both_models():
    """Verify max_output_tokens=800 is passed to both primary and fallback models."""
    client = create_mock_client()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(VALID_REASONING_PAYLOAD)
    mock_resp.parsed = None

    client.models.generate_content.side_effect = [
        MockAPIError(503, "Lite 503", status="UNAVAILABLE"),
        mock_resp,
    ]

    provider = GeminiProvider(api_key="test-api-key", client=client)
    provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    assert client.models.generate_content.call_count == 2
    # Check Primary call config
    primary_config = client.models.generate_content.call_args_list[0][1]["config"]
    assert primary_config.max_output_tokens == 800

    # Check Fallback call config
    fallback_config = client.models.generate_content.call_args_list[1][1]["config"]
    assert fallback_config.max_output_tokens == 800


def test_11_no_secret_or_prompt_leakage_in_logs(caplog):
    """API keys, tokens, and sensitive prompt payloads never appear in logs."""
    client = create_mock_client()
    secret_key = "AIzaSySuperSecretApiKey999"
    sensitive_evidence = "SensitiveTenantRevenueNumbers48291"

    client.models.generate_content.side_effect = MockAPIError(
        503,
        f"Server failed at https://generativelanguage.googleapis.com?key={secret_key}",
        status="UNAVAILABLE",
    )

    provider = GeminiProvider(api_key=secret_key, fallback_model=None, client=client)

    caplog.set_level(logging.DEBUG)
    with pytest.raises(LLMProviderUnavailableError):
        provider.generate_reasoning({"anomaly": {"anomaly_id": "anom-sec"}, "data": sensitive_evidence})

    log_output = caplog.text
    assert secret_key not in log_output
    assert sensitive_evidence not in log_output
    assert "provider=gemini" in log_output


def test_12_numerical_evidence_unmodified_by_gemini():
    """Verify input evidence numerical values remain completely immutable."""
    client = create_mock_client()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(VALID_REASONING_PAYLOAD)
    mock_resp.parsed = None
    client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(api_key="test-api-key", client=client)

    original_package = copy.deepcopy(MOCK_EVIDENCE_PACKAGE)
    input_package = copy.deepcopy(MOCK_EVIDENCE_PACKAGE)

    res = provider.generate_reasoning(input_package)

    assert isinstance(res, AIReasoningResponse)
    assert input_package == original_package
    assert input_package["anomaly"]["actual_value"] == 150000.0


def test_13_sdk_generate_content_config_inspection():
    """Inspect actual google-genai 2.24.0 SDK GenerateContentConfig behavior."""
    cfg = types.GenerateContentConfig(
        system_instruction="You are an explanation layer.",
        temperature=0.2,
        max_output_tokens=800,
        response_mime_type="application/json",
        response_schema=AIReasoningResponse,
    )

    # 1. Pydantic level inspection
    assert cfg.max_output_tokens == 800
    assert cfg.temperature == 0.2
    assert cfg.response_mime_type == "application/json"
    assert cfg.thinking_config is None

    # 2. SDK serialization inspection using internal MLDev converter
    mock_api_client = MagicMock()
    mock_api_client.vertexai = False
    mldev_dict = _GenerateContentConfig_to_mldev(mock_api_client, cfg, {}, {})

    assert mldev_dict["maxOutputTokens"] == 800
    assert mldev_dict["temperature"] == 0.2
    assert mldev_dict["responseMimeType"] == "application/json"
    assert "thinkingConfig" not in mldev_dict


def test_14_lite_timeout_fallback_exactly_two_calls():
    """Lite timeout (TimeoutError) triggers fallback once -> exactly 2 calls."""
    client = create_mock_client()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(VALID_REASONING_PAYLOAD)
    mock_resp.parsed = None

    client.models.generate_content.side_effect = [
        TimeoutError("Connection timed out waiting for Lite model"),
        mock_resp,
    ]

    provider = GeminiProvider(api_key="test-api-key", client=client)
    result = provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    assert isinstance(result, AIReasoningResponse)
    assert client.models.generate_content.call_count == 2
    assert client.models.generate_content.call_args_list[0][1]["model"] == "gemini-3.5-flash-lite"
    assert client.models.generate_content.call_args_list[1][1]["model"] == "gemini-3.8-flash"


def test_15_lite_malformed_response_no_fallback():
    """Malformed provider response raises LLMMalformedResponseError immediately -> exactly 1 call."""
    client = create_mock_client()
    mock_resp = MagicMock()
    mock_resp.text = "INVALID_NON_JSON_OUTPUT"
    mock_resp.parsed = None
    client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(api_key="test-api-key", client=client)

    with pytest.raises(LLMMalformedResponseError):
        provider.generate_reasoning(MOCK_EVIDENCE_PACKAGE)

    assert client.models.generate_content.call_count == 1
