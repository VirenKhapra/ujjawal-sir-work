"""Tests for the constrained LLM column resolution fallback (Tier 5).

Verifies that when deterministic tiers (exact, normalized, synonym, fuzzy)
all produce scores below CONFIDENCE_THRESHOLD, the resolver attempts a
constrained LLM call that can ONLY select from the actual available columns.

The LLM is always mocked — no real API calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finflow_agent.tools.column_resolver import (
    CONFIDENCE_THRESHOLD,
    ColumnResolution,
    resolve_column,
    _LLM_RESOLUTION_CONFIDENCE,
    _LLMColumnChoice,
)
from finflow_agent.tools.dataframe_profile import profile_dataframe


_LLM_PATCH_TARGET = "finflow_agent.llm.get_chat_groq"


@pytest.fixture
def sample_profile():
    """Profile with columns that have NO string similarity to 'merchant'."""
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Reference": ["REF001"],
        "Payment Mode": ["UPI"],
        "Amount": [1500.00],
        "Quantity": [1],
    })
    return profile_dataframe(df, include_samples=False)


@pytest.fixture(autouse=True)
def enable_llm_resolution(monkeypatch):
    """Enable LLM resolution and provide a fake API key for all tests."""
    monkeypatch.setenv("ENABLE_LLM_COLUMN_RESOLUTION", "true")
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
    yield


def _mock_llm_returning(selected_column, reason="test"):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = _LLMColumnChoice(
        selected_column=selected_column, reason=reason,
    )
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


# ---------------------------------------------------------------------------
# 1. LLM picks the right column when fuzzy fails
# ---------------------------------------------------------------------------

def test_llm_fallback_resolves_merchant_to_payment_mode(sample_profile):
    mock_llm = _mock_llm_returning("Payment Mode", "merchant = payment method")
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = resolve_column("__merchant_column__", sample_profile)
    assert result.confidence == _LLM_RESOLUTION_CONFIDENCE
    assert result.matched_column == "Payment Mode"
    assert "llm_semantic_match" in result.reason


def test_llm_fallback_resolves_vendor_to_payment_mode(sample_profile):
    mock_llm = _mock_llm_returning("Payment Mode", "vendor = payment channel")
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = resolve_column("vendor", sample_profile)
    assert result.confidence == _LLM_RESOLUTION_CONFIDENCE
    assert result.matched_column == "Payment Mode"


# ---------------------------------------------------------------------------
# 2. LLM cannot invent columns
# ---------------------------------------------------------------------------

def test_llm_fallback_rejects_invented_column(sample_profile):
    mock_llm = _mock_llm_returning("Merchant Name", "invented")
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = resolve_column("__merchant_column__", sample_profile)
    assert result.confidence < CONFIDENCE_THRESHOLD
    assert "llm_semantic_match" not in result.reason


# ---------------------------------------------------------------------------
# 3. LLM returns null
# ---------------------------------------------------------------------------

def test_llm_fallback_returns_null_falls_through(sample_profile):
    mock_llm = _mock_llm_returning(None, "no match")
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = resolve_column("__merchant_column__", sample_profile)
    assert result.confidence < CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 4. Graceful failure
# ---------------------------------------------------------------------------

def test_llm_fallback_graceful_on_network_error(sample_profile):
    with patch(_LLM_PATCH_TARGET, side_effect=Exception("timeout")):
        result = resolve_column("__merchant_column__", sample_profile)
    assert result.confidence < CONFIDENCE_THRESHOLD


def test_llm_fallback_graceful_on_parse_error(sample_profile):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = "garbage"
    mock_llm.with_structured_output.return_value = mock_structured
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = resolve_column("__merchant_column__", sample_profile)
    assert result.confidence < CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 5. Disabled via env
# ---------------------------------------------------------------------------

def test_llm_fallback_disabled_via_env(sample_profile, monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_COLUMN_RESOLUTION", "false")
    with patch(_LLM_PATCH_TARGET, side_effect=AssertionError("should not call")):
        result = resolve_column("__merchant_column__", sample_profile)
    assert result.confidence < CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 6. LLM not called when deterministic tiers succeed
# ---------------------------------------------------------------------------

def test_llm_not_called_for_exact_match(sample_profile):
    with patch(_LLM_PATCH_TARGET, side_effect=AssertionError("should not call")):
        result = resolve_column("Amount", sample_profile)
    assert result.confidence == 1.0
    assert result.matched_column == "Amount"


def test_llm_not_called_for_fuzzy_above_threshold(monkeypatch):
    df = pd.DataFrame({"Customer_Age": [25, 30], "Name": ["A", "B"]})
    profile = profile_dataframe(df, include_samples=False)
    with patch(_LLM_PATCH_TARGET, side_effect=AssertionError("should not call")):
        result = resolve_column("customer_age", profile)
    assert result.confidence >= CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 7. End-to-end filter agent integration
# ---------------------------------------------------------------------------

def test_filter_agent_succeeds_with_llm_resolved_column(bootstrap_agents):
    from finflow_agent.agents.filter_agent import FilterAgent

    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Payment Mode": ["UPI", "PayPal", "UPI"],
        "Amount": [100, 200, 300],
    })
    mock_llm = _mock_llm_returning("Payment Mode", "merchant = payment method")
    plan = {
        "conditions": [
            {"column": "__merchant_column__", "operator": "eq", "value": "PayPal"}
        ],
        "logic": "and",
    }
    with patch(_LLM_PATCH_TARGET, return_value=mock_llm):
        result = FilterAgent().execute({"plan": plan}, {"input_dataframe": df})

    assert result.status == "success", result.error_message
    assert len(result.data) == 1
    assert result.data.iloc[0]["Payment Mode"] == "PayPal"
    mapping = result.artifacts.get("column_mapping", [])
    assert len(mapping) == 1
    assert mapping[0]["matched_column"] == "Payment Mode"
    assert mapping[0]["confidence"] == _LLM_RESOLUTION_CONFIDENCE
    assert "llm_semantic_match" in mapping[0]["reason"]
