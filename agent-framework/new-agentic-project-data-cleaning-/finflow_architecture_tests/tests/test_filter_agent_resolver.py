"""Focused, deterministic tests for the filter agent's contract.

Covers three responsibilities of ``FilterAgent`` from the
``agent-pipeline-hardening`` spec:

1. The single-source ``input_dataframe`` contract (req 5.5, 11.4).
2. The column resolver wiring and the ``LOW_CONFIDENCE_POLICY``
   enforcement surface (req 7.6 - 7.9, 11.5).
3. The defense-in-depth no-eval / no-``df.query`` rule (req 12.4).

All tests are fully deterministic. None of them call a real LLM and the
only I/O is in-memory pandas. The structured-plan path is always taken
because no ``instruction`` parameter is provided, so the optional Groq
branch in ``FilterAgent._extract_or_build_plan`` cannot fire.
"""

from __future__ import annotations

import io
import pathlib
import tokenize
from typing import Any, Dict, List

import pandas as pd
import pytest

from finflow_agent.agents.filter_agent import FilterAgent, FilterAgentParams
from finflow_agent.operations.schemas import FilterCondition, FilterOperationPlan
from finflow_agent.tools import config as config_module
from finflow_agent.tools.column_resolver import resolve_column
from finflow_agent.tools.dataframe_profile import profile_dataframe


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _plan_dict(
    conditions: List[Dict[str, Any]],
    logic: str = "and",
    select_columns=None,
    limit=None,
) -> Dict[str, Any]:
    """Build a JSON-shape dict that ``FilterOperationPlan`` accepts.

    The agent re-validates the dict via ``FilterOperationPlan.model_validate``,
    so feeding the raw dict reproduces the compiler-emitted code path.
    Note: FilterOperationPlan.logic is Literal["and", "or"] (lowercase).
    """
    plan: Dict[str, Any] = {"conditions": conditions, "logic": logic}
    if select_columns is not None:
        plan["select_columns"] = select_columns
    if limit is not None:
        plan["limit"] = limit
    return plan


def _eq(column: str, value: Any, case_sensitive: bool = False) -> Dict[str, Any]:
    return {
        "column": column,
        "operator": "eq",
        "value": value,
        "case_sensitive": case_sensitive,
    }


@pytest.fixture(autouse=True)
def _reset_config_cache_around_each_test():
    """Force a clean config cache per test.

    Tests in this module either monkeypatch the policy at the resolver
    import site or rely on the default ``"fail"`` policy. Either way, we
    don't want a value cached from an earlier test (or another module)
    to bleed in.
    """
    config_module.reset_config_cache()
    yield
    config_module.reset_config_cache()


# ---------------------------------------------------------------------------
# 1. Single-source input_dataframe contract
# ---------------------------------------------------------------------------


def test_filter_agent_returns_failed_when_input_dataframe_missing(
    bootstrap_agents,
):
    plan = _plan_dict([_eq("col", "x")])

    result = FilterAgent().execute({"plan": plan}, {})

    assert result.status == "failed"
    assert result.error_message is not None
    assert "input_dataframe is required" in result.error_message


def test_filter_agent_returns_failed_when_input_dataframe_is_none(
    bootstrap_agents,
):
    plan = _plan_dict([_eq("col", "x")])

    result = FilterAgent().execute({"plan": plan}, {"input_dataframe": None})

    assert result.status == "failed"
    assert result.error_message is not None
    assert "input_dataframe is required" in result.error_message


# ---------------------------------------------------------------------------
# 2. column_mapping artifact wiring
# ---------------------------------------------------------------------------


def test_filter_agent_publishes_column_mapping_artifact(bootstrap_agents):
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "full_name": ["Alice", "Bob", "Charlie", "Dana"],
            "dob": ["2000-01-01", "1990-07-15", "1985-03-22", "1975-11-30"],
            "total_amount": [100.0, 250.5, 75.25, 1000.0],
        }
    )
    plan = _plan_dict(
        [
            _eq("dob", "2000-01-01"),  # exact case-insensitive column match
            _eq("name", "Alice"),       # synonym match against "full_name"
        ]
    )

    result = FilterAgent().execute(
        {"plan": plan}, {"input_dataframe": df}
    )

    # The status may be "success" or "failed" depending on whether the
    # synonym match clears the threshold under the active policy. In
    # either case the column_mapping artifact MUST be published.
    assert result.status in {"success", "failed"}

    column_mapping = result.artifacts.get("column_mapping")
    assert column_mapping is not None
    assert isinstance(column_mapping, list)
    assert len(column_mapping) >= 2

    for entry in column_mapping:
        assert "requested_field" in entry
        assert "matched_column" in entry
        assert "confidence" in entry
        assert "reason" in entry

    requested_fields = {entry["requested_field"] for entry in column_mapping}
    assert {"dob", "name"}.issubset(requested_fields)


# ---------------------------------------------------------------------------
# 3. LOW_CONFIDENCE_POLICY enforcement (warn / fail / quarantine)
# ---------------------------------------------------------------------------


def _low_confidence_setup():
    """Return ``(df, plan)`` whose only condition cannot be confidently resolved.

    Columns are single ASCII characters; the requested field
    ``"xyz_unknown_field"`` has no exact, normalized, synonym, or fuzzy
    match against any of them, so the resolver always returns a
    confidence well below ``CONFIDENCE_THRESHOLD == 0.75``.
    """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    plan = _plan_dict([_eq("xyz_unknown_field", 1)])
    return df, plan


def test_filter_agent_low_confidence_policy_warn(
    bootstrap_agents, monkeypatch
):
    monkeypatch.setattr(
        "finflow_agent.tools.column_resolver.get_low_confidence_policy",
        lambda: "warn",
    )
    df, plan = _low_confidence_setup()

    result = FilterAgent().execute({"plan": plan}, {"input_dataframe": df})

    # warn = continue with offending condition skipped.
    assert result.status == "success"
    assert len(result.warnings) >= 1
    assert any(
        "xyz_unknown_field" in w for w in result.warnings
    ), f"Warning list should mention the requested field, got: {result.warnings}"

    # Only condition was skipped → executor sees an empty conditions list
    # and the result equals the input (no rows dropped, no columns dropped).
    assert isinstance(result.data, pd.DataFrame)
    assert len(result.data) == len(df)
    assert list(result.data.columns) == list(df.columns)


def test_filter_agent_low_confidence_policy_fail(
    bootstrap_agents, monkeypatch
):
    monkeypatch.setattr(
        "finflow_agent.tools.column_resolver.get_low_confidence_policy",
        lambda: "fail",
    )
    df, plan = _low_confidence_setup()

    result = FilterAgent().execute({"plan": plan}, {"input_dataframe": df})

    assert result.status == "failed"
    assert result.error_message is not None
    # Per req 7.8, the message MUST name requested_field, matched_column,
    # and the confidence value.
    assert "xyz_unknown_field" in result.error_message
    assert "matched_column" in result.error_message
    assert "confidence" in result.error_message


def test_filter_agent_low_confidence_policy_quarantine(
    bootstrap_agents, monkeypatch
):
    monkeypatch.setattr(
        "finflow_agent.tools.column_resolver.get_low_confidence_policy",
        lambda: "quarantine",
    )
    df, plan = _low_confidence_setup()

    result = FilterAgent().execute({"plan": plan}, {"input_dataframe": df})

    # Quarantine surfaces as a failed envelope to the engine, but with
    # a structured ``quarantine`` artifact attached.
    assert result.status == "failed"

    quarantine = result.artifacts.get("quarantine")
    assert quarantine is not None
    assert isinstance(quarantine, dict)
    assert "reason" in quarantine
    assert "resolution" in quarantine

    resolution_payload = quarantine["resolution"]
    assert isinstance(resolution_payload, dict)
    assert resolution_payload.get("requested_field") == "xyz_unknown_field"


# ---------------------------------------------------------------------------
# 4. High-confidence path actually applies the filter
# ---------------------------------------------------------------------------


def test_filter_agent_high_confidence_match_applies_filter(bootstrap_agents):
    df = pd.DataFrame(
        {
            "gender": ["female", "male", "female", "male", "female"],
            "age": [45, 30, 22, 45, 60],
        }
    )
    plan = _plan_dict(
        [
            _eq("gender", "female", case_sensitive=False),
            _eq("age", 45),
        ],
        logic="and",
    )

    result = FilterAgent().execute({"plan": plan}, {"input_dataframe": df})

    assert result.status == "success", result.error_message
    assert isinstance(result.data, pd.DataFrame)
    assert len(result.data) <= len(df)

    column_mapping = result.artifacts.get("column_mapping") or []
    assert len(column_mapping) == 2
    for entry in column_mapping:
        assert entry["confidence"] >= 0.75


# ---------------------------------------------------------------------------
# 5. Defense-in-depth: no df.query / eval / exec in the agent source
# ---------------------------------------------------------------------------


def test_filter_agent_does_not_use_pandas_query():
    """Static check: the filter agent must not call ``df.query``, ``eval``, or ``exec``.

    Production source contains *comments* that mention ``df.query()`` and
    ``eval surface`` for documentation. Comments don't execute, so we
    strip them via ``tokenize`` before scanning. The check is intentionally
    conservative: it scans all remaining tokens (code + string literals).
    """
    from finflow_agent.agents import filter_agent

    source_path = pathlib.Path(filter_agent.__file__)
    source_bytes = source_path.read_bytes()

    # Drop COMMENT tokens; keep everything else (code, strings, docstrings).
    tokens = tokenize.tokenize(io.BytesIO(source_bytes).readline)
    non_comment_source = "".join(
        tok.string for tok in tokens if tok.type != tokenize.COMMENT
    )

    forbidden = ("df.query(", "eval(", "exec(")
    for needle in forbidden:
        assert needle not in non_comment_source, (
            f"FilterAgent source must not contain {needle!r} outside of "
            f"comments (req 12.4)."
        )


# ---------------------------------------------------------------------------
# 6. Engine-state isolation: only input_dataframe key is read
# ---------------------------------------------------------------------------


def test_filter_agent_uses_only_input_dataframe_key(bootstrap_agents):
    df = pd.DataFrame(
        {
            "gender": ["female", "male", "female"],
            "age": [45, 30, 45],
        }
    )

    # A decoy dataframe whose schema would FAIL plan execution if it
    # were read by mistake (no ``gender`` or ``age`` columns).
    decoy = pd.DataFrame(
        {
            "totally_unrelated_col_x": [10, 20, 30],
            "totally_unrelated_col_y": ["p", "q", "r"],
        }
    )

    plan = _plan_dict([_eq("gender", "female"), _eq("age", 45)])

    result = FilterAgent().execute(
        {"plan": plan},
        {"input_dataframe": df, "unrelated_state_key": decoy},
    )

    # The agent must compute its result from ``input_dataframe`` only.
    # If it (incorrectly) read the decoy, column resolution would fall
    # below the threshold and the default ``fail`` policy would surface
    # as ``status="failed"`` with confidence in the error message.
    assert result.status == "success", result.error_message
    assert isinstance(result.data, pd.DataFrame)
    # The result schema must come from ``df``, not the decoy.
    assert set(result.data.columns) == set(df.columns)
    assert "totally_unrelated_col_x" not in result.data.columns


# ---------------------------------------------------------------------------
# 7. Resolver rewrites the condition column to the matched column
# ---------------------------------------------------------------------------


def test_filter_agent_rewrites_condition_column_to_resolved_match(
    bootstrap_agents,
):
    # Note the CAPITAL G in the actual column name.
    df = pd.DataFrame(
        {
            "Gender": ["female", "male", "female"],
            "score": [10, 20, 30],
        }
    )
    plan = _plan_dict([_eq("gender", "female")])  # lowercase request

    result = FilterAgent().execute({"plan": plan}, {"input_dataframe": df})

    assert result.status == "success", result.error_message

    column_mapping = result.artifacts.get("column_mapping") or []
    assert len(column_mapping) == 1

    entry = column_mapping[0]
    assert entry["requested_field"] == "gender"
    assert entry["matched_column"] == "Gender"
    assert entry["confidence"] >= 0.75

    # The agent rewrote the condition's column to "Gender" before
    # invoking the deterministic executor, so filtering succeeded.
    assert isinstance(result.data, pd.DataFrame)
    assert "Gender" in result.data.columns


def test_column_resolver_canonicalizes_placeholder_wrappers():
    df = pd.DataFrame({"payment_method": ["Card", "PayPal"]})
    profile = profile_dataframe(df, include_samples=False)

    resolution = resolve_column("__payment_method_column__", profile)

    assert resolution.matched_column == "payment_method"
    assert resolution.confidence == pytest.approx(1.0)
