from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app.services.canonical_intent import build_canonical_intent
from app.services.data_profile import build_data_profile_from_file
from app.services.new_pipeline_bridge import run_new_semantic_pipeline_sync
from app.services.rule_engine import build_validation_warnings
from app.services.new_pipeline_bridge import _convert_filter


def test_forbidden_substring_flags_normalized_variants():
    frame = pd.DataFrame({"payment_method": ["Card", "PayPal", "pay pal"]})

    warnings = build_validation_warnings(
        frame,
        [{"column": "*", "rule": "forbidden_substring", "value": "PayPal", "severity": "warning"}],
    )

    assert warnings == [
        {
            "column": "payment_method",
            "rule": "forbidden_substring",
            "severity": "warning",
            "reason": "",
            "invalid_count": 2,
            "sample_values": ["PayPal", "pay pal"],
        }
    ]


def test_starts_with_flags_invalid_account_prefix():
    frame = pd.DataFrame({"account_number": ["ACC100", "ZCC200"]})

    warnings = build_validation_warnings(
        frame,
        [{"column": "account_number", "rule": "starts_with", "value": "ACC", "severity": "error"}],
    )

    assert warnings[0]["column"] == "account_number"
    assert warnings[0]["rule"] == "starts_with"
    assert warnings[0]["invalid_count"] == 1
    assert warnings[0]["sample_values"] == ["ZCC200"]


def test_date_not_future_handles_naive_dates_without_crashing():
    future_year = pd.Timestamp.now().year + 3
    frame = pd.DataFrame({"invoice_date": ["2024-01-01", f"{future_year}-01-01"]})

    warnings = build_validation_warnings(
        frame,
        [{"column": "invoice_date", "rule": "date_not_future", "severity": "warning"}],
    )

    assert warnings[0]["column"] == "invoice_date"
    assert warnings[0]["rule"] == "date_not_future"
    assert warnings[0]["invalid_count"] == 1


def test_canonical_intent_distinguishes_projection_drop_and_filter():
    columns = ["Customer_ID", "Customer_Name", "Amount"]

    projection = build_canonical_intent(columns, [], "customer id only")
    drop_columns = build_canonical_intent(columns, [], "remove customer id")
    row_filter = build_canonical_intent(columns, [], "show rows where customer id is 1002")

    assert [action["kind"] for action in projection["actions"]] == ["project_columns"]
    assert projection["actions"][0]["requested_fields"][0]["resolved_column"] == "Customer_ID"

    assert [action["kind"] for action in drop_columns["actions"]] == ["drop_columns"]
    assert drop_columns["actions"][0]["requested_fields"][0]["resolved_column"] == "Customer_ID"

    assert [action["kind"] for action in row_filter["actions"]] == ["filter_rows"]
    assert row_filter["actions"][0]["conditions"][0]["field"]["resolved_column"] == "Customer_ID"
    assert row_filter["actions"][0]["conditions"][0]["value"] == 1002


def test_canonical_intent_expands_projection_families_explicitly():
    columns = ["age", "gender", "loan_amount", "loan_status", "loan_term_months"]

    projection = build_canonical_intent(columns, [], "only show age, gender, loans columns")

    assert [action["kind"] for action in projection["actions"]] == ["project_columns"]
    requested_fields = projection["actions"][0]["requested_fields"]
    assert requested_fields[0]["resolved_column"] == "age"
    assert requested_fields[1]["resolved_column"] == "gender"
    assert requested_fields[2]["selection_mode"] == "semantic_family"
    assert requested_fields[2]["resolved_columns"] == ["loan_amount", "loan_status", "loan_term_months"]
    assert projection["resolution_status"] in {"resolved", "repaired"}


def test_canonical_intent_pauses_on_unresolved_projection_family():
    columns = ["age", "gender"]

    projection = build_canonical_intent(columns, [], "only show loans columns")

    assert [action["kind"] for action in projection["actions"]] == ["project_columns"]
    requested_field = projection["actions"][0]["requested_fields"][0]
    assert requested_field["selection_mode"] == "ambiguous"
    assert projection["resolution_status"] == "needs_clarification"


def test_canonical_intent_select_all_projection_is_resolved():
    columns = ["name", "amount", "status"]

    projection = build_canonical_intent(columns, [], "Clean the data and return all columns.")

    assert [action["kind"] for action in projection["actions"]] == ["clean", "project_columns"]
    requested_field = projection["actions"][1]["requested_fields"][0]
    assert requested_field["resolution_method"] == "all_columns"
    assert requested_field["resolved_columns"] == columns
    assert projection["resolution_status"] in {"resolved", "repaired"}


def test_data_profile_build_is_deterministic_and_bounded():
    csv_path = Path(__file__).with_name(".tmp_profile.csv")
    try:
        csv_path.write_text("name,amount,status\nAlice,10,ok\nBob,20,ok\nCharlie,30,pending\n", encoding="utf-8")
        first = build_data_profile_from_file(csv_path, max_preview_rows=2)
        second = build_data_profile_from_file(csv_path, max_preview_rows=2)
    finally:
        csv_path.unlink(missing_ok=True)

    assert first is not None
    assert second is not None
    assert first[0]["file_fingerprint"] == second[0]["file_fingerprint"]
    assert first[0]["profiler_version"] == second[0]["profiler_version"]
    assert first[0]["row_count"] == 3
    assert len(first[0]["preview_rows"]) == 2
    assert len(first[0]["columns"][0]["sample_values"]) <= 3


def test_bridge_semantic_result_repairs_generic_filter_reference_from_preview(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_BRIDGE_API_KEY", "test-bridge-key")

    unresolved_bridge_result = {
        "schema_version": "2.0",
        "intent_id": "intent-1",
        "intent_revision": 1,
        "intent_hash": "hash-1",
        "parent_intent_id": None,
        "original_prompt": "Clean the data and extract rows which contains paypal or cash as field",
        "normalized_prompt": "clean the data and extract rows which contains paypal or cash as field",
        "resolution_status": "needs_clarification",
        "decision": "filter rows (1 condition(s))",
        "evidence": ["new_pipeline_extraction: 1.0"],
        "alternatives_considered": [],
        "actions": [
            {
                "kind": "filter_rows",
                "mode": "keep",
                "conditions": [
                    {
                        "field": {
                            "raw_reference": "field",
                            "resolved_column": None,
                            "resolution_method": "generic_reference",
                            "candidate_columns": [],
                            "evidence": [],
                            "resolved_columns": [],
                        },
                        "operator": "contains",
                        "value": ["paypal", "cash"],
                    }
                ],
                "logic": "and",
            }
        ],
        "output_format": "xlsx",
        "assumptions": [],
        "repair_notes": [],
        "dataframe_profile": {"columns": ["transaction_id", "payment_method", "transaction_status"]},
        "capability_version": "backend.capability.1",
        "capability_snapshot": {},
    }

    def _fake_bridge(*args, **kwargs):
        return unresolved_bridge_result

    monkeypatch.setattr("app.services.new_pipeline_bridge.run_new_semantic_pipeline_sync", _fake_bridge)

    preview_rows = [
        {
            "transaction_id": "T0001",
            "payment_method": "pay pal",
            "transaction_status": "Pending",
        },
        {
            "transaction_id": "T0002",
            "payment_method": "credit card",
            "transaction_status": "Completed",
        },
    ]

    result = build_canonical_intent(
        ["transaction_id", "payment_method", "transaction_status"],
        preview_rows,
        "Clean the data and extract rows which contains paypal or cash as field",
        detected_types={
            "transaction_id": "string",
            "payment_method": "string",
            "transaction_status": "string",
        },
    )

    condition = result["actions"][0]["conditions"][0]
    assert condition["field"]["resolved_column"] == "payment_method"
    assert condition["field"]["resolution_method"] == "profile_value_evidence"
    assert result["resolution_status"] == "repaired"


def test_canonical_intent_grounds_payment_field_from_observed_values(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_BRIDGE_API_KEY", "test-bridge-key")

    unresolved_bridge_result = {
        "schema_version": "2.0",
        "intent_id": "intent-2",
        "intent_revision": 1,
        "intent_hash": "hash-2",
        "parent_intent_id": None,
        "original_prompt": "Clean the data and extract rows which contains paypal or cash as field",
        "normalized_prompt": "clean the data and extract rows which contains paypal or cash as field",
        "resolution_status": "needs_clarification",
        "decision": "filter rows (1 condition(s))",
        "evidence": ["new_pipeline_extraction: 1.0"],
        "alternatives_considered": [],
        "actions": [
            {
                "kind": "filter_rows",
                "mode": "keep",
                "conditions": [
                    {
                        "field": {
                            "raw_reference": "field",
                            "resolved_column": None,
                            "resolution_method": "generic_reference",
                            "candidate_columns": [],
                            "evidence": [],
                            "resolved_columns": [],
                        },
                        "operator": "in",
                        "value": ["paypal", "cash"],
                    }
                ],
                "logic": "and",
            }
        ],
        "output_format": "xlsx",
        "assumptions": [],
        "repair_notes": [],
        "dataframe_profile": {"columns": ["transaction_id", "payment_method", "transaction_status"]},
        "capability_version": "backend.capability.1",
        "capability_snapshot": {},
    }

    def _fake_bridge(*args, **kwargs):
        return unresolved_bridge_result

    monkeypatch.setattr("app.services.new_pipeline_bridge.run_new_semantic_pipeline_sync", _fake_bridge)

    data_profile = {
        "source_columns": ["transaction_id", "payment_method", "transaction_status"],
        "detected_types": {
            "transaction_id": "string",
            "payment_method": "string",
            "transaction_status": "string",
        },
        "preview_rows": [
            {
                "transaction_id": "T0001",
                "payment_method": "Cash",
                "transaction_status": "Pending",
            },
            {
                "transaction_id": "T0002",
                "payment_method": "PayPal",
                "transaction_status": "Completed",
            },
        ],
        "columns": [
            {
                "name": "transaction_id",
                "sample_values": ["T0001", "T0002"],
                "semantic_type_hint": "transaction_id",
                "distinct_count": 2,
            },
            {
                "name": "payment_method",
                "sample_values": ["Cash", "PayPal"],
                "semantic_type_hint": "payment_method",
                "distinct_count": 2,
            },
            {
                "name": "transaction_status",
                "sample_values": ["Pending", "Completed"],
                "semantic_type_hint": "status",
                "distinct_count": 2,
            },
        ],
    }

    result = build_canonical_intent(
        ["transaction_id", "payment_method", "transaction_status"],
        data_profile["preview_rows"],
        "Clean the data and extract rows which contains paypal or cash as field",
        detected_types=data_profile["detected_types"],
        data_profile=data_profile,
    )

    condition = result["actions"][0]["conditions"][0]
    assert condition["field"]["resolved_column"] == "payment_method"
    assert condition["field"]["resolution_method"] == "profile_value_evidence"
    assert condition["operator"] == "in"
    assert condition["value"] == ["paypal", "cash"]
    assert result["resolution_status"] == "repaired"


def test_canonical_intent_normalizes_membership_filter_without_bridge(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_BRIDGE_API_KEY", raising=False)

    data_profile = {
        "source_columns": ["transaction_id", "payment_method", "transaction_status"],
        "detected_types": {
            "transaction_id": "string",
            "payment_method": "string",
            "transaction_status": "string",
        },
        "preview_rows": [
            {
                "transaction_id": "T0001",
                "payment_method": "Cash",
                "transaction_status": "Pending",
            },
            {
                "transaction_id": "T0002",
                "payment_method": "PayPal",
                "transaction_status": "Completed",
            },
        ],
        "columns": [
            {
                "name": "payment_method",
                "sample_values": ["Cash", "PayPal"],
                "semantic_type_hint": "payment_method",
                "distinct_count": 2,
            },
        ],
    }

    result = build_canonical_intent(
        data_profile["source_columns"],
        data_profile["preview_rows"],
        "Clean the data and extract rows which contains paypal or cash as field",
        detected_types=data_profile["detected_types"],
        data_profile=data_profile,
    )

    filter_action = next(action for action in result["actions"] if action["kind"] == "filter_rows")
    condition = filter_action["conditions"][0]
    assert condition["field"]["resolved_column"] == "payment_method"
    assert condition["operator"] == "in"
    assert condition["value"] == ["paypal", "cash"]
    assert result["resolution_status"] == "repaired"


def test_canonical_intent_keeps_education_ambiguous():
    columns = ["education_id", "education_status", "education_duration"]

    result = build_canonical_intent(columns, [], "only show education")

    assert [action["kind"] for action in result["actions"]] == ["project_columns"]
    requested_field = result["actions"][0]["requested_fields"][0]
    assert requested_field["selection_mode"] == "ambiguous"
    assert set(requested_field["candidate_columns"]) == set(columns)
    assert result["resolution_status"] == "needs_clarification"


def test_new_pipeline_bridge_preserves_in_membership():
    field_ref = SimpleNamespace(
        reference_text="payment method",
        resolved_column="payment_method",
        reference_kind=SimpleNamespace(value="semantic_concept"),
    )
    predicate = SimpleNamespace(
        field_ref=field_ref,
        operator="in",
        value=["paypal", "cash"],
    )
    group = SimpleNamespace(operator="and", predicates=[predicate])
    action = SimpleNamespace(logical_groups=[group])

    converted = _convert_filter(action)

    assert converted["logic"] == "and"
    assert len(converted["conditions"]) == 1
    assert converted["conditions"][0]["operator"] == "in"
    assert converted["conditions"][0]["value"] == ["paypal", "cash"]


def test_new_pipeline_bridge_returns_none_on_rate_limit(monkeypatch):
    from finflow_agent.grounding.llm_adapter import LLMProviderError

    monkeypatch.setenv("GROQ_BRIDGE_API_KEY", "bridge-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def _boom(*args, **kwargs):
        raise LLMProviderError(
            "Groq API returned 429: rate limited",
            error_type="rate_limit",
            call_site="extraction",
        )

    monkeypatch.setattr("app.services.new_pipeline_bridge._run_pipeline", _boom)

    result = run_new_semantic_pipeline_sync(
        "Clean the data and extract rows which contains paypal or cash as field",
        ["payment_method"],
    )

    assert result is None
