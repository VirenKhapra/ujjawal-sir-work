import pytest
from pydantic import ValidationError


def test_orchestrator_rejects_legacy_steps_response(monkeypatch, bootstrap_agents, tmp_path):
    from finflow_agent.planning.orchestrator import Orchestrator

    calls = {"count": 0}

    def fake_llm_response(*args, **kwargs):
        calls["count"] += 1
        return {
            "steps": [
                {
                    "step_id": "ingest",
                    "agent": "ingestion_agent",
                    "params": {
                        "resolved_file_path": str(tmp_path / "input.csv"),
                        "file_type": "csv",
                    },
                    "depends_on": [],
                    "input_from": [],
                    "output_key": "df_ingested",
                }
            ]
        }

    import finflow_agent.orchestrator as root_orchestrator
    monkeypatch.setattr(root_orchestrator, "call_groq_json", fake_llm_response)

    result = Orchestrator().build_plan(
        instruction="make a csv report",
        file_path=str(tmp_path / "input.csv"),
        file_name="input.csv",
        output_format="csv",
    )

    assert isinstance(result, dict)
    assert result["status"] == "quarantined"
    assert "steps" in result["reason"].lower() or "planintent" in result["reason"].lower()
    assert calls["count"] == 1


def test_orchestrator_normalizes_valid_first_response_without_repair(monkeypatch, bootstrap_agents, tmp_path):
    from finflow_agent.planning.orchestrator import Orchestrator
    from finflow_agent.state import ExecutionPlan

    calls = {"count": 0}

    def fake_llm_response(*args, **kwargs):
        calls["count"] += 1
        return {
            "needs_filtering": True,
            "output_format": "XLSX",
            "filter_plan": {
                "logic": "AND",
                "conditions": [
                    {"column": "Age", "operator": "greater_than", "value": 45}
                ],
            },
        }

    import finflow_agent.orchestrator as root_orchestrator
    monkeypatch.setattr(root_orchestrator, "call_groq_json", fake_llm_response)

    result = Orchestrator().build_plan(
        instruction="show age above 45",
        file_path=str(tmp_path / "input.csv"),
        file_name="input.csv",
        output_format="xlsx",
    )

    assert isinstance(result, ExecutionPlan)
    assert calls["count"] == 1


def test_orchestrator_performs_one_schema_aware_repair(monkeypatch, bootstrap_agents, tmp_path):
    from finflow_agent.planning.orchestrator import Orchestrator
    from finflow_agent.state import ExecutionPlan

    responses = [
        {
            "needs_filtering": True,
            "output_format": "xlsx",
            "filter_plan": {
                "conditions": {"column": "Age", "operator": "gt", "value": 45}
            },
        },
        {
            "needs_filtering": True,
            "output_format": "xlsx",
            "filter_plan": {
                "conditions": [
                    {"column": "Age", "operator": "gt", "value": 45}
                ]
            },
        },
    ]

    def fake_llm_response(*args, **kwargs):
        return responses.pop(0)

    import finflow_agent.orchestrator as root_orchestrator
    monkeypatch.setattr(root_orchestrator, "call_groq_json", fake_llm_response)

    result = Orchestrator().build_plan(
        instruction="show age above 45",
        file_path=str(tmp_path / "input.csv"),
        file_name="input.csv",
        output_format="xlsx",
    )

    assert isinstance(result, ExecutionPlan)
    assert responses == []


def test_orchestrator_quarantines_unrepairable_validation_without_repair(monkeypatch, bootstrap_agents, tmp_path):
    from finflow_agent.planning.orchestrator import Orchestrator

    calls = {"count": 0}

    def fake_llm_response(*args, **kwargs):
        calls["count"] += 1
        return {
            "needs_filtering": True,
            "output_format": "xlsx",
            "filter_plan": {
                "conditions": [
                    {"column": "Age", "operator": "gt"}
                ]
            },
        }

    import finflow_agent.orchestrator as root_orchestrator
    monkeypatch.setattr(root_orchestrator, "call_groq_json", fake_llm_response)

    result = Orchestrator().build_plan(
        instruction="show age above 45",
        file_path=str(tmp_path / "input.csv"),
        file_name="input.csv",
        output_format="xlsx",
    )

    assert isinstance(result, dict)
    assert result["status"] == "quarantined"
    assert "unrepairable" in result["reason"]
    assert calls["count"] == 1


@pytest.mark.parametrize(
    "field_name, flag_name, expected_error",
    [
        ("cleaning_plan", "needs_cleaning", "cleaning_plan"),
        ("filter_plan", "needs_filtering", "filter_plan"),
        ("calculation_plan", "needs_calculation", "calculation_plan"),
        ("visualization_plan", "needs_visualization", "visualization_plan"),
    ],
)
def test_plan_intent_requires_requested_stage_plan(
    field_name,
    flag_name,
    expected_error,
):
    from finflow_agent.planning.intent_schema import PlanIntent

    kwargs = {
        "output_format": "xlsx",
        flag_name: True,
    }

    with pytest.raises(ValidationError, match=expected_error):
        PlanIntent(**kwargs)


def test_plan_intent_rejects_unormalized_drop_duplicates_all_columns_subset():
    from finflow_agent.planning.intent_schema import PlanIntent

    with pytest.raises(ValidationError):
        PlanIntent.model_validate(
            {
                "needs_cleaning": True,
                "output_format": "xlsx",
                "cleaning_plan": {
                    "operations": [
                        {"type": "trim_whitespace", "columns": "__all_string_columns__"},
                        {"type": "normalize_column_names", "style": "snake_case"},
                        {
                            "type": "drop_duplicates",
                            "subset": "__all_columns__",
                            "keep": "first",
                        },
                    ]
                },
            }
        )


def test_cleaning_operation_validation_uses_discriminator():
    from finflow_agent.operations.schemas import CleaningOperationPlan

    with pytest.raises(ValidationError) as exc:
        CleaningOperationPlan.model_validate(
            {"operations": [{"type": "not_a_cleaning_operation"}]}
        )

    message = str(exc.value)
    assert "union_tag_invalid" in message
    assert "TrimWhitespaceOperation.columns" not in message


def test_plan_intent_rejects_unormalized_filter_operator_equals():
    from finflow_agent.planning.intent_schema import PlanIntent

    with pytest.raises(ValidationError):
        PlanIntent.model_validate(
            {
                "needs_filtering": True,
                "output_format": "xlsx",
                "filter_plan": {
                    "conditions": [
                        {"column": "status", "operator": "equals", "value": "Paid"}
                    ]
                },
            }
        )


def test_plan_intent_rejects_unormalized_absolute_value_all_numeric_columns():
    from finflow_agent.planning.intent_schema import PlanIntent

    with pytest.raises(ValidationError):
        PlanIntent.model_validate(
            {
                "needs_calculation": True,
                "output_format": "xlsx",
                "calculation_plan": {
                    "operations": [
                        {
                            "type": "absolute_value",
                            "columns": "__all_numeric_columns__",
                        }
                    ]
                },
            }
        )


@pytest.mark.parametrize(
    "field_name, flag_name, expected_error",
    [
        ("cleaning_plan", "needs_cleaning", "cleaning_plan"),
        ("filter_plan", "needs_filtering", "filter_plan"),
        ("calculation_plan", "needs_calculation", "calculation_plan"),
        ("visualization_plan", "needs_visualization", "visualization_plan"),
    ],
)
def test_compiler_rejects_missing_requested_stage_plan(
    field_name,
    flag_name,
    expected_error,
    bootstrap_agents,
    tmp_path,
):
    from finflow_agent.planning.intent_schema import PlanIntent
    from finflow_agent.planning.compiler import compile_intent_to_plan

    intent = PlanIntent.model_construct(
        output_format="xlsx",
        **{flag_name: True, field_name: None},
    )

    with pytest.raises(ValueError, match=expected_error):
        compile_intent_to_plan(
            intent=intent,
            resolved_file_path=str(tmp_path / "input.csv"),
            file_type="csv",
            output_dir=str(tmp_path / "outputs"),
            file_prefix="test",
        )


def test_validate_plan_rejects_missing_input_from(bootstrap_agents):
    from finflow_agent.state import ExecutionPlan, PlanStep
    from finflow_agent.planning.validators import validate_plan

    plan = ExecutionPlan(
        steps=[
            PlanStep(
                step_id="ingest",
                agent="ingestion_agent",
                params={"resolved_file_path": "x.csv", "file_type": "csv"},
                depends_on=[],
                input_from=[],
                output_key="df_ingested",
            ),
            PlanStep(
                step_id="report",
                agent="reporting_agent",
                params={
                    "plan": {"output_format": "xlsx"},
                    "output_dir": "outputs",
                    "file_prefix": "test",
                },
                depends_on=["ingest"],
                input_from=["df_missing"],
                output_key="report_output",
            ),
        ]
    )

    is_valid, error = validate_plan(plan)
    assert not is_valid
    assert "input_from" in error
    assert "df_missing" in error
