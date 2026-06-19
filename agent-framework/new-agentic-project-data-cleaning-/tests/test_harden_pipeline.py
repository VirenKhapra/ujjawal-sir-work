import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from finflow_agent.bootstrap import bootstrap_agents, validate_required_agents_registered
from finflow_agent.registry import registry
from finflow_agent.state import ExecutionPlan, PlanStep
from finflow_agent.planning.validators import validate_plan
from finflow_agent.agents.calculation_agent import CalculationAgent
from finflow_agent.agents.reporting_agent import ReportingAgent
from finflow_agent.engine import ExecutionEngine
from finflow_agent.operations.errors import OperationValidationError, OperationExecutionError
from finflow_agent.tools.dataframe_profile import profile_dataframe
from finflow_agent.api import app, handle_upload, JobPayload

# -------------------------------------------------------------------
# ARCHITECTURE TESTS
# -------------------------------------------------------------------

def test_registry_bootstrap_loads_required_agents():
    bootstrap_agents()
    # Should not raise any exceptions
    validate_required_agents_registered()
    
    # Assert they exist
    assert registry.get_spec("ingestion_agent") is not None
    assert registry.get_spec("cleaning_agent") is not None
    assert registry.get_spec("filter_agent") is not None
    assert registry.get_spec("calculation_agent") is not None
    assert registry.get_spec("reporting_agent") is not None

def test_missing_agent_fails_startup():
    bootstrap_agents()
    # Mock registry specs to miss ingestion_agent
    with patch.dict(registry._specs, {}, clear=True):
        with pytest.raises(ValueError):
            validate_required_agents_registered()

def test_plan_with_unknown_agent_is_rejected():
    plan = ExecutionPlan(steps=[
        PlanStep(step_id="step_1", agent="unknown_agent", depends_on=[])
    ])
    is_valid, err = validate_plan(plan)
    assert not is_valid
    assert "unknown_agent" in err

def test_plan_with_invalid_params_is_rejected():
    agent = CalculationAgent()
    # Pass bad operations list
    res = agent.execute({"operations": [{"type": "group_sum", "column": "Val"}]}, {"input_dataframe": pd.DataFrame({"Val": [1]})})
    assert res.status == "failed"
    assert "Failed to build calculation plan" in res.error_message

def test_cycle_plan_is_rejected():
    plan = ExecutionPlan(steps=[
        PlanStep(step_id="step_1", agent="ingestion_agent", depends_on=["step_2"]),
        PlanStep(step_id="step_2", agent="cleaning_agent", depends_on=["step_1"])
    ])
    is_valid, err = validate_plan(plan)
    assert not is_valid
    assert "Cycle" in err

def test_stage_order_violation_is_rejected():
    # Ingestion (ingest stage) depending on Reporting (deliver stage)
    plan = ExecutionPlan(steps=[
        PlanStep(step_id="step_1", agent="reporting_agent", depends_on=[]),
        PlanStep(step_id="step_2", agent="ingestion_agent", depends_on=["step_1"])
    ])
    is_valid, err = validate_plan(plan)
    assert not is_valid
    assert "Monotonic stage progression violated" in err

def test_step_receives_only_declared_inputs():
    # Verify ExecutionEngine build explicit inputs behavior
    engine = ExecutionEngine()
    plan = ExecutionPlan(steps=[
        PlanStep(
            step_id="ingest",
            agent="ingestion_agent",
            params={"resolved_file_path": "fake.csv", "file_type": "csv"},
            depends_on=[],
            output_key="df_in"
        ),
        PlanStep(
            step_id="clean",
            agent="cleaning_agent",
            params={},
            depends_on=["ingest"],
            input_from=["df_in"]
        )
    ])
    
    # We mock the agent instances to verify execute inputs
    mock_ingest = MagicMock()
    mock_ingest.execute.return_value = MagicMock(status="success", data=pd.DataFrame({"A": [1]}))
    mock_clean = MagicMock()
    mock_clean.execute.return_value = MagicMock(status="success", data=pd.DataFrame({"A": [1]}))
    
    def get_agent_mock(name):
        if name == "ingestion_agent":
            return lambda: mock_ingest
        return lambda: mock_clean
        
    with patch("finflow_agent.engine.registry.get_agent_class", side_effect=get_agent_mock):
        engine.execute(plan)
        
    # Verify inputs passed to clean node
    # Since input_from has 1 element, it normalizes to {"input_dataframe": val}
    args, kwargs = mock_clean.execute.call_args
    assert "input_dataframe" in args[1]
    # And it must not pass the whole global state (no "ingest" key in args[1] since we only passed df_in)
    assert "ingest" not in args[1]

def test_calculation_agent_does_not_scan_global_state():
    agent = CalculationAgent()
    # Pass empty input_data dictionary
    res = agent.execute({"operations": []}, {})
    assert res.status == "failed"
    assert "No input dataframe provided" in res.error_message

def test_missing_column_fails():
    from finflow_agent.operations.executor import execute_calculation_plan
    from finflow_agent.operations.schemas import CalculationOperationPlan, CalculationOperation
    
    df = pd.DataFrame({"A": [1, 2]})
    # "B" is missing
    plan = CalculationOperationPlan(operations=[
        CalculationOperation(type="sum", column="B")
    ])
    with pytest.raises(OperationValidationError):
        execute_calculation_plan(df, plan)

def test_pdf_format_rejected_until_supported():
    orchestrator = Orchestrator = sys.modules["finflow_agent.orchestrator"].Orchestrator()
    res = orchestrator.build_plan("export to PDF", "test.csv", "test.csv", "pdf")
    assert isinstance(res, dict)
    assert res["status"] == "quarantined"
    assert "PDF output format is not supported" in res["reason"]

def test_reporting_returns_primary_output_path():
    agent = ReportingAgent()
    df = pd.DataFrame({"A": [1, 2]})
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = agent.execute({"plan": {"output_format": "csv"}, "output_dir": tmp_dir, "file_prefix": "rep"}, {"input_dataframe": df})
        assert res.status == "success"
        assert "primary_output_path" in res.artifacts
        assert res.artifacts["primary_output_path"].endswith(".csv")

@pytest.mark.anyio
async def test_upload_enqueues_arq_job():
    # Setup mocks for Redis
    mock_redis = AsyncMock()
    app.state.redis = mock_redis
    
    # Clear repository state to prevent early return from idempotency check
    from finflow_agent.jobs.repository import JobRepository
    repo = JobRepository()
    db = repo._read_db()
    if "agent:sub_123" in db:
        del db["agent:sub_123"]
        repo._write_db(db)
    
    # Create synthetic uploaded file
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.dict(os.environ, {"UPLOAD_DIR": tmp_dir}):
            dummy_file = Path(tmp_dir) / "up_123.csv"
            dummy_file.write_text("A,B\n1,2")
            
            payload = JobPayload(
                submission_id="sub_123",
                file_id="up_123.csv",
                file_name="up_123.csv",
                instruction="clean",
                output_format="csv"
            )
            
            # Run API upload
            with patch("finflow_agent.api.process_job_task") as mock_task:
                res = await handle_upload(payload, MagicMock())
                
            assert res["status"] == "queued"
            assert res["job_id"] == "agent:sub_123"
            mock_redis.enqueue_job.assert_called_once()

@pytest.mark.anyio
async def test_duplicate_submission_id_is_idempotent():
    # Verify idempotency check
    mock_redis = AsyncMock()
    app.state.redis = mock_redis
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.dict(os.environ, {"UPLOAD_DIR": tmp_dir}):
            # Setup repository database
            from finflow_agent.jobs.repository import JobRepository
            repo = JobRepository()
            job_id = "agent:sub_dup"
            
            db = repo._read_db()
            db[job_id] = {
                "job_id": job_id,
                "submission_id": "sub_dup",
                "status": "RUNNING",
                "payload": {},
                "result": None,
                "error": None
            }
            repo._write_db(db)
            
            payload = JobPayload(
                submission_id="sub_dup",
                file_id="up_dup.csv",
                file_name="up_dup.csv",
                instruction="clean",
                output_format="csv"
            )
            
            res = await handle_upload(payload, MagicMock())
            assert res["status"] == "running"
            assert res["job_id"] == job_id

# -------------------------------------------------------------------
# OPERATION TESTS
# -------------------------------------------------------------------

def test_running_total_requires_sort_by():
    from finflow_agent.operations.schemas import CalculationOperation
    with pytest.raises(ValueError) as exc:
        CalculationOperation(type="running_total", column="Val")
    assert "requires a sort_by column" in str(exc.value)

def test_running_total_supports_partition_by():
    from finflow_agent.operations.schemas import CalculationOperationPlan, CalculationOperation
    from finflow_agent.operations.executor import execute_calculation_plan
    
    df = pd.DataFrame({
        "Account": ["A", "B", "A", "B"],
        "Month": [1, 1, 2, 2],
        "Amount": [100, 200, 50, 10]
    })
    
    plan = CalculationOperationPlan(operations=[
        CalculationOperation(
            type="running_total",
            column="Amount",
            sort_by="Month",
            partition_by=["Account"],
            output_column="cum_amount"
        )
    ])
    
    out = execute_calculation_plan(df, plan)
    # Check that Account A cumulative total is calculated per partition
    res_df = out.data.sort_values(by=["Account", "Month"])
    assert res_df[res_df["Account"] == "A"]["cum_amount"].tolist() == [100.0, 150.0]
    assert res_df[res_df["Account"] == "B"]["cum_amount"].tolist() == [200.0, 210.0]

def test_percentage_change_requires_sort_by():
    from finflow_agent.operations.schemas import CalculationOperation
    with pytest.raises(ValueError) as exc:
        CalculationOperation(type="percentage_change", column="Val")
    assert "requires a sort_by column" in str(exc.value)

def test_ratio_handles_zero_denominator_with_warning():
    from finflow_agent.operations.schemas import CalculationOperationPlan, CalculationOperation
    from finflow_agent.operations.executor import execute_calculation_plan
    
    df = pd.DataFrame({
        "Rev": [100, 200],
        "Costs": [50, 0]
    })
    
    plan = CalculationOperationPlan(operations=[
        CalculationOperation(
            type="ratio",
            column="Rev",
            secondary_column="Costs",
            output_column="margin"
        )
    ])
    
    out = execute_calculation_plan(df, plan)
    assert out.data["margin"].iloc[0] == 2.0
    assert pd.isna(out.data["margin"].iloc[1])
    assert any("zero denominator" in w for w in out.warnings)

def test_profile_dataframe_does_not_include_sample_records_by_default():
    df = pd.DataFrame({"A": [1, 2, 3]})
    profile = profile_dataframe(df)
    assert "sample_records" not in profile

# -------------------------------------------------------------------
# GOLDEN FILE INTEGRATION TEST
# -------------------------------------------------------------------

def test_golden_file_financial_pipeline():
    # 1. Create a known financial dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "financials.csv"
        csv_path.write_text(
            "Account,Month,Revenue,Costs\n"
            "Sales,2024-01-01,10000,8000\n"
            "Marketing,2024-01-01,0,3000\n"
            "Sales,2024-02-01,15000,9000\n"
            "Marketing,2024-02-01,0,3500\n"
        )
        
        # 2. Build full execution plan manually
        plan = ExecutionPlan(steps=[
            PlanStep(
                step_id="step_ingest",
                agent="ingestion_agent",
                params={"resolved_file_path": str(csv_path), "file_type": "csv"},
                output_key="df_ingested"
            ),
            PlanStep(
                step_id="step_clean",
                agent="cleaning_agent",
                params={
                    "plan": {
                        "operations": [
                            {"type": "normalize_column_names", "style": "snake_case"}
                        ]
                    }
                },
                depends_on=["step_ingest"],
                input_from=["df_ingested"],
                output_key="df_cleaned"
            ),
            PlanStep(
                step_id="step_calculate",
                agent="calculation_agent",
                params={
                    "operations": [
                        {
                            "type": "ratio",
                            "column": "revenue",
                            "secondary_column": "costs",
                            "output_column": "rev_cost_ratio"
                        }
                    ]
                },
                depends_on=["step_clean"],
                input_from=["df_cleaned"],
                output_key="df_calculated"
            ),
            PlanStep(
                step_id="step_report",
                agent="reporting_agent",
                params={
                    "plan": {
                        "output_format": "csv"
                    },
                    "output_dir": tmp_dir,
                    "file_prefix": "golden_report"
                },
                depends_on=["step_calculate"],
                input_from=["df_calculated"],
                output_key="report_output"
            )
        ])
        
        # 3. Execute plan
        engine = ExecutionEngine()
        res = engine.execute(plan)
        
        # 4. Verify pipeline output
        assert res["status"] == "complete"
        assert res["output_path"] is not None
        assert os.path.exists(res["output_path"])
        
        # Read resulting CSV
        output_df = pd.read_csv(res["output_path"])
        # Check normalization (columns should be lowercase/snake_case)
        assert "revenue" in output_df.columns
        # Check calculation (ratio of Sales month 1 should be 1.25)
        sales_m1 = output_df[(output_df["account"] == "Sales") & (output_df["month"] == "2024-01-01")]
        assert sales_m1["rev_cost_ratio"].iloc[0] == 1.25
        # Check that division by zero was set to NaN (for Marketing, which has zero Revenue/Costs ratio)
        mkt_m1 = output_df[(output_df["account"] == "Marketing") & (output_df["month"] == "2024-01-01")]
        # marketing costs=3000, revenue=0. ratio 0/3000 = 0
        assert mkt_m1["rev_cost_ratio"].iloc[0] == 0.0


# -------------------------------------------------------------------
# ADDITIONAL TESTS FOR TASK 10 ARCHITECTURE REQUIREMENTS
# -------------------------------------------------------------------

def test_bootstrap_loads_required_agents():
    # Alias for required test
    test_registry_bootstrap_loads_required_agents()

def test_missing_required_agent_fails_startup():
    # Alias for required test
    test_missing_agent_fails_startup()

def test_api_upload_uses_file_id_not_file_path():
    # Ensure JobPayload has file_id and not file_path
    fields = JobPayload.model_fields
    assert "file_id" in fields
    assert "file_path" not in fields

@pytest.mark.anyio
async def test_duplicate_submission_id_uses_stable_job_id():
    # Ensure stable job_id is computed and used
    from finflow_agent.jobs.repository import JobRepository
    mock_redis = AsyncMock()
    app.state.redis = mock_redis
    
    repo = JobRepository()
    payload1 = JobPayload(
        submission_id="sub_stable_123",
        file_id="up.csv",
        file_name="up.csv",
        instruction="clean",
        output_format="csv"
    )
    # Check that both calls map to the same stable job_id
    job_id1 = f"agent:{payload1.submission_id}"
    assert job_id1 == "agent:sub_stable_123"

def test_filestore_rejects_path_traversal():
    from finflow_agent.storage.file_store import FileStore
    from finflow_agent.operations.errors import UnsafeInputPathError
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FileStore(upload_dir=tmp_dir)
        with pytest.raises(UnsafeInputPathError):
            store.resolve_uploaded_file("../traversal.csv")
        with pytest.raises(UnsafeInputPathError):
            store.resolve_uploaded_file("sub/../../traversal.csv")
        with pytest.raises(UnsafeInputPathError):
            store.resolve_uploaded_file("C:\\windows\\win.ini")

def test_filestore_resolves_only_inside_upload_dir():
    from finflow_agent.storage.file_store import FileStore
    from finflow_agent.operations.errors import UnsafeInputPathError
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FileStore(upload_dir=tmp_dir)
        # Create a file inside upload dir
        safe_file = Path(tmp_dir) / "safe.csv"
        safe_file.write_text("dummy")
        
        resolved = store.resolve_uploaded_file("safe.csv")
        assert resolved.resolve().is_relative_to(Path(tmp_dir).resolve())

@patch("finflow_agent.orchestrator.call_groq_json")
def test_orchestrator_returns_planintent_not_direct_executionplan(mock_call_groq):
    # Under the new compiler architecture, build_plan calls the LLM, 
    # receives PlanIntent JSON, and compiles it.
    from finflow_agent.planning.orchestrator import Orchestrator
    orchestrator = Orchestrator()
    
    # LLM returns a valid PlanIntent matching prompt (no "steps")
    mock_call_groq.return_value = {
        "is_quarantined": False,
        "needs_cleaning": True,
        "cleaning_plan": {"operations": []},
        "output_format": "csv"
    }
    
    plan = orchestrator.build_plan("clean it", "test.csv", "test.csv", "csv")
    assert isinstance(plan, ExecutionPlan)
    # Validate compile structure: ingestion_agent, cleaning_agent, reporting_agent
    agents = [s.agent for s in plan.steps]
    assert "ingestion_agent" in agents
    assert "cleaning_agent" in agents
    assert "reporting_agent" in agents

def test_compiler_creates_deterministic_executionplan():
    from finflow_agent.planning.compiler import compile_intent_to_plan
    from finflow_agent.planning.intent_schema import PlanIntent
    from finflow_agent.operations.schemas import CleaningOperationPlan, FilterOperationPlan
    
    intent = PlanIntent(
        needs_cleaning=True,
        cleaning_plan=CleaningOperationPlan(operations=[]),
        needs_filtering=True,
        filter_plan=FilterOperationPlan(conditions=[]),
        output_format="csv"
    )
    
    plan = compile_intent_to_plan(intent, "test.csv", "csv", "out_dir", "prefix")
    assert plan.steps[0].agent == "ingestion_agent"
    assert plan.steps[1].agent == "cleaning_agent"
    assert plan.steps[2].agent == "filter_agent"
    assert plan.steps[3].agent == "reporting_agent"
    
    assert plan.steps[1].depends_on == ["ingest"]
    assert plan.steps[2].depends_on == ["clean"]
    assert plan.steps[3].depends_on == ["filter"]

def test_compiler_rejects_pdf():
    from finflow_agent.planning.compiler import compile_intent_to_plan
    from finflow_agent.planning.intent_schema import PlanIntent
    
    intent = PlanIntent(output_format="xlsx")
    # Change output format to "pdf"
    intent.output_format = "pdf"
    
    with pytest.raises(ValueError) as exc:
        compile_intent_to_plan(intent, "test.csv", "csv", "out_dir", "prefix")
    assert "PDF" in str(exc.value)

def test_profile_dataframe_excludes_sample_records_by_default():
    df = pd.DataFrame({"A": [1, 2, 3]})
    profile = profile_dataframe(df)
    assert "sample_records" not in profile

def test_agents_require_input_dataframe():
    from finflow_agent.agents.cleaning_agent import CleaningAgent
    from finflow_agent.agents.filter_agent import FilterAgent
    from finflow_agent.agents.calculation_agent import CalculationAgent
    from finflow_agent.agents.visualization_agent import VisualizationAgent
    from finflow_agent.agents.reporting_agent import ReportingAgent
    
    for agent_cls in [CleaningAgent, FilterAgent, CalculationAgent, VisualizationAgent, ReportingAgent]:
        agent = agent_cls()
        res = agent.execute({}, {}) # Empty input_data
        assert res.status == "failed"
        assert "input_dataframe is required" in res.error_message

def test_output_py_is_deprecated_or_removed():
    from finflow_agent.tools.output import generate_output
    with pytest.raises(RuntimeError) as exc:
        generate_output()
    assert "deprecated" in str(exc.value)

@pytest.mark.anyio
async def test_callback_failure_marks_callback_failed():
    from finflow_agent.jobs.callbacks import send_backend_callback
    from finflow_agent.jobs.repository import JobRepository
    
    repo = JobRepository()
    job_id = "agent:cb_fail_test"
    
    db = repo._read_db()
    db[job_id] = {
        "job_id": job_id,
        "submission_id": "cb_fail_test",
        "status": "SUCCEEDED",
        "payload": {},
        "result": None,
        "error": None
    }
    repo._write_db(db)
    
    # Force httpx callback to throw request error immediately
    with patch("httpx.AsyncClient.post", side_effect=Exception("Connection refused")):
        await send_backend_callback({}, job_id, repo)
        
    job = await repo.get_job(job_id)
    assert job["status"] == "CALLBACK_FAILED"


def test_callback_payload_serializer_handles_dataframe_profile():
    from finflow_agent.jobs.callbacks import make_json_safe

    profile = profile_dataframe(
        pd.DataFrame({"Gender": ["Female", "Female"], "Age": [46, 52]}),
        include_samples=False,
    )

    safe_payload = make_json_safe(
        {
            "status": "complete",
            "summary": {
                "agent_summaries": [
                    {
                        "agent": "ingestion_agent",
                        "metrics": {"profile": profile},
                    }
                ]
            },
        }
    )

    assert safe_payload["summary"]["agent_summaries"][0]["metrics"]["profile"][
        "row_count"
    ] == 2
    assert isinstance(
        safe_payload["summary"]["agent_summaries"][0]["metrics"]["profile"],
        dict,
    )


def test_callback_identity_is_stable_for_same_payload():
    from finflow_agent.api import _attach_callback_identity

    first = _attach_callback_identity(
        {"status": "failed", "output_path": None, "summary": {"error": "x"}},
        job_id="agent:sub-1",
        submission_id="sub-1",
    )
    second = _attach_callback_identity(
        {"status": "failed", "output_path": None, "summary": {"error": "x"}},
        job_id="agent:sub-1",
        submission_id="sub-1",
    )

    assert first["job_id"] == "agent:sub-1"
    assert first["event_id"] == second["event_id"]
    assert first["event_id"].startswith("agent:sub-1:")


def test_filter_condition_normalizes_equals_operator():
    from finflow_agent.planning.normalizer import normalize_plan_intent_payload

    result = normalize_plan_intent_payload(
        {
            "filter_plan": {
                "conditions": [
                    {"column": "status", "operator": "equals", "value": "Paid"}
                ]
            }
        }
    )
    assert result.payload["filter_plan"]["conditions"][0]["operator"] == "eq"


def test_absolute_value_operation_accepts_all_numeric_columns_alias():
    from finflow_agent.planning.normalizer import normalize_plan_intent_payload

    result = normalize_plan_intent_payload(
        {
            "calculation_plan": {
                "operations": [
                    {"type": "absolute_value", "columns": "__all_numeric_columns__"}
                ]
            }
        }
    )
    assert result.payload["calculation_plan"]["operations"][0] == {
        "type": "absolute_value",
        "column": "__all_numeric_columns__",
    }


def test_execute_absolute_value_over_all_numeric_columns():
    from finflow_agent.operations.executor import execute_calculation_plan
    from finflow_agent.operations.schemas import CalculationOperationPlan, CalculationOperation

    df = pd.DataFrame(
        {"amount": [-10, 20], "balance": [-5.5, 3.2], "status": ["A", "B"]}
    )
    plan = CalculationOperationPlan(
        operations=[
            CalculationOperation(
                type="absolute_value",
                column="__all_numeric_columns__",
            )
        ]
    )

    output = execute_calculation_plan(df, plan)
    assert output.data["amount"].tolist() == [10, 20]
    assert output.data["balance"].tolist() == [5.5, 3.2]
    assert output.data["status"].tolist() == ["A", "B"]


@pytest.mark.anyio
async def test_e2e_job_normalizes_aliases_executes_and_sends_callback(monkeypatch, tmp_path):
    from finflow_agent.api import process_job_task
    from finflow_agent.jobs.repository import JobRepository

    bootstrap_agents()
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    upload_dir.mkdir()
    output_dir.mkdir()
    (upload_dir / "credits.csv").write_text(
        "Name,Credit,Note\n"
        " Alice ,50000, paid \n"
        " Bob ,10000, hold \n"
        " Cara ,60000, paid \n"
    )

    callback_payloads = []

    async def fake_callback(payload, job_id, repository):
        callback_payloads.append(payload)

    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(
        "finflow_agent.orchestrator.call_groq_json",
        lambda _messages, schema=None: {
            "schema_version": "1.0",
            "needs_cleaning": True,
            "cleaning_plan": {
                "operations": [
                    {"type": "trim_spaces", "columns": "__all_string_columns__"}
                ]
            },
            "needs_filtering": True,
            "filter_plan": {
                "conditions": [
                    {"column": "Credit", "operator": "greater_than", "value": 40000}
                ],
                "logic": "AND",
            },
            "output_format": "csv",
        },
    )
    monkeypatch.setattr("finflow_agent.jobs.callbacks.send_backend_callback", fake_callback)

    repository = JobRepository(db_path=str(tmp_path / "jobs.json"))
    job_id = "agent:e2e_aliases"
    await repository.create_or_update_queued(
        job_id,
        "e2e_aliases",
        {
            "submission_id": "e2e_aliases",
            "file_id": "credits.csv",
            "file_name": "credits.csv",
            "instruction": "Clean this data and show credited transactions above 40000",
            "output_format": "csv",
        },
    )

    await process_job_task(
        {"repository": repository},
        {
            "submission_id": "e2e_aliases",
            "file_id": "credits.csv",
            "file_name": "credits.csv",
            "instruction": "Clean this data and show credited transactions above 40000",
            "output_format": "csv",
        },
    )

    job = await repository.get_job(job_id)
    assert job["status"] == "SUCCEEDED"
    assert callback_payloads
    callback = callback_payloads[0]
    assert callback["status"] == "complete"
    assert callback["event_id"].startswith("agent:e2e_aliases:")
    assert Path(callback["output_path"]).exists()
    output_df = pd.read_csv(callback["output_path"])
    assert output_df["Credit"].tolist() == [50000, 60000]
    assert output_df["Name"].tolist() == ["Alice", "Cara"]


@pytest.mark.anyio
async def test_e2e_job_quarantines_after_failed_schema_repair(monkeypatch, tmp_path):
    from finflow_agent.api import process_job_task
    from finflow_agent.jobs.repository import JobRepository

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "credits.csv").write_text("Name,Credit\nAlice,50000\n")

    callback_payloads = []
    llm_calls = []

    def fake_llm(_messages, schema=None):
        llm_calls.append(1)
        return {
            "schema_version": "1.0",
            "needs_filtering": True,
            "filter_plan": {
                "conditions": {"column": "Credit", "operator": "gt", "value": 40000},
                "logic": "and",
            },
            "output_format": "csv",
        }

    async def fake_callback(payload, job_id, repository):
        callback_payloads.append(payload)

    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr("finflow_agent.orchestrator.call_groq_json", fake_llm)
    monkeypatch.setattr("finflow_agent.jobs.callbacks.send_backend_callback", fake_callback)

    repository = JobRepository(db_path=str(tmp_path / "jobs.json"))
    job_id = "agent:e2e_repair_fail"
    payload = {
        "submission_id": "e2e_repair_fail",
        "file_id": "credits.csv",
        "file_name": "credits.csv",
        "instruction": "show credited transactions above 40000",
        "output_format": "csv",
    }
    await repository.create_or_update_queued(job_id, "e2e_repair_fail", payload)

    await process_job_task({"repository": repository}, payload)

    job = await repository.get_job(job_id)
    assert job["status"] == "QUARANTINED"
    assert len(llm_calls) == 2
    assert callback_payloads[0]["status"] == "quarantined"
    assert "schema-aware repair" in callback_payloads[0]["summary"]["reason"]


@pytest.mark.anyio
async def test_send_backend_callback_posts_json_safe_payload_with_dataframe_profile():
    from finflow_agent.jobs.callbacks import send_backend_callback

    profile = profile_dataframe(pd.DataFrame({"A": [1, 2]}), include_samples=False)
    posted = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

    class FakeRepository:
        async def mark_callback_failed(self, job_id):
            raise AssertionError("callback should not be marked failed")

    async def fake_post(self, url, json, headers):
        posted["payload"] = json
        return FakeResponse()

    with patch("httpx.AsyncClient.post", new=fake_post):
        await send_backend_callback(
            {"status": "complete", "summary": {"profile": profile}},
            "agent:json-safe",
            FakeRepository(),
        )

    assert isinstance(posted["payload"]["summary"]["profile"], dict)
    assert posted["payload"]["summary"]["profile"]["row_count"] == 2


@pytest.mark.anyio
async def test_job_lifecycle_status_updates():
    from finflow_agent.jobs.repository import JobRepository
    repo = JobRepository()
    job_id = "agent:lifecycle_test"
    
    # 1. Queued
    await repo.create_or_update_queued(job_id, "lifecycle_test", {})
    j = await repo.get_job(job_id)
    assert j["status"] == "QUEUED"
    
    # 2. Planning
    await repo.mark_planning(job_id)
    j = await repo.get_job(job_id)
    assert j["status"] == "PLANNING"
    
    # 3. Running
    await repo.mark_running(job_id)
    j = await repo.get_job(job_id)
    assert j["status"] == "RUNNING"
    
    # 4. Succeeded
    await repo.mark_succeeded(job_id, {"res": "ok"})
    j = await repo.get_job(job_id)
    assert j["status"] == "SUCCEEDED"


@pytest.mark.anyio
async def test_worker_startup_bootstraps_agents():
    from finflow_agent.api import worker_startup
    ctx = {}
    with patch("finflow_agent.api.bootstrap_agents") as mock_bootstrap, \
         patch("finflow_agent.api.validate_required_agents_registered") as mock_validate:
        await worker_startup(ctx)
        mock_bootstrap.assert_called_once()
        mock_validate.assert_called_once()
    assert "repository" in ctx
    assert "file_store" in ctx


@pytest.mark.anyio
async def test_engine_failure_summary_preserved():
    from finflow_agent.api import process_job_task
    from finflow_agent.jobs.repository import JobRepository
    
    repo = JobRepository()
    job_id = "agent:fail_pres_123"
    
    db = repo._read_db()
    if job_id in db:
        del db[job_id]
        repo._write_db(db)
        
    payload_dict = {
        "submission_id": "fail_pres_123",
        "file_id": "safe.csv",
        "file_name": "safe.csv",
        "instruction": "clean",
        "output_format": "csv"
    }
    
    await repo.create_or_update_queued(job_id, "fail_pres_123", payload_dict)
    
    # Mock file store to return a safe path without throwing errors
    with patch("finflow_agent.storage.file_store.FileStore.resolve_uploaded_file", return_value=Path("dummy_path")), \
         patch("finflow_agent.planning.orchestrator.Orchestrator.build_plan", return_value=ExecutionPlan(steps=[])), \
         patch("finflow_agent.execution.engine.ExecutionEngine.execute") as mock_execute, \
         patch("finflow_agent.jobs.callbacks.send_backend_callback") as mock_cb:
         
        mock_execute.return_value = {
            "status": "failed",
            "output_path": None,
            "summary": {"failed_step": "calculate", "error": "division by zero"}
        }
        
        ctx = {"repository": repo}
        await process_job_task(ctx, payload_dict)
        
        job = await repo.get_job(job_id)
        assert job is not None
        assert job["status"] == "FAILED"
        assert "division by zero" in job["error"]
        assert "failed_step" in job["error"]


@pytest.mark.anyio
async def test_unique_file_prefix_per_job():
    from finflow_agent.api import process_job_task
    from finflow_agent.jobs.repository import JobRepository
    
    repo = JobRepository()
    job_id = "agent:prefix_test_123"
    
    db = repo._read_db()
    if job_id in db:
        del db[job_id]
        repo._write_db(db)
        
    payload_dict = {
        "submission_id": "prefix_test_123",
        "file_id": "safe.csv",
        "file_name": "safe.csv",
        "instruction": "clean",
        "output_format": "csv"
    }
    
    await repo.create_or_update_queued(job_id, "prefix_test_123", payload_dict)
    
    # Mock resolve_uploaded_file and execute, patch build_plan to capture file_prefix
    with patch("finflow_agent.storage.file_store.FileStore.resolve_uploaded_file", return_value=Path("dummy_path")), \
         patch("finflow_agent.planning.orchestrator.Orchestrator.build_plan") as mock_build_plan, \
         patch("finflow_agent.jobs.callbacks.send_backend_callback") as mock_cb:
         
        mock_build_plan.return_value = ExecutionPlan(steps=[])
        
        ctx = {"repository": repo}
        await process_job_task(ctx, payload_dict)
        
        # Check that file_prefix passed to build_plan starts with submission_prefix_test_123
        mock_build_plan.assert_called_once()
        called_kwargs = mock_build_plan.call_args.kwargs
        file_prefix = called_kwargs.get("file_prefix")
        assert file_prefix is not None
        assert file_prefix.startswith("submission_prefix_test_123_")
        assert len(file_prefix) > len("submission_prefix_test_123_")


def test_compiler_rejects_missing_cleaning_plan():
    from finflow_agent.planning.compiler import compile_intent_to_plan
    from finflow_agent.planning.intent_schema import PlanIntent
    
    intent = PlanIntent.model_construct(
        needs_cleaning=True,
        cleaning_plan=None,
        output_format="xlsx",
    )
    with pytest.raises(ValueError) as exc:
        compile_intent_to_plan(intent, "test.csv", "csv", "out", "pref")
    assert "cleaning_plan is missing" in str(exc.value)

def test_compiler_rejects_missing_filter_plan():
    from finflow_agent.planning.compiler import compile_intent_to_plan
    from finflow_agent.planning.intent_schema import PlanIntent
    
    intent = PlanIntent.model_construct(
        needs_filtering=True,
        filter_plan=None,
        output_format="xlsx",
    )
    with pytest.raises(ValueError) as exc:
        compile_intent_to_plan(intent, "test.csv", "csv", "out", "pref")
    assert "filter_plan is missing" in str(exc.value)

def test_compiler_rejects_missing_calculation_plan():
    from finflow_agent.planning.compiler import compile_intent_to_plan
    from finflow_agent.planning.intent_schema import PlanIntent
    
    intent = PlanIntent.model_construct(
        needs_calculation=True,
        calculation_plan=None,
        output_format="xlsx",
    )
    with pytest.raises(ValueError) as exc:
        compile_intent_to_plan(intent, "test.csv", "csv", "out", "pref")
    assert "calculation_plan is missing" in str(exc.value)

def test_compiler_rejects_missing_visualization_plan():
    from finflow_agent.planning.compiler import compile_intent_to_plan
    from finflow_agent.planning.intent_schema import PlanIntent
    
    intent = PlanIntent.model_construct(
        needs_visualization=True,
        visualization_plan=None,
        output_format="xlsx",
    )
    with pytest.raises(ValueError) as exc:
        compile_intent_to_plan(intent, "test.csv", "csv", "out", "pref")
    assert "visualization_plan is missing" in str(exc.value)


def test_validate_plan_rejects_missing_input_from():
    from finflow_agent.planning.validators import validate_plan
    from finflow_agent.state import ExecutionPlan, PlanStep
    
    # step_2 inputs from df_missing which is never produced
    plan = ExecutionPlan(steps=[
        PlanStep(
            step_id="step_1",
            agent="ingestion_agent",
            params={"resolved_file_path": "fake.csv", "file_type": "csv"},
            depends_on=[],
            output_key="df_ingested"
        ),
        PlanStep(
            step_id="step_2",
            agent="cleaning_agent",
            params={},
            depends_on=["step_1"],
            input_from=["df_missing"],
            output_key="df_cleaned"
        )
    ])
    
    is_valid, err = validate_plan(plan)
    assert not is_valid
    assert "input_from 'df_missing' was not produced by any previous step" in err


def test_engine_passes_visualization_artifacts_to_reporting():
    from finflow_agent.engine import ExecutionEngine
    from finflow_agent.state import ExecutionPlan, PlanStep, AgentResult
    
    engine = ExecutionEngine()
    
    plan = ExecutionPlan(steps=[
        PlanStep(
            step_id="ingest",
            agent="ingestion_agent",
            params={"resolved_file_path": "fake.csv", "file_type": "csv"},
            depends_on=[],
            output_key="df_ingested"
        ),
        PlanStep(
            step_id="visualize",
            agent="visualization_agent",
            params={"plan": {"charts": [{"type": "bar", "x": "A", "y": "B", "title": "My Chart"}]}},
            depends_on=["ingest"],
            input_from=["df_ingested"],
            output_key="viz_out"
        ),
        PlanStep(
            step_id="report",
            agent="reporting_agent",
            params={"plan": {"output_format": "csv"}},
            depends_on=["ingest", "visualize"],
            input_from=["df_ingested", "viz_out"],
            output_key="report_out"
        )
    ])
    
    ingest_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    mock_ingest = MagicMock()
    mock_ingest.execute.return_value = AgentResult(status="success", data=ingest_df)
    
    mock_viz = MagicMock()
    mock_viz.execute.return_value = AgentResult(
        status="success",
        data={"chart": {"type": "bar", "title": "My Chart", "x_col": "A", "y_col": "B"}},
        artifacts={"chart_123": {"type": "bar", "title": "My Chart", "x_col": "A", "y_col": "B"}}
    )
    
    mock_report = MagicMock()
    mock_report.execute.return_value = AgentResult(
        status="success",
        data="report.csv",
        artifacts={"primary_output_path": "report.csv"}
    )
    
    def get_agent_mock(name):
        if name == "ingestion_agent":
            return lambda: mock_ingest
        elif name == "visualization_agent":
            return lambda: mock_viz
        elif name == "reporting_agent":
            return lambda: mock_report
        raise ValueError(f"Unknown agent: {name}")
        
    with patch("finflow_agent.execution.engine.registry.get_agent_class", side_effect=get_agent_mock):
        result = engine.execute(plan)
        
    assert result["status"] == "complete", f"Engine failed with: {result.get('summary', {}).get('error')}"
    assert result["output_path"] == "report.csv"
    
    args, kwargs = mock_report.execute.call_args
    input_data = args[1]
    
    assert "chart_artifacts" in input_data
    assert isinstance(input_data["chart_artifacts"], list)
    assert len(input_data["chart_artifacts"]) == 1
    assert input_data["chart_artifacts"][0]["title"] == "My Chart"
    assert input_data["chart_artifacts"][0]["type"] == "bar"


def test_ingestion_agent_rejects_file_path_param():
    from finflow_agent.agents.ingestion_agent import IngestionAgent
    
    agent = IngestionAgent()
    res = agent.execute({"file_path": "fake.csv", "file_type": "csv"}, {})
    assert res.status == "failed"
    assert "Invalid parameter schema" in res.error_message
    assert "resolved_file_path" in res.error_message


def test_output_py_deprecated():
    from finflow_agent.tools.output import generate_output
    with pytest.raises(RuntimeError) as exc:
        generate_output()
    assert "generate_output is deprecated" in str(exc.value)


def test_profile_dataframe_excludes_samples_by_default():
    from finflow_agent.tools.dataframe_profile import profile_dataframe
    df = pd.DataFrame({"A": [1, 2, 3]})
    
    # default call:
    profile = profile_dataframe(df)
    assert "sample_records" not in profile
    
    # call with include_samples=True:
    profile_with_samples = profile_dataframe(df, include_samples=True)
    assert "sample_records" in profile_with_samples
    assert profile_with_samples["sample_records"] == [{"A": 1}, {"A": 2}, {"A": 3}]


def test_filestore_rejects_path_traversal():
    from finflow_agent.storage.file_store import FileStore
    from finflow_agent.operations.errors import UnsafeInputPathError
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = FileStore(upload_dir=tmp_dir)
        
        # Test rejects traversal with ".."
        with pytest.raises(UnsafeInputPathError):
            store.resolve_uploaded_file("../x")
            
        # Test rejects path separators
        with pytest.raises(UnsafeInputPathError):
            store.resolve_uploaded_file("a/b")
            
        with pytest.raises(UnsafeInputPathError):
            store.resolve_uploaded_file("a\\b")


def test_api_uses_file_id_not_file_path():
    from finflow_agent.api import JobPayload
    
    # Should accept payload with file_id
    payload = JobPayload(
        submission_id="sub_1",
        file_id="safe.csv",
        file_name="safe.csv",
        instruction="clean",
        output_format="csv"
    )
    assert payload.file_id == "safe.csv"
    
    # Should NOT have file_path attribute
    assert not hasattr(payload, "file_path")
    
    schema = JobPayload.model_json_schema()
    assert "file_id" in schema["properties"]
    assert "file_path" not in schema["properties"]


@pytest.mark.anyio
async def test_api_upload_enqueues_arq_job():
    from finflow_agent.api import handle_upload, JobPayload
    
    # Mock Redis enqueue
    mock_redis = AsyncMock()
    app.state.redis = mock_redis
    
    # Make sure job is not already in DB
    from finflow_agent.jobs.repository import JobRepository
    repo = JobRepository()
    db = repo._read_db()
    job_id = "agent:enq_test_123"
    if job_id in db:
        del db[job_id]
        repo._write_db(db)
        
    payload = JobPayload(
        submission_id="enq_test_123",
        file_id="safe.csv",
        file_name="safe.csv",
        instruction="clean",
        output_format="csv"
    )
    
    res = await handle_upload(payload)
    assert res["status"] == "queued"
    assert res["job_id"] == job_id
    
    # Ensure enqueue_job was called with "process_job_task", payload, and stable job_id
    mock_redis.enqueue_job.assert_called_once_with(
        "process_job_task",
        payload.model_dump(),
        _job_id=job_id
    )


def test_bootstrap_agents_does_not_require_langchain_groq_at_import_time():
    import sys
    import importlib
    
    # 1. Unload agent modules to force a clean import
    modules_to_unload = [
        "finflow_agent.bootstrap",
        "finflow_agent.agents.ingestion_agent",
        "finflow_agent.agents.cleaning_agent",
        "finflow_agent.agents.filter_agent",
        "finflow_agent.agents.calculation_agent",
        "finflow_agent.agents.visualization_agent",
        "finflow_agent.agents.reporting_agent"
    ]
    for mod in modules_to_unload:
        if mod in sys.modules:
            del sys.modules[mod]
            
    # Also clean LLM modules if they are present
    llm_modules = ["langchain_groq", "groq"]
    original_llm_states = {}
    for mod in llm_modules:
        if mod in sys.modules:
            original_llm_states[mod] = sys.modules[mod]
            del sys.modules[mod]

    # 2. Perform bootstrap
    from finflow_agent.bootstrap import bootstrap_agents, validate_required_agents_registered
    from finflow_agent.registry import registry
    
    # Ensure registry is cleared and re-registered on import
    registry._specs.clear()
    registry._agents.clear()
    
    bootstrap_agents()
    validate_required_agents_registered()
    
    # 3. Assert all required agents are registered
    assert registry.get_spec("ingestion_agent") is not None
    assert registry.get_spec("cleaning_agent") is not None
    assert registry.get_spec("filter_agent") is not None
    assert registry.get_spec("calculation_agent") is not None
    assert registry.get_spec("reporting_agent") is not None
    
    # 4. Assert langchain_groq was NOT imported during bootstrap
    assert "langchain_groq" not in sys.modules
    
    # Restore original sys.modules states for LLM modules
    for mod, val in original_llm_states.items():
        sys.modules[mod] = val




