import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)

from finflow_agent.api import process_job_task
from finflow_agent.jobs.repository import JobRepository
from finflow_agent.llm import call_groq_json
from finflow_agent.llm_telemetry import reset_runtime_context, set_runtime_context


def _canonical_payload(tmp_path: Path) -> dict:
    upload = tmp_path / "sample.csv"
    upload.write_text("amount,status\n10,ok\n", encoding="utf-8")
    return {
        "submission_id": "canon-123",
        "file_id": "sample.csv",
        "file_name": "sample.csv",
        "resolved_file_path": str(upload),
        "output_format": "csv",
        "audit_context": {"original_instruction": "Clean the data and return all columns."},
        "canonical_intent": {
            "schema_version": "1.0",
            "intent_id": "env-1",
            "intent_revision": 1,
            "intent_hash": "env-hash",
            "parent_intent_id": None,
            "original_instruction": "Clean the data and return all columns.",
            "intent": {
                "schema_version": "2.0",
                "intent_id": "intent-1",
                "intent_revision": 1,
                "intent_hash": "intent-hash",
                "parent_intent_id": None,
                "original_prompt": "Clean the data and return all columns.",
                "normalized_prompt": "clean the data and return all columns",
                "resolution_status": "resolved",
                "decision": "clean",
                "evidence": [],
                "alternatives_considered": [],
                "actions": [{"kind": "clean", "mode": "safe_default", "operations": []}],
                "output_format": "csv",
                "assumptions": [],
                "repair_notes": [],
                "dataframe_profile": {"source_columns": ["amount", "status"]},
                "capability_version": "agent.capability.1",
                "capability_snapshot": {},
            },
            "extractor_version": "1.0",
            "normalizer_version": "1.0",
            "grounding_version": "1.0",
            "capability_version": "agent.capability.1",
            "capability_snapshot": {},
            "repair_notes": [],
            "assumptions": [],
        },
    }


@pytest.mark.anyio
async def test_process_job_task_canonical_payload_compiles_once_without_legacy_planner(monkeypatch, tmp_path):
    repo = JobRepository(db_path=str(tmp_path / "jobs.json"))
    payload = _canonical_payload(tmp_path)
    await repo.create_or_update_queued("agent:canon-123", "canon-123", payload)

    compile_calls: list[str] = []

    def _fake_compile(intent, **kwargs):
        compile_calls.append(intent.intent_id)
        return {"compiled": True}

    class _FakeEngine:
        def execute(self, plan, submission_id=None):
            assert plan == {"compiled": True}
            return {
                "status": "complete",
                "output_path": str(tmp_path / "out.csv"),
                "summary": {"status": "ok"},
            }

    callback = AsyncMock()

    monkeypatch.setattr("finflow_agent.api.compile_canonical_intent", _fake_compile)
    monkeypatch.setattr("finflow_agent.api.ExecutionEngine", _FakeEngine)
    monkeypatch.setattr("finflow_agent.jobs.callbacks.send_backend_callback", callback)
    monkeypatch.setattr(
        "finflow_agent.planning.orchestrator.Orchestrator.build_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy planner should not run")),
    )
    monkeypatch.setattr(
        "finflow_agent.llm.call_groq_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy LLM should not run")),
    )

    await process_job_task({"repository": repo}, payload)

    assert compile_calls == ["intent-1"]
    callback.assert_awaited_once()


def test_call_groq_json_raises_in_strict_mode_for_canonical_job(monkeypatch):
    monkeypatch.setenv("FAIL_ON_CANONICAL_LEGACY_PLANNER", "true")
    token = set_runtime_context(
        submission_id="canon-123",
        job_id="agent:canon-123",
        trigger="worker",
        canonical_intent_present=True,
        instruction_present=True,
    )
    try:
        with pytest.raises(RuntimeError, match="Architecture violation"):
            call_groq_json([], {})
    finally:
        reset_runtime_context(token)


def test_call_groq_json_allows_legacy_context(monkeypatch):
    class _FakeUsage:
        prompt_tokens = 11
        completion_tokens = 7
        total_tokens = 18

    class _FakeMessage:
        content = '{"ok": true}'

    class _FakeChoice:
        message = _FakeMessage()
        finish_reason = "stop"

    class _FakeCompletion:
        usage = _FakeUsage()
        choices = [_FakeChoice()]

    class _FakeChat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return _FakeCompletion()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr("finflow_agent.llm.get_groq_client", lambda: _FakeClient())
    token = set_runtime_context(
        submission_id="legacy-123",
        job_id="agent:legacy-123",
        trigger="worker",
        canonical_intent_present=False,
        instruction_present=True,
    )
    try:
        result = call_groq_json([{"role": "user", "content": "return json"}], {})
    finally:
        reset_runtime_context(token)

    assert result == {"ok": True}
