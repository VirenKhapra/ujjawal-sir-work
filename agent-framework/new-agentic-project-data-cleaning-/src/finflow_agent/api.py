import os
import json
import httpx
import hashlib
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from finflow_agent.engine import ExecutionEngine
from finflow_agent.registry import registry
from finflow_agent.bootstrap import bootstrap_agents, validate_required_agents_registered
from arq import create_pool
from arq.connections import RedisSettings
from finflow_agent.jobs.repository import JobRepository
from finflow_agent.storage.file_store import FileStore
from finflow_agent.llm_telemetry import (
    log_runtime_event,
    reset_runtime_context,
    set_runtime_context,
)
from finflow_agent.planning.canonical_intent import CanonicalIntentEnvelope
from finflow_agent.planning.canonical_intent import compute_intent_hash, upcast_canonical_intent_payload
from finflow_agent.planning.compiler import compile_canonical_intent


logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_agents()
    validate_required_agents_registered()
    # Initialize redis pool for ARQ
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    app.state.redis = await create_pool(RedisSettings.from_dsn(redis_url))
    app.state.repository = JobRepository()
    yield
    if getattr(app.state, "redis", None):
        await app.state.redis.close()

app = FastAPI(title="FinFlow Agent Service", lifespan=lifespan)

class JobPayload(BaseModel):
    submission_id: str
    file_id: str
    file_name: str
    resolved_file_path: str | None = None
    data_profile_id: str | None = None
    canonical_intent_revision_id: str | None = None
    canonical_intent: CanonicalIntentEnvelope | None = None
    output_format: str
    audit_context: dict[str, Any] | None = None


def _extract_failure_reason(result_payload: dict | None) -> str:
    if not isinstance(result_payload, dict):
        return "Job failed without a structured result payload."

    summary = result_payload.get("summary")
    if isinstance(summary, dict):
        failed_step_id = summary.get("failed_step_id")
        for key in ("error_message", "error", "reason"):
            raw = summary.get(key)
            if raw:
                prefix = (
                    f"Step '{failed_step_id}' failed: "
                    if failed_step_id and str(raw) not in str(failed_step_id)
                    else ""
                )
                return prefix + str(raw)

    status_value = result_payload.get("status")
    if status_value:
        return f"Job ended with status {status_value!r} without a detailed failure reason."
    return "Job failed without a detailed failure reason."


def _attach_callback_identity(result_payload: dict, *, job_id: str, submission_id: str) -> dict:
    result_payload["submission_id"] = submission_id
    result_payload["job_id"] = job_id
    stable_payload = {
        "submission_id": submission_id,
        "job_id": job_id,
        "status": result_payload.get("status"),
        "output_path": result_payload.get("output_path"),
        "summary": result_payload.get("summary"),
    }
    stable_json = json.dumps(stable_payload, sort_keys=True, default=str)
    digest = hashlib.sha256(stable_json.encode("utf-8")).hexdigest()
    result_payload["event_id"] = f"{job_id}:{digest}"
    return result_payload


async def process_job_task(ctx, payload_dict: dict):
    try:
        payload = JobPayload.model_validate(payload_dict)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid job payload: {exc}") from exc

    submission_id = payload.submission_id
    job_id = f"agent:{submission_id}"
    runtime_token = set_runtime_context(
        submission_id=submission_id,
        job_id=job_id,
        trigger="worker",
        canonical_intent_present=payload.canonical_intent is not None,
        instruction_present=bool((payload.audit_context or {}).get("original_instruction")),
        legacy_schema_state_present=False,
    )
    
    repository = (ctx or {}).get("repository") or JobRepository()
    await repository.mark_planning(job_id)
    log_runtime_event(
        "worker_entry_entered",
        trigger="worker",
        instruction_present=bool((payload.audit_context or {}).get("original_instruction")),
        canonical_intent_present=payload.canonical_intent is not None,
        legacy_schema_state_present=False,
    )
    
    try:
        # Resolve the file path from the canonical transport payload first.
        try:
            if payload.resolved_file_path:
                resolved_path = Path(payload.resolved_file_path)
                if not resolved_path.exists():
                    raise FileNotFoundError(f"Resolved file path does not exist: {resolved_path}")
            else:
                store = (ctx or {}).get("file_store") or FileStore()
                resolved_path = store.resolve_uploaded_file(payload.file_id)
            resolved_file_path = str(resolved_path)
        except Exception as e:
            reason = f"File store resolution failed: {str(e)}"
            await repository.mark_quarantined(job_id, reason)
            result_payload = {
                "submission_id": submission_id,
                "status": "quarantined",
                "output_path": None,
                "summary": {"reason": reason}
            }
            _attach_callback_identity(result_payload, job_id=job_id, submission_id=submission_id)
            from finflow_agent.jobs.callbacks import send_backend_callback
            await send_backend_callback(result_payload, job_id, repository)
            return

        # Build plan
        from uuid import uuid4
        file_prefix = f"submission_{submission_id}_{uuid4().hex[:8]}"

        plan_or_dict = None
        if payload.canonical_intent is not None:
            try:
                canonical_payload = upcast_canonical_intent_payload(payload.canonical_intent.model_dump(mode="json"))
                if canonical_payload is None:
                    raise ValueError("Canonical intent payload is missing")
                payload.canonical_intent = CanonicalIntentEnvelope.model_validate(canonical_payload)
                logger.info(
                    "event=canonical_job_started job_id=%s submission_id=%s intent_schema_version=%s",
                    job_id,
                    submission_id,
                    payload.canonical_intent.intent.schema_version,
                )
                log_runtime_event(
                    "canonical_compiler_entered",
                    trigger="worker",
                    instruction_present=bool((payload.audit_context or {}).get("original_instruction")),
                    canonical_intent_present=True,
                    legacy_schema_state_present=False,
                    prompt_text=str((payload.audit_context or {}).get("original_instruction", "")),
                    canonical_intent_schema_version=payload.canonical_intent.intent.schema_version,
                )
                plan_or_dict = compile_canonical_intent(
                    payload.canonical_intent.intent,
                    resolved_file_path=resolved_file_path,
                    file_type=Path(resolved_file_path).suffix.lstrip(".").lower(),
                    output_dir=os.environ.get("OUTPUT_DIR", "outputs"),
                    artifact_prefix=file_prefix,
                )
                if isinstance(plan_or_dict, dict):
                    plan_or_dict.setdefault("plan_metadata", {})
                    plan_or_dict["plan_metadata"].update(
                        {
                            "intent_id": payload.canonical_intent.intent_id,
                            "intent_revision": payload.canonical_intent.intent_revision,
                            "intent_hash": payload.canonical_intent.intent_hash or compute_intent_hash(payload.canonical_intent.intent),
                            "compiler_version": "1.0",
                        }
                    )
                logger.info(
                    "event=canonical_job_compiled job_id=%s submission_id=%s",
                    job_id,
                    submission_id,
                )
            except Exception as exc:
                # Extract user-friendly reason from canonical intent evidence if available
                reason = f"Canonical intent compilation failed: {exc}"
                if payload.canonical_intent and payload.canonical_intent.intent:
                    intent_evidence = getattr(payload.canonical_intent.intent, 'evidence', [])
                    intent_status = getattr(payload.canonical_intent.intent, 'resolution_status', '')
                    if intent_status == "needs_clarification" and intent_evidence:
                        reason = "; ".join(str(e) for e in intent_evidence)
                await repository.mark_quarantined(job_id, reason)
                result_payload = {
                    "submission_id": submission_id,
                    "status": "quarantined",
                    "output_path": None,
                    "summary": {"reason": reason},
                }
                _attach_callback_identity(result_payload, job_id=job_id, submission_id=submission_id)
                from finflow_agent.jobs.callbacks import send_backend_callback
                await send_backend_callback(result_payload, job_id, repository)
                return
        else:
            reason = "legacy_payload_not_supported: canonical_intent is required for execution."
            await repository.mark_quarantined(job_id, reason)
            result_payload = {
                "submission_id": submission_id,
                "status": "quarantined",
                "output_path": None,
                "summary": {"reason": reason},
            }
            _attach_callback_identity(result_payload, job_id=job_id, submission_id=submission_id)
            from finflow_agent.jobs.callbacks import send_backend_callback
            await send_backend_callback(result_payload, job_id, repository)
            return

        if isinstance(plan_or_dict, dict) and plan_or_dict.get("status") == "quarantined":
            reason = plan_or_dict.get("reason") or "Request quarantined by canonical planner."
            logger.info(
                "Agent job planning quarantined job_id=%s submission_id=%s reason=%s",
                job_id,
                submission_id,
                reason,
            )
            await repository.mark_quarantined(job_id, reason)
            result_payload = {
                "submission_id": submission_id,
                "status": "quarantined",
                "output_path": None,
                "summary": {"reason": reason}
            }
            _attach_callback_identity(result_payload, job_id=job_id, submission_id=submission_id)
            from finflow_agent.jobs.callbacks import send_backend_callback
            await send_backend_callback(result_payload, job_id, repository)
            return

        try:
            await repository.mark_running(job_id)
            logger.info(
                "Agent job execution started job_id=%s submission_id=%s",
                job_id,
                submission_id,
            )

            engine = ExecutionEngine()
            result_payload = engine.execute(
                plan_or_dict,
                submission_id=submission_id,
            )
            _attach_callback_identity(result_payload, job_id=job_id, submission_id=submission_id)

            if result_payload.get("status") != "complete":
                failure_reason = _extract_failure_reason(result_payload)
                await repository.mark_failed(
                    job_id,
                    failure_reason,
                )
            else:
                output_path = result_payload.get("output_path")
                if not output_path:
                    result_payload = {
                        "submission_id": submission_id,
                        "status": "failed",
                        "output_path": None,
                        "summary": {
                            "error": (
                                "Execution completed but reporting output was missing "
                                "primary output_path."
                            ),
                            "error_message": (
                                "Execution completed but reporting output was missing "
                                "primary output_path."
                            ),
                            "engine_result": result_payload
                        }
                    }
                    _attach_callback_identity(result_payload, job_id=job_id, submission_id=submission_id)
                    await repository.mark_failed(
                        job_id,
                        "Execution completed but reporting output was missing primary output_path.",
                    )
                else:
                    await repository.mark_succeeded(job_id, result_payload)
        except Exception as e:
            error_message = f"Unhandled worker exception while processing job: {e}"
            await repository.mark_failed(job_id, error_message)
            result_payload = {
                "submission_id": submission_id,
                "status": "failed",
                "output_path": None,
                "summary": {
                    "error": error_message,
                    "error_message": error_message,
                }
            }
            _attach_callback_identity(result_payload, job_id=job_id, submission_id=submission_id)

        from finflow_agent.jobs.callbacks import send_backend_callback
        await send_backend_callback(result_payload, job_id, repository)
    finally:
        reset_runtime_context(runtime_token)

# ARQ worker settings
async def worker_startup(ctx):
    bootstrap_agents()
    validate_required_agents_registered()
    ctx["repository"] = JobRepository()
    ctx["file_store"] = FileStore()

class WorkerSettings:
    functions = [process_job_task]
    on_startup = worker_startup
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


@app.post("/api/agent/upload")
async def handle_upload(payload: JobPayload, background_tasks = None):
    job_id = f"agent:{payload.submission_id}"
    repository = JobRepository()
    
    # Idempotency check:
    existing_job = await repository.get_job(job_id)
    if existing_job:
        return {
            "status": existing_job["status"].lower(),
            "job_id": job_id,
            "submission_id": payload.submission_id,
        }
        
    await repository.create_or_update_queued(
        job_id=job_id,
        submission_id=payload.submission_id,
        payload=payload.model_dump()
    )
    
    await app.state.redis.enqueue_job(
        "process_job_task",
        payload.model_dump(exclude_none=True),
        _job_id=job_id
    )
    
    return {
        "status": "queued",
        "job_id": job_id,
        "submission_id": payload.submission_id
    }

@app.get("/api/agent/jobs/{job_id}")
async def get_job_status(job_id: str):
    repository = JobRepository()
    job = await repository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job

@app.get("/api/agent/registry")
def get_registry():
    return {"agents": registry.describe_all()}
