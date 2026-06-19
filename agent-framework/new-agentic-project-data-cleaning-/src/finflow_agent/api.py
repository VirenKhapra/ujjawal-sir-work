import os
import json
import httpx
import hashlib
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from finflow_agent.orchestrator import Orchestrator
from finflow_agent.engine import ExecutionEngine
from finflow_agent.registry import registry
from finflow_agent.bootstrap import bootstrap_agents, validate_required_agents_registered
from arq import create_pool
from arq.connections import RedisSettings
from finflow_agent.jobs.repository import JobRepository
from finflow_agent.storage.file_store import FileStore


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
    instruction: str
    output_format: str


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
    submission_id = payload_dict["submission_id"]
    job_id = f"agent:{submission_id}"
    
    repository = (ctx or {}).get("repository") or JobRepository()
    await repository.mark_planning(job_id)
    
    # Resolve file_id through FileStore inside the worker task
    try:
        store = (ctx or {}).get("file_store") or FileStore()
        resolved_path = store.resolve_uploaded_file(payload_dict["file_id"])
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

    orchestrator = Orchestrator()
    plan_or_dict = orchestrator.build_plan(
        instruction=payload_dict["instruction"],
        file_path=resolved_file_path,
        file_name=payload_dict["file_name"],
        output_format=payload_dict["output_format"],
        output_dir=os.environ.get("OUTPUT_DIR", "outputs"),
        file_prefix=file_prefix,
        job_id=job_id,
        submission_id=submission_id,
    )

    
    if isinstance(plan_or_dict, dict) and plan_or_dict.get("status") == "quarantined":
        reason = plan_or_dict.get("reason") or "Request quarantined by Orchestrator."
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
        result_payload = engine.execute(plan_or_dict)
        _attach_callback_identity(result_payload, job_id=job_id, submission_id=submission_id)
        
        if result_payload.get("status") != "complete":
            await repository.mark_failed(
                job_id,
                json.dumps(result_payload.get("summary", {}), default=str)
            )
        else:
            output_path = result_payload.get("output_path")
            if not output_path:
                result_payload = {
                    "submission_id": submission_id,
                    "status": "failed",
                    "output_path": None,
                    "summary": {
                        "error": "Complete job missing output_path",
                        "engine_result": result_payload
                    }
                }
                _attach_callback_identity(result_payload, job_id=job_id, submission_id=submission_id)
                await repository.mark_failed(job_id, "Complete job missing output_path")
            else:
                await repository.mark_succeeded(job_id, result_payload)
    except Exception as e:
        await repository.mark_failed(job_id, str(e))
        result_payload = {
            "submission_id": submission_id,
            "status": "failed",
            "output_path": None,
            "summary": {"error": str(e)}
        }
        _attach_callback_identity(result_payload, job_id=job_id, submission_id=submission_id)

        
    from finflow_agent.jobs.callbacks import send_backend_callback
    await send_backend_callback(result_payload, job_id, repository)

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
        payload.model_dump(),
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
