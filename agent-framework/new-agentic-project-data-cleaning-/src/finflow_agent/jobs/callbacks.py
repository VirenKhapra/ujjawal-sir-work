import os
import httpx
import asyncio
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from finflow_agent.jobs.repository import JobRepository

logger = logging.getLogger(__name__)


def make_json_safe(value: Any) -> Any:
    """Convert callback payload values to JSON-safe primitives.

    The execution result may contain Pydantic models in nested metrics or
    artifacts. The backend callback must never fail before making the HTTP
    request simply because a Python object reached this boundary.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return make_json_safe(value.item())
    if value.__class__.__module__.startswith("pandas"):
        if value.__class__.__name__ == "DataFrame":
            return {
                "type": "DataFrame",
                "row_count": len(value),
                "columns": [str(column) for column in value.columns],
            }
        if value.__class__.__name__ == "Series":
            return {
                "type": "Series",
                "row_count": len(value),
                "name": str(value.name),
            }
    if isinstance(value, BaseModel):
        return make_json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Path, UUID)):
        return str(value)
    return str(value)


async def send_backend_callback(result_payload: dict, job_id: str, repository: JobRepository) -> None:
    """
    Sends the execution result payload back to the configured backend callback URL.
    Implements timeouts, retries, and transient failure backoff.
    """
    backend_url = os.environ.get("BACKEND_CALLBACK_URL", "http://backend:8000/api/agent/callback")
    secret = os.environ.get("AGENT_CALLBACK_SECRET", "change-agent-callback-secret")
    
    max_retries = 3
    backoff = 1.0
    safe_payload = make_json_safe(result_payload)
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    backend_url,
                    json=safe_payload,
                    headers={"Authorization": f"Bearer {secret}"}
                )
                if 200 <= response.status_code < 300:
                    logger.info(f"Callback succeeded on attempt {attempt + 1}")
                    return
                
                # Check for non-retryable 4xx client errors (excluding 429)
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    logger.error(f"Callback failed with client error {response.status_code}: {response.text}")
                    break
                    
                logger.warning(f"Callback returned status {response.status_code}, retrying...")
        except Exception as e:
            logger.warning(f"Callback request error on attempt {attempt + 1}: {e}")

            
        if attempt < max_retries - 1:
            await asyncio.sleep(backoff)
            backoff *= 2.0
            
    logger.error("Callback failed all retry attempts.")
    # On final failure, mark job CALLBACK_FAILED but do not erase the successful job result
    await repository.mark_callback_failed(job_id)
