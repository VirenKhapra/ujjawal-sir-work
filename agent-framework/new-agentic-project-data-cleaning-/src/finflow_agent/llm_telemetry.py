"""LLM call telemetry — structured logging for every Groq API interaction."""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("llm_telemetry")
# Ensure the telemetry logger outputs to stderr (captured by Docker)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

_LOG_LOCK = threading.Lock()
_DEFAULT_LOG_DIR = "logs"
_DEFAULT_LOG_FILE = "agent_service_llm_telemetry.jsonl"
_RUNTIME_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "finflow_runtime_context",
    default={},
)


def _safe_key_fingerprint(api_key: str) -> str:
    """Return last 4 chars of API key for safe logging."""
    if not api_key or len(api_key) < 8:
        return "NONE"
    return f"...{api_key[-4:]}"


def _prompt_hash(messages: list) -> str:
    """SHA-256 hash of rendered messages for deduplication detection."""
    content = json.dumps(messages, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _prompt_stats(messages: list) -> dict:
    """Safe prompt measurements without logging content."""
    system_chars = sum(len(m.get("content", "") or "") for m in messages if m.get("role") == "system")
    user_chars = sum(len(m.get("content", "") or "") for m in messages if m.get("role") == "user")
    return {
        "system_prompt_characters": system_chars,
        "user_prompt_characters": user_chars,
        "message_count": len(messages),
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_path() -> Path:
    root = Path(
        os.environ.get("FINFLOW_DIAGNOSTIC_LOG_DIR")
        or os.environ.get("LLM_TELEMETRY_LOG_DIR")
        or _DEFAULT_LOG_DIR
    )
    root.mkdir(parents=True, exist_ok=True)
    return root / _DEFAULT_LOG_FILE


def _write_entry(entry: dict[str, Any]) -> None:
    payload = json.dumps(entry, sort_keys=True, default=str)
    logger.info(payload)
    with _LOG_LOCK:
        with _log_path().open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")


def set_runtime_context(**kwargs: Any):
    current = dict(_RUNTIME_CONTEXT.get({}))
    current.update({key: value for key, value in kwargs.items() if value is not None})
    return _RUNTIME_CONTEXT.set(current)


def reset_runtime_context(token) -> None:
    _RUNTIME_CONTEXT.reset(token)


def get_runtime_context() -> dict[str, Any]:
    return dict(_RUNTIME_CONTEXT.get({}))


def safe_summary_keys(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(key) for key in value.keys())


def make_prompt_metadata(
    *,
    messages: list[dict[str, Any]] | None = None,
    prompt_text: str | None = None,
) -> dict[str, Any]:
    if messages:
        stats = _prompt_stats(messages)
        return {
            "prompt_hash": _prompt_hash(messages),
            "prompt_character_count": stats["system_prompt_characters"] + stats["user_prompt_characters"],
            "message_count": stats["message_count"],
            **stats,
        }
    if prompt_text:
        return {
            "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16],
            "prompt_character_count": len(prompt_text),
            "message_count": 1,
            "system_prompt_characters": 0,
            "user_prompt_characters": len(prompt_text),
        }
    return {
        "prompt_hash": "",
        "prompt_character_count": 0,
        "message_count": 0,
        "system_prompt_characters": 0,
        "user_prompt_characters": 0,
    }


def log_runtime_event(
    event: str,
    *,
    service: str = "agent-service",
    trigger: str = "",
    submission_id: str = "",
    job_id: str = "",
    logical_call_id: str = "",
    physical_attempt_id: str = "",
    instruction_present: bool | None = None,
    canonical_intent_present: bool | None = None,
    legacy_schema_state_present: bool | None = None,
    model: str = "",
    api_key: str = "",
    api_key_source: str = "",
    messages: list[dict[str, Any]] | None = None,
    prompt_text: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    runtime = get_runtime_context()
    entry: dict[str, Any] = {
        "event": event,
        "timestamp": _utc_now_iso(),
        "service": service,
        "trigger": trigger or str(runtime.get("trigger", "")),
        "submission_id": submission_id or str(runtime.get("submission_id", "")),
        "job_id": job_id or str(runtime.get("job_id", "")),
        "logical_call_id": logical_call_id,
        "physical_attempt_id": physical_attempt_id,
        "instruction_present": instruction_present if instruction_present is not None else bool(runtime.get("instruction_present")),
        "canonical_intent_present": canonical_intent_present if canonical_intent_present is not None else bool(runtime.get("canonical_intent_present")),
        "legacy_schema_state_present": legacy_schema_state_present if legacy_schema_state_present is not None else bool(runtime.get("legacy_schema_state_present")),
        "model": model,
        "api_key_source": api_key_source,
        "api_key_fingerprint": _safe_key_fingerprint(api_key),
        **make_prompt_metadata(messages=messages, prompt_text=prompt_text),
        **extra,
    }
    _write_entry(entry)
    return entry


def log_llm_started(
    *,
    service: str,
    operation: str,
    caller_file: str,
    caller_function: str,
    model: str,
    api_key_source: str,
    api_key: str,
    attempt: int,
    trigger: str,
    messages: list,
    submission_id: str = "",
    job_id: str = "",
    correlation_id: str = "",
    endpoint_or_worker: str = "",
) -> dict:
    """Log llm_call_started and return context dict for completion/failure logging."""
    logical_call_id = str(uuid.uuid4())[:8]
    physical_attempt_id = str(uuid.uuid4())[:8]

    runtime = get_runtime_context()
    entry = {
        "event": "llm_call_started",
        "logical_call_id": logical_call_id,
        "physical_attempt_id": physical_attempt_id,
        "job_id": job_id or str(runtime.get("job_id", "")),
        "submission_id": submission_id or str(runtime.get("submission_id", "")),
        "correlation_id": correlation_id,
        "service": service,
        "endpoint_or_worker": endpoint_or_worker,
        "operation": operation,
        "caller_file": caller_file,
        "caller_function": caller_function,
        "model": model,
        "api_key_source": api_key_source,
        "api_key_fingerprint": _safe_key_fingerprint(api_key),
        "attempt": attempt,
        "trigger": trigger,
        "prompt_hash": _prompt_hash(messages),
        **_prompt_stats(messages),
        "timestamp": _utc_now_iso(),
        "instruction_present": bool(runtime.get("instruction_present")),
        "canonical_intent_present": bool(runtime.get("canonical_intent_present")),
        "legacy_schema_state_present": bool(runtime.get("legacy_schema_state_present")),
    }
    _write_entry(entry)
    return {
        "logical_call_id": logical_call_id,
        "physical_attempt_id": physical_attempt_id,
        "start_time": time.perf_counter(),
        **entry,
    }


def log_llm_completed(
    ctx: dict,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    finish_reason: str = "",
) -> None:
    """Log llm_call_completed with actual token usage from provider."""
    duration_ms = (time.perf_counter() - ctx["start_time"]) * 1000
    entry = {
        "event": "llm_call_completed",
        "logical_call_id": ctx["logical_call_id"],
        "physical_attempt_id": ctx["physical_attempt_id"],
        "job_id": ctx.get("job_id", ""),
        "submission_id": ctx.get("submission_id", ""),
        "service": ctx["service"],
        "operation": ctx["operation"],
        "model": ctx["model"],
        "attempt": ctx["attempt"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "duration_ms": round(duration_ms, 1),
        "finish_reason": finish_reason,
        "timestamp": _utc_now_iso(),
    }
    _write_entry(entry)

def log_llm_failed(
    ctx: dict,
    *,
    status_code: int = 0,
    error_type: str = "",
    error_message: str = "",
    headers: dict | None = None,
) -> None:
    """Log llm_call_failed with rate-limit headers when available."""
    duration_ms = (time.perf_counter() - ctx["start_time"]) * 1000
    hdrs = headers or {}
    entry = {
        "event": "llm_call_failed",
        "logical_call_id": ctx["logical_call_id"],
        "physical_attempt_id": ctx["physical_attempt_id"],
        "job_id": ctx.get("job_id", ""),
        "submission_id": ctx.get("submission_id", ""),
        "service": ctx["service"],
        "operation": ctx["operation"],
        "model": ctx["model"],
        "attempt": ctx["attempt"],
        "status_code": status_code,
        "error_type": error_type,
        "error_message": error_message[:300],
        "retry_after": hdrs.get("retry-after", ""),
        "rate_limit_limit_tokens": hdrs.get("x-ratelimit-limit-tokens", ""),
        "rate_limit_remaining_tokens": hdrs.get("x-ratelimit-remaining-tokens", ""),
        "rate_limit_reset_tokens": hdrs.get("x-ratelimit-reset-tokens", ""),
        "duration_ms": round(duration_ms, 1),
        "timestamp": _utc_now_iso(),
    }
    _write_entry(entry)
