from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID
import warnings
from typing import Any

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.constants import DEFAULT_OUTPUT_FORMAT_OPTIONS
from app.core.security import require_roles
from app.db.session import get_db
from app.models import AuditLog, DataProfile, Review, Submission, SubmissionRecord, SubmissionStatus, User, UserRole, normalize_submission_status
from app.services.agent_dispatcher import enqueue_submission_dispatch
from app.services.canonical_intent import build_canonical_intent
from app.services.intent_revision import persist_intent_revision
from app.schemas import JobAgentSummaryRead, JobAuditEntryRead, JobDetailRead, JobStepRead, UploadMetadataRead, UploadPreview, UploadSummary, UploadVersionRead
from app.services.excel_parser import SUPPORTED_EXTENSIONS, validate_extension
from app.services.file_validation import validate_file_signature
from app.services.data_profile import get_or_create_data_profile, load_latest_data_profile
from app.services.json_safety import make_json_safe
from app.services.quarantine import is_quarantined_submission, requeue_submission
from app.services.request_security import enforce_rate_limit
from app.services.llm_telemetry import log_runtime_event
from app.services.structured_output import sanitize_structured_row, sanitize_structured_rows
from app.services.websocket_manager import ws_manager
from pydantic import BaseModel
import json
from app.services.request_security import _get_redis_client

router = APIRouter(prefix="/uploads", tags=["uploads"])
CONTENT_TYPE_ALLOWLIST = {
    ".xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
    },
    ".xls": {
        "application/vnd.ms-excel",
        "application/octet-stream",
    },
    ".csv": {"text/csv", "application/csv", "application/vnd.ms-excel", "application/octet-stream"},
    ".tsv": {"text/tab-separated-values", "text/plain", "application/octet-stream"},
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".png": {"image/png", "application/octet-stream"},
    ".jpg": {"image/jpeg", "application/octet-stream"},
    ".jpeg": {"image/jpeg", "application/octet-stream"},
    ".webp": {"image/webp", "application/octet-stream"},
    ".json": {"application/json", "text/json", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
}

SCHEMA_APPROVAL_AGENT_STATUS = "awaiting_schema_approval"
SCHEMA_APPROVAL_TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".txt"}
FAILED_SUBMISSION_STATUSES = {SubmissionStatus.failed.value, SubmissionStatus.callback_failed.value}
EXPLICIT_CLEANING_TERMS = (
    "clean",
    "cleanup",
    "clean up",
    "normalize",
    "normalise",
    "standardize",
    "standardise",
    "deduplicate",
    "de-duplicate",
    "remove duplicates",
    "trim whitespace",
    "remove ",
    "filter ",
    "keep only",
    "only give me",
    "merchant",
    "payment method",
)


def _submission_status_text(value: object) -> str:
    return normalize_submission_status(value)


def _is_submission_status(submission: Submission, *expected: str) -> bool:
    return _submission_status_text(submission.status) in set(expected)


def extract_agent_resolution(submission: Submission) -> dict:
    result_payload = submission.summary if isinstance(submission.summary, dict) else {}
    available_agents = result_payload.get("available_agents")
    return {
        "agent_status": _submission_status_text(submission.status),
        "agent_error": _summary_text(
            submission.summary.get("error") if isinstance(submission.summary, dict) else None,
            default="",
        ),
        "response_reason": result_payload.get("reason"),
        "response_suggestion": result_payload.get("suggestion"),
        "quarantine_status": result_payload.get("quarantine_status"),
        "available_agents": available_agents if isinstance(available_agents, list) else [],
        "preferred_agent_name": submission.preferred_agent_name,
        "output_ready": output_is_available(submission),
        "job_summary": build_job_summary_text(submission, result_payload),
        "agent_summaries": build_agent_summaries(submission, result_payload),
        "preview_token": result_payload.get("preview_token"),
    }


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _first_summary_sentence(value: object) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    return cleaned[:-1] if cleaned.endswith(".") else cleaned


def _summary_text(*values: object, default: str = "") -> str:
    """Return the first non-empty normalized text from the provided values."""
    for value in values:
        cleaned = _first_summary_sentence(value)
        if cleaned and cleaned.lower() not in {"none", "null"}:
            return cleaned
    return default


def _normalize_summary_status(value: object, fallback: str = "complete") -> str:
    status = _clean_text(value).lower()
    if status in {"", "success"}:
        return fallback
    return status


def should_require_schema_approval(*, extension: str, instruction: str) -> bool:
    normalized_extension = str(extension or "").lower()
    if normalized_extension not in SCHEMA_APPROVAL_TABULAR_EXTENSIONS:
        return False
    return True


def is_schema_approval_pending(submission: Submission) -> bool:
    return _submission_status_text(submission.status) == SCHEMA_APPROVAL_AGENT_STATUS


def _data_profile_payload(record: DataProfile | None) -> dict:
    if record is None or not isinstance(record.profile_json, dict):
        return {}
    return make_json_safe(record.profile_json)


async def _load_submission_data_profile(db: AsyncSession, submission: Submission) -> dict[str, Any]:
    return _data_profile_payload(await load_latest_data_profile(db, submission.id))


def _profile_status(submission: Submission, data_profile: dict[str, Any]) -> str:
    if data_profile:
        return str(data_profile.get("profile_status", "ready"))
    if isinstance(submission.summary, dict) and submission.summary:
        return "missing_legacy_state"
    return "missing"


def _compose_clarification_view(submission: Submission, canonical_intent: dict[str, Any] | None) -> dict[str, Any] | None:
    status_text = _submission_status_text(submission.status)
    extraction_preview = get_extraction_preview(submission)
    if extraction_preview and status_text == SubmissionStatus.awaiting_confirmation.value:
        return {
            "status": "awaiting_extraction_confirmation",
            "mode": "extraction_confirmation",
            "reason": "The extracted preview is ready for confirmation before the output is persisted.",
            "extraction_preview": extraction_preview,
        }
    if status_text not in {SubmissionStatus.awaiting_confirmation.value, SubmissionStatus.awaiting_clarification.value}:
        return None
    if isinstance(canonical_intent, dict) and str(canonical_intent.get("resolution_status", "")).strip() in {"resolved", "repaired"}:
        return {
            "status": "awaiting_intent_confirmation",
            "mode": "intent_confirmation",
            "reason": "The canonical interpretation is ready for explicit confirmation.",
        }
    return {
        "status": "needs_clarification",
        "mode": "clarification",
        "reason": "The canonical interpretation still contains unresolved references.",
    }


def _compose_execution_view(submission: Submission) -> dict[str, Any]:
    status_text = _submission_status_text(submission.status)
    summary = submission.summary if isinstance(submission.summary, dict) else {}
    return {
        "status": status_text,
        "output_ready": output_is_available(submission),
        "completed_at": submission.completed_at,
        "error": summary.get("error"),
        "review_status": summary.get("review_status"),
    }


def _latest_submission_profile_json(submission: Submission) -> dict[str, Any]:
    records = getattr(submission, "data_profiles", None)
    if not isinstance(records, list) or not records:
        return {}
    latest = max(
        (record for record in records if isinstance(getattr(record, "profile_json", None), dict)),
        key=lambda record: str(getattr(record, "created_at", "")),
        default=None,
    )
    return _data_profile_payload(latest)


def _infer_display_type(values: list[object]) -> str:
    series = pd.Series(values, dtype="object").dropna()
    if series.empty:
        return "string"
    numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
    if numeric_ratio >= 0.9:
        return "number"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        date_ratio = pd.to_datetime(series, errors="coerce", dayfirst=True).notna().mean()
    if date_ratio >= 0.9:
        return "date"
    return "string"


def get_extraction_preview(submission: Submission) -> dict:
    payload = submission.summary if isinstance(submission.summary, dict) else {}
    if not isinstance(payload, dict):
        return {}
    preview_rows = payload.get("preview")
    columns = payload.get("columns")
    if not isinstance(preview_rows, list) or not isinstance(columns, list):
        return {}
    field_types = payload.get("field_types") if isinstance(payload.get("field_types"), dict) else {}
    validation_warnings = payload.get("validation_warnings") if isinstance(payload.get("validation_warnings"), list) else []
    proposed_fields = payload.get("proposed_fields") if isinstance(payload.get("proposed_fields"), list) else []
    format_profile = payload.get("format_profile") if isinstance(payload.get("format_profile"), dict) else {}
    assumed_date_convention = str(payload.get("assumed_date_convention", "")).strip()
    has_ambiguous_warning = any(
        isinstance(item, dict) and str(item.get("rule", "")).strip() == "ambiguous_date_assumption"
        for item in validation_warnings
    )
    if payload.get("ambiguous_date_count") and not has_ambiguous_warning:
        validation_warnings = [
            *validation_warnings,
            {
                "column": str(payload.get("anchor_column", "")).strip() or "date",
                "rule": "ambiguous_date_assumption",
                "severity": "warning",
                "reason": f"Some slash dates were ambiguous, so the preview assumed {assumed_date_convention or 'DD/MM/YYYY'}.",
                "invalid_count": int(payload.get("ambiguous_date_count", 0) or 0),
                "sample_values": [],
            },
        ]
    return {
        "schema_kind": "unstructured_extraction_preview",
        "proposed_fields": proposed_fields or [
            {
                "source": str(column),
                "target": str(column),
                "detected_type": str(field_types.get(str(column), "text")),
                "confidence": "high",
                "reason": "Extracted directly from the unstructured file preview.",
            }
            for column in columns
        ],
        "validation_warnings": validation_warnings,
        "action_schema": {"actions": []},
        "preview_rows": [row for row in preview_rows if isinstance(row, dict)],
        "source_columns": [str(column) for column in columns],
        "detected_types": {str(key): str(value) for key, value in field_types.items()},
        "anchor_column": str(payload.get("anchor_column", "")).strip(),
        "complete_count": int(payload.get("complete_count", 0) or 0),
        "partial_count": int(payload.get("partial_count", 0) or 0),
        "invalid_count": int(payload.get("invalid_count", 0) or 0),
        "llm_only_count": int(payload.get("llm_only_count", 0) or 0),
        "recovered_count": int(payload.get("recovered_count", 0) or 0),
        "merged_count": int(payload.get("merged_count", 0) or 0),
        "ambiguous_date_count": int(payload.get("ambiguous_date_count", 0) or 0),
        "assumed_date_convention": assumed_date_convention,
        "format_profile": format_profile,
        "total_rows": len(preview_rows),
        "suggestion": "Review the extracted preview, then confirm to persist and download the Excel output.",
    }


def get_result_record_count(submission: Submission) -> int:
    payload = submission.summary if isinstance(submission.summary, dict) else {}
    raw_count = payload.get("record_count", 0)
    try:
        return max(0, int(raw_count or 0))
    except (TypeError, ValueError):
        return 0


import re

def _humanize_orchestrator_summary(value: object) -> str:
    cleaned = _first_summary_sentence(value)
    if not cleaned:
        return "Reviewed the request and selected the best execution path."

    cleaned = cleaned.strip()
    if cleaned.startswith("[Intent:"):
        # The intent block might contain nested brackets like ['foo'].
        # We find the last bracket before the actual message starts.
        # Often the message starts with a capital letter.
        cleaned = re.sub(r'^\[Intent:.*?\]\s*(?=[A-Z])', '', cleaned)
        # Fallback if the regex didn't perfectly strip it (e.g. if it didn't start with a capital letter)
        if cleaned.startswith("] "):
            cleaned = cleaned[2:]

    replacements = {
        "Prompt explicitly requested cleanup or normalization":
            "Detected a cleanup request and planned cleaning before generating the final output",
        "Direct artifact generation":
            "Detected an output request and skipped cleaning because the extracted data was already usable",
        "Data available, no mandatory cleaning stage required":
            "Detected usable extracted data and routed it directly to output generation",
    }
    for source, target in replacements.items():
        if source in cleaned:
            return target

    if "Cleaning forced by data quality signals:" in cleaned:
        signal_text = cleaned.split(":", 1)[1].strip() if ":" in cleaned else "quality issues in the extracted data"
        # remove trailing brackets if any got stuck
        signal_text = signal_text.lstrip("] ")
        return f"Detected data quality issues and added a cleaning step before output generation ({signal_text})"

    return cleaned.lstrip("] ")


def _frontend_push_note(push_report: dict | None) -> str:
    if not isinstance(push_report, dict):
        return ""
    status = _clean_text(push_report.get("status")).lower()
    message = _first_summary_sentence(push_report.get("message"))
    if status == "success":
        return "Live frontend delivery succeeded"
    if status == "skipped":
        return message or "Live frontend delivery was skipped in this environment; the output is still available for download"
    if status == "failed":
        error = _first_summary_sentence(push_report.get("error"))
        if error:
            return f"Live frontend delivery failed: {error}"
        return "Live frontend delivery failed"
    return ""


def _ui_agent_summary(push_report: dict | None) -> str:
    if isinstance(push_report, dict) and _clean_text(push_report.get("status")).lower() == "success":
        return "Prepared the final output and delivered it to the frontend."
    return "Prepared the final output and made it available for download."


def _cleaning_quality_gate_note(cleaning_summary: dict | None, cleaning_report: dict | None) -> str:
    summary = cleaning_summary if isinstance(cleaning_summary, dict) else {}
    report = cleaning_report if isinstance(cleaning_report, dict) else {}
    quality_gate = report.get("quality_gate") if isinstance(report.get("quality_gate"), dict) else {}
    action = _clean_text(report.get("quality_gate_action")).lower()
    passed = summary.get("quality_gate_passed")
    reason = _first_summary_sentence(summary.get("quality_gate_reason") or quality_gate.get("reason"))

    if action == "reverted_to_input" or passed is False:
        if reason:
            return f"Cleaning ran, but the final output was reverted to the original data because the quality gate failed ({reason})"
        return "Cleaning ran, but the final output was reverted to the original data because the quality gate failed"
    return ""


def _normalize_agent_summary(entry: object) -> JobAgentSummaryRead | None:
    if not isinstance(entry, dict):
        return None
    bullets = entry.get("bullets")
    agent_id = _clean_text(entry.get("agent_id") or entry.get("agent") or entry.get("name") or "agent") or "agent"
    summary = _clean_text(entry.get("summary") or entry.get("detail") or entry.get("description") or "No summary available.") or "No summary available."
    normalized_bullets = [_clean_text(item) for item in bullets if _clean_text(item)] if isinstance(bullets, list) else []
    if agent_id == "orchestrator":
        summary = _humanize_orchestrator_summary(summary)
    return JobAgentSummaryRead(
        agent_id=agent_id,
        agent_name=_clean_text(entry.get("agent_name") or entry.get("label") or entry.get("agent_id") or "Agent") or "Agent",
        status=_clean_text(entry.get("status") or "complete") or "complete",
        summary=summary,
        bullets=normalized_bullets,
    )


def build_agent_summaries(submission: Submission, result_payload: dict | None = None) -> list[JobAgentSummaryRead]:
    payload = result_payload if isinstance(result_payload, dict) else (submission.summary if isinstance(submission.summary, dict) else {})
    existing = payload.get("agent_summaries")
    if isinstance(existing, list):
        normalized = [_normalize_agent_summary(entry) for entry in existing]
        summaries = [entry for entry in normalized if entry is not None]
        if summaries:
            return summaries

    summaries: list[JobAgentSummaryRead] = []
    payload_status = _clean_text(payload.get("status")).lower()
    response_reason = _first_summary_sentence(payload.get("reason"))
    response_suggestion = _first_summary_sentence(payload.get("suggestion"))
    orchestration_reason = _first_summary_sentence(payload.get("orchestration_reason"))
    registry_match = payload.get("registry_match") or []
    available_agents = payload.get("available_agents") if isinstance(payload.get("available_agents"), list) else []
    preferred_agent = _clean_text(payload.get("preferred_agent") or submission.preferred_agent_name)

    if orchestration_reason or registry_match or available_agents or preferred_agent:
        bullets: list[str] = []
        if registry_match:
            bullets.append(f"{len(registry_match)} registry match{'es' if len(registry_match) != 1 else ''} evaluated")
        if available_agents:
            bullets.append(f"Available agents: {', '.join(_clean_text(agent) for agent in available_agents if _clean_text(agent))}")
        if preferred_agent:
            bullets.append(f"Preferred agent: {preferred_agent}")
        execution_plan = payload.get("execution_plan")
        if isinstance(execution_plan, list):
            plan = [str(agent).replace("_", " ").title() for agent in execution_plan if _clean_text(agent)]
            if plan:
                bullets.append(f"Execution plan: {' -> '.join(plan)}")
        summaries.append(
            JobAgentSummaryRead(
                agent_id="orchestrator",
                agent_name="Orchestrator",
                status=_normalize_summary_status(payload.get("response_status") or payload_status),
                summary=_humanize_orchestrator_summary(orchestration_reason or response_reason),
                bullets=bullets,
            )
        )

    cleaning_summary = payload.get("cleaning_summary") if isinstance(payload.get("cleaning_summary"), dict) else {}
    cleaning_report = payload.get("cleaning_report") if isinstance(payload.get("cleaning_report"), dict) else {}
    cleaning_bullets = cleaning_summary.get("summary_bullets") if isinstance(cleaning_summary, dict) else []
    if cleaning_report or payload.get("cleaned_data") or payload.get("cleaned_text"):
        quality_gate_note = _cleaning_quality_gate_note(cleaning_summary, cleaning_report)
        bullets = [_clean_text(item) for item in cleaning_bullets if _clean_text(item)] if isinstance(cleaning_bullets, list) else []
        if quality_gate_note and quality_gate_note not in bullets:
            bullets.insert(0, quality_gate_note)
        if not bullets:
            bullets = ["Applied cleaning operations and prepared the transformed data."]
        summaries.append(
            JobAgentSummaryRead(
                agent_id="cleaning_agent",
                agent_name="Data Cleaning Agent",
                status="failed" if payload_status == "failed" or quality_gate_note else "complete",
                summary=quality_gate_note or _first_summary_sentence(bullets[0]),
                bullets=bullets[:5],
            )
        )

    save_report = payload.get("excel_save_report") if isinstance(payload.get("excel_save_report"), dict) else {}
    push_report = payload.get("frontend_push_report") if isinstance(payload.get("frontend_push_report"), dict) else {}
    if save_report or push_report or payload.get("output_path"):
        output_format = _clean_text(payload.get("requested_output_format") or submission.output_format).upper()
        bullets: list[str] = []
        if output_format:
            bullets.append(f"Prepared {output_format} output")
        if save_report.get("status"):
            bullets.append(f"Output save status: {_clean_text(save_report.get('status'))}")
        frontend_note = _frontend_push_note(push_report)
        if frontend_note:
            bullets.append(frontend_note)
        summaries.append(
            JobAgentSummaryRead(
                agent_id="ui_agent",
                agent_name="UI Agent",
                status="failed" if _clean_text(push_report.get("status")).lower() == "failed" else "complete",
                summary=_ui_agent_summary(push_report),
                bullets=bullets,
            )
        )

    return summaries


def build_job_summary_text(submission: Submission, result_payload: dict | None = None) -> str:
    payload = result_payload if isinstance(result_payload, dict) else (submission.summary if isinstance(submission.summary, dict) else {})
    payload_status = _clean_text(payload.get("status")).lower()
    response_reason = _first_summary_sentence(payload.get("reason"))
    response_suggestion = _first_summary_sentence(payload.get("suggestion"))
    job_summary = _first_summary_sentence(payload.get("job_summary"))
    cleaning_summary = payload.get("cleaning_summary") if isinstance(payload.get("cleaning_summary"), dict) else {}
    cleaning_report = payload.get("cleaning_report") if isinstance(payload.get("cleaning_report"), dict) else {}
    quality_gate_note = _cleaning_quality_gate_note(cleaning_summary, cleaning_report)
    workflow_status = _submission_status_text(submission.status)
    if job_summary and not quality_gate_note:
        return job_summary
    if quality_gate_note:
        return f"{quality_gate_note}. Downloaded output matches the original data."

    if workflow_status == SubmissionStatus.running.value:
        return "The workflow is currently being routed and processed."

    if is_schema_approval_pending(submission):
        return "Schema preview is waiting for your approval before processing begins."

    if workflow_status == SubmissionStatus.queued.value and not payload_status:
        return "The workflow is being prepared for execution."

    if payload_status in {"pending_agent_availability", "rejected"}:
        if response_reason and response_suggestion:
            return f"{response_reason}. {response_suggestion}"
        if response_reason:
            return response_reason
        if response_suggestion:
            return response_suggestion
        return "Part of this workflow is quarantined until supported agent coverage is available."

    if payload_status in FAILED_SUBMISSION_STATUSES:
        agent_error = _summary_text(submission.summary.get("error") if isinstance(submission.summary, dict) else None)
        if agent_error:
            return agent_error
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            first_error = _first_summary_sentence(errors[0])
            if first_error:
                return first_error
        return "The workflow could not complete successfully."

    bullets = cleaning_summary.get("summary_bullets") if isinstance(cleaning_summary, dict) else []
    if isinstance(bullets, list):
        cleaned_bullets = [_clean_text(item) for item in bullets if _clean_text(item)]
        if cleaned_bullets:
            leading = "; ".join(cleaned_bullets[:2])
            return f"{leading}. Output is ready."

    agent_summaries = build_agent_summaries(submission, payload)
    for entry in agent_summaries:
        if entry.agent_id != "orchestrator" and _clean_text(entry.summary):
            return f"{_first_summary_sentence(entry.summary)}. Output is ready."

    if response_reason:
        return f"{response_reason}. Output is ready." if payload_status == SubmissionStatus.succeeded.value else response_reason

    return "The workflow completed successfully and the output is ready."


def _recoverable_output_format(submission: Submission) -> str:
    requested = str(submission.output_format or "").strip().upper()
    if requested in {"XLSX", "CSV", "JSON", "TXT"}:
        return requested

    payload = submission.summary if isinstance(submission.summary, dict) else {}
    cleaned_data = payload.get("cleaned_data")
    if isinstance(cleaned_data, list) and cleaned_data:
        return "XLSX"
    if get_result_record_count(submission) > 0:
        return "XLSX"
    return requested


def _load_result_frame(submission: Submission) -> pd.DataFrame | None:
    payload = submission.summary if isinstance(submission.summary, dict) else {}
    cleaned_data = payload.get("cleaned_data")
    if isinstance(cleaned_data, list) and cleaned_data:
        return pd.DataFrame(sanitize_structured_rows(cleaned_data))

    cleaned_text = str(payload.get("cleaned_text") or "").strip()
    if cleaned_text:
        return pd.DataFrame([{"cleaned_text": cleaned_text}])
    return None


def _frame_from_structured_records(rows: list[SubmissionRecord]) -> pd.DataFrame | None:
    payload_rows = [
        sanitize_structured_row(row.payload if isinstance(row.payload, dict) else {})
        for row in rows
    ]
    payload_rows = [row for row in payload_rows if row]
    if not payload_rows:
        return None
    return pd.DataFrame(payload_rows)


def _payload_mentions_output_artifact(submission: Submission) -> bool:
    payload = submission.summary if isinstance(submission.summary, dict) else {}
    return any(
        str(payload.get(key) or "").strip()
        for key in ("output_path", "excel_file_path", "output_file_name", "output_relative_path")
    )


def output_is_available(submission: Submission) -> bool:
    if _submission_status_text(submission.status) != SubmissionStatus.succeeded.value:
        return False
    if submission.output_path and Path(submission.output_path).exists():
        return True

    recoverable_format = _recoverable_output_format(submission)
    if recoverable_format not in {"XLSX", "CSV", "JSON", "TXT"}:
        return False
    if _load_result_frame(submission) is not None:
        return True
    if get_result_record_count(submission) > 0:
        return True
    return _payload_mentions_output_artifact(submission)


async def ensure_output_file(db: AsyncSession, submission: Submission) -> Path | None:
    if _submission_status_text(submission.status) != SubmissionStatus.succeeded.value:
        return None

    if submission.output_path:
        existing = Path(submission.output_path)
        if existing.exists():
            return existing

    output_format = _recoverable_output_format(submission)
    if output_format not in {"XLSX", "CSV", "JSON", "TXT"}:
        return None

    frame = _load_result_frame(submission)
    if frame is None:
        structured_rows = (
            await db.execute(
                select(SubmissionRecord)
                .where(SubmissionRecord.submission_id == submission.id)
                .order_by(SubmissionRecord.record_index)
            )
        ).scalars().all()
        frame = _frame_from_structured_records(structured_rows)
    if frame is None:
        return None

    output_dir = Path(get_settings().output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_map = {
        "XLSX": ".xlsx",
        "CSV": ".csv",
        "JSON": ".json",
        "TXT": ".txt",
    }
    output_path = output_dir / f"{submission.id}-recovered-output{suffix_map[output_format]}"
    if output_format == "XLSX":
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="cleaned_data", index=False)
    elif output_format == "CSV":
        frame.to_csv(output_path, index=False)
    elif output_format == "JSON":
        output_path.write_text(frame.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")
    elif output_format == "TXT":
        output_path.write_text(frame.to_string(index=False), encoding="utf-8")

    submission.output_path = str(output_path)
    submission.output_file_path = str(output_path)
    await db.commit()
    await db.refresh(submission)
    return output_path


@router.get("/metadata", response_model=UploadMetadataRead)
async def get_upload_metadata(
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> UploadMetadataRead:
    settings = get_settings()
    return UploadMetadataRead(
        accepted_file_types=sorted(ext.removeprefix(".").upper() for ext in SUPPORTED_EXTENSIONS),
        output_format_options=DEFAULT_OUTPUT_FORMAT_OPTIONS,
        max_upload_size_mb=settings.max_upload_size_mb,
    )

@router.post("", response_model=UploadPreview)
async def create_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    instruction: str = Form(default=""),
    output_format: str = Form(default="XLSX"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee)),
) -> UploadPreview:
    await enforce_rate_limit(request=request, bucket="upload_create", limit=20, window_seconds=60)
    return await save_upload(
        file=file,
        instruction=instruction,
        output_format=output_format,
        db=db,
        user=user,
        background_tasks=background_tasks,
    )


async def save_upload(
    *,
    file: UploadFile,
    instruction: str,
    output_format: str,
    db: AsyncSession,
    user: User,
    background_tasks: BackgroundTasks,
    parent_submission: Submission | None = None,
) -> UploadPreview:
    settings = get_settings()
    try:
        ext = validate_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    normalized_name = Path(file.filename or "upload").name
    if normalized_name != (file.filename or "upload"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if len(normalized_name) > 255:
        raise HTTPException(status_code=400, detail="Filename is too long")
    content_type = (file.content_type or "application/octet-stream").lower()
    allowed_content_types = CONTENT_TYPE_ALLOWLIST.get(ext, {"application/octet-stream"})
    if content_type not in allowed_content_types:
        raise HTTPException(status_code=400, detail="File content type does not match the selected file type")
    normalized_output_format = (output_format or "XLSX").upper().strip()
    if normalized_output_format not in DEFAULT_OUTPUT_FORMAT_OPTIONS:
        raise HTTPException(status_code=400, detail="Unsupported output format")

    await ws_manager.broadcast("uploads", "upload_progress", {"filename": file.filename, "progress": 10})
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(contents) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File size limit is {settings.max_upload_size_mb} MB")
    try:
        validate_file_signature(extension=ext, contents=contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    root_submission_id = None
    version_number = 1
    if parent_submission:
        root_submission_id = parent_submission.parent_submission_id or parent_submission.id
        version_number = (
            await db.scalar(
                select(func.max(Submission.version_number)).where(
                    (Submission.id == root_submission_id) | (Submission.parent_submission_id == root_submission_id)
                )
            )
            or parent_submission.version_number
            or 1
        ) + 1

    submission = Submission(
        user_id=user.id,
        file_name=normalized_name,
        file_path="pending",
        file_size_bytes=len(contents),
        original_filename=normalized_name,
        instruction=instruction.strip(),
        output_format=normalized_output_format,
        version_number=version_number,
        parent_submission_id=root_submission_id,
        status=SubmissionStatus.queued,
    )
    db.add(submission)
    await db.flush()
    count = await db.scalar(select(func.count()).select_from(Submission))
    submission.sub_id = count

    path = upload_dir / f"{submission.id}{ext}"
    path.write_bytes(contents)
    submission.file_path = str(path)

    preview_records: list[dict[str, Any]] = []
    data_profile_payload: dict[str, Any] = {}
    canonical_intent_payload: dict[str, Any] = {}
    if should_require_schema_approval(extension=ext, instruction=instruction.strip()):
        _profile_record, data_profile_payload, preview_records = await get_or_create_data_profile(
            db,
            submission=submission,
            path=path,
            max_preview_rows=settings.max_preview_rows,
        )
        if data_profile_payload:
            canonical_intent_payload = build_canonical_intent(
                list(data_profile_payload.get("source_columns", [])),
                list(data_profile_payload.get("preview_rows", [])),
                instruction.strip(),
                output_format=normalized_output_format.lower(),
                detected_types=data_profile_payload.get("detected_types", {}),
                data_profile=data_profile_payload,
                submission_id=str(submission.id),
                trigger="upload",
            )
            await persist_intent_revision(
                db,
                submission=submission,
                canonical_intent=canonical_intent_payload,
                original_instruction=instruction.strip(),
            )

    if canonical_intent_payload and str(canonical_intent_payload.get("resolution_status", "")).strip() not in {"resolved", "repaired"}:
        submission.status = SubmissionStatus.awaiting_clarification
        submission.summary = {
            "status": "awaiting_clarification",
            "suggestion": canonical_intent_payload.get("suggestion") or "Clarify the intended operation before processing continues.",
            "canonical_intent": canonical_intent_payload,
        }
    elif canonical_intent_payload:
        submission.summary = {
            "status": "canonical_ready",
            "suggestion": None,
            "canonical_intent": canonical_intent_payload,
        }

    await db.commit()
    await db.refresh(submission)

    payload = {
        "upload_id": submission.id,
        "filename": submission.file_name,
        "status": _submission_status_text(submission.status),
        "total_rows": int(data_profile_payload.get("row_count", 0) or 0),
    }
    payload["agent_status"] = _submission_status_text(submission.status)
    await ws_manager.broadcast("uploads", "upload_progress", {**payload, "progress": 40})
    await ws_manager.broadcast(
        "uploads",
        "upload.processing" if _submission_status_text(submission.status) == SubmissionStatus.queued.value else "upload.intent_review",
        payload,
    )
    await ws_manager.broadcast("uploads", "upload_status", payload)
    await ws_manager.broadcast("dashboard", "dashboard_refresh", payload)

    if canonical_intent_payload and str(canonical_intent_payload.get("resolution_status", "")).strip() in {"resolved", "repaired"}:
        submission.status = SubmissionStatus.queued
        await db.commit()
        await db.refresh(submission)
        payload["agent_status"] = _submission_status_text(submission.status)
        try:
            await enqueue_submission_dispatch(submission.id, persist_revision=False)
        except Exception as exc:
            submission.status = SubmissionStatus.failed
            submission.summary = {'error': f"Unable to enqueue agent dispatch: {exc}"} if not isinstance(submission.summary, dict) else {**submission.summary, 'error': f"Unable to enqueue agent dispatch: {exc}"}
            submission.completed_at = datetime.utcnow()
            await db.commit()
            await db.refresh(submission)
            failed_payload = {
                "upload_id": submission.id,
                "filename": submission.file_name,
                "status": _submission_status_text(submission.status),
                "agent_status": _submission_status_text(submission.status),
                "error": _summary_text(
                    submission.summary.get("error") if isinstance(submission.summary, dict) else None,
                    default=f"Unable to enqueue agent dispatch: {exc}",
                ),
            }
            await ws_manager.broadcast("uploads", "upload.failed", failed_payload)
            await ws_manager.broadcast("uploads", "upload_status", failed_payload)
            await ws_manager.broadcast("dashboard", "dashboard_refresh", failed_payload)

    return UploadPreview(
        upload_id=submission.id,
        sub_id=submission.sub_id,
        filename=submission.file_name,
        instruction=submission.instruction,
        output_format=submission.output_format,
        status=_submission_status_text(submission.status),
        version_number=submission.version_number,
        parent_submission_id=submission.parent_submission_id,
        total_rows=int(data_profile_payload.get("row_count", 0) or 0),
        total_columns=len(data_profile_payload.get("source_columns", [])) if data_profile_payload else 0,
        created_at=submission.uploaded_at,
        columns=data_profile_payload.get("source_columns", []) if data_profile_payload else [],
        detected_types=data_profile_payload.get("detected_types", {}) if data_profile_payload else {},
        reviewed_at=submission.completed_at,
        validation={
            "valid": None if _submission_status_text(submission.status) in {SubmissionStatus.queued.value, SubmissionStatus.planning.value, SubmissionStatus.awaiting_confirmation.value, SubmissionStatus.awaiting_clarification.value} else True,
            "status": _submission_status_text(submission.status),
        },
        preview_rows=preview_records,
        version_history=await get_version_history(db, submission),
        preferred_agent_name=submission.preferred_agent_name,
        job_summary=(
            "Clarify or confirm the canonical interpretation before processing continues."
            if _submission_status_text(submission.status) in {SubmissionStatus.awaiting_confirmation.value, SubmissionStatus.awaiting_clarification.value}
            else None
        ),
        data_profile=data_profile_payload,
        profile_status=str(data_profile_payload.get("profile_status", "ready")) if data_profile_payload else None,
        canonical_intent=canonical_intent_payload,
        intent_status=str(canonical_intent_payload.get("resolution_status", "")) if canonical_intent_payload else None,
        clarification=_compose_clarification_view(submission, canonical_intent_payload),
        execution=_compose_execution_view(submission),
        repair_available=False,
    )


class ConfirmExtractionRequest(BaseModel):
    preview_token: str


class ConfirmInterpretationRequest(BaseModel):
    reason: str | None = None


class ReplaceColumnMappingRequest(BaseModel):
    mapping: dict[str, str | list[str]]
    reason: str | None = None


class RejectInterpretationRequest(BaseModel):
    reason: str | None = None


class ResumeJobRequest(BaseModel):
    reason: str | None = None

@router.post("/{upload_id}/confirm-extraction")
async def confirm_extraction(
    upload_id: UUID,
    payload: ConfirmExtractionRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
):
    submission = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.user))
            .where(Submission.id == upload_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    verify_upload_access(submission, user)

    agent_result = dict(submission.summary) if isinstance(submission.summary, dict) else {}
    expected_preview_token = str(agent_result.get("preview_token", "")).strip()
    if _submission_status_text(submission.status) == SubmissionStatus.succeeded.value:
        return {"status": "already_confirmed"}
    if _submission_status_text(submission.status) != SubmissionStatus.awaiting_confirmation.value:
        raise HTTPException(status_code=409, detail="This submission is not awaiting extraction confirmation")
    if not expected_preview_token or payload.preview_token != expected_preview_token:
        raise HTTPException(status_code=400, detail="Preview token does not match this submission")

    redis = await _get_redis_client()
    key = f"extraction_preview:{payload.preview_token}"
    data_str = await redis.get(key)
    if not data_str:
        if _submission_status_text(submission.status) == SubmissionStatus.succeeded.value:
            return {"status": "already_confirmed"}
        raise HTTPException(status_code=400, detail="Preview expired. Please re-upload.")

    data = json.loads(data_str)
    if str(data.get("job_id", "")).strip() != str(upload_id):
        raise HTTPException(status_code=400, detail="Preview token is not valid for this submission")
    rows = sanitize_structured_rows((data.get("complete_rows", []) + data.get("partial_rows", [])))

    await db.execute(delete(SubmissionRecord).where(SubmissionRecord.submission_id == upload_id))
    for i, row in enumerate(rows):
        db.add(SubmissionRecord(submission_id=upload_id, record_index=i, payload=row))

    submission.status = SubmissionStatus.succeeded
    submission.completed_at = datetime.now(UTC).replace(tzinfo=None)
    agent_result["cleaned_data"] = rows
    agent_result.pop("preview_token", None)
    submission.summary = agent_result
    submission.status = SubmissionStatus.succeeded

    await db.commit()
    await redis.delete(key)

    await ws_manager.broadcast("uploads", "extraction.confirmed", {"upload_id": str(upload_id)})
    return {"status": SubmissionStatus.succeeded.value}


async def retry_submission_dispatch(*, submission: Submission, db: AsyncSession, user: User | None = None, preferred_agent_name: str | None = None) -> None:
    await requeue_submission(db, submission=submission, actor=user, preferred_agent_name=preferred_agent_name)


def _current_canonical_intent_payload(submission: Submission) -> dict[str, Any] | None:
    if isinstance(submission.canonical_intent, dict):
        return deepcopy(submission.canonical_intent)
    summary = submission.summary if isinstance(submission.summary, dict) else {}
    canonical_intent = summary.get("canonical_intent")
    if isinstance(canonical_intent, dict):
        return deepcopy(canonical_intent)
    return None


def _canonical_intent_has_unresolved_fields(canonical_intent: dict[str, Any]) -> bool:
    for action in canonical_intent.get("actions", []):
        if not isinstance(action, dict):
            continue
        if action.get("kind") in {"project_columns", "drop_columns"}:
            for field in action.get("requested_fields", []):
                if not isinstance(field, dict):
                    return True
                if not field.get("resolved_column") and not field.get("resolved_columns"):
                    return True
        elif action.get("kind") == "filter_rows":
            for condition in action.get("conditions", []):
                if not isinstance(condition, dict):
                    return True
                field = condition.get("field")
                if isinstance(field, dict) and not field.get("resolved_column") and not field.get("resolved_columns"):
                    return True
    return False


def _normalise_column_mapping(mapping: dict[str, str | list[str]]) -> dict[str, list[str]]:
    normalised: dict[str, list[str]] = {}
    for raw_key, raw_value in mapping.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        if isinstance(raw_value, list):
            values = [str(item).strip() for item in raw_value if str(item).strip()]
        else:
            values = [str(raw_value).strip()] if str(raw_value or "").strip() else []
        if values:
            normalised[key] = values
    return normalised


def _match_column_mapping(field: dict[str, Any], mapping: dict[str, list[str]]) -> list[str] | None:
    candidates = [
        str(field.get("raw_reference", "")).strip(),
        str(field.get("resolved_column", "")).strip(),
    ]
    candidates.extend(str(value).strip() for value in field.get("candidate_columns", []) if str(value).strip())
    normalized_candidates = {str(candidate).lower(): candidate for candidate in candidates if candidate}
    normalized_candidates.update({str(candidate).replace("_", " ").lower(): candidate for candidate in candidates if candidate})
    for key, value in mapping.items():
        key_normalized = str(key).strip().lower()
        if key_normalized in normalized_candidates or key_normalized.replace("_", " ") in normalized_candidates:
            return value
    return None


def _apply_column_mapping_to_canonical_intent(canonical_intent: dict[str, Any], mapping: dict[str, str | list[str]]) -> tuple[dict[str, Any], bool]:
    normalised_mapping = _normalise_column_mapping(mapping)
    revised = deepcopy(canonical_intent)
    changed = False
    for action in revised.get("actions", []):
        if not isinstance(action, dict):
            continue
        kind = action.get("kind")
        if kind in {"project_columns", "drop_columns"}:
            for field in action.get("requested_fields", []):
                if not isinstance(field, dict):
                    continue
                resolved = _match_column_mapping(field, normalised_mapping)
                if not resolved:
                    continue
                changed = True
                field["candidate_columns"] = list(dict.fromkeys(resolved))
                field["resolved_columns"] = list(dict.fromkeys(resolved))
                field["selection_mode"] = "semantic_family" if len(resolved) > 1 else "single"
                field["resolved_column"] = resolved[0] if len(resolved) == 1 else None
                field["resolution_method"] = "human_correction"
            continue
        if kind == "filter_rows":
            for condition in action.get("conditions", []):
                if not isinstance(condition, dict):
                    continue
                field = condition.get("field")
                if not isinstance(field, dict):
                    continue
                resolved = _match_column_mapping(field, normalised_mapping)
                if not resolved:
                    continue
                if len(resolved) != 1:
                    raise HTTPException(status_code=400, detail="Filter conditions must resolve to exactly one column")
                changed = True
                field["candidate_columns"] = [resolved[0]]
                field["resolved_columns"] = [resolved[0]]
                field["selection_mode"] = "single"
                field["resolved_column"] = resolved[0]
                field["resolution_method"] = "human_correction"
            continue
        if kind == "sort_rows":
            for key in action.get("sort_keys", []):
                if not isinstance(key, dict):
                    continue
                field = key.get("column")
                if not isinstance(field, dict):
                    continue
                resolved = _match_column_mapping(field, normalised_mapping)
                if not resolved:
                    continue
                if len(resolved) != 1:
                    raise HTTPException(status_code=400, detail="Sort keys must resolve to exactly one column")
                changed = True
                field["candidate_columns"] = [resolved[0]]
                field["resolved_columns"] = [resolved[0]]
                field["selection_mode"] = "single"
                field["resolved_column"] = resolved[0]
                field["resolution_method"] = "human_correction"
            continue
        if kind == "visualize":
            for field in action.get("fields", []):
                if not isinstance(field, dict):
                    continue
                resolved = _match_column_mapping(field, normalised_mapping)
                if not resolved:
                    continue
                changed = True
                field["candidate_columns"] = list(dict.fromkeys(resolved))
                field["resolved_columns"] = list(dict.fromkeys(resolved))
                field["selection_mode"] = "semantic_family" if len(resolved) > 1 else "single"
                field["resolved_column"] = resolved[0] if len(resolved) == 1 else None
                field["resolution_method"] = "human_correction"
            continue
        if kind == "rename_columns":
            for mapping_item in action.get("mapping", []):
                if not isinstance(mapping_item, dict):
                    continue
                source = mapping_item.get("source")
                if not isinstance(source, dict):
                    continue
                resolved = _match_column_mapping(source, normalised_mapping)
                if not resolved or len(resolved) != 1:
                    continue
                changed = True
                source["candidate_columns"] = [resolved[0]]
                source["resolved_columns"] = [resolved[0]]
                source["selection_mode"] = "single"
                source["resolved_column"] = resolved[0]
                source["resolution_method"] = "human_correction"
    if changed:
        revised["resolution_status"] = "repaired"
        revised["grounded_at"] = datetime.now(UTC)
    return revised, changed


async def _persist_reviewed_canonical_intent(
    db: AsyncSession,
    submission: Submission,
    canonical_intent: dict[str, Any],
    *,
    original_instruction: str,
    review_status: str,
    review_reason: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    parent_intent_id = UUID(str(submission.intent_id)) if submission.intent_id else None
    revision_payload = await persist_intent_revision(
        db,
        submission=submission,
        canonical_intent=canonical_intent,
        original_instruction=original_instruction,
        parent_intent_id=parent_intent_id,
    )
    summary = submission.summary if isinstance(submission.summary, dict) else {}
    reviewed_intent = deepcopy(canonical_intent)
    reviewed_intent.update(
        {
            "intent_id": revision_payload["intent_id"],
            "intent_revision": revision_payload["intent_revision"],
            "intent_hash": revision_payload["intent_hash"],
            "parent_intent_id": revision_payload["parent_intent_id"],
            "capability_version": revision_payload["capability_version"],
            "capability_snapshot": revision_payload["capability_snapshot"],
            "created_at": revision_payload["created_at"],
            "grounded_at": revision_payload["grounded_at"],
        }
    )
    submission.summary = {
        **summary,
        "canonical_intent": reviewed_intent,
        "canonical_intent_schema_version": reviewed_intent.get("schema_version", "2.0"),
        "canonical_intent_status": reviewed_intent.get("resolution_status", "resolved"),
        "review_status": review_status,
        "review_reason": review_reason,
        "original_instruction": original_instruction,
        "intent_id": revision_payload["intent_id"],
        "intent_revision": revision_payload["intent_revision"],
        "intent_hash": revision_payload["intent_hash"],
        "parent_intent_id": revision_payload["parent_intent_id"],
        "grounded_at": revision_payload["grounded_at"],
        "capability_version": revision_payload["capability_version"],
    }
    submission.status = SubmissionStatus.queued if resume else SubmissionStatus.awaiting_confirmation
    submission.agent_task_id = None
    await db.commit()
    if resume:
        await enqueue_submission_dispatch(submission.id, persist_revision=False)
    return reviewed_intent


@router.post("/{upload_id}/confirm-interpretation", response_model=UploadPreview)
async def confirm_interpretation(
    upload_id: UUID,
    payload: ConfirmInterpretationRequest | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> UploadPreview:
    payload = payload or ConfirmInterpretationRequest()
    submission = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.user), selectinload(Submission.review))
            .where(Submission.id == upload_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    verify_upload_access(submission, user)
    if _submission_status_text(submission.status) not in {
        SubmissionStatus.awaiting_confirmation.value,
        SubmissionStatus.awaiting_clarification.value,
    }:
        raise HTTPException(status_code=409, detail="This submission is not awaiting interpretation review")

    canonical_intent = _current_canonical_intent_payload(submission)
    if canonical_intent is None:
        raise HTTPException(status_code=409, detail="No canonical intent is available for confirmation")
    if _canonical_intent_has_unresolved_fields(canonical_intent):
        raise HTTPException(status_code=409, detail="Canonical intent still contains unresolved fields. Replace the mapping first.")

    await _persist_reviewed_canonical_intent(
        db,
        submission,
        canonical_intent,
        original_instruction=str(submission.instruction or "").strip(),
        review_status="confirmed",
        review_reason=payload.reason,
        resume=True,
    )
    payload_out = {
        "upload_id": submission.id,
        "filename": submission.file_name,
        "status": _submission_status_text(submission.status),
        "agent_status": _submission_status_text(submission.status),
        "review_status": "confirmed",
    }
    await ws_manager.broadcast("uploads", "canonical_intent.confirmed", payload_out)
    await ws_manager.broadcast("uploads", "upload_status", payload_out)
    await ws_manager.broadcast("dashboard", "dashboard_refresh", payload_out)
    return await get_upload(upload_id, db, user)


@router.post("/{upload_id}/replace-column-mapping", response_model=UploadPreview)
async def replace_column_mapping(
    upload_id: UUID,
    payload: ReplaceColumnMappingRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> UploadPreview:
    submission = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.user), selectinload(Submission.review))
            .where(Submission.id == upload_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    verify_upload_access(submission, user)

    canonical_intent = _current_canonical_intent_payload(submission)
    if canonical_intent is None:
        raise HTTPException(status_code=409, detail="No canonical intent is available for replacement")

    revised_intent, changed = _apply_column_mapping_to_canonical_intent(canonical_intent, payload.mapping)
    if not changed:
        raise HTTPException(status_code=400, detail="No fields matched the provided mapping")

    await _persist_reviewed_canonical_intent(
        db,
        submission,
        revised_intent,
        original_instruction=str(submission.instruction or "").strip(),
        review_status="corrected",
        review_reason=payload.reason,
        resume=True,
    )
    payload_out = {
        "upload_id": submission.id,
        "filename": submission.file_name,
        "status": _submission_status_text(submission.status),
        "agent_status": _submission_status_text(submission.status),
        "review_status": "corrected",
    }
    await ws_manager.broadcast("uploads", "canonical_intent.corrected", payload_out)
    await ws_manager.broadcast("uploads", "upload_status", payload_out)
    await ws_manager.broadcast("dashboard", "dashboard_refresh", payload_out)
    return await get_upload(upload_id, db, user)


@router.post("/{upload_id}/reject-interpretation", response_model=UploadPreview)
async def reject_interpretation(
    upload_id: UUID,
    payload: RejectInterpretationRequest | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> UploadPreview:
    payload = payload or RejectInterpretationRequest()
    submission = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.user), selectinload(Submission.review))
            .where(Submission.id == upload_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    verify_upload_access(submission, user)
    if _submission_status_text(submission.status) not in {
        SubmissionStatus.awaiting_confirmation.value,
        SubmissionStatus.awaiting_clarification.value,
    }:
        raise HTTPException(status_code=409, detail="This submission is not awaiting interpretation review")

    canonical_intent = _current_canonical_intent_payload(submission)
    if canonical_intent is None:
        raise HTTPException(status_code=409, detail="No canonical intent is available for rejection")

    summary = submission.summary if isinstance(submission.summary, dict) else {}
    submission.summary = {
        **summary,
        "canonical_intent_status": "rejected",
        "review_status": "rejected",
        "review_reason": payload.reason,
        "status": "awaiting_clarification",
        "suggestion": payload.reason or "Clarify the intended fields or operations so the canonical interpretation can be rebuilt.",
    }
    submission.status = SubmissionStatus.awaiting_clarification
    submission.agent_task_id = None
    await db.commit()

    payload_out = {
        "upload_id": submission.id,
        "filename": submission.file_name,
        "status": _submission_status_text(submission.status),
        "agent_status": _submission_status_text(submission.status),
        "review_status": "rejected",
        "error": payload.reason or "The canonical interpretation was rejected by the user.",
    }
    await ws_manager.broadcast("uploads", "canonical_intent.rejected", payload_out)
    await ws_manager.broadcast("uploads", "upload_status", payload_out)
    await ws_manager.broadcast("dashboard", "dashboard_refresh", payload_out)
    return await get_upload(upload_id, db, user)


@router.post("/{upload_id}/resume-job", response_model=UploadPreview)
async def resume_job(
    upload_id: UUID,
    payload: ResumeJobRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> UploadPreview:
    submission = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.user), selectinload(Submission.review))
            .where(Submission.id == upload_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    verify_upload_access(submission, user)

    canonical_intent = _current_canonical_intent_payload(submission)
    if canonical_intent is None:
        raise HTTPException(status_code=409, detail="No canonical intent is available to resume")
    if _canonical_intent_has_unresolved_fields(canonical_intent):
        raise HTTPException(status_code=409, detail="Canonical intent still contains unresolved fields. Replace the mapping first.")

    summary = submission.summary if isinstance(submission.summary, dict) else {}
    submission.summary = {
        **summary,
        "canonical_intent_status": canonical_intent.get("resolution_status", "resolved"),
        "review_status": "resumed",
        "review_reason": payload.reason,
    }
    submission.status = SubmissionStatus.queued
    await db.commit()
    await enqueue_submission_dispatch(submission.id, persist_revision=False)

    payload_out = {
        "upload_id": submission.id,
        "filename": submission.file_name,
        "status": _submission_status_text(submission.status),
        "agent_status": _submission_status_text(submission.status),
        "review_status": "resumed",
    }
    await ws_manager.broadcast("uploads", "canonical_intent.resumed", payload_out)
    await ws_manager.broadcast("uploads", "upload_status", payload_out)
    await ws_manager.broadcast("dashboard", "dashboard_refresh", payload_out)
    return await get_upload(upload_id, db, user)


@router.get("", response_model=list[UploadSummary])
async def list_uploads(
    status: str | None = None,
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> list[UploadSummary]:
    structured_count_sq = (
        select(
            SubmissionRecord.submission_id.label("submission_id"),
            func.count(SubmissionRecord.id).label("structured_count"),
        )
        .group_by(SubmissionRecord.submission_id)
        .subquery()
    )
    stmt = (
        select(
            Submission,
            func.coalesce(structured_count_sq.c.structured_count, 0).label("row_count"),
        )
        .join(User, User.id == Submission.user_id)
        .join(Review, Review.submission_id == Submission.id, isouter=True)
        .join(structured_count_sq, structured_count_sq.c.submission_id == Submission.id, isouter=True)
        .options(selectinload(Submission.user), selectinload(Submission.review), selectinload(Submission.data_profiles))
        .group_by(
            Submission.id,
            Review.reviewed_at,
            structured_count_sq.c.structured_count,
        )
        .order_by(desc(Submission.uploaded_at))
        .limit(100)
    )
    if status:
        normalized_status = _submission_status_text(status)
        if normalized_status == "awaiting_agent":
            stmt = stmt.where(Submission.status == SubmissionStatus.quarantined.value)
        else:
            stmt = stmt.where(Submission.status == normalized_status)
    if date_from:
        stmt = stmt.where(Submission.uploaded_at >= date_from)
    if date_to:
        stmt = stmt.where(Submission.uploaded_at <= date_to)
    if user.role == UserRole.employee:
        stmt = stmt.where(Submission.user_id == user.id)
    elif user.role == UserRole.manager:
        stmt = stmt.where(User.manager_id == user.id)

    submissions = (await db.execute(stmt)).all()
    return [
        UploadSummary(
            id=submission.id,
            sub_id=submission.sub_id,
            filename=submission.file_name,
            instruction=submission.instruction,
            output_format=submission.output_format,
            status=_submission_status_text(submission.status),
            version_number=submission.version_number,
            parent_submission_id=submission.parent_submission_id,
            total_rows=max(
                int(row_count or 0),
                int(_latest_submission_profile_json(submission).get("row_count", 0) or 0),
                get_result_record_count(submission),
            ),
            total_columns=0,
            uploader_name=submission.user.full_name if submission.user else None,
            validation_passed=_submission_status_text(submission.status) not in {
                SubmissionStatus.failed.value,
                SubmissionStatus.callback_failed.value,
                SubmissionStatus.quarantined.value,
                SubmissionStatus.declined.value,
            },
            created_at=submission.uploaded_at,
            reviewed_at=submission.completed_at or (submission.review.reviewed_at if submission.review else None),
            **extract_agent_resolution(submission),
        )
        for submission, row_count in submissions
    ]


@router.get("/{upload_id}", response_model=UploadPreview)
async def get_upload(
    upload_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> UploadPreview:
    submission = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.user), selectinload(Submission.review))
            .where(Submission.id == upload_id)
        )
    ).scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    verify_upload_access(submission, user)

    structured_rows = (
        await db.execute(
            select(SubmissionRecord)
            .where(SubmissionRecord.submission_id == upload_id)
            .order_by(SubmissionRecord.record_index)
            .limit(get_settings().max_preview_rows)
        )
    ).scalars().all()
    structured_row_count = await db.scalar(
        select(func.count()).select_from(SubmissionRecord).where(SubmissionRecord.submission_id == upload_id)
    )
    status_text = _submission_status_text(submission.status)
    validation = {"valid": True, "status": SubmissionStatus.succeeded.value}
    if status_text == SubmissionStatus.running.value:
        validation = {"valid": None, "status": SubmissionStatus.running.value}
    elif status_text in FAILED_SUBMISSION_STATUSES:
        validation = {"valid": False, "status": SubmissionStatus.failed.value}
    elif is_schema_approval_pending(submission):
        validation = {"valid": None, "status": SCHEMA_APPROVAL_AGENT_STATUS}

    data_profile_payload = await _load_submission_data_profile(db, submission)
    canonical_intent_payload = _current_canonical_intent_payload(submission) or {}
    detected_types = (
        data_profile_payload.get("detected_types")
        if isinstance(data_profile_payload.get("detected_types"), dict)
        else {}
    )
    preview_rows = [record.payload for record in structured_rows]
    columns = build_structured_columns(structured_rows)
    if not columns and isinstance(data_profile_payload.get("source_columns"), list):
        columns = [str(column) for column in data_profile_payload.get("source_columns", [])]
    if not preview_rows and isinstance(data_profile_payload.get("preview_rows"), list):
        preview_rows = [row for row in data_profile_payload.get("preview_rows", []) if isinstance(row, dict)]
    total_rows = max(
        int(structured_row_count or 0),
        int(data_profile_payload.get("row_count", 0) or 0),
        get_result_record_count(submission),
    )

    return UploadPreview(
        upload_id=submission.id,
        sub_id=submission.sub_id,
        filename=submission.file_name,
        instruction=submission.instruction,
        output_format=submission.output_format,
        status=status_text,
        version_number=submission.version_number,
        parent_submission_id=submission.parent_submission_id,
        total_rows=total_rows,
        total_columns=len(columns),
        created_at=submission.uploaded_at,
        reviewed_at=submission.completed_at or (submission.review.reviewed_at if submission.review else None),
        columns=columns,
        detected_types=detected_types,
        validation=validation,
        preview_rows=preview_rows,
        version_history=await get_version_history(db, submission),
        preferred_agent_name=submission.preferred_agent_name,
        data_profile=data_profile_payload,
        profile_status=_profile_status(submission, data_profile_payload),
        canonical_intent=canonical_intent_payload,
        intent_status=str(canonical_intent_payload.get("resolution_status", "missing")) if canonical_intent_payload else "missing",
        clarification=_compose_clarification_view(submission, canonical_intent_payload),
        execution=_compose_execution_view(submission),
        repair_available=not data_profile_payload and bool(submission.file_path),
    )


@router.get("/{upload_id}/job-detail", response_model=JobDetailRead)
async def get_job_detail(
    upload_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> JobDetailRead:
    submission = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.user), selectinload(Submission.review))
            .where(Submission.id == upload_id)
        )
    ).scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    verify_upload_access(submission, user)

    logs = (
        await db.execute(
            select(AuditLog)
            .where(AuditLog.target_id == str(submission.id))
            .order_by(AuditLog.created_at)
        )
    ).scalars().all()

    structured_rows = (
        await db.execute(
            select(SubmissionRecord)
            .where(SubmissionRecord.submission_id == upload_id)
            .order_by(SubmissionRecord.record_index)
            .limit(get_settings().max_preview_rows)
        )
    ).scalars().all()
    status_text = _submission_status_text(submission.status)
    data_profile_payload = await _load_submission_data_profile(db, submission)
    canonical_intent_payload = _current_canonical_intent_payload(submission) or {}
    detected_types = (
        data_profile_payload.get("detected_types")
        if isinstance(data_profile_payload.get("detected_types"), dict)
        else {}
    )
    preview_rows = [record.payload for record in structured_rows]
    columns = build_structured_columns(structured_rows)
    if not columns and isinstance(data_profile_payload.get("source_columns"), list):
        columns = [str(column) for column in data_profile_payload.get("source_columns", [])]
    if not preview_rows and isinstance(data_profile_payload.get("preview_rows"), list):
        preview_rows = [row for row in data_profile_payload.get("preview_rows", []) if isinstance(row, dict)]

    validation = {"valid": True, "status": SubmissionStatus.succeeded.value}
    if status_text == SubmissionStatus.running.value:
        validation = {"valid": None, "status": SubmissionStatus.running.value}
    elif status_text in FAILED_SUBMISSION_STATUSES:
        validation = {"valid": False, "status": SubmissionStatus.failed.value}
    elif is_schema_approval_pending(submission):
        validation = {"valid": None, "status": SCHEMA_APPROVAL_AGENT_STATUS}

    return JobDetailRead(
        id=submission.id,
        sub_id=submission.sub_id,
        title=build_job_title(submission),
        instruction=submission.instruction,
        file_name=submission.file_name,
        output_format=submission.output_format,
        status=status_text,
        submitted_by=submission.user.full_name if submission.user else None,
        submitted_at=submission.uploaded_at,
        completed_at=submission.completed_at or (submission.review.reviewed_at if submission.review else None),
        **extract_agent_resolution(submission),
        columns=columns,
        detected_types=detected_types,
        validation=validation,
        preview_rows=preview_rows,
        data_profile=data_profile_payload,
        profile_status=_profile_status(submission, data_profile_payload),
        canonical_intent=canonical_intent_payload,
        intent_status=str(canonical_intent_payload.get("resolution_status", "missing")) if canonical_intent_payload else "missing",
        clarification=_compose_clarification_view(submission, canonical_intent_payload),
        execution=_compose_execution_view(submission),
        repair_available=not data_profile_payload and bool(submission.file_path),
        steps=build_job_steps(submission),
        audit=build_job_audit(submission, logs),
    )


@router.post("/{upload_id}/retry", response_model=UploadPreview)
async def retry_upload(
    upload_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> UploadPreview:
    submission = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.user), selectinload(Submission.review))
            .where(Submission.id == upload_id)
        )
    ).scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    verify_upload_access(submission, user)
    if _submission_status_text(submission.status) not in FAILED_SUBMISSION_STATUSES | {SubmissionStatus.succeeded.value} and not is_quarantined_submission(submission):
        payload = submission.summary if isinstance(submission.summary, dict) else {}
        payload_status = str(payload.get("status", "")).strip().lower() or "none"
        raise HTTPException(
            status_code=409,
            detail=(
                "This workflow cannot be retried right now "
                f"(submission_status={_submission_status_text(submission.status)!r}, "
                f"payload_status={payload_status!r})."
            ),
        )

    try:
        await requeue_submission(db, submission=submission, actor=user)
    except Exception as exc:
        submission.status = SubmissionStatus.failed
        submission.summary = {'error': f"Unable to requeue workflow: {exc}"} if not isinstance(submission.summary, dict) else {**submission.summary, 'error': f"Unable to requeue workflow: {exc}"}
        submission.completed_at = datetime.utcnow()
        await db.commit()
        raise HTTPException(
            status_code=502,
            detail=_summary_text(
                submission.summary.get("error") if isinstance(submission.summary, dict) else None,
                default=f"Unable to requeue workflow: {exc}",
            ),
        ) from exc
    refreshed = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.review))
            .where(Submission.id == upload_id)
        )
    ).scalar_one()
    structured_record_count = await db.scalar(
        select(func.count()).select_from(SubmissionRecord).where(SubmissionRecord.submission_id == upload_id)
    )

    return UploadPreview(
        upload_id=refreshed.id,
        sub_id=refreshed.sub_id,
        filename=refreshed.file_name,
        instruction=refreshed.instruction,
        output_format=refreshed.output_format,
        status=_submission_status_text(refreshed.status),
        version_number=refreshed.version_number,
        parent_submission_id=refreshed.parent_submission_id,
        total_rows=structured_record_count or 0,
        total_columns=0,
        created_at=refreshed.uploaded_at,
        columns=[],
        detected_types={},
        reviewed_at=refreshed.completed_at,
        validation={"valid": None, "status": _submission_status_text(refreshed.status)},
        preview_rows=[],
        version_history=await get_version_history(db, refreshed),
        preferred_agent_name=refreshed.preferred_agent_name,
    )


async def get_version_history(db: AsyncSession, submission: Submission) -> list[UploadVersionRead]:
    root_submission_id = submission.parent_submission_id or submission.id
    versions = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.review))
            .where(
                (Submission.id == root_submission_id) | (Submission.parent_submission_id == root_submission_id)
            )
            .order_by(Submission.version_number)
        )
    ).scalars().all()
    return [
        UploadVersionRead(
            id=version.id,
            filename=version.file_name,
            status=_submission_status_text(version.status),
            version_number=version.version_number,
            created_at=version.uploaded_at,
            reviewed_at=version.completed_at or (version.review.reviewed_at if version.review else None),
        )
        for version in versions
    ]


def verify_upload_access(submission: Submission, user: User) -> None:
    if user.role == UserRole.admin:
        return
    if user.role == UserRole.employee and submission.user_id == user.id:
        return
    if user.role == UserRole.manager and submission.user and submission.user.manager_id == user.id:
        return
    raise HTTPException(status_code=404, detail="Submission not found")


def build_job_title(submission: Submission) -> str:
    instruction = (submission.instruction or "").strip()
    if instruction:
        return instruction[:72] + ("..." if len(instruction) > 72 else "")
    return submission.file_name


def format_display_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    settings = get_settings()
    normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return normalized.astimezone(settings.display_tzinfo).strftime("%H:%M")


def build_agent_execution_summary(submission: Submission) -> str:
    result_payload = submission.summary if isinstance(submission.summary, dict) else {}
    status_text = _submission_status_text(submission.status)
    selected_agent = result_payload.get("selected_agent")
    completed_agents = result_payload.get("completed_agents")
    payload_status = str(result_payload.get("status", "")).strip().lower()
    rejection_reason = str(result_payload.get("reason", "")).strip()
    suggestion = str(result_payload.get("suggestion", "")).strip()

    if payload_status in {"pending_agent_availability", "rejected"}:
        if rejection_reason and suggestion:
            return f"{rejection_reason}. {suggestion}"
        if rejection_reason:
            return rejection_reason
        if suggestion:
            return suggestion
        return "Part of this workflow is quarantined until supported agent coverage is available."

    if isinstance(completed_agents, list) and completed_agents:
        agents = ", ".join(str(agent).strip() for agent in completed_agents if str(agent).strip())
        if agents:
            return f"Completed by {agents}."
    if is_schema_approval_pending(submission):
        return "Waiting for clarification or explicit confirmation before agent execution starts."
    if status_text in {SubmissionStatus.awaiting_confirmation.value, SubmissionStatus.awaiting_clarification.value}:
        return "Waiting for clarification or explicit confirmation before dispatch."
    if isinstance(selected_agent, str) and selected_agent.strip():
        if status_text == SubmissionStatus.running.value:
            return f"{selected_agent.strip()} is currently processing this workflow."
        return f"Processed by {selected_agent.strip()}."
    error_text = _summary_text(submission.summary.get("error") if isinstance(submission.summary, dict) else None)
    if error_text:
        return error_text
    if status_text == SubmissionStatus.running.value:
        return "The workflow is being interpreted and processed by the selected agents."
    if status_text in FAILED_SUBMISSION_STATUSES:
        return "The workflow could not produce a valid result."
    if status_text == SubmissionStatus.succeeded.value:
        return "The requested workflow completed successfully."
    return "The workflow instructions were accepted successfully."


def build_job_steps(submission: Submission) -> list[JobStepRead]:
    uploaded_time = format_display_time(submission.uploaded_at)
    dispatched_time = format_display_time(submission.dispatched_at)
    reviewed_at = submission.completed_at or (submission.review.reviewed_at if submission.review else None)
    reviewed_time = format_display_time(reviewed_at)
    result_payload = submission.summary if isinstance(submission.summary, dict) else {}
    payload_status = str(result_payload.get("status", "")).strip().lower()
    status_text = _submission_status_text(submission.status)
    quarantined_workflow = status_text == SubmissionStatus.quarantined.value and payload_status in {
        "pending_agent_availability",
        "rejected",
    }
    schema_review_pending = is_schema_approval_pending(submission)
    interpretation_review_pending = status_text in {
        SubmissionStatus.awaiting_confirmation.value,
        SubmissionStatus.awaiting_clarification.value,
    }

    queue_status = "complete" if submission.dispatched_at else "running"
    queue_summary = (
        f"Dispatched to agent task {submission.agent_task_id}."
        if submission.dispatched_at and submission.agent_task_id
        else "Preparing the workflow for execution."
    )

    execution_status = "running"
    execution_time = None
    if status_text == SubmissionStatus.running.value and submission.dispatched_at:
        execution_status = "running"
        execution_time = dispatched_time
    elif status_text == SubmissionStatus.succeeded.value:
        execution_status = "complete"
        execution_time = reviewed_time
    elif status_text in FAILED_SUBMISSION_STATUSES:
        execution_status = "failed"
        execution_time = reviewed_time

    output_status = "running"
    output_summary = "Final output will be prepared after agent execution completes."
    output_time = None
    if status_text == SubmissionStatus.succeeded.value:
        output_status = "complete"
        output_summary = (
            "The workflow finished successfully and output is ready."
            if submission.output_path
            else "The workflow finished successfully."
        )
        output_time = reviewed_time
    elif status_text in FAILED_SUBMISSION_STATUSES:
        output_status = "blocked"
        output_summary = "No output could be generated because execution failed."
        output_time = reviewed_time

    if status_text == SubmissionStatus.queued.value:
        queue_status = "running"
        execution_status = "running"
    if schema_review_pending:
        queue_status = "complete"
        queue_summary = "Canonical interpretation prepared and waiting for user confirmation."
        execution_status = "blocked"
        execution_time = None
        output_status = "blocked"
        output_time = None
        output_summary = "Processing will begin after the interpretation is confirmed or clarified."
    if interpretation_review_pending:
        queue_status = "complete"
        queue_summary = "Canonical interpretation prepared and waiting for clarification or confirmation."
        execution_status = "blocked"
        execution_time = None
        output_status = "blocked"
        output_time = None
        output_summary = "Processing will begin after the interpretation is confirmed or clarified."
    if quarantined_workflow:
        queue_status = "complete" if submission.dispatched_at else "running"
        queue_summary = (
            f"Submitted to task {submission.agent_task_id} for quarantine review."
            if submission.agent_task_id
            else "Submitted for quarantine review."
        )
        execution_status = "blocked"
        execution_time = reviewed_time or dispatched_time
        output_status = "blocked"
        output_time = reviewed_time or dispatched_time
        output_summary = (
            result_payload.get("suggestion")
            or "Output is paused because part of the workflow is quarantined."
        )

    return [
        JobStepRead(name="Ingestion", status="complete", summary="File upload accepted and staged.", time=uploaded_time),
        JobStepRead(name="Workflow routing", status=queue_status, summary=queue_summary, time=dispatched_time or (reviewed_time if queue_status == "complete" else None)),
        JobStepRead(
            name="Agent execution" if not schema_review_pending and not interpretation_review_pending else "Interpretation review",
            status=execution_status,
            summary=(
                build_agent_execution_summary(submission)
                if not schema_review_pending and not interpretation_review_pending
                else "Review the canonical interpretation and confirm or clarify it to continue."
            ),
            time=execution_time,
        ),
        JobStepRead(name="Output preparation", status=output_status, summary=output_summary, time=output_time),
    ]


def build_job_audit(submission: Submission, logs: list[AuditLog]) -> list[JobAuditEntryRead]:
    entries: list[tuple[datetime | None, str, str]] = [
        (
            submission.uploaded_at,
            "upload created",
            submission.file_name,
        ),
    ]

    if submission.dispatched_at:
        dispatch_detail = (
            f"Workflow dispatched to task {submission.agent_task_id}."
            if submission.agent_task_id
            else "Workflow dispatched to the agent queue."
        )
        entries.append((submission.dispatched_at, "agent dispatched", dispatch_detail))

    if submission.completed_at:
        result_payload = submission.summary if isinstance(submission.summary, dict) else {}
        status_text = _submission_status_text(submission.status)
        if status_text == SubmissionStatus.succeeded.value:
            detail = build_agent_execution_summary(submission)
            if submission.output_path:
                detail = f"{detail} Output file is ready."
            entries.append((submission.completed_at, "workflow completed", detail))
        elif status_text in FAILED_SUBMISSION_STATUSES:
            detail = _summary_text(
                submission.summary.get("error") if isinstance(submission.summary, dict) else None,
                result_payload.get("error"),
                default="Agent execution failed.",
            )
            entries.append((submission.completed_at, "workflow failed", detail))
        elif status_text in {
            SubmissionStatus.awaiting_confirmation.value,
            SubmissionStatus.awaiting_clarification.value,
        }:
            detail = "Canonical interpretation is awaiting clarification or explicit confirmation."
            entries.append((submission.uploaded_at, "interpretation review requested", detail))
        elif status_text == SubmissionStatus.quarantined.value:
            if is_schema_approval_pending(submission):
                detail = "Canonical interpretation is awaiting user review."
                entries.append((submission.uploaded_at, "interpretation review requested", detail))
            else:
                detail = _summary_text(
                    submission.summary.get("error") if isinstance(submission.summary, dict) else None,
                    result_payload.get("suggestion"),
                    default="Part of the workflow is quarantined pending review.",
                )
                entries.append((submission.completed_at, "workflow quarantined", detail))

    for log in logs:
        detail = log.detail or log.target_label or submission.file_name
        normalized_action = log.action.value.replace("_", " ")
        if normalized_action == "upload created" and detail in {"workflow_created", submission.file_name}:
            continue
        if normalized_action in {"upload approved", "upload declined"} and submission.completed_at:
            continue
        entries.append((log.created_at, normalized_action, detail))

    entries.sort(key=lambda item: item[0] or datetime.min.replace(tzinfo=UTC))
    return [
        JobAuditEntryRead(
            time=format_display_time(occurred_at) or "--:--",
            action=action,
            detail=detail,
        )
        for occurred_at, action, detail in entries
    ]

def build_structured_columns(rows: list[SubmissionRecord]) -> list[str]:
    if not rows:
        return []
    first_payload = sanitize_structured_row(rows[0].payload if isinstance(rows[0].payload, dict) else {})
    return [str(key) for key in first_payload.keys()]


@router.get("/{upload_id}/download")
async def download_output(
    upload_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee, UserRole.manager, UserRole.admin)),
) -> FileResponse:
    submission = (
        await db.execute(
            select(Submission)
            .options(selectinload(Submission.user))
            .where(Submission.id == upload_id)
        )
    ).scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    verify_upload_access(submission, user)
    if _submission_status_text(submission.status) != SubmissionStatus.succeeded.value:
        raise HTTPException(status_code=404, detail="Output not ready")

    output_path = await ensure_output_file(db, submission)
    if output_path is None or not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    suffix = output_path.suffix or f".{submission.output_format.lower()}"
    filename = f"{Path(submission.original_filename).stem}_processed{suffix}"
    return FileResponse(path=output_path, filename=filename, media_type="application/octet-stream")
