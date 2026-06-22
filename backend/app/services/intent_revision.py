from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import CanonicalIntentRevision, IntentDispatchOutbox, Submission
from app.services.json_safety import make_json_safe

CURRENT_INTENT_SCHEMA_VERSION = "2.0"
CURRENT_ENVELOPE_VERSION = "1.0"
CAPABILITY_VERSION = "backend.capability.1"

SUPPORTED_CANONICAL_OUTPUT_FORMATS = {"xlsx", "csv", "json", "txt"}
SUPPORTED_CANONICAL_ACTIONS = {
    "clean",
    "project_columns",
    "drop_columns",
    "rename_columns",
    "filter_rows",
    "sort_rows",
    "limit_rows",
    "calculate",
    "visualize",
    "report",
}
SUPPORTED_CANONICAL_OPERATORS = {
    "eq",
    "neq",
    "gt",
    "lt",
    "gte",
    "lte",
    "contains",
    "in",
    "not_in",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def _strip_runtime_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    for key in (
        "intent_id",
        "intent_revision",
        "intent_hash",
        "parent_intent_id",
        "created_at",
        "grounded_at",
        "execution_plan_id",
        "execution_plan_hash",
    ):
        sanitized.pop(key, None)
    return sanitized


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def compute_intent_hash(intent_payload: dict[str, Any]) -> str:
    sanitized = _strip_runtime_metadata(intent_payload)
    digest = hashlib.sha256(stable_json_dumps(sanitized).encode("utf-8")).hexdigest()
    return digest


def compute_plan_hash(plan_payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(stable_json_dumps(plan_payload).encode("utf-8")).hexdigest()
    return digest


def build_capability_snapshot() -> dict[str, Any]:
    settings = get_settings()
    return {
        "capability_version": CAPABILITY_VERSION,
        "available_action_kinds": sorted(SUPPORTED_CANONICAL_ACTIONS),
        "available_operators": sorted(SUPPORTED_CANONICAL_OPERATORS),
        "available_output_formats": sorted(SUPPORTED_CANONICAL_OUTPUT_FORMATS),
        "registered_agent_versions": {
            "cleaning": getattr(settings, "agent_name", "finflow_cleaning_agent"),
            "filtering": getattr(settings, "agent_name", "finflow_filter_agent"),
            "calculation": getattr(settings, "agent_name", "finflow_calculation_agent"),
            "reporting": getattr(settings, "agent_name", "finflow_reporting_agent"),
            "visualization": getattr(settings, "agent_name", "finflow_visualization_agent"),
        },
        "operation_schema_versions": {
            "cleaning": "1.0",
            "filtering": "1.0",
            "calculation": "1.0",
            "visualization": "1.0",
            "reporting": "1.0",
        },
    }


def upcast_canonical_intent(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None

    version = str(payload.get("schema_version") or CURRENT_INTENT_SCHEMA_VERSION).strip()
    if version == CURRENT_INTENT_SCHEMA_VERSION:
        return payload

    if version != "1.0":
        raise ValueError(f"Unsupported canonical intent schema version: {version}")

    # v1 -> v2: keep the action payload intact, add metadata defaults, and
    # normalize the name of the schema version.
    upgraded = dict(payload)
    upgraded["schema_version"] = CURRENT_INTENT_SCHEMA_VERSION
    upgraded.setdefault("capability_version", CAPABILITY_VERSION)
    upgraded.setdefault("repair_notes", [])
    upgraded.setdefault("assumptions", [])
    upgraded.setdefault("dataframe_profile", {})
    return upgraded


def _next_revision(submission: Submission) -> int:
    current = submission.intent_revision or 0
    return current + 1


def build_revision_payload(
    *,
    canonical_intent: dict[str, Any],
    original_instruction: str,
    parent_intent_id: uuid.UUID | None,
    revision: int,
    capability_snapshot: dict[str, Any] | None = None,
    intent_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    capability_snapshot = capability_snapshot or build_capability_snapshot()
    intent_id = intent_id or uuid.uuid4()
    safe_canonical_intent = make_json_safe(canonical_intent)
    intent_hash = compute_intent_hash(safe_canonical_intent)
    grounded_at = safe_canonical_intent.get("grounded_at") or utc_now().isoformat()
    payload = {
        "schema_version": CURRENT_ENVELOPE_VERSION,
        "intent_id": str(intent_id),
        "intent_revision": revision,
        "intent_hash": intent_hash,
        "parent_intent_id": str(parent_intent_id) if parent_intent_id else None,
        "intent": safe_canonical_intent,
        "original_instruction": original_instruction,
        "extractor_version": safe_canonical_intent.get("extractor_version"),
        "normalizer_version": safe_canonical_intent.get("normalizer_version"),
        "grounding_version": safe_canonical_intent.get("grounding_version"),
        "repair_notes": list(safe_canonical_intent.get("repair_notes", [])) if isinstance(safe_canonical_intent.get("repair_notes"), list) else [],
        "assumptions": list(safe_canonical_intent.get("assumptions", [])) if isinstance(safe_canonical_intent.get("assumptions"), list) else [],
        "capability_version": capability_snapshot.get("capability_version", CAPABILITY_VERSION),
        "capability_snapshot": capability_snapshot,
        "created_at": utc_now().isoformat(),
        "grounded_at": grounded_at,
    }
    return payload


async def persist_intent_revision(
    db: AsyncSession,
    *,
    submission: Submission,
    canonical_intent: dict[str, Any],
    original_instruction: str,
    parent_intent_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    revision = _next_revision(submission)
    intent_id = submission.intent_id or uuid.uuid4()
    capability_snapshot = build_capability_snapshot()
    safe_canonical_intent = make_json_safe(canonical_intent)
    revision_payload = build_revision_payload(
        canonical_intent=safe_canonical_intent,
        original_instruction=original_instruction,
        parent_intent_id=parent_intent_id,
        revision=revision,
        capability_snapshot=capability_snapshot,
        intent_id=intent_id,
    )

    intent_record = CanonicalIntentRevision(
        submission_id=submission.id,
        intent_id=uuid.UUID(str(revision_payload["intent_id"])),
        intent_revision=revision,
        intent_hash=revision_payload["intent_hash"],
        parent_intent_id=parent_intent_id,
        canonical_intent=safe_canonical_intent,
        original_instruction=original_instruction,
        grounded_at=utc_now(),
        capability_version=revision_payload["capability_version"],
        extractor_version=safe_canonical_intent.get("extractor_version"),
        normalizer_version=safe_canonical_intent.get("normalizer_version"),
        grounding_version=safe_canonical_intent.get("grounding_version"),
    )
    db.add(intent_record)

    submission.canonical_intent = safe_canonical_intent
    submission.canonical_intent_schema_version = safe_canonical_intent.get("schema_version", CURRENT_INTENT_SCHEMA_VERSION)
    submission.intent_status = safe_canonical_intent.get("resolution_status", "resolved")
    submission.intent_id = intent_record.intent_id
    submission.intent_revision = revision
    submission.intent_hash = intent_record.intent_hash
    submission.parent_intent_id = parent_intent_id
    submission.grounded_at = intent_record.grounded_at
    submission.intent_extractor_version = safe_canonical_intent.get("extractor_version")
    submission.intent_normalizer_version = safe_canonical_intent.get("normalizer_version")
    submission.intent_grounding_version = safe_canonical_intent.get("grounding_version")
    submission.intent_created_at = datetime.fromisoformat(revision_payload["created_at"])
    submission.capability_version = revision_payload["capability_version"]

    await db.flush()
    return revision_payload


async def create_dispatch_outbox(
    db: AsyncSession,
    *,
    submission: Submission,
    payload: dict[str, Any],
) -> IntentDispatchOutbox:
    intent_hash = submission.intent_hash or compute_intent_hash(payload.get("canonical_intent") or {})
    existing_stmt = (
        select(IntentDispatchOutbox)
        .where(
            IntentDispatchOutbox.submission_id == submission.id,
            IntentDispatchOutbox.intent_hash == intent_hash,
        )
        .limit(1)
    )
    existing = (await db.execute(existing_stmt)).scalars().first()
    if existing is not None:
        existing.intent_id = submission.intent_id or existing.intent_id
        existing.intent_revision = submission.intent_revision or existing.intent_revision
        existing.payload = payload
        existing.status = "pending"
        existing.last_error = None
        existing.delivered_at = None
        await db.flush()
        return existing

    outbox = IntentDispatchOutbox(
        submission_id=submission.id,
        intent_id=submission.intent_id or uuid.uuid4(),
        intent_revision=submission.intent_revision or 1,
        intent_hash=intent_hash,
        payload=payload,
        status="pending",
    )
    db.add(outbox)
    await db.flush()
    return outbox


async def mark_outbox_delivered(db: AsyncSession, outbox_id: uuid.UUID) -> None:
    outbox = await db.get(IntentDispatchOutbox, outbox_id)
    if outbox is None:
        return
    outbox.status = "delivered"
    outbox.delivered_at = utc_now()
    outbox.last_error = None
    await db.flush()


async def mark_outbox_failed(db: AsyncSession, outbox_id: uuid.UUID, error: str) -> None:
    outbox = await db.get(IntentDispatchOutbox, outbox_id)
    if outbox is None:
        return
    outbox.status = "pending"
    outbox.last_error = error
    await db.flush()


async def latest_intent_revision_for_submission(db: AsyncSession, submission_id: uuid.UUID) -> CanonicalIntentRevision | None:
    stmt = (
        select(CanonicalIntentRevision)
        .where(CanonicalIntentRevision.submission_id == submission_id)
        .order_by(CanonicalIntentRevision.intent_revision.desc(), CanonicalIntentRevision.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalars().first()
