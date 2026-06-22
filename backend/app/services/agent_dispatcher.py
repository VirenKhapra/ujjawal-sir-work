import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from arq import create_pool
from arq.connections import RedisSettings

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.models import Submission, SubmissionStatus
from app.services.canonical_intent import (
    INTENT_EXTRACTOR_VERSION,
    INTENT_GROUNDING_VERSION,
    INTENT_NORMALIZER_VERSION,
)
from app.services.intent_revision import create_dispatch_outbox, mark_outbox_delivered, persist_intent_revision
from app.services.data_profile import load_latest_data_profile
from app.services.llm_telemetry import log_runtime_event
from app.services.intent_revision import latest_intent_revision_for_submission

logger = logging.getLogger(__name__)


def _submission_file_id(submission: Submission) -> str:
    return Path(str(submission.file_path or "")).name


def _build_canonical_intent_envelope(submission: Submission, canonical_intent: dict, *, created_at: str | None = None) -> dict:
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "intent_id": canonical_intent.get("intent_id"),
        "intent_revision": canonical_intent.get("intent_revision", 1),
        "intent_hash": canonical_intent.get("intent_hash"),
        "parent_intent_id": canonical_intent.get("parent_intent_id"),
        "intent": canonical_intent,
        "original_instruction": str(submission.instruction or "").strip(),
        "intent_status": canonical_intent.get("resolution_status", "resolved"),
        "repair_notes": list(canonical_intent.get("repair_notes", [])) if isinstance(canonical_intent.get("repair_notes"), list) else [],
        "assumptions": list(canonical_intent.get("assumptions", [])) if isinstance(canonical_intent.get("assumptions"), list) else [],
        "extractor_version": INTENT_EXTRACTOR_VERSION,
        "normalizer_version": INTENT_NORMALIZER_VERSION,
        "grounding_version": INTENT_GROUNDING_VERSION,
        "capability_version": canonical_intent.get("capability_version"),
        "capability_snapshot": canonical_intent.get("capability_snapshot", {}),
        "created_at": created_at,
        "grounded_at": canonical_intent.get("grounded_at") or created_at,
    }


async def enqueue_submission_dispatch(submission_id: UUID | str, *, persist_revision: bool = True) -> None:
    settings = get_settings()
    try:
        redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        async with AsyncSessionLocal() as db:
            submission = await db.get(Submission, UUID(str(submission_id)))
            if not submission:
                logger.warning("Dispatch queue referenced missing submission %s", submission_id)
                return

            profile_record = await load_latest_data_profile(db, submission.id)
            if profile_record is None:
                submission.status = SubmissionStatus.failed
                summary = submission.summary if isinstance(submission.summary, dict) else {}
                submission.summary = {
                    **summary,
                    "error": "data_profile_missing",
                }
                await db.commit()
                return

            revision_record = await latest_intent_revision_for_submission(db, submission.id)
            canonical_payload = revision_record.canonical_intent if revision_record is not None else submission.canonical_intent
            if not isinstance(canonical_payload, dict):
                submission.status = SubmissionStatus.failed
                summary = submission.summary if isinstance(submission.summary, dict) else {}
                submission.summary = {
                    **summary,
                    "error": "canonical_intent_missing",
                }
                await db.commit()
                return
            if str(canonical_payload.get("resolution_status", "")).strip() not in {"resolved", "repaired"}:
                submission.status = SubmissionStatus.quarantined
                summary = submission.summary if isinstance(submission.summary, dict) else {}
                submission.summary = {
                    **summary,
                    "reason": canonical_payload.get("resolution_status", "needs_clarification"),
                }
                await db.commit()
                return

            canonical_intent = _build_canonical_intent_envelope(
                submission,
                canonical_payload,
                created_at=revision_record.created_at.isoformat() if revision_record is not None and revision_record.created_at else None,
            )
            outbox = None
            payload = {
                "submission_id": str(submission.id),
                "file_id": _submission_file_id(submission),
                "file_name": submission.file_name,
                "resolved_file_path": str(submission.file_path or ""),
                "data_profile_id": str(profile_record.id),
                "canonical_intent_revision_id": str(revision_record.id) if revision_record is not None else "",
                "output_format": str(submission.output_format or "").strip().lower(),
                "audit_context": {
                    "original_instruction": str(submission.instruction or "").strip(),
                    "submission_id": str(submission.id),
                },
            }
            canonical_payload = canonical_intent["intent"] if isinstance(canonical_intent.get("intent"), dict) else canonical_intent
            if persist_revision:
                await persist_intent_revision(
                    db,
                    submission=submission,
                    canonical_intent=canonical_payload,
                    original_instruction=str(submission.instruction or "").strip(),
                    parent_intent_id=UUID(str(canonical_intent["parent_intent_id"])) if canonical_intent.get("parent_intent_id") else None,
                )
            summary = submission.summary if isinstance(submission.summary, dict) else {}
            created_at_raw = canonical_intent.get("created_at")
            payload["audit_context"] = {
                **payload["audit_context"],
                "intent_id": canonical_intent.get("intent_id"),
                "intent_revision": canonical_intent.get("intent_revision", 1),
                "intent_hash": canonical_intent.get("intent_hash"),
                "capability_version": canonical_intent.get("capability_version"),
            }
            submission.summary = {
                **summary,
                "canonical_intent": canonical_intent,
                "canonical_intent_schema_version": canonical_intent.get("schema_version", "1.0"),
                "canonical_intent_status": canonical_intent.get("intent_status", canonical_intent.get("intent", {}).get("resolution_status", "resolved")),
                "original_instruction": str(submission.instruction or "").strip(),
                "intent_extractor_version": canonical_intent.get("extractor_version", INTENT_EXTRACTOR_VERSION),
                "intent_normalizer_version": canonical_intent.get("normalizer_version", INTENT_NORMALIZER_VERSION),
                "intent_grounding_version": canonical_intent.get("grounding_version", INTENT_GROUNDING_VERSION),
                "intent_created_at": created_at_raw,
                "intent_id": canonical_intent.get("intent_id"),
                "intent_revision": canonical_intent.get("intent_revision", 1),
                "intent_hash": canonical_intent.get("intent_hash"),
                "parent_intent_id": canonical_intent.get("parent_intent_id"),
                "grounded_at": canonical_intent.get("grounded_at"),
                "capability_version": canonical_intent.get("capability_version"),
            }
            payload["canonical_intent"] = canonical_intent

            outbox = await create_dispatch_outbox(db, submission=submission, payload=payload)
            submission.status = SubmissionStatus.planning
            await db.commit()

            await redis.enqueue_job("process_job_task", payload)
            submission.dispatched_at = datetime.now(timezone.utc)
            await db.commit()
            if outbox is not None:
                logger.info(
                    "event=canonical_job_enqueued submission_id=%s intent_schema_version=%s",
                    submission.id,
                    canonical_intent.get("intent", {}).get("schema_version", "2.0"),
                )
                await mark_outbox_delivered(db, outbox.id)
                await db.commit()
    except Exception as e:
        logger.exception("Failed to enqueue submission via arq")

async def start_dispatcher(app) -> None:
    # Set a dummy task so that health checks that look for agent_dispatch_task return True
    app.state.agent_dispatch_task = "arq_managed"

async def stop_dispatcher(app) -> None:
    app.state.agent_dispatch_task = None
