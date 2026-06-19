import logging
import json
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, Request, Response, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.uploads import save_upload
from app.core.config import get_settings
from app.core.security import create_access_token, find_user_by_email, get_optional_user, require_roles, verify_password
from app.db.session import get_db
from app.models import AuditAction, CallbackEvent, DeadLetterJob, NeedsReviewJob, RegisteredAgent, ReviewStatus, Submission, SubmissionStatus, User, UserRole
from app.schemas import AgentRead, AgentRegisterRequest, AgentTokenResponse, LoginRequest, UploadPreview
from app.services.alerts import create_quarantine_alert
from app.services.audit import log_action
from app.services.quarantine import requeue_quarantined_submissions
from app.services.request_security import enforce_rate_limit
from app.services.schema_proposal import make_json_safe
from app.services.submission_results import persist_submission_results
from app.services.websocket_manager import ws_manager

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _clean_error_text(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "null", "false"}:
        return ""
    return text


def resolve_agent_artifact_path(result_payload: dict | None) -> Path | None:
    if not isinstance(result_payload, dict):
        return None

    settings = get_settings()
    output_dir = Path(settings.output_dir)

    output_file_name = str(result_payload.get("output_file_name", "")).strip()
    if output_file_name:
        candidate = output_dir / output_file_name
        if candidate.exists():
            return candidate

    output_relative_path = str(
        result_payload.get("output_relative_path")
        or result_payload.get("output_path")
        or result_payload.get("excel_file_path")
        or result_payload.get("file_path")
        or ""
    ).strip()
    if output_relative_path:
        candidate = output_dir / Path(output_relative_path).name
        if candidate.exists():
            return candidate

    return None


def map_agent_status_to_submission_status(status_value: str) -> SubmissionStatus:
    normalized = _clean_error_text(status_value).lower()
    if normalized in {"complete", "completed", "succeeded"}:
        return SubmissionStatus.succeeded
    if normalized in {"failed", "partial"}:
        return SubmissionStatus.failed
    if normalized == "quarantined":
        return SubmissionStatus.quarantined
    if normalized == "callback_failed":
        return SubmissionStatus.callback_failed
    if normalized == "declined":
        return SubmissionStatus.declined
    if normalized in {status.value for status in SubmissionStatus}:
        return SubmissionStatus(normalized)
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown agent status: {status_value}")


def reconcile_callback_status(status_value: str, result_payload: dict | None, error: str | None) -> str:
    mapped = map_agent_status_to_submission_status(status_value)
    if _clean_error_text(error):
        if mapped == SubmissionStatus.succeeded:
            return SubmissionStatus.failed.value
    if isinstance(result_payload, dict):
        payload_status = str(result_payload.get("status", "")).strip().lower()
        payload_errors = result_payload.get("errors")
        if mapped == SubmissionStatus.succeeded and isinstance(payload_errors, list) and any(_clean_error_text(item) for item in payload_errors):
            return SubmissionStatus.failed.value
        if payload_status in {"failed", "partial"}:
            return SubmissionStatus.failed.value
        if payload_status == "quarantined":
            return SubmissionStatus.quarantined.value
        if payload_status in {"complete", "completed", "succeeded"}:
            return SubmissionStatus.succeeded.value
    return mapped.value


async def _table_exists(db: AsyncSession, table_name: str) -> bool:
    def _check(sync_session) -> bool:
        connection = sync_session.connection()
        return inspect(connection).has_table(table_name)

    return await db.run_sync(_check)


@router.post("/login", response_model=AgentTokenResponse)
async def agent_login(
    payload: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AgentTokenResponse:
    await enforce_rate_limit(request=request, bucket="agent_login", limit=10, window_seconds=60)
    user = await find_user_by_email(db, payload.email)
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    if user.role != UserRole.employee:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Agent access requires an employee account")

    return AgentTokenResponse(access_token=create_access_token(user))


@router.post("/upload", response_model=UploadPreview)
async def agent_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    instruction: str = Form(default=""),
    output_format: str = Form(default="XLSX"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.employee)),
) -> UploadPreview:
    return await save_upload(
        file=file,
        instruction=instruction,
        output_format=output_format,
        db=db,
        user=user,
        background_tasks=background_tasks,
    )


@router.get("/registry", response_model=list[AgentRead])
async def list_registered_agents(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles(UserRole.manager, UserRole.admin)),
) -> list[AgentRead]:
    agents = (
        await db.execute(
            select(RegisteredAgent).order_by(RegisteredAgent.total_invocations.desc(), RegisteredAgent.name.asc())
        )
    ).scalars().all()

    return [
        AgentRead(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            capability_tags=agent.capability_tags or [],
            input_formats=agent.input_formats or [],
            output_formats=agent.output_formats or [],
            endpoint_url=agent.endpoint_url,
            status=agent.status,
            last_heartbeat=agent.last_heartbeat,
            total_invocations=agent.total_invocations,
            registered_at=agent.registered_at,
        )
        for agent in agents
    ]


@router.post("/register", response_model=AgentRead, status_code=status.HTTP_201_CREATED)
async def register_agent(
    payload: AgentRegisterRequest,
    response: Response,
    x_agent_registry_secret: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> AgentRead:
    settings = get_settings()
    if user is not None and user.role not in {UserRole.manager, UserRole.admin}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
    if user is None and x_agent_registry_secret != settings.agent_registry_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent registry secret")

    existing = (
        await db.execute(select(RegisteredAgent).where(RegisteredAgent.name == payload.name.strip()))
    ).scalar_one_or_none()
    if existing:
        existing.description = payload.description.strip()
        existing.capability_tags = [tag.strip() for tag in payload.capability_tags if tag.strip()]
        existing.input_formats = [item.strip().upper() for item in payload.input_formats if item.strip()]
        existing.output_formats = [item.strip().upper() for item in payload.output_formats if item.strip()]
        existing.endpoint_url = payload.endpoint_url.strip() if payload.endpoint_url else None
        existing.status = payload.status.strip().lower() or existing.status or "active"
        existing.last_heartbeat = _utc_now()
        await db.commit()
        await db.refresh(existing)
        requeued_ids = await requeue_quarantined_submissions(db)
        if requeued_ids:
            await ws_manager.broadcast(
                "dashboard",
                "dashboard_refresh",
                {"reprocessed_count": len(requeued_ids), "agent_name": existing.name},
            )
        response.status_code = status.HTTP_200_OK
        return AgentRead(
            id=existing.id,
            name=existing.name,
            description=existing.description,
            capability_tags=existing.capability_tags or [],
            input_formats=existing.input_formats or [],
            output_formats=existing.output_formats or [],
            endpoint_url=existing.endpoint_url,
            status=existing.status,
            last_heartbeat=existing.last_heartbeat,
            total_invocations=existing.total_invocations,
            registered_at=existing.registered_at,
        )

    agent = RegisteredAgent(
        name=payload.name.strip(),
        description=payload.description.strip(),
        capability_tags=[tag.strip() for tag in payload.capability_tags if tag.strip()],
        input_formats=[item.strip().upper() for item in payload.input_formats if item.strip()],
        output_formats=[item.strip().upper() for item in payload.output_formats if item.strip()],
        endpoint_url=payload.endpoint_url.strip() if payload.endpoint_url else None,
        status=payload.status.strip().lower() or "active",

    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)

    requeued_ids = await requeue_quarantined_submissions(db)
    if requeued_ids:
        await ws_manager.broadcast(
            "dashboard",
            "dashboard_refresh",
            {"reprocessed_count": len(requeued_ids), "agent_name": agent.name},
        )

    return AgentRead(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        capability_tags=agent.capability_tags or [],
        input_formats=agent.input_formats or [],
        output_formats=agent.output_formats or [],
        endpoint_url=agent.endpoint_url,
        status=agent.status,
        last_heartbeat=agent.last_heartbeat,
        total_invocations=agent.total_invocations,
        registered_at=agent.registered_at,
    )

class AgentCallbackPayload(BaseModel):
    submission_id: str
    status: str
    output_path: str | None = None
    summary: dict | None = None
    event_id: str | None = None
    job_id: str | None = None


def _callback_payload_hash(payload: AgentCallbackPayload) -> str:
    payload_json = json.dumps(payload.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _derive_callback_event_id(payload: AgentCallbackPayload) -> str:
    if _clean_error_text(payload.event_id):
        return _clean_error_text(payload.event_id)
    stable_payload = {
        "submission_id": payload.submission_id,
        "status": payload.status,
        "output_path": payload.output_path,
        "summary": payload.summary,
    }
    stable_json = json.dumps(stable_payload, sort_keys=True, default=str)
    digest = hashlib.sha256(stable_json.encode("utf-8")).hexdigest()
    return f"callback:{payload.submission_id}:{digest}"


async def _mark_callback_side_effect_failed(
    db: AsyncSession,
    *,
    event_id: str,
    error: str,
) -> None:
    event = (
        await db.execute(select(CallbackEvent).where(CallbackEvent.event_id == event_id))
    ).scalar_one_or_none()
    if event:
        event.processing_status = "side_effect_failed"
        event.last_error = error[:2000]
        await db.commit()


async def _persist_needs_review_side_effect(
    *,
    db: AsyncSession,
    payload: AgentCallbackPayload,
    submission: Submission,
    event_id: str,
    settings,
) -> str:
    if not settings.enable_needs_review_jobs:
        logger.info(
            "Skipping needs_review_jobs insert event_id=%s submission_id=%s feature_enabled=false",
            event_id,
            submission.id,
        )
        return "disabled"

    try:
        if not await _table_exists(db, NeedsReviewJob.__tablename__):
            raise RuntimeError("Database schema missing needs_review_jobs table")
        review_reason = "Domain not supported"
        if isinstance(payload.summary, dict):
            review_reason = (
                _clean_error_text(payload.summary.get("reason"))
                or _clean_error_text(payload.summary.get("error"))
                or review_reason
            )
        db.add(
            NeedsReviewJob(
                submission_id=submission.id,
                source_event_id=event_id,
                reason=review_reason,
            )
        )
        await db.commit()
        return "created"
    except IntegrityError:
        await db.rollback()
        logger.info(
            "Callback side effect duplicate event_id=%s job_id=%s submission_id=%s side_effect_type=needs_review_job side_effect_status=duplicate",
            event_id,
            payload.job_id,
            submission.id,
        )
        return "duplicate"
    except Exception as exc:
        await db.rollback()
        logger.exception(
            "Callback side effect failed event_id=%s submission_id=%s side_effect_type=needs_review_job table=%s exception=%s message=%s",
            event_id,
            submission.id,
            NeedsReviewJob.__tablename__,
            exc.__class__.__name__,
            _clean_error_text(exc),
        )
        await _mark_callback_side_effect_failed(db, event_id=event_id, error=str(exc))
        return "failed"


async def _persist_dead_letter_side_effect(
    *,
    db: AsyncSession,
    payload: AgentCallbackPayload,
    submission: Submission,
    event_id: str,
) -> str:
    try:
        if not await _table_exists(db, DeadLetterJob.__tablename__):
            raise RuntimeError("Database schema missing dead_letter_jobs table")
        error_detail = "Execution failed"
        if isinstance(payload.summary, dict):
            error_detail = _clean_error_text(payload.summary.get("error")) or error_detail
        db.add(
            DeadLetterJob(
                submission_id=submission.id,
                source_event_id=event_id,
                error_detail=error_detail,
            )
        )
        await db.commit()
        return "created"
    except IntegrityError:
        await db.rollback()
        logger.info(
            "Callback side effect duplicate event_id=%s job_id=%s submission_id=%s side_effect_type=dead_letter_job side_effect_status=duplicate",
            event_id,
            payload.job_id,
            submission.id,
        )
        return "duplicate"
    except Exception as exc:
        await db.rollback()
        logger.exception(
            "Callback side effect failed event_id=%s submission_id=%s side_effect_type=dead_letter_job table=%s exception=%s message=%s",
            event_id,
            submission.id,
            DeadLetterJob.__tablename__,
            exc.__class__.__name__,
            _clean_error_text(exc),
        )
        await _mark_callback_side_effect_failed(db, event_id=event_id, error=str(exc))
        return "failed"

@router.post("/callback")
async def agent_callback(
    payload: AgentCallbackPayload,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    # Use Bearer token or custom header
    auth_header = request.headers.get("Authorization")
    settings = get_settings()
    expected_token = f"Bearer {settings.agent_callback_secret}"
    
    if auth_header != expected_token:
        # Fallback to header
        if request.headers.get("x-agent-callback-secret") != settings.agent_callback_secret:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent callback secret")

    submission_id = UUID(payload.submission_id)
    event_id = _derive_callback_event_id(payload)
    payload_hash = _callback_payload_hash(payload)

    existing_event = (
        await db.execute(select(CallbackEvent).where(CallbackEvent.event_id == event_id))
    ).scalar_one_or_none()
    if existing_event and existing_event.processing_status in {"completed", "side_effect_failed"}:
        logger.info(
            "Callback idempotent replay event_id=%s job_id=%s submission_id=%s primary_commit_status=already_completed idempotent_replay=true",
            event_id,
            payload.job_id,
            payload.submission_id,
        )
        submission = (
            await db.execute(
                select(Submission)
                .options(selectinload(Submission.user))
                .where(Submission.id == submission_id)
            )
        ).scalar_one_or_none()
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        ws_payload = {
            "upload_id": str(submission.id),
            "filename": submission.file_name,
            "status": submission.status,
            "output_ready": bool(submission.output_path),
        }
        return {
            "message": "Callback accepted",
            "accepted": True,
            "primary_state_persisted": True,
            "idempotent_replay": True,
            "side_effects": {},
            **ws_payload,
        }

    try:
        submission = (
            await db.execute(
                select(Submission)
                .options(selectinload(Submission.user))
                .where(Submission.id == submission_id)
            )
        ).scalar_one_or_none()
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")

        callback_event = existing_event or CallbackEvent(
            event_id=event_id,
            job_id=payload.job_id,
            submission_id=submission.id,
            event_type=payload.status,
            payload_hash=payload_hash,
            processing_status="processing",
        )
        if existing_event is None:
            db.add(callback_event)

        submission.status = map_agent_status_to_submission_status(payload.status)
        summary_has_error = False
        if isinstance(payload.summary, dict):
            summary_has_error = bool(_clean_error_text(payload.summary.get("error")))
            payload_errors = payload.summary.get("errors")
            if isinstance(payload_errors, list):
                summary_has_error = summary_has_error or any(_clean_error_text(item) for item in payload_errors)
        if summary_has_error and submission.status == SubmissionStatus.succeeded:
            submission.status = SubmissionStatus.failed

        submission.summary = payload.summary
        if payload.output_path:
            submission.output_path = payload.output_path
        submission.completed_at = _utc_now()
        callback_event.processing_status = "completed"
        callback_event.processed_at = _utc_now()
        callback_event.last_error = None

        await db.commit()
        logger.info(
            "Callback primary persisted event_id=%s job_id=%s submission_id=%s primary_commit_status=committed idempotent_replay=false",
            event_id,
            payload.job_id,
            submission.id,
        )
    except HTTPException:
        raise
    except IntegrityError as exc:
        await db.rollback()
        logger.warning(
            "Duplicate callback event race event_id=%s submission_id=%s error=%s",
            event_id,
            payload.submission_id,
            exc.__class__.__name__,
        )
        return {
            "message": "Callback accepted",
            "accepted": True,
            "primary_state_persisted": True,
            "idempotent_replay": True,
            "side_effects": {},
        }
    except Exception as exc:
        await db.rollback()
        logger.exception(
            "Callback primary persistence failed event_id=%s job_id=%s submission_id=%s primary_commit_status=failed exception=%s message=%s",
            event_id,
            payload.job_id,
            payload.submission_id,
            exc.__class__.__name__,
            _clean_error_text(exc),
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to persist agent callback") from exc

    await db.refresh(submission)

    side_effects: dict[str, str] = {}
    if submission.status == SubmissionStatus.quarantined:
        side_effects["needs_review_job"] = await _persist_needs_review_side_effect(
            db=db,
            payload=payload,
            submission=submission,
            event_id=event_id,
            settings=settings,
        )
        logger.info(
            "Callback side effect completed event_id=%s job_id=%s submission_id=%s side_effect_type=needs_review_job side_effect_status=%s",
            event_id,
            payload.job_id,
            submission.id,
            side_effects["needs_review_job"],
        )
    elif submission.status in {SubmissionStatus.failed, SubmissionStatus.callback_failed}:
        side_effects["dead_letter_job"] = await _persist_dead_letter_side_effect(
            db=db,
            payload=payload,
            submission=submission,
            event_id=event_id,
        )
        logger.info(
            "Callback side effect completed event_id=%s job_id=%s submission_id=%s side_effect_type=dead_letter_job side_effect_status=%s",
            event_id,
            payload.job_id,
            submission.id,
            side_effects["dead_letter_job"],
        )

    ws_payload = {
        "upload_id": str(submission.id),
        "filename": submission.file_name,
        "status": submission.status,
        "output_ready": bool(submission.output_path),
    }
    await ws_manager.broadcast("uploads", "upload_status", ws_payload)
    await ws_manager.broadcast("dashboard", "dashboard_refresh", ws_payload)

    return {
        "message": "Callback accepted",
        "accepted": True,
        "primary_state_persisted": True,
        "idempotent_replay": False,
        "side_effects": side_effects,
        **ws_payload,
    }
