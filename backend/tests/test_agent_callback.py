import asyncio
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.agent import AgentCallbackPayload, agent_callback
from app.models import CallbackEvent, DeadLetterJob, NeedsReviewJob, Submission, SubmissionStatus, User


class DummyRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers


class FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class FakeInspector:
    def __init__(self, table_exists: bool):
        self.table_exists = table_exists

    def has_table(self, table_name: str) -> bool:
        return self.table_exists if table_name == "needs_review_jobs" else True


class FakeSyncSession:
    def connection(self):
        return object()


class FakeCallbackDb:
    def __init__(self, submission, *, table_exists: bool):
        self.submission = submission
        self.callback_event = None
        self.table_exists = table_exists
        self.added = []
        self.commit_count = 0
        self.rollback_count = 0
        self.refreshed = []

    async def execute(self, statement, *_args, **_kwargs):
        if "callback_events" in str(statement):
            return FakeResult(self.callback_event)
        return FakeResult(self.submission)

    async def run_sync(self, fn):
        return fn(FakeSyncSession())

    def add(self, obj):
        self.added.append(obj)
        if isinstance(obj, CallbackEvent):
            self.callback_event = obj

    async def commit(self):
        self.commit_count += 1

    async def rollback(self):
        self.rollback_count += 1

    async def refresh(self, obj):
        self.refreshed.append(obj)


def _build_submission() -> Submission:
    return Submission(
        id=uuid4(),
        file_name="sample.csv",
        file_path="uploads/sample.csv",
        file_size_bytes=1,
        original_filename="sample.csv",
        instruction="review this",
        output_format="XLSX",
        user_id=uuid4(),
        status=SubmissionStatus.running,
    )


def test_agent_callback_creates_needs_review_job_when_table_exists(monkeypatch):
    async def run() -> dict:
        submission = _build_submission()
        fake_db = FakeCallbackDb(submission, table_exists=True)
        settings = type(
            "Settings",
            (),
            {
                "agent_callback_secret": "test-secret",
                "enable_needs_review_jobs": True,
            },
        )()

        monkeypatch.setattr("app.api.agent.get_settings", lambda: settings)
        monkeypatch.setattr("app.api.agent.inspect", lambda _connection: FakeInspector(True))
        monkeypatch.setattr("app.api.agent.ws_manager.broadcast", lambda *args, **kwargs: asyncio.sleep(0))

        payload = AgentCallbackPayload(
            submission_id=str(submission.id),
            status="quarantined",
            summary={"reason": "Needs human review"},
        )
        response = await agent_callback(payload, DummyRequest({"Authorization": "Bearer test-secret"}), fake_db)

        assert fake_db.commit_count >= 2
        assert fake_db.rollback_count == 0
        assert any(isinstance(obj, NeedsReviewJob) for obj in fake_db.added)
        assert response["status"] == SubmissionStatus.quarantined
        assert response["upload_id"] == str(submission.id)
        assert response["primary_state_persisted"] is True
        assert response["side_effects"]["needs_review_job"] == "created"
        return response

    asyncio.run(run())


def test_agent_callback_preserves_primary_state_when_needs_review_table_missing(monkeypatch):
    async def run() -> None:
        submission = _build_submission()
        fake_db = FakeCallbackDb(submission, table_exists=False)
        settings = type(
            "Settings",
            (),
            {
                "agent_callback_secret": "test-secret",
                "enable_needs_review_jobs": True,
            },
        )()

        monkeypatch.setattr("app.api.agent.get_settings", lambda: settings)
        monkeypatch.setattr("app.api.agent.inspect", lambda _connection: FakeInspector(False))

        payload = AgentCallbackPayload(
            submission_id=str(submission.id),
            status="quarantined",
            summary={"reason": "Needs human review"},
        )

        monkeypatch.setattr("app.api.agent.ws_manager.broadcast", lambda *args, **kwargs: asyncio.sleep(0))

        response = await agent_callback(payload, DummyRequest({"Authorization": "Bearer test-secret"}), fake_db)

        assert response["primary_state_persisted"] is True
        assert response["side_effects"]["needs_review_job"] == "failed"
        assert fake_db.commit_count >= 1
        assert fake_db.rollback_count >= 1
        assert submission.status == SubmissionStatus.quarantined

    asyncio.run(run())


def test_agent_callback_creates_dead_letter_job_for_failed_status(monkeypatch):
    async def run() -> None:
        submission = _build_submission()
        fake_db = FakeCallbackDb(submission, table_exists=True)
        settings = type(
            "Settings",
            (),
            {
                "agent_callback_secret": "test-secret",
                "enable_needs_review_jobs": True,
            },
        )()

        monkeypatch.setattr("app.api.agent.get_settings", lambda: settings)
        monkeypatch.setattr("app.api.agent.inspect", lambda _connection: FakeInspector(True))
        monkeypatch.setattr("app.api.agent.ws_manager.broadcast", lambda *args, **kwargs: asyncio.sleep(0))

        payload = AgentCallbackPayload(
            submission_id=str(submission.id),
            status="failed",
            summary={"error": "Low-confidence column match"},
            event_id="agent:test-failed",
        )

        response = await agent_callback(payload, DummyRequest({"Authorization": "Bearer test-secret"}), fake_db)

        assert response["primary_state_persisted"] is True
        assert response["side_effects"]["dead_letter_job"] == "created"
        assert submission.status == SubmissionStatus.failed
        assert any(isinstance(obj, DeadLetterJob) for obj in fake_db.added)

    asyncio.run(run())


def test_agent_callback_duplicate_event_returns_idempotent_success(monkeypatch):
    async def run() -> None:
        submission = _build_submission()
        submission.status = SubmissionStatus.failed
        fake_db = FakeCallbackDb(submission, table_exists=True)
        fake_db.callback_event = CallbackEvent(
            event_id="agent:duplicate",
            submission_id=submission.id,
            event_type="failed",
            payload_hash="hash",
            processing_status="completed",
        )
        settings = type(
            "Settings",
            (),
            {
                "agent_callback_secret": "test-secret",
                "enable_needs_review_jobs": True,
            },
        )()

        monkeypatch.setattr("app.api.agent.get_settings", lambda: settings)

        payload = AgentCallbackPayload(
            submission_id=str(submission.id),
            status="failed",
            summary={"error": "Already processed"},
            event_id="agent:duplicate",
        )

        response = await agent_callback(payload, DummyRequest({"Authorization": "Bearer test-secret"}), fake_db)

        assert response["primary_state_persisted"] is True
        assert response["idempotent_replay"] is True
        assert fake_db.commit_count == 0
        assert not any(isinstance(obj, DeadLetterJob) for obj in fake_db.added)

    asyncio.run(run())


def test_agent_callback_rejects_invalid_secret(monkeypatch):
    async def run() -> None:
        submission = _build_submission()
        fake_db = FakeCallbackDb(submission, table_exists=True)
        settings = type(
            "Settings",
            (),
            {
                "agent_callback_secret": "test-secret",
                "enable_needs_review_jobs": True,
            },
        )()

        monkeypatch.setattr("app.api.agent.get_settings", lambda: settings)

        payload = AgentCallbackPayload(
            submission_id=str(submission.id),
            status="failed",
            summary={"error": "should not persist"},
        )

        with pytest.raises(HTTPException) as exc:
            await agent_callback(payload, DummyRequest({"Authorization": "Bearer wrong"}), fake_db)

        assert exc.value.status_code == 401
        assert fake_db.commit_count == 0
        assert fake_db.rollback_count == 0
        assert fake_db.added == []

    asyncio.run(run())
