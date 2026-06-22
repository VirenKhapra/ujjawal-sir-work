from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from app.api import uploads
from app.models import Submission, SubmissionStatus
from app.services.canonical_intent import build_canonical_intent


class FakeDbSession:
    def __init__(self, submission: Submission):
        self.submission = submission
        self.committed = False
        self.added = []

    async def execute(self, *_args, **_kwargs):
        return SimpleNamespace(scalar_one_or_none=lambda: self.submission)

    def add(self, item):
        self.added.append(item)

    async def flush(self):
        return None

    async def commit(self):
        self.committed = True
def _user(role_name: str = "admin") -> SimpleNamespace:
    from app.models import UserRole

    return SimpleNamespace(id=uuid4(), role=getattr(UserRole, role_name))


def test_replace_column_mapping_creates_revision_and_resumes(monkeypatch):
    async def run() -> tuple[Submission, list[tuple[str, str, bool]], dict]:
        canonical_intent = build_canonical_intent(["age", "gender"], [], "only show loans columns")
        submission = Submission(
            id=uuid4(),
            file_name="input.csv",
            file_path="input.csv",
            file_size_bytes=10,
            original_filename="input.csv",
            instruction="only show loans columns",
            output_format="XLSX",
            user_id=uuid4(),
            version_number=1,
            status=SubmissionStatus.awaiting_confirmation,
            summary={"canonical_intent": canonical_intent},
        )

        fake_db = FakeDbSession(submission)
        enqueue_calls: list[tuple[str, str, bool]] = []
        broadcast_calls: list[tuple[str, str, bool]] = []

        async def fake_enqueue(submission_id, *, persist_revision=True):
            enqueue_calls.append((str(submission_id), submission.status.value, persist_revision))

        async def fake_broadcast(*args, **kwargs):
            broadcast_calls.append((str(args[0]), str(args[1]), True))

        async def fake_get_upload(*_args, **_kwargs):
            return {"status": "ok"}

        monkeypatch.setattr(uploads, "enqueue_submission_dispatch", fake_enqueue)
        monkeypatch.setattr(uploads.ws_manager, "broadcast", fake_broadcast)
        monkeypatch.setattr(uploads, "get_upload", fake_get_upload)

        result = await uploads.replace_column_mapping(
            submission.id,
            uploads.ReplaceColumnMappingRequest(mapping={"loans": ["loan_amount", "loan_status"]}, reason="user correction"),
            db=fake_db,
            user=_user("admin"),
        )
        return submission, enqueue_calls, result

    submission, enqueue_calls, result = asyncio.run(run())

    assert submission.status == SubmissionStatus.queued
    assert submission.summary["review_status"] == "corrected"
    assert submission.summary["canonical_intent"]["resolution_status"] == "repaired"
    assert submission.summary["canonical_intent"]["actions"][0]["requested_fields"][0]["selection_mode"] == "semantic_family"
    assert submission.summary["canonical_intent"]["actions"][0]["requested_fields"][0]["resolved_columns"] == ["loan_amount", "loan_status"]
    assert enqueue_calls == [(str(submission.id), SubmissionStatus.queued.value, False)]
    assert result == {"status": "ok"}


def test_reject_interpretation_pauses_without_resuming(monkeypatch):
    async def run() -> Submission:
        canonical_intent = build_canonical_intent(["age", "gender"], [], "only show loans columns")
        submission = Submission(
            id=uuid4(),
            file_name="input.csv",
            file_path="input.csv",
            file_size_bytes=10,
            original_filename="input.csv",
            instruction="only show loans columns",
            output_format="XLSX",
            user_id=uuid4(),
            version_number=1,
            status=SubmissionStatus.awaiting_confirmation,
            summary={"canonical_intent": canonical_intent},
        )

        fake_db = FakeDbSession(submission)
        enqueue_calls: list[tuple[str, str, bool]] = []
        async def fake_broadcast(*_args, **_kwargs):
            return None

        async def fake_get_upload(*_args, **_kwargs):
            return {"status": "ok"}

        monkeypatch.setattr(uploads, "enqueue_submission_dispatch", lambda *args, **kwargs: enqueue_calls.append(("called", "called", False)))
        monkeypatch.setattr(uploads.ws_manager, "broadcast", fake_broadcast)
        monkeypatch.setattr(uploads, "get_upload", fake_get_upload)

        await uploads.reject_interpretation(
            submission.id,
            uploads.RejectInterpretationRequest(reason="needs clarification"),
            db=fake_db,
            user=_user("admin"),
        )
        return submission

    submission = asyncio.run(run())

    assert submission.status == SubmissionStatus.awaiting_clarification
    assert submission.summary["review_status"] == "rejected"
    assert submission.summary["canonical_intent_status"] == "rejected"


def test_resume_job_enqueues_existing_canonical_intent(monkeypatch):
    async def run() -> list[tuple[str, str, bool]]:
        canonical_intent = build_canonical_intent(["age", "gender", "loan_amount"], [], "only show age and gender")
        submission = Submission(
            id=uuid4(),
            file_name="input.csv",
            file_path="input.csv",
            file_size_bytes=10,
            original_filename="input.csv",
            instruction="only show age and gender",
            output_format="XLSX",
            user_id=uuid4(),
            version_number=1,
            status=SubmissionStatus.awaiting_confirmation,
            summary={"canonical_intent": canonical_intent},
        )
        fake_db = FakeDbSession(submission)
        enqueue_calls: list[tuple[str, str, bool]] = []

        async def fake_enqueue(submission_id, *, persist_revision=True):
            enqueue_calls.append((str(submission_id), submission.status.value, persist_revision))

        async def fake_broadcast(*_args, **_kwargs):
            return None

        async def fake_get_upload(*_args, **_kwargs):
            return {"status": "ok"}

        monkeypatch.setattr(uploads, "enqueue_submission_dispatch", fake_enqueue)
        monkeypatch.setattr(uploads.ws_manager, "broadcast", fake_broadcast)
        monkeypatch.setattr(uploads, "get_upload", fake_get_upload)

        await uploads.resume_job(
            submission.id,
            uploads.ResumeJobRequest(reason="approved"),
            db=fake_db,
            user=_user("admin"),
        )
        return enqueue_calls

    enqueue_calls = asyncio.run(run())
    assert enqueue_calls == [(str(enqueue_calls[0][0]), SubmissionStatus.queued.value, False)]
