import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import BackgroundTasks, HTTPException

from app.api.agent import map_agent_status_to_submission_status, reconcile_callback_status, resolve_agent_artifact_path
from app.api.uploads import _frame_from_structured_records, list_uploads, output_is_available, retry_upload, save_upload
from app.models import Submission, SubmissionRecord, SubmissionStatus, User, UserRole
from app.services.quarantine import requeue_submission


class DummyUploadFile:
    def __init__(self, filename: str, content_type: str, contents: bytes):
        self.filename = filename
        self.content_type = content_type
        self._contents = contents

    async def read(self) -> bytes:
        return self._contents


class FakeExecuteResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def first(self):
        if isinstance(self._rows, list):
            return self._rows[0] if self._rows else None
        return self._rows

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        if isinstance(self._rows, list):
            return self._rows[0] if self._rows else None
        return self._rows

    def scalar_one(self):
        if isinstance(self._rows, list):
            return self._rows[0]
        return self._rows


class FakeDb:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.added = []

    async def execute(self, *_args, **_kwargs):
        return FakeExecuteResult(self.rows)

    async def scalar(self, *_args, **_kwargs):
        return 1

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid4()
        return None

    async def commit(self):
        return None

    async def refresh(self, *_args, **_kwargs):
        return None


def test_requeue_submission_error_includes_current_status():
    async def run() -> str:
        submission = Submission(
            id=uuid4(),
            file_name="sample.xlsx",
            file_path="uploads/sample.xlsx",
            file_size_bytes=1,
            original_filename="sample.xlsx",
            instruction="clean this",
            output_format="XLSX",
            user_id=uuid4(),
            status=SubmissionStatus.running,
        )

        try:
            await requeue_submission(FakeDb(), submission=submission)
        except HTTPException as exc:
            assert exc.status_code == 409
            return exc.detail
        raise AssertionError("Expected HTTPException")

    detail = asyncio.run(run())
    assert "cannot be requeued right now" in detail
    assert "submission_status='running'" in detail
    assert "payload_status='none'" in detail


def test_retry_upload_error_includes_current_status(monkeypatch):
    async def run() -> str:
        user = User(
            id=uuid4(),
            full_name="Tester",
            email="tester@example.com",
            hashed_password="hashed",
            role=UserRole.employee,
        )
        submission = Submission(
            id=uuid4(),
            file_name="sample.xlsx",
            file_path="uploads/sample.xlsx",
            file_size_bytes=1,
            original_filename="sample.xlsx",
            instruction="clean this",
            output_format="XLSX",
            user_id=user.id,
            status=SubmissionStatus.running,
        )
        submission.user = user

        monkeypatch.setattr("app.api.uploads.verify_upload_access", lambda submission, user: None)

        try:
            await retry_upload(submission.id, db=FakeDb(submission), user=user)
        except HTTPException as exc:
            assert exc.status_code == 409
            return exc.detail
        raise AssertionError("Expected HTTPException")

    detail = asyncio.run(run())
    assert "cannot be retried right now" in detail
    assert "submission_status='running'" in detail
    assert "payload_status='none'" in detail


def test_submission_status_enum_contains_all_runtime_statuses():
    assert [status.value for status in SubmissionStatus] == [
        "queued",
        "planning",
        "running",
        "succeeded",
        "failed",
        "quarantined",
        "callback_failed",
        "awaiting_schema_approval",
        "awaiting_confirmation",
        "declined",
        "awaiting_clarification",
    ]


def test_resolve_agent_artifact_path_prefers_output_file_name(monkeypatch):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "outputs"
        output_dir.mkdir()
        artifact = output_dir / "cleaned_data.xlsx"
        artifact.write_text("ok", encoding="utf-8")
        settings = type("Settings", (), {"output_dir": str(output_dir)})()

        monkeypatch.setattr("app.api.agent.get_settings", lambda: settings)

        resolved = resolve_agent_artifact_path({"output_file_name": "cleaned_data.xlsx"})

        assert resolved == artifact


def test_resolve_agent_artifact_path_uses_relative_name_not_absolute_container_path(monkeypatch):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "outputs"
        output_dir.mkdir()
        artifact = output_dir / "cleaned_data.xlsx"
        artifact.write_text("ok", encoding="utf-8")
        settings = type("Settings", (), {"output_dir": str(output_dir)})()

        monkeypatch.setattr("app.api.agent.get_settings", lambda: settings)

        resolved = resolve_agent_artifact_path({"output_path": "/app/outputs/cleaned_data.xlsx"})

        assert resolved == artifact


def test_output_is_available_when_completed_payload_has_record_count_but_no_inline_rows():
    submission = Submission(
        file_name="sample.xlsx",
        file_path="uploads/sample.xlsx",
        file_size_bytes=1,
        original_filename="sample.xlsx",
        instruction="clean this",
        output_format="XLSX",
        user_id="00000000-0000-0000-0000-000000000001",
        status=SubmissionStatus.succeeded,
        summary={"record_count": 3},
    )

    assert output_is_available(submission) is True


def test_output_is_available_when_requested_pdf_can_fallback_to_recovered_xlsx():
    submission = Submission(
        file_name="sample.csv",
        file_path="uploads/sample.csv",
        file_size_bytes=1,
        original_filename="sample.csv",
        instruction="filter this",
        output_format="PDF",
        user_id="00000000-0000-0000-0000-000000000001",
        status=SubmissionStatus.succeeded,
        summary={"cleaned_data": [{"gender": "FEMALE", "marital_status": "SINGLE"}], "record_count": 1},
    )

    assert output_is_available(submission) is True


def test_frame_from_structured_records_builds_dataframe_from_saved_rows():
    rows = [
        SubmissionRecord(record_index=0, payload={"invoice_id": "INV-1", "amount": 100}),
        SubmissionRecord(record_index=1, payload={"invoice_id": "INV-2", "amount": 200}),
    ]

    frame = _frame_from_structured_records(rows)

    assert frame is not None
    assert frame.to_dict(orient="records") == [
        {"invoice_id": "INV-1", "amount": 100},
        {"invoice_id": "INV-2", "amount": 200},
    ]


def test_agent_callback_complete_maps_to_succeeded():
    assert map_agent_status_to_submission_status("complete") == SubmissionStatus.succeeded
    assert reconcile_callback_status("complete", {"status": "success"}, None) == "succeeded"


def test_agent_callback_quarantined_survives():
    assert map_agent_status_to_submission_status("quarantined") == SubmissionStatus.quarantined
    assert reconcile_callback_status("quarantined", {"status": "quarantined"}, None) == "quarantined"


def test_reconcile_callback_status_partial_maps_to_failed():
    assert reconcile_callback_status("complete", {"status": "partial"}, None) == "failed"
    assert reconcile_callback_status("partial", {"status": "success"}, None) == "failed"
    assert reconcile_callback_status("partial", {"status": "partial"}, None) == "failed"


def test_reconcile_callback_status_standard_flows():
    assert reconcile_callback_status("complete", {"status": "success"}, None) == "succeeded"
    assert reconcile_callback_status("complete", {"status": "failed"}, None) == "failed"
    assert reconcile_callback_status("complete", None, "some error") == "failed"
    assert reconcile_callback_status("complete", {"status": "success", "errors": ["err"]}, None) == "failed"


def test_upload_can_set_queued_status(tmp_path, monkeypatch):
    async def run() -> str:
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        settings = type(
            "Settings",
            (),
            {
                "upload_dir": str(upload_dir),
                "max_upload_size_mb": 1,
                "max_preview_rows": 5,
            },
        )()

        async def noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr("app.api.uploads.get_settings", lambda: settings)
        monkeypatch.setattr("app.api.uploads.enqueue_submission_dispatch", noop)
        monkeypatch.setattr("app.api.uploads.ws_manager.broadcast", noop)

        fake_db = FakeDb()
        user = User(
            id=uuid4(),
            full_name="Tester",
            email="tester@example.com",
            hashed_password="hashed",
            role=UserRole.employee,
        )
        file = DummyUploadFile("sample.txt", "text/plain", b"line one\nline two\n")

        result = await save_upload(
            file=file,
            instruction="clean this",
            output_format="XLSX",
            db=fake_db,
            user=user,
            background_tasks=BackgroundTasks(),
        )

        assert result.status == SubmissionStatus.queued.value
        assert fake_db.added[0].status == SubmissionStatus.queued
        return result.status

    assert asyncio.run(run()) == SubmissionStatus.queued.value


def test_uploads_list_handles_all_statuses():
    async def run() -> list[str]:
        admin = User(
            id=uuid4(),
            full_name="Admin",
            email="admin@example.com",
            hashed_password="hashed",
            role=UserRole.admin,
        )
        now = datetime(2026, 1, 1, 12, 0, 0)
        statuses = [
            SubmissionStatus.queued,
            SubmissionStatus.planning,
            SubmissionStatus.running,
            SubmissionStatus.succeeded,
            SubmissionStatus.failed,
            SubmissionStatus.quarantined,
            SubmissionStatus.callback_failed,
            SubmissionStatus.awaiting_schema_approval,
            SubmissionStatus.awaiting_confirmation,
            SubmissionStatus.declined,
        ]
        rows = []
        for index, status in enumerate(statuses):
            submission = Submission(
                id=uuid4(),
                file_name=f"sample-{index}.txt",
                file_path=f"uploads/sample-{index}.txt",
                file_size_bytes=10,
                original_filename=f"sample-{index}.txt",
                instruction="clean this",
                output_format="XLSX",
                user_id=admin.id,
                version_number=1,
                status=status,
                summary={"status": status.value},
            )
            submission.user = admin
            submission.uploaded_at = now + timedelta(minutes=index)
            rows.append((submission, index))

        fake_db = FakeDb(rows)
        result = await list_uploads(
            status=None,
            date_from=None,
            date_to=None,
            db=fake_db,
            user=admin,
        )

        return [item.status for item in result]

    assert asyncio.run(run()) == [
        "queued",
        "planning",
        "running",
        "succeeded",
        "failed",
        "quarantined",
        "callback_failed",
        "awaiting_schema_approval",
        "awaiting_confirmation",
        "declined",
    ]
