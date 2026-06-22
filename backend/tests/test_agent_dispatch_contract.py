import ast
import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4

from app.models import Submission, SubmissionStatus
from app.services import agent_dispatcher


class FakeRedis:
    def __init__(self):
        self.jobs: list[tuple[str, dict]] = []

    async def enqueue_job(self, job_name: str, payload: dict) -> None:
        self.jobs.append((job_name, payload))


class FakeDbSession:
    def __init__(self, submission: Submission):
        self.submission = submission
        self.committed = False
        self.items = []
        self.executed = []

    async def get(self, model, submission_id):
        if model is Submission:
            assert str(submission_id) == str(self.submission.id)
            return self.submission
        return None

    async def execute(self, *_args, **_kwargs):
        self.executed.append((_args, _kwargs))

        class _EmptyResult:
            def scalars(self):
                return self

            def first(self):
                return None

        return _EmptyResult()

    def add(self, item):
        self.items.append(item)

    async def flush(self):
        return None

    async def commit(self):
        self.committed = True


class FakeSessionManager:
    def __init__(self, session: FakeDbSession):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _agent_service_job_payload_fields() -> list[str]:
    source = (_repo_root() / "agent-framework" / "new-agentic-project-data-cleaning-" / "src" / "finflow_agent" / "api.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "JobPayload":
            fields: list[str] = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append(stmt.target.id)
            return fields
    raise AssertionError("JobPayload class was not found in the agent service API")


def _resolved_canonical_intent() -> dict:
    return {
        "schema_version": "2.0",
        "intent_id": "intent-1",
        "intent_revision": 1,
        "intent_hash": "hash-1",
        "parent_intent_id": None,
        "original_prompt": "clean this",
        "normalized_prompt": "clean this",
        "resolution_status": "resolved",
        "decision": "clean",
        "evidence": [],
        "alternatives_considered": [],
        "actions": [{"kind": "clean", "mode": "safe_default", "operations": []}],
        "output_format": "xlsx",
        "assumptions": [],
        "repair_notes": [],
        "dataframe_profile": {"source_columns": ["a", "b"]},
        "capability_version": "backend.capability.1",
        "capability_snapshot": {},
    }


def test_agent_dispatcher_sends_file_id_not_file_path(monkeypatch):
    async def run() -> tuple[str, dict, bool, str, str]:
        with tempfile.TemporaryDirectory(dir=_repo_root()) as temp_dir:
            upload_dir = Path(temp_dir) / "uploads"
            upload_dir.mkdir()
            stored_file = upload_dir / "stored-input.csv"
            stored_file.write_text("a,b\n1,2\n", encoding="utf-8")

            submission = Submission(
                id=uuid4(),
                file_name="original-input.csv",
                file_path=str(stored_file),
                file_size_bytes=10,
                original_filename="original-input.csv",
                instruction="clean this",
                output_format="XLSX",
                user_id=uuid4(),
                version_number=1,
                status=SubmissionStatus.queued,
            )
            submission.canonical_intent = _resolved_canonical_intent()

            fake_redis = FakeRedis()
            fake_db = FakeDbSession(submission)
            data_profile = type("Profile", (), {"id": uuid4()})()
            revision_record = type("Revision", (), {"id": uuid4(), "created_at": None, "canonical_intent": submission.canonical_intent})()

            async def fake_create_pool(*_args, **_kwargs):
                return fake_redis

            async def fake_load_profile(*_args, **_kwargs):
                return data_profile

            async def fake_latest_revision(*_args, **_kwargs):
                return revision_record

            monkeypatch.setattr(agent_dispatcher, "create_pool", fake_create_pool)
            monkeypatch.setattr(agent_dispatcher, "AsyncSessionLocal", lambda: FakeSessionManager(fake_db))
            monkeypatch.setattr(agent_dispatcher, "load_latest_data_profile", fake_load_profile)
            monkeypatch.setattr(agent_dispatcher, "latest_intent_revision_for_submission", fake_latest_revision)
            monkeypatch.setattr(
                agent_dispatcher,
                "get_settings",
                lambda: type("Settings", (), {"redis_url": "redis://localhost:6379/0"})(),
            )

            await agent_dispatcher.enqueue_submission_dispatch(submission.id)
            assert fake_redis.jobs
            job_name, payload = fake_redis.jobs[0]
            return job_name, payload, fake_db.committed, submission.status.value, str(submission.id)

    job_name, payload, committed, status, submission_id = asyncio.run(run())

    assert job_name == "process_job_task"
    assert payload["submission_id"] == submission_id
    assert payload["file_id"] == "stored-input.csv"
    assert payload["file_name"] == "original-input.csv"
    assert payload["resolved_file_path"].endswith("stored-input.csv")
    assert payload["output_format"] == "xlsx"
    assert payload["audit_context"]["original_instruction"] == "clean this"
    assert isinstance(payload["canonical_intent"], dict)
    assert payload["canonical_intent"]["schema_version"] == "1.0"
    assert payload["canonical_intent"]["intent"]["schema_version"] == "2.0"
    assert "file_path" not in payload
    assert "instruction" not in payload
    assert committed is True
    assert status == SubmissionStatus.planning.value


def test_agent_payload_matches_agent_service_job_payload(monkeypatch):
    async def run() -> dict:
        with tempfile.TemporaryDirectory(dir=_repo_root()) as temp_dir:
            upload_dir = Path(temp_dir) / "uploads"
            upload_dir.mkdir()
            stored_file = upload_dir / "job-input.csv"
            stored_file.write_text("a,b\n1,2\n", encoding="utf-8")

            submission = Submission(
                id=uuid4(),
                file_name="job-input.csv",
                file_path=str(stored_file),
                file_size_bytes=10,
                original_filename="job-input.csv",
                instruction="normalize values",
                output_format="CSV",
                user_id=uuid4(),
                version_number=1,
                status=SubmissionStatus.queued,
            )
            submission.canonical_intent = _resolved_canonical_intent()

            fake_redis = FakeRedis()
            fake_db = FakeDbSession(submission)
            data_profile = type("Profile", (), {"id": uuid4()})()
            revision_record = type("Revision", (), {"id": uuid4(), "created_at": None, "canonical_intent": submission.canonical_intent})()

            async def fake_create_pool(*_args, **_kwargs):
                return fake_redis

            async def fake_load_profile(*_args, **_kwargs):
                return data_profile

            async def fake_latest_revision(*_args, **_kwargs):
                return revision_record

            monkeypatch.setattr(agent_dispatcher, "create_pool", fake_create_pool)
            monkeypatch.setattr(agent_dispatcher, "AsyncSessionLocal", lambda: FakeSessionManager(fake_db))
            monkeypatch.setattr(agent_dispatcher, "load_latest_data_profile", fake_load_profile)
            monkeypatch.setattr(agent_dispatcher, "latest_intent_revision_for_submission", fake_latest_revision)
            monkeypatch.setattr(
                agent_dispatcher,
                "get_settings",
                lambda: type("Settings", (), {"redis_url": "redis://localhost:6379/0"})(),
            )

            await agent_dispatcher.enqueue_submission_dispatch(submission.id)
            return fake_redis.jobs[0][1]

    payload = asyncio.run(run())
    agent_fields = _agent_service_job_payload_fields()

    assert sorted(payload.keys()) == sorted(agent_fields)
    assert "file_path" not in payload
    assert payload["file_id"] == "job-input.csv"
    assert payload["resolved_file_path"].endswith("job-input.csv")
    assert payload["output_format"] == "csv"
    assert agent_fields == [
        "submission_id",
        "file_id",
        "file_name",
        "resolved_file_path",
        "data_profile_id",
        "canonical_intent_revision_id",
        "canonical_intent",
        "output_format",
        "audit_context",
    ]


def test_shared_upload_dir_contract_documented():
    compose_text = (_repo_root() / "docker-compose.yml").read_text(encoding="utf-8")
    deploy_text = (_repo_root() / "docs" / "RAILWAY_DEPLOYMENT.md").read_text(encoding="utf-8")

    assert compose_text.count("UPLOAD_DIR: /app/storage/uploads") >= 2
    assert "backend_storage:/app/storage" in compose_text
    assert "UPLOAD_DIR=/app/storage/uploads" in deploy_text


def test_dispatch_requires_persisted_data_profile_when_missing(monkeypatch):
    async def run() -> dict:
        with tempfile.TemporaryDirectory(dir=_repo_root()) as temp_dir:
            upload_dir = Path(temp_dir) / "uploads"
            upload_dir.mkdir()
            stored_file = upload_dir / "stored-input.csv"
            stored_file.write_text("a,b\n1,2\n", encoding="utf-8")

            submission = Submission(
                id=uuid4(),
                file_name="original-input.csv",
                file_path=str(stored_file),
                file_size_bytes=10,
                original_filename="original-input.csv",
                instruction="clean this",
                output_format="XLSX",
                user_id=uuid4(),
                version_number=1,
                status=SubmissionStatus.queued,
            )
            submission.summary = {}

            fake_redis = FakeRedis()
            fake_db = FakeDbSession(submission)

            async def fake_create_pool(*_args, **_kwargs):
                return fake_redis

            async def fake_load_profile(*_args, **_kwargs):
                return None

            async def fake_latest_revision(*_args, **_kwargs):
                return None

            monkeypatch.setattr(agent_dispatcher, "create_pool", fake_create_pool)
            monkeypatch.setattr(agent_dispatcher, "AsyncSessionLocal", lambda: FakeSessionManager(fake_db))
            monkeypatch.setattr(agent_dispatcher, "load_latest_data_profile", fake_load_profile)
            monkeypatch.setattr(agent_dispatcher, "latest_intent_revision_for_submission", fake_latest_revision)
            monkeypatch.setattr(
                agent_dispatcher,
                "get_settings",
                lambda: type("Settings", (), {"redis_url": "redis://localhost:6379/0", "max_preview_rows": 50})(),
            )

            await agent_dispatcher.enqueue_submission_dispatch(submission.id)
            return {"jobs": fake_redis.jobs, "status": submission.status.value, "summary": submission.summary}

    result = asyncio.run(run())

    assert result["jobs"] == []
    assert result["status"] == SubmissionStatus.failed.value
    assert result["summary"]["error"] == "data_profile_missing"
