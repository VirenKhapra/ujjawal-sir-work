import enum
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def enum_values(enum_type: type[enum.Enum]) -> list[str]:
    return [member.value for member in enum_type]


class UserRole(str, enum.Enum):
    employee = "employee"
    manager = "manager"
    admin = "admin"


class SubmissionStatus(str, enum.Enum):
    queued = "queued"
    planning = "planning"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    quarantined = "quarantined"
    callback_failed = "callback_failed"
    awaiting_schema_approval = "awaiting_schema_approval"
    awaiting_confirmation = "awaiting_confirmation"
    declined = "declined"


SUBMISSION_STATUS_ALIASES: dict[str, str] = {
    "pending": SubmissionStatus.queued.value,
    "processing": SubmissionStatus.running.value,
    "complete": SubmissionStatus.succeeded.value,
    "success": SubmissionStatus.succeeded.value,
    "partial": SubmissionStatus.failed.value,
    "rejected": SubmissionStatus.quarantined.value,
}


def normalize_submission_status(value: SubmissionStatus | str | None) -> str:
    if value is None:
        return SubmissionStatus.queued.value
    if isinstance(value, SubmissionStatus):
        return value.value
    normalized = str(value).strip().lower()
    if not normalized:
        return SubmissionStatus.queued.value
    return SUBMISSION_STATUS_ALIASES.get(normalized, normalized)

class ReviewStatus(str, enum.Enum):
    processing = "processing"
    pending = "pending"
    complete = "complete"
    failed = "failed"


class ReviewAction(str, enum.Enum):
    complete = "complete"
    failed = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role", values_callable=enum_values),
        default=UserRole.employee,
        nullable=False,
    )
    manager_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    submissions: Mapped[list["Submission"]] = relationship(back_populates="user", foreign_keys="Submission.user_id")
    reviews: Mapped[list["Review"]] = relationship(back_populates="manager", foreign_keys="Review.manager_id")
    comments: Mapped[list["SubmissionComment"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    manager: Mapped["User | None"] = relationship(remote_side=[id], back_populates="employees", foreign_keys=[manager_id])
    employees: Mapped[list["User"]] = relationship(back_populates="manager", foreign_keys=[manager_id])
    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    @property
    def name(self) -> str:
        return self.full_name


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sub_id: Mapped[int] = mapped_column(Integer, nullable=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    instruction: Mapped[str] = mapped_column(Text, nullable=False, default="")
    output_format: Mapped[str] = mapped_column(String(32), nullable=False, default="XLSX")
    version_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    parent_submission_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("submissions.id"))
    agent_task_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[SubmissionStatus] = mapped_column(
        Enum(SubmissionStatus, name="submission_status", values_callable=enum_values),
        default=SubmissionStatus.queued,
        nullable=False,
    )
    summary: Mapped[dict | None] = mapped_column(JSONB)
    output_path: Mapped[str | None] = mapped_column(String(500))
    preferred_agent_name: Mapped[str | None] = mapped_column(String(120))
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="submissions", foreign_keys=[user_id])
    parent_submission: Mapped["Submission | None"] = relationship(remote_side=[id])
    review: Mapped["Review | None"] = relationship(back_populates="submission", cascade="all, delete-orphan")
    structured_records: Mapped[list["SubmissionRecord"]] = relationship(back_populates="submission", cascade="all, delete-orphan")
    comments: Mapped[list["SubmissionComment"]] = relationship(back_populates="submission", cascade="all, delete-orphan")


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("submission_id", name="uq_reviews_submission_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False)
    manager_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    action: Mapped[ReviewAction] = mapped_column(
        Enum(ReviewAction, name="review_action", values_callable=enum_values),
        nullable=False,
    )
    comment: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    submission: Mapped[Submission] = relationship(back_populates="review")
    manager: Mapped[User] = relationship(back_populates="reviews")


class SubmissionComment(Base):
    __tablename__ = "submission_comments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    submission: Mapped[Submission] = relationship(back_populates="comments")
    user: Mapped[User] = relationship(back_populates="comments")


class SubmissionRecord(Base):
    __tablename__ = "submission_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False)
    record_index: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    submission: Mapped["Submission"] = relationship(back_populates="structured_records")


class NeedsReviewJob(Base):
    __tablename__ = "needs_review_jobs"
    __table_args__ = (
        UniqueConstraint("source_event_id", name="uq_needs_review_jobs_source_event_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False)
    source_event_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

class DeadLetterJob(Base):
    __tablename__ = "dead_letter_jobs"
    __table_args__ = (
        UniqueConstraint("source_event_id", name="uq_dead_letter_jobs_source_event_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False)
    source_event_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class CallbackEvent(Base):
    __tablename__ = "callback_events"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_callback_events_event_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    submission_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    processing_status: Mapped[str] = mapped_column(String(40), nullable=False, default="processing")
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="refresh_tokens")


class PendingPasswordChange(Base):
    __tablename__ = "pending_password_changes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    new_password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(foreign_keys=[user_id])


class AuditAction(str, enum.Enum):
    upload_created = "upload_created"
    upload_approved = "upload_approved"
    upload_declined = "upload_declined"
    reupload_requested = "reupload_requested"
    reupload_submitted = "reupload_submitted"
    comment_added = "comment_added"
    user_assigned = "user_assigned"
    user_reassigned = "user_reassigned"
    login = "login"
    logout = "logout"
    password_change = "password_change"


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    actor_name: Mapped[str] = mapped_column(String(120), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[AuditAction] = mapped_column(
        Enum(AuditAction, name="audit_action", values_callable=enum_values),
        nullable=False,
    )
    target_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    target_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    actor: Mapped["User | None"] = relationship(foreign_keys=[actor_id])


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False, default="transaction_validation")
    title: Mapped[str | None] = mapped_column(String(255))
    message: Mapped[str | None] = mapped_column(Text)
    upload_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("submissions.id"))
    entry_no: Mapped[str] = mapped_column(String(80), nullable=False)
    account_code: Mapped[str] = mapped_column(String(80), nullable=False)
    sub_account: Mapped[str] = mapped_column(String(255), nullable=False)
    difference: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="FAILED")
    is_read: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RegisteredAgent(Base):
    __tablename__ = "registered_agents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    capability_tags: Mapped[list[str]] = mapped_column(ARRAY(String(64)), nullable=False, default=list)
    input_formats: Mapped[list[str]] = mapped_column(ARRAY(String(32)), nullable=False, default=list)
    output_formats: Mapped[list[str]] = mapped_column(ARRAY(String(32)), nullable=False, default=list)
    endpoint_url: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="active")
    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_invocations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
