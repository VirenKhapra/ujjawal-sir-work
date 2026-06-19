"""callback persistence hardening

Revision ID: 0020_callback_hardening
Revises: 0019_add_needs_review_jobs_table
Create Date: 2026-06-19 10:20:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0020_callback_hardening"
down_revision = "0019_add_needs_review_jobs_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "needs_review_jobs",
        sa.Column("source_event_id", sa.String(length=128), nullable=True),
    )
    op.create_unique_constraint(
        "uq_needs_review_jobs_source_event_id",
        "needs_review_jobs",
        ["source_event_id"],
    )

    op.create_table(
        "dead_letter_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("submission_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_event_id", sa.String(length=128), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["submission_id"], ["submissions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_event_id", name="uq_dead_letter_jobs_source_event_id"),
    )
    op.create_index(
        op.f("ix_dead_letter_jobs_submission_id"),
        "dead_letter_jobs",
        ["submission_id"],
        unique=False,
    )

    op.create_table(
        "callback_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_id", sa.String(length=128), nullable=False),
        sa.Column("job_id", sa.String(length=128), nullable=True),
        sa.Column("submission_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("processing_status", sa.String(length=40), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["submission_id"], ["submissions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id", name="uq_callback_events_event_id"),
    )
    op.create_index(
        op.f("ix_callback_events_submission_id"),
        "callback_events",
        ["submission_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_callback_events_submission_id"), table_name="callback_events")
    op.drop_table("callback_events")
    op.drop_index(op.f("ix_dead_letter_jobs_submission_id"), table_name="dead_letter_jobs")
    op.drop_table("dead_letter_jobs")
    op.drop_constraint(
        "uq_needs_review_jobs_source_event_id",
        "needs_review_jobs",
        type_="unique",
    )
    op.drop_column("needs_review_jobs", "source_event_id")
