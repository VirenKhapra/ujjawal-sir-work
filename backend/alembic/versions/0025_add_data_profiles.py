"""Add deterministic data profile persistence.

Revision ID: 0025_add_data_profiles
Revises: 0024_add_clarification_tables
Create Date: 2026-06-22 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0025_add_data_profiles"
down_revision = "0024_add_clarification_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "data_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("submission_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("file_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("profiler_version", sa.String(length=32), nullable=False),
        sa.Column("profile_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="ready"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("submission_id", "file_fingerprint", "profiler_version", name="uq_data_profiles_submission_fingerprint"),
    )


def downgrade() -> None:
    op.drop_table("data_profiles")
