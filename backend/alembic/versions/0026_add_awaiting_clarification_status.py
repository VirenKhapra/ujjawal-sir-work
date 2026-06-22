"""Add awaiting_clarification to submission_status.

Revision ID: 0026_add_clarify_status
Revises: 0025_add_data_profiles
Create Date: 2026-06-22 00:00:00.000000
"""

from __future__ import annotations

from alembic import op


revision = "0026_add_clarify_status"
down_revision = "0025_add_data_profiles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE submission_status ADD VALUE IF NOT EXISTS 'awaiting_clarification'")


def downgrade() -> None:
    op.execute("ALTER TABLE submissions ALTER COLUMN status DROP DEFAULT")
    op.execute(
        """
        ALTER TABLE submissions
        ALTER COLUMN status TYPE TEXT
        USING status::text
        """
    )
    op.execute(
        """
        UPDATE submissions
        SET status = CASE
            WHEN status = 'awaiting_clarification' THEN 'awaiting_confirmation'
            ELSE status
        END
        """
    )
    op.execute("ALTER TYPE submission_status RENAME TO submission_status_old")
    op.execute(
        """
        CREATE TYPE submission_status AS ENUM (
            'queued',
            'planning',
            'running',
            'succeeded',
            'failed',
            'quarantined',
            'callback_failed',
            'awaiting_schema_approval',
            'awaiting_confirmation',
            'declined'
        )
        """
    )
    op.execute(
        """
        ALTER TABLE submissions
        ALTER COLUMN status TYPE submission_status
        USING status::submission_status
        """
    )
    op.execute("ALTER TABLE submissions ALTER COLUMN status SET DEFAULT 'queued'")
    op.execute("DROP TYPE submission_status_old")
