"""enforce callback timestamp nullability

Revision ID: 0021_callback_nullability
Revises: 0020_callback_hardening
Create Date: 2026-06-19 11:05:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0021_callback_nullability"
down_revision = "0020_callback_hardening"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "dead_letter_jobs",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.func.now(),
        nullable=False,
    )
    op.alter_column(
        "callback_events",
        "received_at",
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.func.now(),
        nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "callback_events",
        "received_at",
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.func.now(),
        nullable=True,
    )
    op.alter_column(
        "dead_letter_jobs",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.func.now(),
        nullable=True,
    )
