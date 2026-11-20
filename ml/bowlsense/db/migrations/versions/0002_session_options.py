"""session options and original file name

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-20 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply this migration."""
    with op.batch_alter_table("sessions") as batch:
        batch.add_column(sa.Column("options", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("original_name", sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Revert this migration."""
    with op.batch_alter_table("sessions") as batch:
        batch.drop_column("original_name")
        batch.drop_column("options")
