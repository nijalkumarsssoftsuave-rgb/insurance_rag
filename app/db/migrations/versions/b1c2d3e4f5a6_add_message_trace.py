"""Add messages.trace - the replay bundle for a single turn.

Nullable and additive: every row written before this migration keeps its audit
fields and simply has no replay bundle. Backfilling it would mean inventing
model parameters that were never recorded, and an audit trail with fabricated
values is worse than one with an honest gap.

Revision ID: b1c2d3e4f5a6
Revises: 20907b7121db
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b1c2d3e4f5a6"
down_revision = "20907b7121db"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("trace", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "trace")
