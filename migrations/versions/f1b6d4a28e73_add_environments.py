"""add environments

Revision ID: f1b6d4a28e73
Revises: e5a3c1f09d62
Create Date: 2026-09-29 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1b6d4a28e73'
down_revision: Union[str, Sequence[str], None] = 'e5a3c1f09d62'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Items with an environment: initial and final state, and every tool call made through it.
    op.execute("ALTER TABLE eval_results ADD COLUMN environment jsonb;")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("ALTER TABLE eval_results DROP COLUMN environment;")
