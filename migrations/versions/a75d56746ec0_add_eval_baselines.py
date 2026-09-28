"""add eval baselines

Revision ID: a75d56746ec0
Revises: fb8c9ce469e7
Create Date: 2026-09-28 18:05:53.963423

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a75d56746ec0'
down_revision: Union[str, Sequence[str], None] = 'fb8c9ce469e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TABLE eval_runs ADD COLUMN is_baseline boolean NOT NULL DEFAULT false;")
    # At most one baseline per dataset + agent.
    op.execute("""
        CREATE UNIQUE INDEX eval_runs_one_baseline_idx
        ON eval_runs (dataset_name, agent) WHERE is_baseline;
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX eval_runs_one_baseline_idx;")
    op.execute("ALTER TABLE eval_runs DROP COLUMN is_baseline;")
