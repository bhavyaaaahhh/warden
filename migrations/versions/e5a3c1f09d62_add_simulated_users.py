"""add simulated users

Revision ID: e5a3c1f09d62
Revises: d7f2b8e41c09
Create Date: 2026-09-29 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5a3c1f09d62'
down_revision: Union[str, Sequence[str], None] = 'd7f2b8e41c09'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Scenario items: which simulated user ran, and why the conversation ended.
    op.execute("ALTER TABLE eval_results ADD COLUMN simulation jsonb;")
    # A simulated conversation can end by running out of turns; that's an outcome, not an error.
    op.execute("ALTER TABLE eval_results DROP CONSTRAINT eval_results_termination_check;")
    op.execute("""
        ALTER TABLE eval_results ADD CONSTRAINT eval_results_termination_check
        CHECK (termination IN ('completed', 'agent_error', 'infra_error', 'max_turns'));
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("UPDATE eval_results SET termination = 'completed' WHERE termination = 'max_turns';")
    op.execute("ALTER TABLE eval_results DROP CONSTRAINT eval_results_termination_check;")
    op.execute("""
        ALTER TABLE eval_results ADD CONSTRAINT eval_results_termination_check
        CHECK (termination IN ('completed', 'agent_error', 'infra_error'));
    """)
    op.execute("ALTER TABLE eval_results DROP COLUMN simulation;")
