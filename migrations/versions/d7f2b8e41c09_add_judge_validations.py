"""add judge validations

Revision ID: d7f2b8e41c09
Revises: c3e1a9d27b40
Create Date: 2026-09-29 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd7f2b8e41c09'
down_revision: Union[str, Sequence[str], None] = 'c3e1a9d27b40'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # How well a judge version agreed with human labels. Kept as history;
    # the latest row per version is the one that counts.
    op.execute("""
        CREATE TABLE judge_validations (
            validation_id  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            scorer_version text NOT NULL,
            labels         integer NOT NULL,
            compared       integer NOT NULL,
            kappa          double precision,
            kappa_ci       jsonb,
            confusion      jsonb NOT NULL DEFAULT '{}',
            flip_rate      double precision,
            repeats        integer NOT NULL,
            unscored       integer NOT NULL,
            validated      boolean NOT NULL,
            created_at     timestamptz NOT NULL DEFAULT now()
        );
    """)
    op.execute("CREATE INDEX judge_validations_version_idx ON judge_validations (scorer_version, created_at DESC);")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE judge_validations;")
