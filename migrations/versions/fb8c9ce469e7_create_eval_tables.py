"""create eval tables

Revision ID: fb8c9ce469e7
Revises: 28845ced45fc
Create Date: 2026-09-28 17:48:01.868577

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fb8c9ce469e7'
down_revision: Union[str, Sequence[str], None] = '28845ced45fc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
        CREATE TABLE eval_runs (
            run_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            dataset_name  text NOT NULL,
            dataset_hash  text NOT NULL,
            agent         text NOT NULL,
            version_tag   text,
            started_at    timestamptz NOT NULL DEFAULT now(),
            ended_at      timestamptz,
            status        text NOT NULL DEFAULT 'running'
                          CHECK (status IN ('running', 'completed', 'failed')),
            metadata      jsonb NOT NULL DEFAULT '{}'
        );
    """)
    op.execute("""
        CREATE TABLE eval_results (
            result_id   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id      uuid NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
            item_id     text NOT NULL,
            trace_id    uuid REFERENCES traces(trace_id) ON DELETE SET NULL,
            input       jsonb,
            expected    jsonb,
            output      jsonb,
            error       text,
            created_at  timestamptz NOT NULL DEFAULT now(),
            UNIQUE (run_id, item_id)
        );
    """)
    op.execute("""
        CREATE TABLE scores (
            score_id   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            result_id  uuid NOT NULL REFERENCES eval_results(result_id) ON DELETE CASCADE,
            scorer     text NOT NULL,
            value      numeric,
            passed     boolean,
            reason     text,
            UNIQUE (result_id, scorer)
        );
    """)
    op.execute("CREATE INDEX eval_runs_version_idx ON eval_runs (version_tag, started_at DESC);")
    op.execute("CREATE INDEX eval_results_trace_id_idx ON eval_results (trace_id);")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE scores;")
    op.execute("DROP TABLE eval_results;")
    op.execute("DROP TABLE eval_runs;")
