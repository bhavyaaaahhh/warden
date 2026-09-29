"""add trials, outcomes and multi-turn results

Revision ID: c3e1a9d27b40
Revises: a75d56746ec0
Create Date: 2026-09-29 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3e1a9d27b40'
down_revision: Union[str, Sequence[str], None] = 'a75d56746ec0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TABLE eval_runs ADD COLUMN trials integer NOT NULL DEFAULT 1 CHECK (trials >= 1);")

    # One result per item per trial; existing results are trial 0.
    op.execute("""
        ALTER TABLE eval_results
            ADD COLUMN trial       integer NOT NULL DEFAULT 0,
            ADD COLUMN case_hash   text,
            ADD COLUMN termination text NOT NULL DEFAULT 'completed'
                       CHECK (termination IN ('completed', 'agent_error', 'infra_error')),
            ADD COLUMN transcript  jsonb,
            ADD COLUMN turns       jsonb;
    """)
    op.execute("ALTER TABLE eval_results DROP CONSTRAINT eval_results_run_id_item_id_key;")
    op.execute("ALTER TABLE eval_results ADD UNIQUE (run_id, item_id, trial);")

    # pass/fail/unscored for checks, NULL for metrics. Backfill from passed.
    op.execute("""
        ALTER TABLE scores
            ADD COLUMN outcome        text CHECK (outcome IN ('pass', 'fail', 'unscored')),
            ADD COLUMN criterion      text NOT NULL DEFAULT '',
            ADD COLUMN scorer_version text;
    """)
    op.execute("UPDATE scores SET outcome = CASE WHEN passed THEN 'pass' ELSE 'fail' END WHERE passed IS NOT NULL;")
    op.execute("ALTER TABLE scores DROP CONSTRAINT scores_result_id_scorer_key;")
    op.execute("ALTER TABLE scores ADD UNIQUE (result_id, scorer, criterion);")

    # Verdicts cached for the runs list. compare_runs is deterministic and
    # completed runs don't change, so a cached verdict is exact; `method`
    # changes when the comparison does.
    op.execute("""
        CREATE TABLE eval_comparisons (
            baseline_run_id  uuid NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
            candidate_run_id uuid NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
            method           text NOT NULL,
            verdict          text NOT NULL,
            broke            integer NOT NULL,
            PRIMARY KEY (baseline_run_id, candidate_run_id, method)
        );
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE eval_comparisons;")
    # Keeps only trial 0 and one score per scorer, so the old unique keys hold again.
    op.execute("DELETE FROM eval_results WHERE trial <> 0;")
    op.execute("DELETE FROM scores WHERE criterion <> '';")
    op.execute("ALTER TABLE scores DROP CONSTRAINT scores_result_id_scorer_criterion_key;")
    op.execute("ALTER TABLE scores ADD UNIQUE (result_id, scorer);")
    op.execute("ALTER TABLE scores DROP COLUMN outcome, DROP COLUMN criterion, DROP COLUMN scorer_version;")
    op.execute("ALTER TABLE eval_results DROP CONSTRAINT eval_results_run_id_item_id_trial_key;")
    op.execute("ALTER TABLE eval_results ADD UNIQUE (run_id, item_id);")
    op.execute("""
        ALTER TABLE eval_results
            DROP COLUMN trial, DROP COLUMN case_hash, DROP COLUMN termination,
            DROP COLUMN transcript, DROP COLUMN turns;
    """)
    op.execute("ALTER TABLE eval_runs DROP COLUMN trials;")
