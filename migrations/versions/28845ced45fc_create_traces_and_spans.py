"""create traces and spans

Revision ID: 28845ced45fc
Revises: 
Create Date: 2026-09-28 17:17:09.945884

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '28845ced45fc'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
        CREATE TABLE traces (
            trace_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_name   text NOT NULL,
            version_tag  text,
            input        jsonb,
            started_at   timestamptz NOT NULL DEFAULT now(),
            ended_at     timestamptz,
            status       text NOT NULL CHECK (status IN ('running', 'success', 'error')),
            metadata     jsonb NOT NULL DEFAULT '{}'
        );
    """)
    op.execute("""
        CREATE TABLE spans (
            span_id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            trace_id        uuid NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
            parent_span_id  uuid REFERENCES spans(span_id) ON DELETE CASCADE,
            span_type       text NOT NULL CHECK (span_type IN ('llm_call', 'tool_call', 'retrieval')),
            name            text NOT NULL,
            input           jsonb,
            output          jsonb,
            started_at      timestamptz NOT NULL,
            ended_at        timestamptz,
            tokens_input    integer,
            tokens_output   integer,
            cost_usd        numeric(12, 6),
            error           text
        );
    """)
    op.execute("CREATE INDEX spans_trace_id_idx ON spans (trace_id);")
    op.execute("CREATE INDEX spans_parent_span_id_idx ON spans (parent_span_id);")
    op.execute("CREATE INDEX traces_agent_version_idx ON traces (agent_name, version_tag, started_at DESC);")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE spans;")
    op.execute("DROP TABLE traces;")
