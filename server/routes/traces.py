from uuid import UUID

from fastapi import APIRouter, HTTPException
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from server.db import jsonb, pool
from server.schemas import SpanIn, TraceIn

router = APIRouter(tags=["traces"])


@router.post("/traces", status_code=201)
def create_trace(trace: TraceIn):
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO traces
                (trace_id, agent_name, version_tag, input,
                 started_at, ended_at, status, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                trace.trace_id,
                trace.agent_name,
                trace.version_tag,
                jsonb(trace.input),
                trace.started_at,
                trace.ended_at,
                trace.status,
                Jsonb(trace.metadata),
            ),
        )
    return {"trace_id": trace.trace_id}


@router.post("/spans", status_code=201)
def create_spans(spans: list[SpanIn]):
    # Callers must send spans sorted by started_at so parents land before
    # children (parent_span_id FK). One transaction: all spans or none.
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO spans
                    (span_id, trace_id, parent_span_id, span_type, name,
                     input, output, started_at, ended_at,
                     tokens_input, tokens_output, cost_usd, error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        s.span_id,
                        s.trace_id,
                        s.parent_span_id,
                        s.span_type,
                        s.name,
                        jsonb(s.input),
                        jsonb(s.output),
                        s.started_at,
                        s.ended_at,
                        s.tokens_input,
                        s.tokens_output,
                        s.cost_usd,
                        s.error,
                    )
                    for s in spans
                ],
            )
    return {"inserted": len(spans)}


@router.get("/traces")
def list_traces(agent_name: str | None = None, limit: int = 100):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT t.trace_id, t.agent_name, t.version_tag, t.status,
                       t.input, t.started_at, t.ended_at,
                       t.metadata->>'eval_run_id' AS eval_run_id,
                       count(s.span_id) AS span_count,
                       sum(s.tokens_input) AS tokens_input,
                       sum(s.tokens_output) AS tokens_output,
                       sum(s.cost_usd) AS cost_usd
                FROM traces t
                LEFT JOIN spans s USING (trace_id)
                WHERE %(agent_name)s::text IS NULL OR t.agent_name = %(agent_name)s
                GROUP BY t.trace_id
                ORDER BY t.started_at DESC
                LIMIT %(limit)s
                """,
                {"agent_name": agent_name, "limit": min(limit, 500)},
            )
            return cur.fetchall()


@router.get("/traces/{trace_id}")
def get_trace(trace_id: UUID):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM traces WHERE trace_id = %s", (trace_id,))
            trace = cur.fetchone()
            if trace is None:
                raise HTTPException(status_code=404, detail="trace not found")
            cur.execute(
                "SELECT * FROM spans WHERE trace_id = %s ORDER BY started_at",
                (trace_id,),
            )
            trace["spans"] = cur.fetchall()
            return trace
