from uuid import UUID

from fastapi import APIRouter, HTTPException
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from server.db import jsonb, pool
from server.schemas import EvalResultIn, EvalRunIn, EvalRunStatus, EvalRunUpdate

router = APIRouter(prefix="/eval_runs", tags=["evals"])


@router.post("", status_code=201)
def create_eval_run(run: EvalRunIn):
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO eval_runs
                (run_id, dataset_name, dataset_hash, agent, version_tag, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                run.run_id,
                run.dataset_name,
                run.dataset_hash,
                run.agent,
                run.version_tag,
                Jsonb(run.metadata),
            ),
        )
    return {"run_id": run.run_id}


@router.patch("/{run_id}")
def update_eval_run(run_id: UUID, update: EvalRunUpdate):
    with pool.connection() as conn:
        cur = conn.execute(
            """
            UPDATE eval_runs
            SET status = %(status)s,
                version_tag = COALESCE(%(version_tag)s, version_tag),
                ended_at = CASE WHEN %(status)s = 'running' THEN NULL ELSE now() END
            WHERE run_id = %(run_id)s
            """,
            {"status": update.status, "version_tag": update.version_tag, "run_id": run_id},
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="eval run not found")
    return {"run_id": run_id, "status": update.status}


@router.post("/{run_id}/results", status_code=201)
def create_eval_result(run_id: UUID, result: EvalResultIn):
    # One item's result and all its scores in one transaction.
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO eval_results
                    (result_id, run_id, item_id, trace_id, input, expected, output, error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    result.result_id,
                    run_id,
                    result.item_id,
                    result.trace_id,
                    jsonb(result.input),
                    jsonb(result.expected),
                    jsonb(result.output),
                    result.error,
                ),
            )
            if result.scores:
                cur.executemany(
                    """
                    INSERT INTO scores (result_id, scorer, value, passed, reason)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    [(result.result_id, s.scorer, s.value, s.passed, s.reason) for s in result.scores],
                )
    return {"result_id": result.result_id}


@router.get("")
def list_eval_runs(
    version_tag: str | None = None,
    status: EvalRunStatus | None = None,
    limit: int = 50,
):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT r.*, count(res.result_id) AS item_count
                FROM eval_runs r
                LEFT JOIN eval_results res USING (run_id)
                WHERE (%(version_tag)s::text IS NULL OR r.version_tag = %(version_tag)s)
                  AND (%(status)s::text IS NULL OR r.status = %(status)s)
                GROUP BY r.run_id
                ORDER BY r.started_at DESC
                LIMIT %(limit)s
                """,
                {"version_tag": version_tag, "status": status, "limit": min(limit, 500)},
            )
            return cur.fetchall()


@router.get("/{run_id}")
def get_eval_run(run_id: UUID):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM eval_runs WHERE run_id = %s", (run_id,))
            run = cur.fetchone()
            if run is None:
                raise HTTPException(status_code=404, detail="eval run not found")
            cur.execute(
                "SELECT * FROM eval_results WHERE run_id = %s ORDER BY created_at",
                (run_id,),
            )
            results = cur.fetchall()
            cur.execute(
                """
                SELECT s.result_id, s.scorer, s.value, s.passed, s.reason
                FROM scores s JOIN eval_results r USING (result_id)
                WHERE r.run_id = %s
                ORDER BY s.scorer
                """,
                (run_id,),
            )
            scores_by_result: dict = {}
            for s in cur.fetchall():
                scores_by_result.setdefault(s.pop("result_id"), []).append(s)
            for r in results:
                r["scores"] = scores_by_result.get(r["result_id"], [])
            run["results"] = results
            return run
