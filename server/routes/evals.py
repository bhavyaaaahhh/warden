from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from server.db import jsonb, pool
from server.routes.judges import judge_warnings
from server.schemas import EvalResultIn, EvalRunIn, EvalRunStatus, EvalRunUpdate
from warden_sdk.evals.compare import METHOD_VERSION, compare_runs

router = APIRouter(prefix="/eval_runs", tags=["evals"])

# The current baseline for a run's dataset + agent, excluding the run itself.
BASELINE_FOR_RUN = """
    SELECT b.run_id FROM eval_runs b
    WHERE b.dataset_name = r.dataset_name AND b.agent = r.agent
      AND b.is_baseline AND b.run_id <> r.run_id
"""


@router.post("", status_code=201)
def create_eval_run(run: EvalRunIn):
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO eval_runs
                (run_id, dataset_name, dataset_hash, agent, version_tag, trials, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run.run_id,
                run.dataset_name,
                run.dataset_hash,
                run.agent,
                run.version_tag,
                run.trials,
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
                    (result_id, run_id, item_id, trial, case_hash, termination, trace_id,
                     input, expected, output, error, transcript, turns)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    result.result_id,
                    run_id,
                    result.item_id,
                    result.trial,
                    result.case_hash,
                    result.termination,
                    result.trace_id,
                    jsonb(result.input),
                    jsonb(result.expected),
                    jsonb(result.output),
                    result.error,
                    jsonb(result.transcript),
                    jsonb(result.turns),
                ),
            )
            if result.scores:
                cur.executemany(
                    """
                    INSERT INTO scores
                        (result_id, scorer, criterion, value, passed, outcome, reason, scorer_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (result.result_id, s.scorer, s.criterion, s.value, s.passed, s.outcome, s.reason,
                         s.scorer_version)
                        for s in result.scores
                    ],
                )
    return {"result_id": result.result_id}


@router.put("/{run_id}/baseline")
def set_baseline(run_id: UUID):
    # Replaces any existing baseline for the same dataset + agent.
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT dataset_name, agent, status FROM eval_runs WHERE run_id = %s FOR UPDATE",
                (run_id,),
            )
            run = cur.fetchone()
            if run is None:
                raise HTTPException(status_code=404, detail="eval run not found")
            if run["status"] != "completed":
                raise HTTPException(status_code=409, detail="only a completed run can be a baseline")
            cur.execute(
                """
                UPDATE eval_runs SET is_baseline = false
                WHERE dataset_name = %s AND agent = %s AND is_baseline
                """,
                (run["dataset_name"], run["agent"]),
            )
            cur.execute("UPDATE eval_runs SET is_baseline = true WHERE run_id = %s", (run_id,))
    return {"run_id": run_id, "is_baseline": True}


@router.delete("/{run_id}/baseline")
def clear_baseline(run_id: UUID):
    with pool.connection() as conn:
        cur = conn.execute("UPDATE eval_runs SET is_baseline = false WHERE run_id = %s", (run_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="eval run not found")
    return {"run_id": run_id, "is_baseline": False}


@router.get("")
def list_eval_runs(
    version_tag: str | None = None,
    status: EvalRunStatus | None = None,
    dataset_name: str | None = None,
    agent: str | None = None,
    baseline: bool | None = None,
    limit: int = 50,
    verdicts: bool = True,
):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT r.*,
                    (SELECT count(DISTINCT res.item_id) FROM eval_results res WHERE res.run_id = r.run_id) AS item_count,
                    (SELECT coalesce(jsonb_object_agg(scorer, jsonb_build_object(
                                'passed', passed, 'total', total)), '{{}}')
                     FROM (SELECT s.scorer,
                                  count(*) FILTER (WHERE s.passed) AS passed,
                                  count(*) AS total
                           FROM scores s JOIN eval_results res USING (result_id)
                           WHERE res.run_id = r.run_id AND s.passed IS NOT NULL
                           GROUP BY s.scorer) x) AS checks,
                    bl.run_id AS baseline_run_id
                FROM eval_runs r
                LEFT JOIN LATERAL ({BASELINE_FOR_RUN}) bl ON true
                WHERE (%(version_tag)s::text IS NULL OR r.version_tag = %(version_tag)s)
                  AND (%(status)s::text IS NULL OR r.status = %(status)s)
                  AND (%(dataset_name)s::text IS NULL OR r.dataset_name = %(dataset_name)s)
                  AND (%(agent)s::text IS NULL OR r.agent = %(agent)s)
                  AND (%(baseline)s::boolean IS NULL OR r.is_baseline = %(baseline)s)
                ORDER BY r.started_at DESC
                LIMIT %(limit)s
                """,
                {
                    "version_tag": version_tag,
                    "status": status,
                    "dataset_name": dataset_name,
                    "agent": agent,
                    "baseline": baseline,
                    "limit": min(limit, 500),
                },
            )
            runs = cur.fetchall()
    for r in runs:
        r["vs_baseline"] = None
    if verdicts:
        _add_verdicts(runs)
    return runs


def _add_verdicts(runs: list[dict]) -> None:
    """Attach each run's verdict against its baseline.

    The verdict comes from compare_runs, not SQL, so the list can't disagree
    with `check`. Computing it is slow, so it's cached per (baseline, candidate).
    """
    pending = [r for r in runs if r["baseline_run_id"] and not r["is_baseline"] and r["status"] == "completed"]
    if not pending:
        return
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT baseline_run_id, candidate_run_id, verdict, broke FROM eval_comparisons
                WHERE method = %s AND candidate_run_id = ANY(%s)
                """,
                (METHOD_VERSION, [r["run_id"] for r in pending]),
            )
            cached = {(c["baseline_run_id"], c["candidate_run_id"]): c for c in cur.fetchall()}
    fetched: dict = {}

    def load(run_id: UUID) -> dict:
        if run_id not in fetched:
            fetched[run_id] = jsonable_encoder(get_eval_run(run_id))
        return fetched[run_id]

    for r in pending:
        hit = cached.get((r["baseline_run_id"], r["run_id"]))
        if hit is None:
            baseline = load(r["baseline_run_id"])
            if baseline["status"] != "completed":
                continue
            cmp = compare_runs(baseline, load(r["run_id"]), intervals=False)
            hit = {"verdict": cmp["verdict"], "broke": sum(t["label"] == "broke" for t in cmp["transitions"])}
            with pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO eval_comparisons (baseline_run_id, candidate_run_id, method, verdict, broke)
                    VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
                    """,
                    (r["baseline_run_id"], r["run_id"], METHOD_VERSION, hit["verdict"], hit["broke"]),
                )
        r["vs_baseline"] = {"verdict": hit["verdict"], "broke": hit["broke"]}


@router.get("/compare")
def compare_eval_runs(baseline: UUID, candidate: UUID):
    # jsonable_encoder turns Decimal/UUID/datetime into the same JSON types the CLI sees.
    a = jsonable_encoder(get_eval_run(baseline))
    b = jsonable_encoder(get_eval_run(candidate))
    cmp = compare_runs(a, b)
    cmp["warnings"] += judge_warnings([a, b])
    return cmp


@router.get("/{run_id}")
def get_eval_run(run_id: UUID):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT r.*, bl.run_id AS baseline_run_id
                FROM eval_runs r LEFT JOIN LATERAL ({BASELINE_FOR_RUN}) bl ON true
                WHERE r.run_id = %s
                """,
                (run_id,),
            )
            run = cur.fetchone()
            if run is None:
                raise HTTPException(status_code=404, detail="eval run not found")
            cur.execute(
                "SELECT * FROM eval_results WHERE run_id = %s ORDER BY created_at, trial",
                (run_id,),
            )
            results = cur.fetchall()
            cur.execute(
                """
                SELECT s.result_id, s.scorer, s.criterion, s.value, s.passed, s.outcome, s.reason,
                       s.scorer_version
                FROM scores s JOIN eval_results r USING (result_id)
                WHERE r.run_id = %s
                ORDER BY s.scorer, s.criterion
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
