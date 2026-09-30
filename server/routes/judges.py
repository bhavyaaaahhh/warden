from fastapi import APIRouter
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from server.db import pool
from server.schemas import JudgeValidationIn
from warden_sdk.evals.validation import judge_versions, validation_warning

router = APIRouter(prefix="/judge_validations", tags=["judges"])


@router.post("", status_code=201)
def create_judge_validation(v: JudgeValidationIn):
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO judge_validations
                (scorer_version, labels, compared, kappa, kappa_ci, confusion,
                 flip_rate, repeats, unscored, validated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (v.scorer_version, v.labels, v.compared, v.kappa, Jsonb(v.kappa_ci), Jsonb(v.confusion),
             v.flip_rate, v.repeats, v.unscored, v.validated),
        )
    return {"scorer_version": v.scorer_version, "validated": v.validated}


@router.get("")
def latest_judge_validation(scorer_version: str):
    """The most recent validation of this judge version, or null if it was never validated."""
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM judge_validations WHERE scorer_version = %s ORDER BY created_at DESC LIMIT 1",
                (scorer_version,),
            )
            return cur.fetchone()


def judge_warnings(runs: list[dict]) -> list[str]:
    """Warnings for judge scorers in these runs that haven't passed validation."""
    warnings = [validation_warning(v, latest_judge_validation(v)) for v in judge_versions(runs)]
    return [w for w in warnings if w]
