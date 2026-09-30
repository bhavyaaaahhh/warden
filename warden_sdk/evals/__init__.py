"""Offline evals for agents: datasets, scorers, trials, and noise-aware regression checks.

Layout:
  dataset.py   loading JSONL datasets
  runner/      calling the agent (harness) and running a whole eval (run)
  scorers/     Case/Score, built-in checks and metrics, the LLM judge
  compare/     comparing two runs (compare), statistics (stats), the report
  models.py    the LLM behind judges
  validation.py  checking a judge against human labels
  cli/         the `python -m warden_sdk.evals` commands
"""

from warden_sdk.evals.runner import EvalContext, InfraError, run_eval
from warden_sdk.evals.scorers import SCORERS, Case, Score, Scorer
from warden_sdk.evals.scorers.judge import llm_judge

__all__ = ["SCORERS", "Case", "EvalContext", "InfraError", "Score", "Scorer", "llm_judge", "run_eval"]
