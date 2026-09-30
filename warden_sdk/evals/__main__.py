"""Usage:
  python -m warden_sdk.evals check DATASET --agent module:function [--trials K] [--baseline REF] [--version TAG]
  python -m warden_sdk.evals run DATASET --agent module:function [--trials K] [--version TAG] [--scorers a,b]
  python -m warden_sdk.evals calibrate DATASET --agent module:function [--trials K]

run, check and calibrate also take --concurrency N (trials run at once, default 4)
and --scorer module:function (repeatable) to add your own scorers.
  python -m warden_sdk.evals diff BASELINE CANDIDATE
  python -m warden_sdk.evals baseline REF
  python -m warden_sdk.evals runs [--dataset NAME]

REF is a run id or a version tag (the latest completed run with that tag).

Exit codes for check, diff and calibrate: 0 no regression, 1 regression,
2 could not decide (too little was scored, a scorer changed, or a run failed).
"""

import argparse
import sys
from pathlib import Path

import httpx

from warden_sdk.evals.commands import list_runs, run_calibrate, run_check, set_baseline
from warden_sdk.evals.diff import run_diff
from warden_sdk.evals.runner import DEFAULT_CONCURRENCY, load_scorer, run_eval
from warden_sdk.evals.scorers import SCORERS
from warden_sdk.tracer import WARDEN_URL


def _add_run_args(p: argparse.ArgumentParser, trials: int, version: bool = True) -> None:
    p.add_argument("dataset", type=Path, help="path to a .jsonl dataset")
    p.add_argument("--agent", required=True, help="agent entrypoint, e.g. examples.demo_agent:run")
    p.add_argument("--trials", type=int, default=trials, help=f"runs of each item (default {trials})")
    if version:
        p.add_argument("--version", help="version tag (defaults to the tag the agent's traces report)")
    p.add_argument("--scorers", help=f"comma-separated subset of the built-in scorers: {', '.join(SCORERS)}")
    p.add_argument("--scorer", action="append", default=[], metavar="MODULE:FUNCTION",
                   help="add a custom scorer (repeatable)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"trials to run at once (default {DEFAULT_CONCURRENCY})")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m warden_sdk.evals")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="run a dataset and check it against the baseline")
    _add_run_args(check, trials=3)
    check.add_argument("--baseline", help="compare against this run id or version tag instead")

    run = sub.add_parser("run", help="run a dataset through an agent and score it")
    _add_run_args(run, trials=1)

    calibrate = sub.add_parser("calibrate", help="run the same agent twice to measure run-to-run noise")
    _add_run_args(calibrate, trials=3, version=False)

    diff = sub.add_parser("diff", help="compare a candidate run against a baseline")
    diff.add_argument("baseline", help="run id or version tag")
    diff.add_argument("candidate", help="run id or version tag")

    baseline = sub.add_parser("baseline", help="make a run the baseline for its dataset + agent")
    baseline.add_argument("ref", help="run id or version tag")

    runs = sub.add_parser("runs", help="list recent eval runs")
    runs.add_argument("--dataset", help="only runs of this dataset")
    runs.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    try:
        return _dispatch(parser, args)
    except httpx.ConnectError:
        print(f"warden: cannot reach the server at {WARDEN_URL} (is it running? set WARDEN_URL?)", file=sys.stderr)
        return 2


def _scorers(parser: argparse.ArgumentParser, subset: str | None, custom: list[str]):
    scorers = dict(SCORERS)
    if subset:
        names = [n.strip() for n in subset.split(",")]
        unknown = [n for n in names if n not in SCORERS]
        if unknown:
            parser.error(f"unknown scorers: {', '.join(unknown)}")
        scorers = {n: SCORERS[n] for n in names}
    for target in custom:
        try:
            name, scorer = load_scorer(target)
        except (ValueError, ImportError, AttributeError) as e:
            parser.error(str(e))
        if name in scorers:
            parser.error(f"--scorer {target}: a scorer named {name!r} already exists")
        scorers[name] = scorer
    return scorers


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.command == "diff":
        return run_diff(args.baseline, args.candidate)
    if args.command == "baseline":
        return set_baseline(args.ref)
    if args.command == "runs":
        return list_runs(args.dataset, args.limit)

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    scorers = _scorers(parser, args.scorers, args.scorer)
    common = {"scorers": scorers, "trials": args.trials, "concurrency": args.concurrency}
    if args.command == "calibrate":
        return run_calibrate(args.dataset, args.agent, **common)
    if args.command == "check":
        return run_check(args.dataset, args.agent, version_tag=args.version, baseline_ref=args.baseline, **common)
    run_eval(args.dataset, args.agent, version_tag=args.version, **common)
    return 0


if __name__ == "__main__":
    sys.exit(main())
