"""Usage:
  python -m warden_sdk.evals run DATASET --agent module:function [--version TAG] [--scorers a,b]
  python -m warden_sdk.evals diff BASELINE CANDIDATE     (run ids or version tags)
"""

import argparse
import sys
from pathlib import Path

import httpx

from warden_sdk.evals.diff import run_diff
from warden_sdk.evals.runner import run_eval
from warden_sdk.evals.scorers import SCORERS
from warden_sdk.tracer import WARDEN_URL


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m warden_sdk.evals")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a dataset through an agent and score it")
    run.add_argument("dataset", type=Path, help="path to a .jsonl dataset")
    run.add_argument("--agent", required=True, help="agent entrypoint, e.g. examples.demo_agent:run")
    run.add_argument("--version", help="version tag (defaults to the tag the agent's traces report)")
    run.add_argument("--scorers", help=f"comma-separated subset of: {', '.join(SCORERS)}")

    diff = sub.add_parser("diff", help="compare a candidate run against a baseline")
    diff.add_argument("baseline", help="run id or version tag")
    diff.add_argument("candidate", help="run id or version tag")

    args = parser.parse_args()
    try:
        return _dispatch(parser, args)
    except httpx.ConnectError:
        print(f"warden: cannot reach the server at {WARDEN_URL} (is it running? set WARDEN_URL?)", file=sys.stderr)
        return 2


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.command == "diff":
        return run_diff(args.baseline, args.candidate)

    scorers = SCORERS
    if args.scorers:
        names = [n.strip() for n in args.scorers.split(",")]
        unknown = [n for n in names if n not in SCORERS]
        if unknown:
            parser.error(f"unknown scorers: {', '.join(unknown)}")
        scorers = {n: SCORERS[n] for n in names}
    run_eval(args.dataset, args.agent, version_tag=args.version, scorers=scorers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
