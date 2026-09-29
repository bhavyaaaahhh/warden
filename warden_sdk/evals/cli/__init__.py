"""Usage:
  python -m warden_sdk.evals run DATASET --agent module:function [--trials K] [--version TAG]
  python -m warden_sdk.evals check DATASET --agent module:function [--trials K] [--baseline REF] [--version TAG]
  python -m warden_sdk.evals calibrate DATASET --agent module:function [--trials K]
  python -m warden_sdk.evals diff BASELINE CANDIDATE
  python -m warden_sdk.evals baseline REF
  python -m warden_sdk.evals runs [--dataset NAME]
  python -m warden_sdk.evals rescore REF [--scorer module:function] [--judge]
  python -m warden_sdk.evals export-labels REF --scorer judge -o labels.jsonl
  python -m warden_sdk.evals validate-judge labels.jsonl [--judge module:name]
  python -m warden_sdk.evals rescore REF --labels labels.jsonl     people's labels as a "human" check
  python -m warden_sdk.evals from-traces [TRACE_ID ...] [--agent-name NAME] [--errors-only] -o dataset.jsonl
  python -m warden_sdk.evals generate --about "what the agent does" -n 10 -o scenarios.jsonl
  python -m warden_sdk.evals pairwise BASELINE CANDIDATE [--question "..."]

run, check and calibrate also take:
  --concurrency N           trials to run at once (default 4)
  --scorers a,b             a subset of the built-in scorers
  --scorer module:function  add your own scorer (repeatable)
  --judge                   add the LLM judge (grades items' "criteria"; needs Anthropic credentials)
  --simulator-model M       model playing the user in "scenario" items (default claude-opus-5-5)
  --simulator module:name   your own simulated user
  --faults FILE             run each item with an "environment" again under each tool fault in FILE

REF is a run id or a version tag (the latest completed run with that tag).
check --baseline git compares against the run on the commit your branch forked from.

Exit codes for check, diff and calibrate: 0 no regression, 1 regression,
2 could not decide (too little was scored, a scorer changed, or a run failed).
validate-judge exits 0 if the judge passed validation, 1 if not.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

from warden_sdk.evals.cli.commands import (
    from_traces,
    generate,
    list_runs,
    run_calibrate,
    run_check,
    run_diff,
    run_pairwise,
    set_baseline,
)
from warden_sdk.evals.environment import Fault
from warden_sdk.evals.loading import load_scorer, load_target
from warden_sdk.evals.models import DEFAULT_MODEL, ClaudeModel
from warden_sdk.evals.runner import DEFAULT_CONCURRENCY, run_eval
from warden_sdk.evals.runner.rescore import rescore
from warden_sdk.evals.runner.simulator import UserSimulator
from warden_sdk.evals.scorers import SCORERS
from warden_sdk.evals.scorers.human import human_labels
from warden_sdk.evals.scorers.judge import LLMJudge, llm_judge
from warden_sdk.evals.validation import (
    export_labels,
    print_validation,
    store_validation,
    validate_judge,
)
from warden_sdk.tracer import WARDEN_URL


def _add_run_args(p: argparse.ArgumentParser, trials: int, version: bool = True) -> None:
    p.add_argument("dataset", type=Path, help="path to a .jsonl dataset")
    p.add_argument("--agent", required=True, help="agent entrypoint, e.g. examples.demo_agent:run")
    p.add_argument("--trials", type=int, default=trials, help=f"runs of each item (default {trials})")
    if version:
        p.add_argument("--version", help="version tag (defaults to the tag the agent's traces report)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"trials to run at once (default {DEFAULT_CONCURRENCY})")
    p.add_argument("--faults", type=Path, metavar="FILE",
                   help="JSON list of tool faults; each item with an environment is also run once per fault")
    _add_scorer_args(p)
    p.add_argument("--simulator", dest="simulator_target", metavar="MODULE:NAME",
                   help="your own simulated user: a UserSimulator, or a model to give the built-in one")
    p.add_argument("--simulator-model", default=DEFAULT_MODEL,
                   help=f"model playing the user in scenario items (default {DEFAULT_MODEL})")
    p.add_argument("--simulator-effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                   help="effort for the simulated user (default low)")


def _add_scorer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scorers", help=f"comma-separated subset of the built-in scorers: {', '.join(SCORERS)}")
    p.add_argument("--scorer", action="append", default=[], metavar="MODULE:FUNCTION",
                   help="add a custom scorer (repeatable)")
    p.add_argument("--judge", action="store_true", help="add the LLM judge, which grades items' criteria")
    _add_judge_model_args(p)


def _add_judge_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-model", default=DEFAULT_MODEL, help=f"model for the LLM judge (default {DEFAULT_MODEL})")
    p.add_argument("--judge-effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                   help="effort for the LLM judge (default low)")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m warden_sdk.evals")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="run a dataset and check it against the baseline")
    _add_run_args(check, trials=3)
    check.add_argument("--baseline", help="compare against this run id or version tag, or 'git' for the run "
                                          "where this branch forked from the default branch")

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

    rescore = sub.add_parser("rescore", help="score a stored run again, without re-running the agent")
    rescore.add_argument("ref", help="run id or version tag")
    rescore.add_argument("--version", help="version tag for the new run (default: the original's)")
    rescore.add_argument("--labels", type=Path, help="people's labels for this run's outputs, scored as 'human'")
    _add_scorer_args(rescore)

    export = sub.add_parser("export-labels", help="write a run's judged criteria to a file for people to label")
    export.add_argument("ref", help="run id or version tag")
    export.add_argument("--scorer", default="judge", help="the judge scorer's name (default judge)")
    export.add_argument("-o", "--out", type=Path, required=True, help="JSONL file to write")
    export.add_argument("--per-item", action="store_true", help="one row per item instead of per judged criterion")

    traces = sub.add_parser("from-traces", help="turn recorded traces into dataset items")
    traces.add_argument("trace_ids", nargs="*", help="trace ids (default: the agent's recent traces)")
    traces.add_argument("--agent-name", help="only traces from this agent_name")
    traces.add_argument("--limit", type=int, default=20, help="how many recent traces (default 20)")
    traces.add_argument("--errors-only", action="store_true", help="only traces that ended in an error")
    traces.add_argument("--expect-tools", action="store_true", help="expect the tool calls the trace made")
    traces.add_argument("--as-turns", action="store_true",
                        help="write each as a one-turn conversation, for agents called with messages")
    traces.add_argument("-o", "--out", type=Path, required=True, help="dataset to append to")

    generate = sub.add_parser("generate", help="generate scenario items with an LLM")
    generate.add_argument("--about", required=True, help="what the agent does, in a sentence or two")
    generate.add_argument("-n", type=int, default=10, help="how many scenarios (default 10)")
    generate.add_argument("--topic", action="append", default=[], help="a kind of request to cover (repeatable)")
    generate.add_argument("--persona", action="append", default=[], help="a persona to use (repeatable)")
    generate.add_argument("--max-turns", type=int, default=8)
    generate.add_argument("--model", default=DEFAULT_MODEL, help=f"model that writes them (default {DEFAULT_MODEL})")
    generate.add_argument("-o", "--out", type=Path, required=True, help="dataset to append to")

    pair = sub.add_parser("pairwise", help="judge two runs' conversations side by side")
    pair.add_argument("baseline", help="run id or version tag")
    pair.add_argument("candidate", help="run id or version tag")
    pair.add_argument("--question", default="Which agent handled the user's request better?",
                      help="what the judge should compare")
    pair.add_argument("--judge", dest="judge_target", metavar="MODULE:NAME",
                      help="your own judge model (with `name` and `json()`)")
    _add_judge_model_args(pair)

    validate = sub.add_parser("validate-judge", help="check a judge against human labels")
    validate.add_argument("labels", type=Path, help="labelled JSONL, e.g. from export-labels")
    validate.add_argument("--judge", dest="judge_target", metavar="MODULE:NAME",
                          help="a judge built with llm_judge() (default: the built-in judge)")
    _add_judge_model_args(validate)
    validate.add_argument("--repeats", type=int, default=3, help="times to judge each row (default 3)")
    validate.add_argument("--no-store", action="store_true", help="don't save the result to the server")

    args = parser.parse_args()
    try:
        return _dispatch(parser, args)
    except httpx.ConnectError:
        print(f"warden: cannot reach the server at {WARDEN_URL} (is it running? set WARDEN_URL?)", file=sys.stderr)
        return 2
    except SystemExit as e:
        # Commands stop with a message when they can't go on (no such run, no git baseline).
        # That's "could not decide", exit 2, never "regression".
        if isinstance(e.code, str):
            print(f"warden: {e.code}", file=sys.stderr)
            return 2
        raise


def _scorers(parser: argparse.ArgumentParser, subset: str | None, custom: list[str], judge: LLMJudge | None):
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
    if judge is not None:
        if judge.name in scorers:
            parser.error(f"--judge: a scorer named {judge.name!r} already exists")
        scorers[judge.name] = judge
    return scorers


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.command == "diff":
        return run_diff(args.baseline, args.candidate)
    if args.command == "baseline":
        return set_baseline(args.ref)
    if args.command == "runs":
        return list_runs(args.dataset, args.limit)
    if args.command == "rescore":
        scorers = _scorers(parser, args.scorers, args.scorer, _judge(args) if args.judge else None)
        if args.labels:
            try:
                labels = human_labels(args.labels)
            except (OSError, ValueError) as e:
                parser.error(f"--labels: {e}")
            scorers[labels.name] = labels
        rescore(args.ref, scorers, version_tag=args.version)
        return 0
    if args.command == "export-labels":
        return export_labels(args.ref, args.scorer, args.out, per_item=args.per_item)
    if args.command == "from-traces":
        return from_traces(args.trace_ids, args.agent_name, args.limit, args.errors_only, args.expect_tools,
                           args.as_turns, args.out)
    if args.command == "generate":
        return generate(args.about, args.n, args.topic, args.persona, args.max_turns, args.model, args.out)
    if args.command == "pairwise":
        model = (load_target(args.judge_target, "--judge") if args.judge_target
                 else ClaudeModel(args.judge_model, args.judge_effort, lazy=True))
        return run_pairwise(args.baseline, args.candidate, args.question, model)
    if args.command == "validate-judge":
        judge = load_target(args.judge_target, "--judge") if args.judge_target else _judge(args)
        if not isinstance(judge, LLMJudge):
            parser.error("--judge must name a judge built with llm_judge()")
        report = validate_judge(args.labels, judge, repeats=args.repeats)
        if not args.no_store:
            store_validation(report)
        return print_validation(report)

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    scorers = _scorers(parser, args.scorers, args.scorer, _judge(args) if args.judge else None)
    simulator = _simulator(parser, args)
    faults = _faults(parser, args.faults)
    common = {"scorers": scorers, "trials": args.trials, "concurrency": args.concurrency, "simulator": simulator,
              "faults": faults}
    if args.command == "calibrate":
        return run_calibrate(args.dataset, args.agent, **common)
    if args.command == "check":
        return run_check(args.dataset, args.agent, version_tag=args.version, baseline_ref=args.baseline, **common)
    run_eval(args.dataset, args.agent, version_tag=args.version, **common)
    return 0


def _judge(args: argparse.Namespace) -> LLMJudge:
    return llm_judge(model=ClaudeModel(args.judge_model, args.judge_effort, lazy=True))


def _simulator(parser: argparse.ArgumentParser, args: argparse.Namespace) -> UserSimulator:
    if not args.simulator_target:
        return UserSimulator(ClaudeModel(args.simulator_model, args.simulator_effort, lazy=True))
    target = load_target(args.simulator_target, "--simulator")
    if isinstance(target, UserSimulator):
        return target
    if callable(getattr(target, "json", None)) and hasattr(target, "name"):
        return UserSimulator(target)
    parser.error("--simulator must name a UserSimulator or a model with `name` and `json()`")


def _faults(parser: argparse.ArgumentParser, path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    try:
        faults = json.loads(path.read_text())
        if not isinstance(faults, list):
            raise ValueError("expected a JSON list of faults")
        for f in faults:
            Fault.parse(f)
    except (OSError, ValueError) as e:
        parser.error(f"--faults {path}: {e}")
    return faults
