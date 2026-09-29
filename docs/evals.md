# Offline evals

Run a dataset through your agent, score it, and find out whether a change made it worse, better, or no different from run-to-run noise.

Everything here is a command you run when you want it. Nothing is wired into CI. Exit codes are meaningful if you want to wire them in yourself: `0` no regression, `1` regression, `2` couldn't decide.

```bash
docker compose up -d && uv run alembic upgrade head && uv run uvicorn server.app:app --reload
uv run python -m warden_sdk.evals check examples/datasets/support.jsonl --agent examples.support_agent:chat
```

The first `check` saves its run as the baseline; later ones are compared against it. The viewer at http://localhost:8000 shows runs, comparisons and every conversation.

## Datasets

A dataset is a JSONL file, one item per line. Each item has an `id` that stays the same when you edit it, and exactly one of:

| Shape | Use it for |
|---|---|
| `"input": ...` | one call: `agent(input)` |
| `"turns": [{"user": ...}, {"expect": {...}}, ...]` | a scripted conversation: `agent(messages)` each turn |
| `"scenario": {...}` | a conversation with a simulated user ([below](#simulated-users)) |

`"expected"` says what to check for the whole item; an `{"expect": ...}` step after a user turn checks the reply to that turn.

| Key | Checked by | Meaning |
|---|---|---|
| `contains`, `exact` | `contains`, `exact_match`, `turn_expectations` | text in the reply |
| `tools`, `tools_match` | `tool_calls`, `turn_expectations` | tools called: names, or `{"name", "args"}` ([below](#tool-calls)) |
| `criteria` | the LLM judge | things a grader should check, in plain language |
| `reference` | the LLM judge | a good answer to compare against |
| `state` | `end_state` | the environment's final state ([below](#environments-end-state-mocks-and-faults)) |

Items are hashed: an item you edit isn't compared against its old results.

## Your agent

```python
def chat(messages, context):   # `context` is optional; declare it to get it
    ...
    return "a reply"            # or a message dict, or a list of messages (OpenAI shape, tool calls included)
```

`context.thread_id` is stable across one conversation's turns, for agents that keep state on a server. `context.env` is the item's environment, if it has one. Raise `warden_sdk.evals.InfraError` for failures that aren't the agent's fault (rate limits, outages): the trial is retried, then left out rather than failed.

Trace your agent with `warden_sdk.trace` to get latency, cost and tokens, and tool calls when it doesn't return tool messages.

## Trials and the verdict

LLM agents give different answers on different runs, so `check` runs every item 3 times (`--trials`) and asks whether the change between baseline and candidate is bigger than that noise:

- **Checks** (pass/fail) are compared per item with an exact McNemar test (1 trial) or a paired sign-flip test (several), with a 95% CI and a correction for testing several checks at once.
- **Metrics** (latency, cost, tokens) are *worse* or *better* only when the whole 95% CI of the change is past ±10%.
- Every item that moved is listed: **broke**, **degraded**, **fixed**, **stabilised**, **improved**. With few items, a real break can show up here before it's significant overall.
- A check that couldn't decide (a judge that gave no verdict, a scorer that raised, no trace to read tool calls from) is **unscored**: left out of pass rates, and reported as coverage. Below 90% coverage the verdict is *inconclusive*.
- Comparing across a changed judge or simulated user is *incomparable*: the ruler moved, so the numbers can't be compared.

The method and its sources are in [research/evaluation-methodology.md](research/evaluation-methodology.md).

**Calibrate first.** `calibrate` runs the same agent twice and reports how much the suite moves on its own. If an identical rerun is reported as a regression, add items or trials before trusting `check`.

## Commands

| Command | What it does |
|---|---|
| `run DATASET --agent m:f` | run and score, store the results |
| `check DATASET --agent m:f` | run, then compare against the baseline |
| `calibrate DATASET --agent m:f` | run twice with no change, to see the noise |
| `diff BASELINE CANDIDATE` | compare two stored runs |
| `baseline REF` | make a run the baseline for its dataset and agent |
| `runs [--dataset NAME]` | recent runs with their verdicts |
| `rescore REF` | score a stored run again without re-running the agent |
| `export-labels REF -o FILE` | rows for people to label |
| `validate-judge FILE` | check a judge against human labels |
| `from-traces [IDS] -o FILE` | turn recorded traces into items |
| `generate --about "..." -o FILE` | generate scenario items with an LLM |
| `pairwise BASELINE CANDIDATE` | judge two runs side by side |

`REF` is a run id or a version tag. Common options for `run`, `check` and `calibrate`: `--trials`, `--concurrency` (default 4), `--scorer module:function` for your own scorers, `--judge`, `--faults FILE`, `--version`. `check --baseline git` compares against the run on the commit your branch forked from.

## Scorers

Built in: `contains`, `exact_match`, `no_errors`, `tool_calls`, `turn_expectations`, `end_state`, `user_goal_met`, and the metrics `latency_ms`, `cost_usd`, `total_tokens`, `turns`. Each only applies to items that ask for it.

Your own:

```python
from warden_sdk.evals import Case, Score

def mentions_order(case: Case) -> Score | None:
    if case.output is None:
        return None                                   # doesn't apply
    ok = "order" in str(case.output)
    return Score(value=float(ok), passed=ok, reason=None if ok else "no order mentioned")

mentions_order.version = "1"                          # bump when its meaning changes
```

Return a list of `Score(..., criterion="...")` to report several checks, and `Score.unscored("why")` when you can't decide. A `Case` has the item, output, error, traces and spans, and for conversations the transcript and per-turn records.

## LLM judges

`--judge` grades the `criteria` in items and turns with Claude (`claude-opus-5-5` by default; needs `ANTHROPIC_API_KEY` or `ant auth login`). Each criterion gets pass, fail or inconclusive, with reasoning. A refusal, a missing verdict or an inconclusive one is unscored, never a pass.

**Don't let an unvalidated judge decide anything.** Compare it to people first:

```bash
python -m warden_sdk.evals export-labels REF -o labels.jsonl   # judge's verdicts left out on purpose
# fill in "label": "pass" or "fail" for each row (50 or more)
python -m warden_sdk.evals validate-judge labels.jsonl
```

This reports agreement (Cohen's kappa with a CI), the confusion matrix, and how often the judge contradicts itself. Until a judge has kappa ≥ 0.6 on at least 50 labels, `check`, `diff` and the viewer warn about it.

Build a judge with your own criteria or model with `llm_judge(criteria=[...], model=...)` and pass it with `--scorer`.

## Simulated users

```json
{"id": "refund-broken-kettle",
 "scenario": {"persona": "An impatient customer", "goal": "Get a refund for order 1234",
              "known_info": ["order number 1234"], "unknown_info": ["the purchase date"],
              "opening": ["hi, my kettle arrived broken"], "max_turns": 6},
 "expected": {"tools": ["lookup_order", "refund"], "criteria": ["checks the order before refunding it"]}}
```

An LLM plays that user until its goal is met, it gives up, or `max_turns`. It only sees what a person would (the agent's replies, not its tool calls) and never invents facts beyond `known_info`. `user_goal_met` reports whether the simulated user thought it was done; that's only as good as the simulator, so check what matters with tools, `end_state` or the judge. Simulated users are a real source of noise: run `calibrate` on scenario suites.

## Tool calls

```json
"tools": ["lookup_order", {"name": "refund", "args": {"order": "55"}}],
"tools_match": "superset"
```

`args` must all match; extra arguments are fine unless `"args_match": "exact"`. `tools_match`: `superset` (the default: every expected call made, extras fine), `strict` (same calls, same order), `unordered`, or `subset`.

## Environments: end state, mocks and faults

```json
"environment": {
  "state": {"orders": {"55": {"status": "delivered", "refunded": false}}},
  "mocks": {"weather": {"returns": {"temp_c": 18}}}
},
"expected": {"state": {"orders": {"55": {"refunded": true}}}}
```

The agent calls tools through `context.env.call("refund", order="55")`. Implement them with `@warden_sdk.evals.environment.tool`, taking the environment first and changing `env.state`, or mock them. `end_state` checks the final state: only the keys you list, lists in any order (unless `{"$ordered": [...]}`), `{"$absent": true}` for keys that must be gone, and each difference reported by path.

`--faults faults.json` runs each environment item again under each fault, and reports how they did next to the originals:

```json
[{"tool": "refund", "effect": "error", "message": "payments down"},
 {"tool": "lookup_order", "on_call": 1, "effect": "timeout"}]
```

Effects: `error` and `timeout` (the call fails), `empty` and `corrupt` (the tool runs, but the agent sees a damaged result; seeded, so reruns match).

## Growing a suite

- **From real traffic:** `from-traces --agent-name support_agent --errors-only -o regressions.jsonl` turns traces into items (`--as-turns` for chat agents, `--expect-tools` to expect what they called). Then write down what "correct" is in each.
- **Generated:** `generate --about "..." --topic refunds --persona "angry customer" -n 20 -o scenarios.jsonl`. Generated items are marked unreviewed and `run` warns about them until someone reads them and sets `"reviewed": true`.
- **People's judgement:** `export-labels REF --per-item -o labels.jsonl`, label, then `rescore REF --labels labels.jsonl` to get a `human` check on that run.
- **Side by side:** `pairwise BASELINE CANDIDATE --question "which reply is more helpful?"`. Every case is judged in both orders and counted as a tie when they disagree, so a judge that favours whichever answer comes first can't decide it.
