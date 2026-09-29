# MVP spec (Phase 1)

**One use case, end to end:** a user changes their agent and runs `check` on a suite with single-turn and **scripted multi-turn** cases, **k trials** each, and gets a statistically sound regression verdict with per-case transitions. (UC1, UC2, UC4 in `proposed-architecture.md` §5.)

## In

1. **Case model:** `id`, `input` | `turns`, `expected`; per-case content hash; `turns` steps `{"user": ...}` and `{"expect": {...}}` where `expect` supports `contains`, `exact`, `tools` (names called in that turn).
2. **Agent interface:** `agent(input)` for single-turn; `agent(messages)` for multi-turn, returning a string, a message dict, or a list of messages.
3. **Harness:** scripted loop, per-turn records (user message, agent messages, tool calls, duration), typed termination (`completed`, `agent_error`, `infra_error`).
4. **Trials:** `--trials k` (default 1 for `run`, 3 for `check`), deterministic trial ids, infra-error retry ×2.
5. **Scorer contract:** outcome pass/fail/unscored, value, reason, criterion; `name`, `version`, `kind`, `direction`; raising scorer → unscored.
6. **Scorers:** `contains`, `exact_match`, `no_errors`, `tool_calls` (strict/unordered/subset/superset on names), `turn_expectations`, `latency_ms`, `cost_usd`, `total_tokens`, `turns`.
7. **Storage:** trial index, case hash, termination reason on results; outcome, criterion, scorer version on scores; trials + per-case hashes on runs.
8. **Comparison:** everything in methodology §§2–6 (pairing, coverage, per-case summaries, pass^k, McNemar / sign-flip, bootstrap CI, BH, verdicts, transitions, metric rule), pure and deterministic.
9. **Commands:** `run --trials`, `check --trials`, `compare`, `calibrate`; exit codes 0/1/2.
10. **Viewer:** shows the new comparison (verdict per scorer with CI and p, transitions, per-turn failures).

## Out (Phase 2+)

- Simulated users (`scenario` cases) and the `UserSimulator` interface
- LLM judges and the judge validation workflow
- Git merge-base baseline selection
- End-state / environment grading, mocked tools, fault injection
- Re-score without re-run
- Online evaluation of production traces
- Pairwise comparison

## Acceptance

- Unit tests for the statistics against hand-computed values (McNemar exact p, pass^k estimator, BH ordering, sign-flip on a known case).
- A demo suite with scripted multi-turn cases runs end to end through `check`, and an A/A `calibrate` on the demo agent prints no regression.
- A deliberately broken demo agent produces `regression`, exit 1, with the broken cases listed as **broke**.
