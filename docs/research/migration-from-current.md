# Migration from the current repo

Written after `proposed-architecture.md` was frozen. For each target component: **keep**, **change**, **replace**, or **build**. Scope is the Phase 1 MVP (`mvp-spec.md`).

| Target component | Current repo | Action |
|---|---|---|
| JSONL suite, stable `id`, dataset hash | `runner.py:load_dataset` validates `id` + `input`, hashes the file | **Change**: accept `turns` as an alternative to `input`; compute a per-case content hash |
| Trace + span model | `tracer.py` `Trace`/`Span`, `tool_call` span type; eval context collects traces per item | **Keep**. Collect all traces per trial (multi-turn opens one per turn) |
| Agent interface | `agent(input)` | **Change**: add `agent(messages)` for `turns` cases |
| Harness / scripted loop | none | **Build** in `runner.py` |
| Trials | one result per item; `UNIQUE (run_id, item_id)` | **Change**: add `trial` column, unique on `(run_id, item_id, trial)` |
| Infra-error retry | `WardenError` aborts the run; agent exceptions become `error` | **Build**: `InfraError` an agent can raise to mean "not my fault, retry"; retried ×2, then stored with no scores |
| Termination reason | none | **Build**: `termination` column (`completed`, `agent_error`, `infra_error`) |
| Scorer contract | `Scorer = Callable[[Case], Score \| None]`; `Score(value, passed, reason)` | **Change, compatible**: add `outcome` (derived from `passed` when not given), `criterion`, `Score.unscored()`; a scorer may return a list; optional `version` attribute. Existing custom scorers keep working |
| Raising scorer → `unscored` | an exception in a scorer aborts the run | **Change** in `runner.py` |
| Built-in scorers | contains, exact_match, no_errors, latency_ms, cost_usd, total_tokens | **Keep**, adjusted for multi-trace cases; **build** `tool_calls`, `turn_expectations`, `turns` |
| Score storage | `scores(result_id, scorer, value, passed, reason)`, unique `(result_id, scorer)` | **Change**: add `outcome`, `criterion` (default `''`), `scorer_version`; unique `(result_id, scorer, criterion)` |
| Per-turn records, transcript | none | **Build**: `turns jsonb`, `transcript jsonb` on `eval_results` |
| Comparison | `diff.py:compare_runs` — pass→fail per item, metric +20% flag, pure, shared by CLI and viewer | **Replace** the logic with the trial-aware paired comparison; keep it pure and shared; keep `fetch_run` |
| Statistics | none | **Build** `warden_sdk/evals/stats.py` (no new dependencies) |
| Runs list "vs baseline" | SQL counts pass→fail items (duplicates the old diff rule) | **Replace**: compute with `compare_runs` in Python so the list can't disagree with `check` |
| Baselines per (dataset, agent) | `is_baseline` + unique partial index; `check` auto-sets the first | **Keep** |
| Commands | run, check, diff, baseline, runs | **Change**: `--trials` on run/check; **build** `calibrate`; exit codes 0/1/2 |
| Viewer comparison | `index.html` renders `regressions`/`fixes`/`checks`/`metrics` | **Change** to the new comparison shape |
| Old stored runs | k = 1, no case hashes, no outcomes | Migration defaults: `trial = 0`, `outcome` backfilled from `passed`, `termination = 'completed'`; pairing without a case hash falls back to the dataset-hash warning |
| Tests | none in the repo | **Build**: pytest for stats, comparison, scorers, harness |

## Cost

Moderate. One additive migration, one module replaced (`compare_runs`), and one viewer function rewritten. No target element had to be softened. The main cost is listed in `risk-register.md`: computing the runs-list verdict in Python fetches each baseline run, which is slower than the old SQL count.
