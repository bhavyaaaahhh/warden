# Gap analysis: offline evaluation of multi-turn, tool-using agents

## Answer to the framed question

The best-known approach is a composition of primitives from different projects [inferred]:

- **Test case as data.** Use a hidden user scenario (`known_info`/`unknown_info`), an initial environment state, and evaluation criteria with a declared gating basis (τ2-bench). Keep the persona separate from the scenario (DeepEval).
- **Execution.** The harness owns the loop. Open with a scripted prefix, then continue with the LLM (openevals `fixed_responses`). Put ordered, per-turn assertions and judge checkpoints in a script (Scenario), but store the script as data. The simulator emits a structured stop reason (Strands, τ2) and is grounded in environment state, which roughly halved critical simulator errors in τ2's paper.
- **Scoring, outcome first.** Derive the gold end state by replaying a reference trajectory, and compare it with a structural diff rather than a hash (τ2, fixing #514). Declare deterministic trajectory match modes on the case (agentevals vocabulary, promptfoo arg ignore/partial). For judges, use per-criterion fail-closed verdicts (Scenario), forced-enum output (Braintrust), an `unscored` sentinel on parse failure (Inspect), and kappa-validated alignment (Ragas).
- **Trials.** Run epochs with pass^k/pass@k reducers and clustered stderr (Inspect, τ-bench). Use typed termination so infra errors are retried, not scored (τ2, promptfoo tri-state).
- **Regression.** Use stable row ids with pinned dataset versions (Braintrust `origin`/`_xact_id`), a merge-base baseline (Braintrust), and re-scoring without re-running (Inspect, LangSmith).

The missing piece is a **paired, trial-aware significance test between runs.** None of the nine projects implements one [code/doc]. The method is described in Miller 2024 [paper].

**Design implication.** Aggregate trials by row identity first, then compare.

---

**Evidence base.** This analysis draws on the nine dossiers in `projects/` (commit-pinned) and on `capability-matrix.md`. Papers listed in `ecosystem-map.md` (GAUGE, "Catching One in Five", Miller 2024 error bars) were read at **abstract level only** during the breadth pass. No dossier examined them. They are cited as [paper, abstract-only], and I do not build on any specific number from them.

**Product constraint.** Regression checks are tools users run on demand, not an automatic gate on every PR. Where projects ship CI gates (promptfoo exit codes, Strands `--fail-on`, DeepEval pytest), they are noted as prior art for on-demand checks only.

**Legend.** Well solved = at least one project does it correctly and it is reusable. Partially solved = it exists but has known defects or is limited to one context. Fragmented = good pieces exist, but in different projects that don't compose. Poorly solved = mostly absent or wrong. Emerging = research or very new code. Unknown = the dossiers don't cover it.

## Classification by capability

| Capability | Status | Best existing reference | Why not "well solved" |
|---|---|---|---|
| Multi-turn test-case model | **Partially solved** | τ2 `Task` (`user_scenario` + `initial_state` + `evaluation_criteria`) [code] | Only a benchmark has it. Platforms use opaque JSON (Braintrust, LangSmith) or code arguments (Scenario). DeepEval's `ConversationalGolden` has no expected tools. |
| User simulator | **Partially solved** | τ2 `UserSimulator` (layered prompt, env-grounded); DeepEval (persona/scenario, `stopping_controller`) [code] | Stop signals are magic substrings (τ, promptfoo `###STOP###`). Scenario's simulator sees the agent's tool traffic. Strands seeds with an unseeded random greeting. |
| Hidden-info / visibility boundary | **Poorly solved** | τ2 `known_info`/`unknown_info` [code] | Only one project. Scenario sends `description` to both simulator and judge. Nobody has explicit "judge-only" or "user-only" visibility. |
| Simulator validity measurement | **Emerging** | τ2 `ConversationReviewer` `critical_helped/hindered` plus paper Table 2 (16–47% simulator error rates) [code]/[paper] | Diagnostic only; it does not affect reward. No other project measures simulator fidelity. GAUGE and related 2026 papers [paper, abstract-only] indicate the area is still being defined. |
| Scripted turns / per-turn checks | **Partially solved** | Scenario script DSL [code] | Scripts are closures over a private executor, so they can't be stored or diffed. Elsewhere per-turn checks are post-hoc windows (DeepEval) or absent. |
| Expected tool trajectory in **multi-turn** data | **Poorly solved** | τ2 `actions` with `requestor` (conversation-level) [code] | No project attaches expected tool calls *per turn*. DeepEval `Turn` has none. Ragas flattens every call. openevals' app contract drops intra-turn tool calls. |
| Trajectory match modes | **Fragmented** | agentevals/openevals (strict/unordered/subset/superset); promptfoo (in_order, globs, counts); DeepEval (weighted LCS) [code] | Each project has a different subset, and each has bugs: greedy first-fit (LS), judge-chosen mode (Strands), length mismatch → 0 (Ragas). None has bipartite matching together with partial credit and declared modes. |
| Tool-arg matching | **Well solved (in pieces)** | openevals per-tool overrides (mode / key paths / callable); promptfoo `ignore` globs + `defaults` stripping [code] | Needs a merge. τ2 `compare_args=None` lets omitted args pass (#568). Ragas has inconsistent equality between metrics. |
| End-state (environment) grading | **Partially solved** | τ2 gold-replay → DB compare + `EnvAssertion` [code] | See the dedicated section below. It exists only in a benchmark with a fully mockable DB, it is brittle, and it misses procedure. Platforms substitute LLM "goal accuracy" judges. |
| Simulated / mocked tools | **Fragmented** | τ2 stateful toolkits (real state); Strands `ToolSimulator` (LLM, no state) [code] | Stateful fakes need a hand-built domain. LLM fakes can't support end-state checks. Five of nine dossiers are Unknown here. |
| Fault injection | **Emerging** | Strands `ChaosCase` + `ChaosPlugin` with a baseline variant [code] | One project, and fault payloads are unseeded. No dossier shows fault injection combined with end-state grading. |
| LLM judge (single / rubric) | **Well solved** | Scenario per-criterion verdicts; Braintrust forced-enum; DeepEval DAG [code] | Solved as mechanism. Calibration is not (see below). |
| Pairwise judge | **Poorly solved** | LangSmith `randomize_order` (buggy, #3582) [code] | promptfoo `select-best` and Braintrust `Battle` use a fixed order with no tie option. Position bias is unmitigated almost everywhere. |
| Judge parse-failure semantics | **Fragmented** | Inspect `Score.unscored(reason)` excluded from metrics; Scenario fail-closed [code] | DeepEval (all-dropped → 1.0), Ragas (NaN + nanmean), τ2 (`all([])` passes, #554) and promptfoo (missing `pass` → true) all **inflate** scores on judge failure. |
| Judge versioning / provenance | **Partially solved** | Inspect `EvalSpec` (model_roles, git, packages) + grading prompt in metadata; Ragas prompt SHA [code] | No project stores a judge-prompt hash *per score*. DeepEval does not record which score path (logprob vs raw) produced a number. |
| Judge calibration vs humans | **Poorly solved** | Ragas `align()` + Cohen's kappa [code] | Only one library. It is absent from every agent-focused project. |
| Repeats / epochs | **Partially solved** | Inspect `Epochs(n, reducer)` [code] | Most tools repeat rows flatly, without an index or aggregation (promptfoo, LangSmith, Braintrust, DeepEval). |
| pass^k / pass@k | **Well solved (math), fragmented (bookkeeping)** | Inspect `pass_k`, `pass_at`; τ2 `pass_hat_k` [code] | Handling of missing or failed trials differs (τ1 global n, τ2 per-task surviving n, leaderboard counts infra errors as failures, #497). |
| Uncertainty (stderr / CI) | **Partially solved** | Inspect clustered + bootstrap stderr [code] | Only Inspect. τ2 reports no intervals despite a 5–11 pt noise floor (#540). |
| **Run-vs-run significance** | **Poorly solved** | none [code/doc] | See the dedicated section. |
| Baseline selection | **Partially solved** | Braintrust git merge-base ancestor [code] + server [doc] | Silent fallback to "most recent" when git collection is off. Everyone else uses manual selection or nothing. |
| Row identity across runs | **Partially solved** | τ2 stable task ids; Braintrust configurable comparison key + `origin` [code/doc] | promptfoo aligns by position. Braintrust defaults to `input` (volatile). LangSmith pytest hashes the return value. Ragas has no id. |
| Dataset versioning | **Well solved (closed servers), poorly solved (local OSS)** | Braintrust `_xact_id` + pinned reads + environments [code] | LangSmith records `max(modified_at)` loosely. DeepEval versions server-side only. Ragas, Scenario and Strands have none. |
| Error vs fail distinction | **Partially solved** | τ2 typed `TerminationReason`; promptfoo NONE/ASSERT/ERROR; Inspect `fail_on_error`/`retry_on_error` [code] | Ragas (NaN), Strands (score 0) and τ1 (reward 0) conflate the two. promptfoo counts ERROR against pass rate. |
| Limits as outcomes | **Partially solved** | Inspect `EvalSampleLimit`; τ2 MAX_STEPS/TIMEOUT [code] | Scenario fails at `max_turns` without consulting the judge (#934). Ragas and Braintrust have no per-row limits. |
| Re-score without re-run | **Well solved** | Inspect `inspect score`; LangSmith `evaluate(experiment_id)` [code] | τ2 lenient replay can shift scores silently (#502). Strands keys stored outputs by case name. |
| Caching correctness | **Poorly solved** | Inspect (key includes epoch) [code] | DeepEval omits tools, trace and prompt from the key. Scenario keys positional args only (#926). Strands keys by name. promptfoo applies a 14-day TTL to judges. Inspect uses pickle, which isn't portable. |
| Concurrency / retries | **Partially solved** | promptfoo AIMD + Retry-After; Inspect adaptive + multi-level retries [code] | Braintrust and Ragas-new are unbounded. LangSmith has no retries or timeouts. Ragas-legacy retries every exception 10×. |
| Trace-based tool extraction (OTel) | **Partially solved** | promptfoo attribute-family tables; Strands session mappers [code] | It depends on the system under test emitting correct spans (promptfoo #10518). Braintrust passes "no trace" as "no tool calls". The dialects differ. |
| Test-runner integration | **Well solved** | DeepEval, LangSmith (pytest/vitest + git-tagged dataset versions), Scenario [code] | Scenario monkeypatches the runner. Note: prior art for on-demand runs only (product constraint). |

## Cross-cutting gap 1: run-vs-run regression is statistically naive everywhere

What the dossiers actually show, rather than what their docs imply:
- **LangSmith** flags a row as regressed on *any* per-example score decrease against the source experiment. The docs describe no noise threshold and no significance test [doc] (`langsmith-openevals.md` §Regression comparison).
- **Braintrust** `summarize` returns `improvements`/`regressions` as raw counts of matched rows that moved, computed server-side. Local runs get `improvements=0, regressions=0` [code] (`braintrust.md` §Comparison).
- **promptfoo** aligns rows by `testIdx` *position* and places columns side by side, with no statistic [code].
- **Ragas** prints a delta of means, and its CLI is broken at the pinned SHA [code].
- **Inspect** has the right ingredients (clustered and bootstrap stderr, pass^k, content-hash `task_identifier`) but no diff command (#1327) [code].
- **DeepEval, Scenario:** comparison happens only in their hosted platforms [doc/inferred]. **Strands, τ-bench:** none.

Why this matters: τ2 #540 measured a noise floor of 5–11 points between runs of the same configuration at temperature 0. It also measured simulator prompt wording alone moving scores by about 15 points (#182) [code/issue]. A "row got worse" rule therefore flags noise as regressions. Meanwhile the tools that aggregate trials (Inspect) don't diff, and the tools that diff (LangSmith, Braintrust) don't aggregate trials (Braintrust autoevals #219; LangSmith repetitions are flat) [code]. **Status: Poorly solved. The pieces are fragmented across tools that don't compose.**

Prerequisites the dossiers show are needed before any test is meaningful [inferred]:
1. Stable row identity: user-assigned or content-hash ids, not position and not raw `input`.
2. A trial index per row, with per-row aggregation before comparison.
3. Pinned dataset version and judge provenance on both runs, so that a changed judge isn't reported as an agent regression.
4. Infra errors excluded *and counted*, with a fixed n per row (the τ2 #497 failure mode).

Method choice (paired test over per-row trial means, clustered SE, multiple-comparison control across metrics) is an **`eval-scientist` question**. Miller 2024 [paper, abstract-only] is the cited starting point.

## Cross-cutting gap 2: end-state grading, from the dossiers rather than assumed

It is tempting to treat "check the final environment state" as solved because τ-bench is famous for it. The dossiers show it is narrower than that:
- **Only τ-bench does it** [code]. Every platform library substitutes an LLM judge over the transcript. DeepEval `TaskCompletion`, Ragas `AgentGoalAccuracy*` and promptfoo `trajectory:goal-success` all infer the outcome from text. Strands `StateEquals` compares only what the task function chooses to return.
- **Its preconditions are strong.** It needs a fully mockable, resettable, deterministic backend (pydantic DB plus Python toolkits) and a task authored so that exactly one outcome is valid (τ1 paper §3) [code/paper].
- **The comparison is brittle.** SHA-256 over `json.dumps(sort_keys=True)` is list-order sensitive (#514), lets free-text args enter the hash (#485), and breaks on time-dependent output (#387). When it fails, it gives no field-level diff [code].
- **It misses procedure.** Confirm-before-write, authentication order and required handoffs pass unchecked (#298, #320, #384) unless an author adds an `EnvAssertion` [code].
- **Replay can be silently wrong.** Lenient replay moved one score from 0.72 to 0.10 without an error (#502) [code].
- **Simulated tools can't support it.** Strands' LLM `ToolSimulator` keeps a 20-entry prompt log, not materialized state [code].
- **Nobody combines it with fault injection.** Strands has chaos effects but no state. τ2 has state but no runtime faults [code].

**Status: Partially solved.** The pattern (derive the gold state by replaying a reference trajectory, then compare outcomes) is sound and generalizes to any sandboxable backend [inferred]. The comparison, the procedural checks and the backend requirement are all open.

## Cross-cutting gap 3: judge-failure and error semantics inflate scores in 4 of 9 projects

DeepEval, Ragas, τ2 (NL assertions) and promptfoo (missing `pass`) each have a code path where a judge that fails to produce a usable verdict *raises* the reported score [code]. Inspect (`unscored` NaN with a reason, kept out of metrics but counted) and Scenario (fail-closed per criterion) are the correct references. Braintrust's forced-enum tool call avoids most parse failures upstream. The same pattern applies to task errors: Ragas NaN + `nanmean` means the score can *rise* as the system degrades (#3028). **Status: Fragmented.** The fix is known, but it isn't applied consistently.

## Cross-cutting gap 4: simulator validity is unmeasured in products

- τ2 is the only project that measures simulator error. It found 16–47% of conversations had simulator errors, with 6–13% critical (paper Table 2) [paper].
- It is also the only one that tags errors that made the agent *pass* by mistake (`critical_helped`) [code].
- No platform records simulator model and prompt version as first-class provenance well enough to separate "the agent regressed" from "the simulator changed" (τ2 #540, τ1 #13) [code].
- Leakage paths exist: Scenario's simulator receives agent tool calls and tool results (`reverse_roles`), and Scenario's judge sees the hidden `description` [code].
- **Status: Emerging.** Research is active (GAUGE, Catching One in Five, SAGE, VISTA in `ecosystem-map.md` §5 [paper, abstract-only]). No dossier project implements their methods.

## Cross-cutting gap 5: multi-turn tool expectations have no home

- Single-turn tool matching is well served (DeepEval ToolCorrectness, Ragas ToolCallAccuracy/F1, openevals matchers).
- In multi-turn data, **no project** lets a case say "at turn 3 the agent must call `lookup_order` with `{id}`" as data. The closest options are Scenario code assertions (not data), τ2 conversation-level `actions` (order-free and rarely gating) and promptfoo trace assertions over the whole run [code].
- Ragas and openevals lose per-turn tool structure entirely: Ragas has no tool-call ids and openevals' app returns one message per turn [code].
- **Status: Poorly solved.**

## Smaller gaps

- **Caching** (Poorly solved). Only Inspect includes the epoch in the key. The others either collapse repeated trials or serve stale scores after a tool, trace or prompt change [code].
- **Pairwise judging** (Poorly solved). Position bias is unmitigated in promptfoo and Braintrust, and LangSmith's mitigation races (#3582) [code].
- **Calibration** (Poorly solved). Ragas `align()` + kappa is the only implementation [code].
- **Row identity** (Partially solved). The failure modes are well documented: position (promptfoo), volatile `input` (Braintrust), return-value hash (LangSmith pytest) [code/doc].

## Questions to hand off

**To `eval-scientist`:**
- Which paired test to use across runs with k trials per row: a per-row mean diff with clustered SE, or a permutation/bootstrap?
- How to control for multiple metrics.
- How to set a minimum detectable effect given τ2's 5–11 pt noise floor.
- pass^k when n varies per row.
- Simulator-noise attribution: is `critical_helped` usable as a filter?
- Minimum human-label set for judge validation (kappa target).

**To `eval-architect`:**
- How to store a conversation case, script and expectations as data. Scenario's DSL shows the semantics, and its closure form is the anti-pattern.
- A structural state diff to replace the hash.
- Where trial aggregation sits relative to row identity and comparison.

**To `github-code-researcher`:** resolve the Unknown cells listed in `capability-matrix.md` if a design decision depends on them. The most decision-relevant are simulated/mocked tools in DeepEval and Scenario, and the τ2 cache key.
