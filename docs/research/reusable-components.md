# Reusable components: integrate, adapt, don't copy

This file draws on the nine commit-pinned dossiers in `projects/`, together with `capability-matrix.md` and `gap-analysis.md`. The tags mean:
- **[code]**: observed in a dossier at its pinned SHA.
- **[doc]**: documentation only.
- **[paper]**: read at abstract level unless a dossier covers it.
- **[inferred]** / **[hypothesis]**: my interpretation, or something that still needs validation.

This file stops at "what to reuse and why". The architecture belongs to `eval-architect`.

## License summary

| Project | License | What cannot be reused |
|---|---|---|
| Inspect AI | MIT | — |
| DeepEval | Apache-2.0 | Confident AI backend (regression views, dataset versions) is proprietary |
| promptfoo | MIT | Hosted simulated-user and GOAT attacker services (the OSS code POSTs to them) |
| τ-bench / τ2-bench | MIT (code and bundled task data) | — |
| LangSmith SDK, openevals, agentevals | MIT | LangSmith server (comparison view, dataset versioning) is proprietary |
| Braintrust SDKs / autoevals | Apache-2.0 / MIT | Server (summarize/diff, thread preprocessor, online scoring) is closed |
| Ragas | Apache-2.0 | — (note: telemetry is on by default; opt out with `RAGAS_DO_NOT_TRACK=true`) |
| LangWatch Scenario | Apache-2.0 | LangWatch platform (cross-run comparison) is a separate product |
| Strands Evals | Apache-2.0 | — |

**Implications [inferred; not legal advice]:**
- **All nine OSS codebases are permissive.** Taking their code verbatim requires keeping copyright and license notices.
  - For Apache-2.0 sources (DeepEval, Ragas, Scenario, Strands, Braintrust SDKs), carry over any `NOTICE` file and mark modified files.
  - Apache-2.0 also includes an express patent grant. MIT does not.
- **Re-implementing a pattern from a description** (most of the "Adapt" list) carries no license obligation. Crediting the source in design docs is still good practice.
- **Closed server behavior is knowable only through [doc].** Where a feature exists only server-side (LangSmith and Braintrust diffing, Confident AI baselines, LangWatch comparison), nothing can be integrated, and even the algorithm is unverified.
- **τ2-bench task data is MIT**, so it can be used as fixtures. It carries a known backlog of ground-truth disputes (τ1 #35/#37/#41/#65/#85; τ2 #84/#89/#156/#321/#496). Its per-task `issues` field records some of them.

---

## 1. Integrate (standards and specifications as-is)

**No evaluation library is recommended as an as-is runtime dependency for the core.** Each candidate has a defect in exactly the area it would own:
- openevals trajectory matchers use greedy first-fit.
- Ragas tool metrics have inconsistent semantics.
- Inspect requires the system under test to run inside Inspect.
- DeepEval metrics are stateful objects with an incomplete cache key.

What is worth integrating is formats and formulas.

| Component | Evidence | How to integrate | Caveats |
|---|---|---|---|
| **OpenTelemetry GenAI semantic conventions** (`gen_ai.*` span attributes) as the ingestion format for tool calls and agent steps | promptfoo `trajectory:*` reads `gen_ai.tool.*` among other attribute families (`promptfoo.md` §Tool-call and trajectory assertions) [code]. Braintrust probes `metadata["gen_ai.tool.name"]` (`braintrust.md` §Execution record) [code]. Strands emits `gen_ai.evaluation.*` per case and maps several OTel dialects (`strands-evals.md` §Execution model, §Purpose) [code]. Scenario collects in-process OTel spans for the judge (`langwatch-scenario.md` §Tool calls inside a turn) [code]. Spec URL in `sources.md` [doc]. | Accept OTel spans as one source of the tool trajectory, and normalize to an internal step model. | **Dialects are not uniform.** promptfoo needs an attribute-family table (`gen_ai.*`, `ai.toolCall.*`, `codex.mcp.*`, `tool.name`). Strands needs per-framework mappers. Propagation fails in practice (promptfoo #10518). "No spans" must mean *error*, not "no tool calls" (Braintrust does the latter). The semconv stability status is **Unknown** in these dossiers. |
| **OpenAI chat message format (with `tool_calls[].id` and `tool_call_id`)** as the transcript interchange | openevals, agentevals and Scenario (Python) all normalize to it (`langsmith-openevals.md` §Core abstractions; `langwatch-scenario.md` §Unit of work) [code]. | Use it as the import/export shape for transcripts. | Ragas shows what goes wrong without ids: results can't be attributed to calls, and there are no parallel calls and no system message (`ragas.md` §Message types). Scenario JS uses the Vercel `ModelMessage` format instead, so transcripts don't port between its SDKs [code]. |
| **pass@k and pass^k unbiased estimators**: pass@k (Chen 2021), pass^k = E_task[C(c,k)/C(n,k)] (Yao 2024) | τ-bench paper §3 [paper]. Implementations: Inspect `pass_at`, `pass_k`, `at_least` (`inspect-ai.md` §Scorers and aggregation) and τ2 `pass_hat_k` (`tau-bench.md` §pass^k) [code]. | Implement the formula directly (it is math, with no license). Inspect's MIT code is a reference. | The formula is settled. **The bookkeeping is not.** Fix n per task, and retry infra errors until n valid trials exist or report the exclusion (τ2 #497, #493). Return "unscored" when fewer than k trials exist, as Inspect does. |
| **Clustered and bootstrap standard errors; paired-difference comparison** | Inspect ships `stderr` (clustered via sample metadata) and `bootstrap_stderr` (`inspect-ai.md` §Scorers and aggregation) [code]. Miller 2024, "Adding Error Bars to Evals" (arXiv 2411.00640) [paper, abstract-only]. | Use as the statistical basis for uncertainty and run-vs-run comparison. | The choice of test and the multiple-comparison policy are for `eval-scientist`. No dossier project implements the paired run-vs-run test. |
| **τ2-bench domains as a conformance fixture set** [hypothesis] | MIT code and data. Stateful toolkits, gold actions, `EnvAssertion`s (`tau-bench.md` §Environment) [code]. | Use them to validate a simulator, end-state and pass^k pipeline against published behavior. | Ground-truth disputes (see License). Hash comparison brittleness (#514, #485, #387) is part of what you would be testing against. |

## 2. Adapt (patterns: source, what to keep, what to change)

### Test-case model
| Pattern | Source | Keep | Change |
|---|---|---|---|
| Hidden-scenario task: `user_scenario` + `initial_state` + `evaluation_criteria` | τ2 `Task` (`tau-bench.md` §Test case) [code] | Structured `known_info`/`unknown_info`. A seeded `message_history` prefix. A per-task `issues` audit trail. | Generalize beyond a DB (any resettable backend). Add explicit **visibility** per field (user-only, judge-only). Scenario shows the leak that results without it (`langwatch-scenario.md` §Weaknesses). |
| Persona separate from scenario and outcome | DeepEval `Persona` / `ConversationalGolden` (`deepeval.md` §Core abstractions) [code] | Reuse a scenario across personas. | Make persona structured, not free-text `characteristics`. τ2 `PersonaConfig` (verbosity, interrupt tendency) is a model for this. |
| Expected/actual slot per dimension (output, trajectory, interactions, env state) | Strands `EvaluationData` (`strands-evals.md` §Evaluation record) [code] | One record pairs every expectation with its actual. | Type the trajectory slot (it is `list[Any]` today). Keep expectations metric-agnostic, **not** Ragas-style per-metric fields. |
| Declared gating basis + always-on diagnostics | τ2 `reward_basis` (`tau-bench.md` §tau2) [code]; promptfoo weighted assert-sets with `weight: 0` metric-only (`promptfoo.md` §Assertion implementations) [code] | Gating components are declared per case. Every component's detail is stored. | Make the composition pluggable. τ2 hard-codes it in an if/elif chain. |

### Conversation execution
| Pattern | Source | Keep | Change |
|---|---|---|---|
| Script as an ordered list of steps: fixed turn, generated turn, assertion, judge checkpoint, `proceed(N)` | Scenario (`langwatch-scenario.md` §Script DSL) [code] | The semantics, and assertion failures surfacing as normal test failures. | Make it **data** (serializable, diffable), not closures over `state._executor`. |
| Scripted prefix, then LLM continuation | openevals `fixed_responses` (`langsmith-openevals.md` §Multi-turn simulation) [code] | Reproducible openers. | Keep intra-turn tool calls in the transcript. openevals' app contract returns only one message. |
| Structured stop signal from the simulator, stamped by the harness | Strands `ActorResponse.stop` + harness-set `stop_reason` (`strands-evals.md` §User simulator) [code]; τ2 three stop reasons (`tau-bench.md` §tau2 user simulator) [code] | Distinct reasons: done, transfer, out-of-scope. | Replace magic `###STOP###` substrings with a field or tool. Don't collapse the reasons into USER_STOP (τ2 #517). |
| Stopping hook `(Context) -> Decision{should_end, reason}` | DeepEval `stopping_controller` (`deepeval.md` §Conversation simulator) [code] | A small interface for scripted and judged termination. | Bound the cost: DeepEval's full-history completion check grows about quadratically. |
| Two-phase judge that can end the conversation (cheap continue/verdict gate, then a verdict) | Scenario (`langwatch-scenario.md` §Judge implementation) [code] | Early termination on violation. | Make per-turn invocation opt-in and budgeted. Scenario calls it every turn (#980, #934). At `max_turns`, still consult the judge. |
| Environment-grounded simulator (user has tools over observable state) | τ2 telecom `user_tools` + `sync_tools` (`tau-bench.md` §World-state coupling) [code]/[paper] | The single measured lever for simulator fidelity (critical errors halved, paper Table 2). | Needs a stateful backend, so it is domain work, not a library feature. |
| Filter the simulator's view to what a user would observe | openevals `_is_internal_message` filter (`langsmith-openevals.md` §Multi-turn simulation) [code] | Drop tool calls and tool results from the simulator's history. | Scenario is the counter-example (`reverse_roles` leak). |
| Callback-signature introspection for the app under test (`{input, turns, thread_id}`) | DeepEval (`deepeval.md` §Conversation simulator) [code] | Supports stateless and stateful apps without adapters. | Fail loudly on unknown parameter names. LangSmith's silent positional fallback is the risk. |
| Typed fault effects with a baseline variant (`ChaosCase.expand`) | Strands (`strands-evals.md` §Chaos) [code] | Discriminated-union effects (pre/post hooks), case × fault expansion, resilience measured as a delta. | **Seed** fault payloads. They use module-global `random`. Decouple from the Strands SDK hooks. |
| Record a simulator reviewer label (`critical_helped` / `critical_hindered`) | τ2 `ConversationReviewer` (`tau-bench.md` §Simulator noise) [code] | Attribute false passes and false fails to the simulator. | Its validity as a filter is an `eval-scientist` question. |

### Scoring
| Pattern | Source | Keep | Change |
|---|---|---|---|
| Gold state derived by replaying a reference trajectory, then an outcome comparison | τ2 `EnvironmentEvaluator` (`tau-bench.md` §tau2 evaluate_simulation) [code] | Author one valid path and derive the target. Keep the path for diagnostic overlap scores. | Replace the SHA-256 hash with a **structural diff** (order-insensitive collections, ignored fields, per-field report). Add targeted `EnvAssertion`s for procedure (confirm-before-write, #298). Make lenient replay loud (#502). |
| Trajectory match vocabulary: strict / unordered / subset / superset | agentevals / openevals (`langsmith-openevals.md` §Trajectory match) [code] | Clear set-relation modes, declared on the case. | Use **bipartite matching** instead of greedy first-fit. Add partial credit (precision, recall, LCS as in DeepEval `should_consider_ordering`). Add promptfoo's `in_order` subsequence and `{min,max}` counts. |
| Per-tool arg matching: mode / key paths / callable, plus ignore globs and defaults stripping | openevals `_get_matcher_for_tool_name` [code]; promptfoo `tool-args-match` (`promptfoo.md`) [code] | Handles volatile args (timestamps, ids). | Pick the openevals missing-key semantics (sentinel), not agentevals'. Require the *reference's* keys, not the agent's (τ2 #568). |
| Cross-vendor tool-call extraction | promptfoo `tool-call-f1` extractor (OpenAI chat, Responses, Anthropic, Gemini) (`promptfoo.md`) [code] | Linear-time, bounded parsing. | Keep arguments (F1 there is name-only). |
| Per-criterion verdict `{requirement, reasoning, status: passed \| failed \| inconclusive}`, fail-closed | Scenario (`langwatch-scenario.md` §Judge implementation) [code] | Reasoning before status. A missing criterion → failed. Negative criteria restated as positive requirements. | Make `inconclusive` a reportable state, not an automatic run failure. |
| Forced function-call judge with an enum of choice labels; YAML specs as data | autoevals `LLMClassifier` (`braintrust.md` §Judge implementation) [code] | It removes most parse failures, and judges become diffable data. | Add position swap and a tie option for pairwise (`Battle` has neither). |
| `unscored` sentinel with reason code, excluded from metrics but counted | Inspect `Score.unscored` (`inspect-ai.md` §Judge implementation) [code] | Separates measurement failure from model failure. | Report the unscored count in every aggregate. |
| Two-layer judge failure handling: transport retry and a bounded output-repair call | Ragas `FixOutputFormat` + tenacity (`ragas.md` §Judge implementation) [code] | Separation of concerns. | Classify retries. Ragas retries every exception 10×. After repair fails, emit unscored, not NaN. |
| DAG rubric: the LLM classifies, code assigns the score | DeepEval `DAGMetric` (`deepeval.md` §Judge implementation) [code] | Reproducible rubric grading. | — |
| Typed, hashed judge prompts (pydantic I/O, examples, SHA-256) | Ragas `PydanticPrompt` (`ragas.md` §Judge implementation) [code] | Prompt identity becomes provenance. | Store the hash **per score**. Ragas experiments store no run metadata. |
| Judge alignment loop: human labels → dynamic few-shot → Cohen's kappa on a held-out split | Ragas `align()` / `validate_alignment` (`ragas.md` §Judge implementation) [code] | The only calibration loop in the set. | Method details go to `eval-scientist`. |
| Confusion-matrix scope adherence (answered/refused × in/out of scope) | Ragas `TopicAdherence` (`ragas.md` §Multi-turn metrics) [code] | Precision measures over-answering and recall measures over-refusing. | Fix the list-length padding and truncation. |
| Unit-interaction sliding window for per-turn metrics | DeepEval `get_unit_interactions` (`deepeval.md` §Multi-turn metrics) [code] | O(n) calls, and a user/assistant pair is never split. | Add an inner concurrency bound. DeepEval fans out unbounded. |
| Dict-valued score (one scorer, named dimensions) | Inspect `Score.value` mapping (`inspect-ai.md` §Core abstractions) [code] | Per-key metrics. | Don't force every score into [0,1]. Braintrust's `Score` validator rejects latency and counts. |

### Trials, execution, storage, comparison
| Pattern | Source | Keep | Change |
|---|---|---|---|
| Epochs with pluggable `list[Score] -> Score` reducers (mean, pass_at, pass_k, at_least) | Inspect (`inspect-ai.md` §Scorers and aggregation) [code] | Reduce per sample, then compute metrics. | — |
| Cache key includes the epoch; repeat-scoped namespaces | Inspect `CachePolicy` [code]; promptfoo `withCacheNamespace('repeat:N')` [code] | Repeated trials remain real re-samples. | Hash the **full** scored input and judge config, including the template (DeepEval's partial key is the counter-example). Portable format, not pickle. Label cached judge results explicitly. |
| Typed termination reasons; infra errors retried, not scored | τ2 `TerminationReason` + `run_with_retry` (`tau-bench.md` §tau2) [code]; promptfoo NONE/ASSERT/ERROR (`promptfoo.md`) [code]; Inspect `retry_on_error` [code] | A tri-state or richer outcome, persisted and indexed. | Don't count ERROR against pass rate (promptfoo does). Keep n fixed per row. |
| Limits as scored terminal states with the limit type recorded | Inspect `EvalSampleLimit` (`inspect-ai.md` §Execution model) [code] | Agents that "fail by timeout" stay visible. | — |
| Re-score without re-run | Inspect `score(log, scorers, action="append")` [code]; LangSmith `evaluate(experiment_id)` [code] | Separates expensive generation from cheap scoring. | Record the scorer version on appended scores. |
| Row provenance `origin{dataset id, row id, version}`, dataset version pinned on the run | Braintrust (`braintrust.md` §Dataset) [code] | Every result traces back to the exact row version. | Add named, immutable snapshots. Braintrust "version" is just the max xact id. |
| Git merge-base ancestor as the default baseline | Braintrust `get_past_n_ancestors` (`braintrust.md` §Comparison) [code] | The right default for "compare my branch to main" in an on-demand check. | Fail loudly when there is no git info or no base. Braintrust silently falls back to "most recent". |
| Deterministic ids per (case, trial); content-hash task identity excluding transport knobs | Braintrust `uuid5(...:trial:i)` [code]; Inspect `task_identifier` [code] | Idempotent re-runs and cross-run pairing. | Row ids must be user-assigned or content-hashed. Never position, never raw `input`. |
| Normalized results tables plus a lossless payload blob; local store | DeepEval SQLite store (`deepeval.md` §Data model) [code]; promptfoo SQLite (`promptfoo.md`) [code] | SQL over results for local comparison. | Row-per-(case, trial, scorer) with stable ids, not parallel arrays (Strands). |
| Adaptive concurrency (AIMD on 429) + Retry-After backoff + per-step timeout | promptfoo `scheduler/` (`promptfoo.md` §Execution model) [code] | Operationally the most solid in the set. | Per-conversation serialization, not a global concurrency of 1 for multi-turn. |
| Tag simulated traffic (`origin=simulation`) | Scenario (`langwatch-scenario.md` §Online / production path) [code] | Keeps simulated traces out of production analytics. | — |
| pytest integration that tags dataset versions with the git commit | LangSmith `_end_tests` (`langsmith-openevals.md` §Data model) [code] | Traceable runs. | Do **not** mutate the dataset on every test run (LangSmith syncs examples). Don't hash return values into example ids. Use it for on-demand runs, not a default PR gate (product constraint). |

## 3. Don't copy (with the reason)

| Anti-pattern | Where | Why |
|---|---|---|
| Unparseable judge verdicts dropped, with all-dropped → 1.0 | DeepEval `score_qag_verdicts` (`deepeval.md` §Multi-turn metrics) [code] | Judge failure inflates the score. |
| Errors → NaN, then averaged with `nanmean` and no failure count | Ragas (`ragas.md` §Execution model) [code] | The score can rise as the system degrades (#3028). |
| Missing `pass` defaults to true; first integer anywhere wins | promptfoo judge parsing (`promptfoo.md` §Don't copy) [code] | Hides judge malfunctions. |
| `all([])` passing on empty judge results | τ2 NL assertions (#554) (`tau-bench.md`) [code] | A vacuous pass. |
| "No trace" treated as "no tool calls" (`usedNoTools` passes) | Braintrust JS agent assertions (`braintrust.md` §Multi-turn) [code] | Tracing misconfiguration becomes a silent pass. |
| Silent fallback between two estimators (logprob-weighted vs raw) in one column | DeepEval G-Eval (#1029) [code] | The score distribution shifts with the provider, and nothing records the path. |
| Position-based row identity (`testIdx`) for comparison and resume | promptfoo (`promptfoo.md` §Data model) [code] | Reordering or adding tests silently misaligns diffs. |
| `input` as the default comparison key | Braintrust [doc] | Volatile fields give blank comparison rows. |
| Hashing the test's return value into example ids | LangSmith pytest (`langsmith-openevals.md` §Data model) [code] | Non-deterministic outputs fork examples every run. |
| Regression = any per-row score decrease, no noise model | LangSmith compare view [doc]; Braintrust improvements/regressions counts [doc] | τ2 #540 measured a 5–11 pt noise floor at temperature 0, so this flags noise. |
| Comparison only as a difference of means | Ragas `--baseline` (`ragas.md`) [code] | No pairing, variance or significance. |
| Whole-DB hash equality as the only state check | τ2 `get_db_hash` (`tau-bench.md` §Don't copy) [code] | Order-sensitive (#514), free-text args in the hash (#485), no field-level diff. |
| Substring match for "communicated info" | τ2 COMMUNICATE (`tau-bench.md`) [code] | Gameable, and breaks on formatting ("$54.04"). |
| Magic stop tokens inside free text; collapsing stop reasons | τ, promptfoo `###STOP###` [code] | Fragile, and loses the reason (τ2 #517). |
| Excluding failed trials from pass^k while letting per-task n vary | τ2 `get_metrics_df` (#497) [code] | Biases toward tasks with flaky infrastructure. |
| Judge chooses the trajectory match mode at runtime | Strands `TrajectoryEvaluator` + scorer tools (`strands-evals.md` §Trajectory matching) [code] | The same case can be scored under different modes on different runs. |
| LLM-generated tool responses as the default "environment" | Strands `ToolSimulator` [code] | No materialized state, so no end-state check and no reproducibility. |
| Unseeded module-global `random` in fault payloads and simulator setup | Strands (`strands-evals.md`) [code] | Reruns differ. |
| Cache keys that are partial or volatile: hand-picked fields, case name only, positional args only, or keys that include per-turn trace ids | DeepEval [code]; Strands [code]; Scenario (#926, [inferred] trace_id) | Stale scores come back silently, or "deterministic mode" never hits. |
| Long-TTL judge cache shared across runs | promptfoo 14-day TTL (`promptfoo.md`) [code] | Judge verdicts become sticky, which masks judge changes. |
| Pickle-on-disk cache | Inspect (`inspect-ai.md` §Don't copy) [code] | Not portable, and unsafe to load from untrusted sources. |
| Conversation as a global mutable var keyed by test order, forcing concurrency 1 | promptfoo `_conversation` (#385) [code] | Serializes the whole eval. |
| Flattening a multi-turn transcript into one output string | promptfoo simulated user [code] | Per-turn and tool-level assertions become impossible. |
| Scripts as closures over a private executor; scenarios as function arguments | Scenario (`langwatch-scenario.md` §Don't copy) [code] | Can't be stored, versioned, diffed or authored in a UI. |
| One shared prose field for hidden info and persona; simulator sees raw tool traffic | Scenario [code] | Leaks information both to the judge and to the simulated user. |
| Stateful metric objects holding score/reason | DeepEval `BaseMetric` [code] | Forces deep copies and blocks reentrancy. Return immutable results instead. |
| Parallel single-turn and conversational type hierarchies; parallel metric APIs | DeepEval [code]; Ragas legacy vs collections [code] | Duplicated code that drifts. |
| One flat sample schema that grows a field per metric | Ragas `MultiTurnSample` (`ragas.md` §Don't copy) [code] | Couples the dataset schema to the metric catalogue. |
| Messages without tool-call ids or a system message | Ragas `messages.py` [code] | Can't express parallel calls or attribute tool outputs. |
| Closed enum + static handler map for scorer types | promptfoo `ASSERTION_HANDLERS` [code] | Every new scorer is a core edit. |
| Hard-coded if/elif evaluator composition | τ2 `evaluate_simulation` [code] | Same problem. |
| Unbounded default concurrency with eager task creation; whole-run-only timeout; 429-only retry up to 100× | Braintrust Python SDK [code]; Ragas `ExperimentWrapper.arun` [code] | Resource blowups and hangs. |
| Single save at end, failed rows printed and dropped | Ragas experiments [code] | A crash loses everything, and failures vanish. |
| Silently dropping result upload batches | Braintrust background logger [code] | Offline results must be durable. |
| Shared mutable list shuffled across concurrent comparators | LangSmith `randomize_order` (#3582) [code] | Scores get attributed to the wrong run. |
| Fixed-order pairwise judge with no tie option | promptfoo `select-best`; autoevals `Battle` [code] | Position bias goes unmeasured. |
| Signature-sniffing with a silent positional fallback | LangSmith evaluators; Braintrust scorers [code] | A typo changes semantics without an error. |
| Monkeypatching the test runner for reporting | Scenario pytest plugin [code] | Fragile under concurrency and plugin ordering. |
| Hard dependency on a vendor-hosted model for core simulation | promptfoo simulated-user / GOAT [code] | The OSS feature isn't self-contained. |
| Comparison and dataset versioning only in a hosted backend | DeepEval, Scenario, LangSmith, Braintrust [code/doc] | OSS users can't compare runs locally, and the algorithms can't be audited. |
| Default-on telemetry; silently defaulting to a paid judge model | Ragas (`ragas.md` §Weaknesses) [code] | Surprising data egress and cost. |
