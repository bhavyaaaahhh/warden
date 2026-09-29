# Inspect AI
Repo: https://github.com/UKGovernmentBEIS/inspect_ai @ f753add3909e0a2974249f0fa2c0896b5691f0e5 · License: MIT · Lang: Python (viewer in TS, git submodule `meridianlabs-ai/ts-mono`) · Last commit: 2026-09-28 · Activity: very high; 0.3.272 released 28 Sep 2026, 0.3.271 two days before (docs/CHANGELOG.md), ~2.9k stars, ~340 open issues.

Permalink base used below: `B = https://github.com/UKGovernmentBEIS/inspect_ai/blob/f753add3909e0a2974249f0fa2c0896b5691f0e5/src/inspect_ai`

## Purpose
Inspect is an offline eval framework from the UK AI Security Institute. Its main users are researchers who evaluate frontier models on benchmarks and agentic tasks, where the tasks often run inside sandboxes. You describe an eval in Python as `Task(dataset, solver, scorer)` and run it with the `inspect eval` CLI or `eval()`. Each run writes a self-contained `.eval` log file, and you browse those with `inspect view`. The system under test is "a model plus a solver/agent" that Inspect drives itself. Inspect does not score traces from an external application. There is no production or online ingestion path. The only exception is the "agent bridge", which runs third-party agent code inside Inspect.

## Core abstractions

**Batch execution: `Task`.** `_eval/task/task.py:Task.__init__` ([_eval/task/task.py#L81-L125](https://github.com/UKGovernmentBEIS/inspect_ai/blob/f753add3909e0a2974249f0fa2c0896b5691f0e5/src/inspect_ai/_eval/task/task.py#L81-L125))
```python
Task(dataset: Dataset|Sequence[Sample]|SampleSource|None, setup: Solver|list[Solver]|None,
     solver: Solver|Agent|list[Solver] = generate(), cleanup=..., scorer: Scorers|None,
     metrics=..., model=..., config: GenerateConfig, model_roles=..., sandbox=...,
     approval=..., epochs: int|Epochs|None, fail_on_error: bool|float|None,
     continue_on_fail=..., score_on_error=..., message_limit, token_limit, turn_limit,
     time_limit, working_limit, cost_limit, early_stopping, version: int|str = 0, ...)
```
- `Task` is a declarative bundle. It carries data, the solving strategy, scoring, limits and error policy all at once, plus `version` for tracking changes to the task spec.
- `setup` always runs, even when the CLI swaps out `solver` (`--solver`). That separation lets you hold the environment fixed while comparing agents.
- `Scorers` (`scorer/_scorers.py:Scorers`) is `Scorer | Scanner[Transcript] | Sequence[...]`, so a task can carry several scorers. "Scanners" over transcripts count as scorers too.
- Trade-off: task identity is a Python function plus its args, not a stored dataset version.

**Unit of data: `Sample`.** `dataset/_dataset.py:Sample` ([dataset/_dataset.py#L29-L43](https://github.com/UKGovernmentBEIS/inspect_ai/blob/f753add3909e0a2974249f0fa2c0896b5691f0e5/src/inspect_ai/dataset/_dataset.py#L29-L43))
```python
Sample(input: str | list[ChatMessage], choices: list[str]|None, target: str | list[str] = "",
       id: int|str|None, metadata: dict|None, sandbox: SandboxEnvironmentType|None,
       files: dict[str,str]|None, setup: str|None, checkpoint: ...)
```
- `input` can be a full `list[ChatMessage]`, so you can seed a multi-turn prefix (earlier user, assistant and tool turns) and evaluate the next turn(s).
- `target` is a string or a list of strings. For a model grader it is free-form "criterion" text. There is no structured expected tool-call trajectory type. Anything like that goes in `metadata`.
- The sandbox can be set per sample, along with files and a setup script.
- `Dataset` is an abstract `Sequence[Sample]` (`dataset/_dataset.py:Dataset`, L144). `MemoryDataset` is the concrete class. Loaders handle CSV, JSON and HF.

**Per-sample state: `TaskState`.** `solver/_task_state.py:TaskState` ([solver/_task_state.py#L134-L180](https://github.com/UKGovernmentBEIS/inspect_ai/blob/f753add3909e0a2974249f0fa2c0896b5691f0e5/src/inspect_ai/solver/_task_state.py#L134-L180))
```python
TaskState(model, sample_id, epoch, input, messages: list[ChatMessage], target, choices,
          output: ModelOutput|None, message_limit, token_limit, cost_limit, completed,
          metadata, store: dict|None, scores: dict[str, Score]|None, sample_uuid)
```
- `messages` is the mutable conversation. The first message comes from `Sample.input`.
- `output` is the last `ModelOutput`. Most scorers read `output.completion`.
- `store` is a typed key/value scratchpad that is also persisted into the log.
- `tools` sits on the state, so solvers can add or remove tools mid-run.
- Scorers receive the entire `TaskState`, not just the output. A scorer can therefore inspect the full message history and tool calls, but the framework gives it no helpers for doing so.

**Execution step: `Solver` and `Generate`.** `solver/_solver.py:Solver` ([solver/_solver.py#L79-L85](https://github.com/UKGovernmentBEIS/inspect_ai/blob/f753add3909e0a2974249f0fa2c0896b5691f0e5/src/inspect_ai/solver/_solver.py#L79-L85)), `Generate` (L37-L62)
```python
class Solver(Protocol):
    async def __call__(self, state: TaskState, generate: Generate) -> TaskState: ...
class Generate(Protocol):
    async def __call__(self, state, tool_calls: Literal["loop","single","none"]="loop", **cfg) -> TaskState: ...
```
- Composition is sequential through `solver/_chain.py:chain(*solvers)` (L13), or by passing a list to `Task`.
- `generate(tool_calls="loop")` is the simplest agent loop: keep generating and running tools until there are no more tool calls or a limit is hit.
- The design is a state-transformer pipeline. Because every solver can rewrite `messages`, a solver can inject a prompt, run a critique, or play a fake user turn.

**Agent: `Agent` and `AgentState`.** `agent/_agent.py:Agent` ([agent/_agent.py#L94-L120](https://github.com/UKGovernmentBEIS/inspect_ai/blob/f753add3909e0a2974249f0fa2c0896b5691f0e5/src/inspect_ai/agent/_agent.py#L94-L120)), `AgentState` (L36)
```python
class Agent(Protocol):
    async def __call__(self, state: AgentState, *args, **kwargs) -> AgentState: ...
# AgentState holds only messages + output
```
- This is a newer and narrower interface than `Solver`: it sees only messages and output. You can pass an agent directly as a Task solver (`resolve_solver` accepts `Agent`), and `agent/_as_solver.py:as_solver` adapts it with its own limits.

**Unit of scoring: `Scorer`, returning `Score`.** `scorer/_scorer.py:Scorer` ([scorer/_scorer.py#L35-L40](https://github.com/UKGovernmentBEIS/inspect_ai/blob/f753add3909e0a2974249f0fa2c0896b5691f0e5/src/inspect_ai/scorer/_scorer.py#L35-L40)), `scorer/_metric.py:Score` (L112), `Value` (L67)
```python
class Scorer(Protocol):
    async def __call__(self, state: TaskState, target: Target) -> Score | None: ...
Value = Union[str|int|float|bool, Sequence[str|int|float|bool], Mapping[str, str|int|float|bool|None]]
class Score(BaseModel): value: Value; answer: str|None; explanation: str|None;
                        reason: ScoreReason|str|None; metadata: dict|None; history: list[ScoreEdit]
```
- `Value` can be a scalar, a list or a dict. A dict lets one scorer emit several named dimensions, and metrics then apply per key.
- `Score.unscored(reason=...)` (L158) stores a NaN value. Metrics and reducers skip it, but the explanation is kept. This separates "the grader failed" from "the answer was wrong".
- `history: list[ScoreEdit]` with `provenance` (L82-L110) records human or post-hoc score edits as an audit trail.
- `@scorer(metrics=[accuracy(), stderr()])` attaches default metrics to each scorer. `Task(metrics=...)` overrides them.

**Metric and reducer.** `scorer/_metric.py:MetricProtocol` (L349) is `(scores: list[SampleScore]) -> Value`. A `ScoreReducer` is `list[Score] -> Score`, and reducers combine the epochs of a single sample (`scorer/_reducer/reducer.py`).

## Multi-turn / agent evaluation
- **ReAct loop.** `agent/_react.py:react()` ([agent/_react.py#L53-L68](https://github.com/UKGovernmentBEIS/inspect_ai/blob/f753add3909e0a2974249f0fa2c0896b5691f0e5/src/inspect_ai/agent/_react.py#L53-L68)). Its parameters are `tools`, `attempts`, `submit`, `on_continue`, `retry_refusals`, `compaction`, `truncation`, `approval` and `review`.
  - The main loop ([L248-L400](https://github.com/UKGovernmentBEIS/inspect_ai/blob/f753add3909e0a2974249f0fa2c0896b5691f0e5/src/inspect_ai/agent/_react.py#L248-L400)) runs: drain channel messages, checkpoint tick, `_agent_generate`, handle `model_length` overflow (compaction or truncation), then `execute_tools`, then check for a `submit()` call.
  - With `attempts>1`, the loop calls the task scorer inside the loop (`answer_scores = await score(state)`, ~L322). If the score isn't 1.0 it tells the model the answer was wrong and continues.
  - When the model stops calling tools, `on_continue` plays back a user message such as "please continue / call {submit}".
  - After three consecutive `content_filter` stops the loop breaks.
- **Tools.** `tool/_tool.py:Tool` (L82) is an async callable. Its schema comes from the signature and docstring.
  - `model/_call_tools.py:execute_tools` (L225) runs tool calls. Tool exceptions are mapped to `ToolCallError` kinds (`parsing`, `approval`, `limit`, `unknown`, ...) and returned to the model instead of crashing the sample (L195-L222).
  - Approval policies (`approval/`) gate tool calls, and review policies check tool results.
- **Sandboxes.** `util/_sandbox/environment.py:SandboxEnvironment` (L138) is an ABC with `exec`, `write_file`, `read_file` and `connection`, plus lifecycle classmethods `task_init`, `sample_init`, `sample_cleanup` and `task_cleanup`. It is registered as registry type `sandboxenv`, and Docker and local backends ship with Inspect.
- **Handoffs and multi-agent.** `agent/_handoff.py:handoff(agent, input_filter, output_filter=content_only, tool_name, limits)` (L19) wraps an agent as a `transfer_to_<name>` tool (`AgentTool`, L83).
  - `execute_tools` intercepts the call and runs the sub-agent over the shared history. Input and output filters rewrite that history, and the default output filter strips reasoning and system messages.
  - `as_tool()` is the alternative. It exposes the agent as a string-in/string-out tool.
- **Agent bridge.** `agent/_bridge/` runs external agents written against the OpenAI, Anthropic or Google SDKs and redirects their model calls through Inspect, so their conversations are logged and limited.
- **Intervention channel.** `agent/_channel/` (`AgentChannel`, `AgentRef.post/interrupt`) lets an operator inject user messages or cancel a running agent between turns. It is used by the ACP transport and human-in-the-loop setups (`agent/_human/`).
- **User simulation: not built in.** Searching `src/` for "simulated user" or "user simulation" finds nothing. The only hit is `docs/evals/evals.json` ("UserBench (SWE user simulation)"), which lives in the separate inspect_evals catalog. To simulate a user you write a custom solver or agent that alternates between the agent under test and a second model acting as the user (a `model_roles` entry). You can also use a custom `on_continue` callback that generates the next user turn. Both are user-written patterns. There is no first-class simulator with a persona, goal and stop condition.

## Scorers and aggregation
- **Built-ins** (`scorer/`): `match`, `includes`, `pattern`, `answer`, `choice`, `exact`, `f1`, `math`, `model_graded_qa`, `model_graded_fact`, `multi_scorer`, precomputed and perplexity scorers.
- **Multiple scorers.** Pass a list to `Task(scorer=[...])`. Each scorer's scores are keyed by registry name in `TaskState.scores` and `EvalSample.scores`. Scorers can also be re-run after the fact on an existing log with `_eval/score.py:score(log, scorers, epochs_reducer, action="append"|"overwrite")` (L81), which is the `inspect score` CLI.
- **`multi_scorer(scorers, reducer)`** (`scorer/_multi.py#L19`) runs scorers in parallel (`tg_collect`) and reduces them to one `Score`, for example a majority vote across graders. If every sub-scorer returns None, it produces `unscored(reason="scoring_failed")`.
- **Epoch reducers** (`scorer/_reducer/reducer.py`): `mean`, `median`, `mode`, `majority`, `max`, `collect`, `at_least(k)` (L130), `pass_at(k)` (L164, the Chen 2021 unbiased estimator) and `pass_k(k)` (L209, `C(correct,k)/C(total,k)`, the "pass^k" all-k-succeed estimator from tau-bench). Both return NaN (unscored) when fewer than k scored epochs remain. `Task(epochs=Epochs(n, reducer=[...]))` accepts several reducers, and each one produces its own metric view.
- **Aggregation order** (`_eval/task/results.py:eval_results` L90, `compute_eval_scores` L251, `reduce_scores` L703): per-sample epoch scores go through each reducer, then each scorer's metrics run over the reduced `SampleScore`s. `EvalSampleReductions` holds the reduced per-sample values in the log.
- **Metrics**: `accuracy`, `mean`, `std`, `stderr` (with clustered SE via sample metadata, `scorer/_metrics/std.py:_cluster_partition` L57), `bootstrap_stderr`, grouped metrics, categorical, and Krippendorff agreement (`scorer/_metrics/`).

## Judge implementation
- **Entry point.** `scorer/_model.py:model_graded_qa(template, instructions, grade_pattern, include_history=False, partial_credit, model: list|str|Model|None, model_role="grader", reducer="majority")` ([scorer/_model.py#L115-L124](https://github.com/UKGovernmentBEIS/inspect_ai/blob/f753add3909e0a2974249f0fa2c0896b5691f0e5/src/inspect_ai/scorer/_model.py#L115-L124)). `model_graded_fact` (L35) is a variant for fact inclusion.
- **Grader resolution.** The grader model comes from the `grader` model role when scoring runs. If that role is bound to a list of models, the scorer fans out and majority-votes through `multi_scorer` (~L230-L243).
- **Prompts.** Templates are in the same file (`DEFAULT_MODEL_GRADED_QA_TEMPLATE`, L366), adapted from OpenAI evals `closedqa`.
- **History.** `include_history=True` passes the full `chat_history(state)` as the question, which is how you judge a multi-turn conversation. The default grades only the original input.
- **Parsing** (`_model_graded_qa_single`, L246):
  - The regex `DEFAULT_GRADE_PATTERN` (L432) uses a greedy leading `.*`, so it binds to the last `GRADE: X` in the grader's output. This is a deliberate defence against prompt injection through the submission.
  - With the default instructions, the verdict is also checked against the grades that were actually offered (`C/I`, or `C/P/I` with partial credit).
  - If parsing fails, the result is `Score.unscored(reason="grader_failed")` rather than "incorrect". The grading prompt and grader message are kept in `metadata["grading"]`.
- **Calibration.** Nothing beyond majority voting across graders and agreement metrics (Krippendorff). There is no built-in validation against human labels.

## Execution model
- **Async.** anyio task groups, one process. Sample concurrency uses a task-scoped semaphore (`_eval/task/run.py:create_sample_semaphore`, L3720). When `max_samples` isn't set, it follows the model's adaptive connection controller. `GenerateConfig.max_connections` and adaptive concurrency (min 10, start 20, max 100 by default) are in `model/_generate_config.py` (L93-L97). `max_tasks`, `max_subprocesses` and `max_sandboxes` are further `eval()` knobs (`_eval/eval.py`, L166-L170).
- **Retries, at two levels:**
  - Model API calls use tenacity (`model/_model.py` imports `retry`). `ModelAPI.should_retry` returns a `RetryDecision` (transient or rate_limit, with retry-after), and `GenerateConfig` has `max_retries`, `timeout`, `attempt_timeout` and `stream_idle_timeout`.
  - Samples retry through `retry_on_error=N` (`task_run_sample`, L2190, `SampleAttempt`). A retry re-queues behind the semaphore.
  - Whole failed evals are retried with `eval_retry()` (`_eval/eval.py`, L1307) and `eval_set` retries, which reuse completed samples from the prior log.
- **Failure handling** (`task_run`, `SampleErrorHandler`, L909):
  - `fail_on_error`: True means stop at the first error, a float is a proportion, an int >1 is a count. `continue_on_fail` defers the failure to the end of the run.
  - `score_on_error` scores errored samples anyway.
  - `LimitExceededError` is not an error. It records `EvalSampleLimit(type, limit)` and still scores the last state (L2923-L2930), so hitting a limit becomes a scored outcome and not a crash.
- **Limits.** Per sample: `message_limit`, `token_limit` (formulas such as `"output:1m"` or `"(input*0.1)+output:1m"`), `turn_limit`, `time_limit`, `working_limit` (excludes time waiting on retries), `cost_limit`. Scoped limits are also available per handoff and per agent.
- **Caching** (`model/_cache.py`):
  - `CachePolicy(expiry="1W", per_epoch=True, scopes={})` (L58). Entries are keyed by an md5 of generate config, input messages (without ids), base_url, tool_choice, tools, expiry, scopes and, by default, the epoch (`_cache_key`, L185).
  - The cache is a pickle-on-disk store under the Inspect cache dir. `per_epoch=True` means cache hits don't collapse epochs, which keeps variance estimates honest.
- **Checkpointing and resume.** `Task(checkpoint=..., on_checkpoint, on_resume)` takes mid-sample snapshots (every 500k tokens by default), and the react loop calls `cp.tick()` each turn.

## Data model & storage
- **`.eval` file.** A zip (`log/_recorders/eval.py`). It contains `header.json`, `start.json`, `results.json`, `reductions.json`, `summaries.json`, a `_journal/` dir for incremental writes, and one member per sample per epoch (`_sample_filename(id, epoch)`). It is written incrementally, with a `log_buffer` option. A `.json` recorder also exists.
- **`EvalLog`** (`log/_log.py`, L1218) has `version=2, status, eval: EvalSpec, plan, results, stats, error, invalidated, log_updates, tags, metadata, samples, reductions`. The field order is part of the format.
- **`EvalSpec`** (L1013) records eval_set_id, eval_id, run_id, task and task_version, task_args, solver and solver_args, dataset (name, location, sample ids), model and `model_generate_config`, `model_roles`, `config: EvalConfig`, `revision: EvalRevision(type="git", origin, commit, dirty)` (L997), and `packages` (installed versions). Enough is stored to reproduce and attribute a run.
- **`EvalSample`** (L423) holds id, epoch, input, target, `messages`, `output`, `scores: dict[str, Score]`, metadata, `store`, `events` (the full transcript of model, tool, sandbox, approval, limit and error events), `timelines`, `model_usage`, `role_usage` and limit.
- **Comparing runs.** The core has no log-diff command. Comparison works like this:
  - (a) The `inspect view` web viewer lists logs in a directory. Its UI source is in the `ts-mono` submodule, which I didn't check out, so I couldn't verify its compare features.
  - (b) `inspect_ai.analysis` dataframes (`evals_df`, `samples_df`, `messages_df`, `events_df` in `analysis/_dataframe/`) let you do the joins yourself in pandas.
  - (c) Eval sets (`_eval/evalset.py:eval_set`, L229) are an idempotent matrix of tasks × models in a single `log_dir`. `task_identifier` (L2058) hashes task file, name, args, model, solver, generate config, model args, roles, version and limits, and uses that to pair tasks with existing logs so completed work is skipped on re-run.
  - The open issue [#1327 "Side by side comparison of two models"](https://github.com/UKGovernmentBEIS/inspect_ai/issues/1327) (Feb 2025) confirms that per-sample A/B diffing isn't a first-class feature.

## Online / production path
None. Evals only run inside Inspect's own runtime. Bridged agents are still run by Inspect, not observed in production.

## Extension model
- **Registry** (`_util/registry.py`). `RegistryType` (L41) covers agent, approver, hooks, reviewer, metric, modelapi, plan, sandboxenv, score_reducer, scorer, solver, task, task_source, tool, loader, scanner and validation_predicate. Decorators (`@task`, `@solver`, `@scorer`, `@metric`, `@score_reducer`, `@agent`, `@tool`, `@modelapi`, `@sandboxenv`, `@hooks`) register objects under names, so the CLI and logs can refer to them as strings (`SolverSpec`, `ScorerSpec`) and rebuild them from a log.
- **Packaging.** Setuptools entry-point group `inspect_ai` (`_util/entrypoints.py:ensure_entry_points`). Third-party packages expose names such as `pkg/scorer_name`, which are loaded lazily when first looked up.
- **Hooks** (`hooks/`) are lifecycle callbacks for eval, task and sample events, typically used for external logging.

## Strengths
- **Limits are recorded as outcomes.** An agent that runs out of turns, tokens or time is still scored, and the limit type is stored with the sample (`run.py`, L2923; `EvalSampleLimit`). That is the right semantics for agent regression tests.
- **The scoring failure taxonomy is principled.** `Score.unscored` / NaN with a `reason` (`_metric.py`, L158) keeps grader failures out of accuracy while preserving them for audit. The judge parser binds to the last grade and validates it against the offered set (`_model.py`, L432).
- **Repeated-trial statistics are built in:** `pass_at`, `pass_k` and `at_least` reducers (`reducer.py`, L130-L246), several reducers at once, and clustered and bootstrap stderr. There is a rigorous path from epochs to a number with error bars.
- **The logs are rich and reproducible.** A per-sample event transcript, git revision, package versions and the full config are in `EvalSpec` and `EvalSample`. `inspect score` re-scores an old log without re-running the agent (`_eval/score.py`, L81), which lets you iterate on scorers cheaply.
- **The agent building blocks are strong:** react loop, handoffs with history filters, sandbox ABC, approval policies, compaction, checkpoint/resume, and adaptive concurrency.

## Weaknesses
- **No built-in user simulator.** For multi-turn conversational agents (as opposed to autonomous tool-use agents), you have to build the user-turn loop yourself (grep result above).
- **No structured tool-trajectory target or scorer.** `Sample.target` is `str | list[str]`. Checking "called tool X with args Y in order" needs a custom scorer walking `state.messages`.
- **No first-class run-to-run comparison or regression gate** in the core ([#1327](https://github.com/UKGovernmentBEIS/inspect_ai/issues/1327)). Comparison means dataframes and your own analysis.
- **Log scale pain.** Large `.eval` files are slow to read ([#4931](https://github.com/UKGovernmentBEIS/inspect_ai/issues/4931), [#5074](https://github.com/UKGovernmentBEIS/inspect_ai/issues/5074)), and re-scoring has memory issues ([#2735](https://github.com/UKGovernmentBEIS/inspect_ai/issues/2735)). File-per-run storage doesn't give you cross-run queries.
- **The system under test must run inside Inspect.** The model, solver and agent are driven by the framework, so evaluating an existing production agent means bridging or re-hosting it.
- **The API surface is huge and fast-moving.** Roughly 40 `Task` kwargs, a deprecated `Metric` signature (`_metric.py`, ~L370), and releases every few days.

## Worth adapting
- **Treat limit hits as scored terminal states** with a recorded limit type. This avoids hiding agents that "fail by timeout".
- **An `unscored` NaN sentinel with a reason code**, which metrics and reducers skip. It separates measurement failure from model failure.
- **Epoch reducers as pluggable `list[Score] -> Score`** with `pass@k`, `pass^k` and `at_least`. This gives consistency metrics for nondeterministic agents.
- **Dict-valued `Score.value`**, so one scorer emits several named dimensions with per-key metrics.
- **Re-score an existing run** (`score(log, scorers, action="append")`), which separates expensive generation from cheap scoring iteration.
- **A content-hash task identity** (`task_identifier`) that excludes transport knobs. It makes re-runs idempotent and lets you pair results across runs.
- **A per-epoch cache key**, so caching doesn't silently collapse repeated trials.
- **Handoff input/output message filters**, which control what a sub-agent sees and returns.

## Don't copy
- **`TaskState` as a mutable grab-bag.** Solvers can rewrite `messages`, `tools`, `output`, `store` and `completed` at will, so it is hard to tell after the fact what the "system under test" produced versus what the harness injected. The event transcript makes up for this only partly.
- **Scorers called inside the agent loop** (`react(attempts>1)` calls `score(state)`, ~L322). This mixes ground-truth leakage into the trajectory. It fits benchmarks with retries but is dangerous for regression testing.
- **Zip-file-per-run as the only store.** It's portable, but cross-run queries, per-sample diffs and dashboards turn into full-file scans (issues #4931, #5074).
- **The pickle-on-disk generate cache** (`cache_store`). It isn't portable, isn't shareable across machines, and is unsafe to load from untrusted sources.

## Open questions
- What the `inspect view` UI supports for multi-log comparison. The `ts-mono` submodule wasn't checked out.
- Whether `inspect_evals` (a separate repo) contains a reusable user-simulator solver, for example in its UserBench or tau-bench ports. I didn't inspect it.
- How `Scanner[Transcript]` scorers (`inspect_scout` integration) differ in execution from regular scorers.
- Exact semantics of `early_stopping` callbacks and `SampleSource` dynamic datasets. I saw these in signatures only.
