# LangSmith SDK (Python) + openevals + agentevals

Repos:
- LangSmith SDK: https://github.com/langchain-ai/langsmith-sdk @ `3e225e3a1aec4f0b0e2b0bd3a188dc79da56d320` · License: MIT · Lang: Python + TS · Last commit: 2026-09-28 · Activity: very active, v0.14.1 released 2026-09-25, weekly releases, ~1.1k stars
- openevals: https://github.com/langchain-ai/openevals @ `efaf799535b2f7b8a0ca8fe2e0d14d355e719f40` · License: MIT · Lang: Python + TS · Last commit: 2026-09-28 · Activity: openevals 0.2.1 / js 0.2.2 (2026-08-18), ~1.2k stars
- agentevals: https://github.com/langchain-ai/agentevals @ `946ad15b9f2ce9bc9843edd2bb773bd07d78ca7e` · License: MIT · Lang: Python + TS · Last commit: 2026-07-13 · Activity: slowing. Last release js 0.0.7 (2026-03). openevals now carries its own copy of the trajectory evaluators (see below).

LangSmith's server is closed source. Server behavior is labeled `[doc]` (docs.langchain.com, fetched 2026-09-29) or `[inferred]`. Everything else is `[code]` at the SHAs above.

Permalink bases used below:
- `LS` = https://github.com/langchain-ai/langsmith-sdk/blob/3e225e3a1aec4f0b0e2b0bd3a188dc79da56d320/python/langsmith
- `OE` = https://github.com/langchain-ai/openevals/blob/efaf799535b2f7b8a0ca8fe2e0d14d355e719f40/python/openevals
- `AE` = https://github.com/langchain-ai/agentevals/blob/946ad15b9f2ce9bc9843edd2bb773bd07d78ca7e/python/agentevals

## Purpose

The LangSmith SDK is the client for a hosted platform. It covers tracing plus offline evaluation (`evaluate`/`aevaluate` over a dataset, plus a pytest/vitest integration). Online evaluation of production runs and threads happens server-side `[doc]`. openevals is a library of evaluator factories with no server dependency: LLM-as-judge, exact/JSON/string/embedding match, code checks, a multi-turn user simulator, and (now) trajectory match. agentevals is the older agent-specific library: trajectory match, LangGraph graph-trajectory evaluators and trajectory judges. All three return plain `{key, score, comment}` dicts that LangSmith ingests as "feedback". They are aimed at app developers running offline experiments. openevals and agentevals judges run anywhere, but they auto-log to LangSmith when called inside a LangSmith pytest case.

## Core abstractions

**Unit of data: `Example`**. `LS/schemas.py:ExampleBase` (L82) / `Example` (L139).
```python
class ExampleBase(BaseModel):
    dataset_id: UUID
    inputs: Optional[dict[str, Any]]
    outputs: Optional[dict[str, Any]]      # "reference outputs"
    metadata: Optional[dict[str, Any]]     # splits live here: metadata["dataset_split"]
class Example(ExampleBase):
    id: UUID; created_at: datetime; modified_at: Optional[datetime]
    source_run_id: Optional[UUID]; attachments: Optional[dict[str, AttachmentInfo]]
```
Trade-off: an example is untyped `inputs`/`outputs` dicts. That is flexible, but a multi-turn conversation or an expected tool trajectory is just a convention inside `outputs`. The data model has no first-class "conversation" or "expected trajectory" type. Splits are stored as a list in `metadata["dataset_split"]`, and an example can be in several splits (read in `_ExperimentManager._get_dataset_splits`, `LS/evaluation/_runner.py` L1905-1920).

**Dataset + version**: `LS/schemas.py:Dataset` (L248), `DatasetVersion` (L296: `tags: list[str]`, `as_of: datetime`), `DataType` kv/llm/chat. Versioning is implicit and timestamp-based. Every example add/update/delete creates a new version `[doc]` (manage-datasets). The SDK addresses a version by `as_of` (a timestamp or a tag name) in `Client.list_examples(..., as_of=, splits=, metadata=, filter=)` (`LS/client.py` L7365). It tags a version with `Client.update_dataset_tag(as_of=, tag=)` (L6135) and edits splits with `Client.update_dataset_splits(split_name, example_ids, remove)` (L7898). Trade-off: versions come for free with no user action. The cost is that the experiment has to *record* which version it used, and it does so loosely (see Data model).

**Execution record: `Run`**. `LS/schemas.py:RunBase` (L307). It has `id, name, run_type, inputs, outputs, error, reference_example_id, parent_run_id, trace_id, dotted_order, session_id` plus token and cost fields. A run is a node in a trace tree, and `reference_example_id` links the root run to an example. Tool calls are child runs, not a structured field.

**Batch execution: `Experiment` = `TracerSession`**. `LS/schemas.py:TracerSession` (L731). An experiment is a tracing project with `reference_dataset_id` set. Trade-off: experiments reuse the tracing store wholesale, so every eval row is a full trace. The downside is that experiment-level facts (dataset version, splits, num_repetitions, git info) sit in free-form project `metadata`.

**Unit of scoring: evaluator → `EvaluationResult`**. `LS/evaluation/evaluator.py:EvaluationResult` (L71):
```python
key: str; score: SCORE_TYPE = None; value: VALUE_TYPE = None
comment: Optional[str]; correction: Optional[dict]; metadata: Optional[dict]
feedback_config: Optional[FeedbackConfig]   # continuous(min,max) | categorical | freeform
source_run_id  # the evaluator's own trace
target_run_id  # defaults to root run
extra: Optional[dict]                       # {"error": True} marks evaluator failures
```
Evaluator signature is resolved by parameter *name* in `_normalize_evaluator_func` (L663). The function may take any subset of `run, example, inputs, outputs, reference_outputs, attachments`. `(run, example)` or any 2-positional-arg function falls back to legacy `(Run, Example)`. Return values are coerced in `_format_evaluator_result` (L932): bool/int/float→`{"score"}`, str→`{"value"}`, list→`{"results": [...]}`, dict as-is. Trade-off: the ergonomics are very good. The binding is magic, though: rename a parameter and the evaluator changes meaning. `outputs` is `run.outputs` of the *root* run only, so a trajectory evaluator has to either get the trajectory from the target's return value or walk `run.child_runs`.

**Summary evaluator**: `_normalize_summary_evaluator` (L972) takes `runs, examples, inputs, outputs, reference_outputs` as *lists*. It is called once per experiment in `_apply_summary_evaluators` (`_runner.py` L1842) and logged as project-level feedback (`create_feedback(run_id=None, project_id=...)`).

**Comparing executions: `ComparativeExperiment` + `ComparisonEvaluationResult`**. `LS/evaluation/evaluator.py:ComparisonEvaluationResult` (L170): `key, scores: dict[run_id, score], comment: str | dict[run_id, str]`. Comparative evaluators accept `runs, example, inputs, outputs (list), reference_outputs` (`_normalize_comparison_evaluator_func` L801). This is an N-way ranking primitive, not strictly pairwise.

**openevals unit of scoring**: `OE/types.py:SimpleEvaluator` protocol: `(*, inputs=None, outputs, reference_outputs=None, **kwargs) -> EvaluatorResult | list[EvaluatorResult]`, with `EvaluatorResult = {key, score, comment, metadata, source_run_id}`. The kwargs mirror LangSmith's named-arg binding, so openevals evaluators plug straight into `evaluate()`.

**agentevals trajectory types**: messages are normalized to OpenAI chat format (`ChatCompletionMessage`: role, content, tool_calls, id). `AE/types.py:GraphTrajectory` (L14) = `{inputs: list[dict]|None, results: list[dict], steps: list[list[str]]}`, i.e. node names per graph invocation.

## Execution model

`evaluate()` (`LS/evaluation/_runner.py:evaluate`, L137) is a dispatcher:
- `target` is an experiment id → `evaluate_existing` (re-score stored runs).
- `target` is a 2-tuple of experiments → `evaluate_comparative`.
- `target` is a callable → `_evaluate` (L1105). An async callable is rejected with a pointer to `aevaluate` (L375-388).

`_evaluate` builds an immutable-ish `_ExperimentManager` (L1392) and chains `.start() → .with_predictions() → .with_evaluators() → .with_summary_evaluators()`. Each step returns a `_copy` with lazily `itertools.tee`'d generators (L1940). `ExperimentResults(manager, blocking=...)` (L551) drains the stream, in a background thread when `blocking=False`.

- **Target invocation**: `_forward` (L2029). `_get_target_args` (L2234) passes `inputs` plus, optionally, `attachments` and `metadata` by parameter name. Each call is wrapped as a traced run with `reference_example_id` and `metadata.example_version = example.modified_at` (L2050-2054).
- **Concurrency (sync)**: `max_concurrency=0` (the default since 0.2.0) runs sequentially. `N` uses `ContextThreadPoolExecutor(N)` for predictions (`_predict` L1653) and a *separate* pool of the same size for evaluators (`_score` L1797). The two stages are pipelined: `_score` polls `as_completed(timeout=0.001)` so early results stream (L1830-1839). `None` means unbounded threads. Evaluators for one row run sequentially inside `_run_evaluators` (L1697).
- **Concurrency (async)**: `aevaluate` (`LS/evaluation/_arunner.py` L79). `awith_predictions_and_evaluators` (L824) creates one coroutine per example that runs target → all evaluators, bounded by an `asyncio.Semaphore(max_concurrency)` in `_internal/_aiter.py:aiter_with_concurrency` (L258). Feedback upload goes through a separate 4-worker thread pool (L838-839). Sync evaluators passed to `aevaluate` run via `aio_to_thread` `[inferred from imports]`.
- **Repetitions**: `num_repetitions` just repeats the example iterator `N` times (`_ExperimentManager.examples`, L1523-1532). Every repetition is a separate run of the same example in the same experiment. The SDK does no aggregation. The UI shows the per-key mean and lets you open the stddev `[doc]` (repetition.md).
- **Retries and timeouts**: none in the eval loop. I found no retry or timeout around `fn(...)` in `_forward` or around `evaluator.evaluate_run` (grep for `timeout`/`wait_for` in `_arunner.py` returns only the 0.001 s polling constant). Rate limiting is left to the user's clients.
- **Caching**: `LANGSMITH_TEST_CACHE` wraps the whole run in a vcrpy cassette per dataset (`_evaluate` L1142-1146 → `utils.with_optional_cache` / `with_cache`, `record_mode="new_episodes"`, `utils.py` ~L632-655). LangSmith's own API host is excluded. This is HTTP-level record/replay, not semantic caching.
- **Target failure**: `_forward` catches the exception and logs it (L2073-2077). The run is still traced with `error` set. `error_handling="log"` (default) keeps it in the experiment. `"ignore"` sets `reference_example_id` only on success (`_on_success`, L2055-2061), so failed runs drop out of the experiment entirely. Evaluators still run on errored rows with `run.outputs` possibly `None`.
- **Evaluator failure**: `_run_evaluators` catches it and synthesizes one `EvaluationResult(key, comment=repr(e), extra={"error": True})` per declared `feedback_keys` (L1748-1784). Errors are recorded as feedback, not dropped. If the keys can't be inferred, the error is only logged.
- **Summary-evaluator failure**: logged and skipped (L1890-1894).
- **Comparative**: `evaluate_comparative` (L688) loads the runs of both experiments and intersects on `reference_example_id` (L917-929). It fetches examples `as_of=projects[0].metadata["dataset_version"]` (L937-941, first experiment's version only, with the comment "We aren't providing any training wheels here"). It then runs each comparator per example in a thread pool (default 5) and writes one feedback per run id, sharing a `feedback_group_id` and `comparative_experiment_id` (L953-987). `randomize_order=True` shuffles position to fight position bias (L960-961).

**openevals multi-turn simulation**: `run_multiturn_simulation` (`OE/simulators/multiturn.py` L131). It is a plain synchronous `while True` loop:
1. `user(trajectory, turn_counter=)` produces a message.
2. The message is merged into the trajectory.
3. `app(latest_user_message, thread_id=)` produces a message.
4. `turn_counter += 1`, then merge.
5. Check `max_turns` or `stopping_condition(trajectory, turn_counter=)`.

After the loop, each `trajectory_evaluators` entry is called with `outputs=trajectory, reference_outputs=`. An evaluator exception is `print`ed and swallowed (L255-256). The whole loop is `@traceable(name="multiturn_simulator")`, and app and user are traced child runs sharing `thread_id` (L22-39).

## Data model & storage

What the SDK writes per experiment `[code]`:
- **Project (experiment)**: created in `_create_experiment` (L1333) with `reference_dataset_id`. On completion `_end` (L1922) patches metadata with `dataset_version = max(example.modified_at)` over the examples actually run (`_get_dataset_version`, L1897) and `dataset_splits` (L1905; `"base"` for examples with no split).
- **Runs**: one root trace per (example × repetition), with `metadata.example_version` per run.
- **Feedback rows**: per (run, key) via `client._log_evaluation_feedback`. Summary feedback is attached to the project. Comparative feedback is attached to runs plus a `comparative_experiment_id`.

Versioning caveats:
- The experiment's "dataset version" is *derived* from example timestamps, not a server-issued version id.
- If you pass a filtered or split iterable, the recorded version is the max `modified_at` of that subset, which can predate edits to other examples.
- `evaluate_comparative` does not check that the two experiments used the same version (L933-934 TODO).

Server `[doc]`: datasets are versioned on every mutation. Past versions are read-only. Tags ("prod") can pin a version, and CI should target tags. Experiments are shown per dataset version in the Tests tab. Open issue #2430 reports that experiment snapshots show pre-update example outputs after `update_example()` (https://github.com/langchain-ai/langsmith-sdk/issues/2430).

**pytest integration** (`LS/testing/_internal.py`):
- `@pytest.mark.langsmith` / `@t.test` → `_run_test` (L1030).
- The test *suite* is a dataset named from `LANGSMITH_TEST_SUITE` or the test module (`_get_test_suite_name` L441, `_get_test_suite` L456, auto-created with `metadata.__ls_runner="pytest"`).
- Each run creates a new experiment. Under pytest-xdist the name is deterministic from `PYTEST_XDIST_TESTRUNUID`, so workers share one experiment (`_get_experiment_name` L423).
- **Example identity**: for datasets created by SDK ≥0.4.33, the example id is `uuid5(dataset_id, hash(inputs), hash(outputs))` (`_get_example_id` L500).
  - Python `_TestCase.end_run` (L918-925) passes the test function's *return value* as `outputs`. The JS side hashes `referenceOutputs` (`js/src/utils/jestlike/index.ts` L666-671).
  - So in Python, a test that returns a non-deterministic value gets a new example id each run. Pass an explicit `id=` to avoid that.
  - The legacy scheme was `uuid5(suite_id + file::func [+ params])` (`_get_example_id_legacy` L511).
- **Example sync**: `sync_example` reads the example, then creates or updates it. So running tests *mutates the dataset*, which bumps its version. After the session, `_end_tests` (L528) tags the dataset version with `git:commit:<sha>` and `git:branch:<name>` and writes git info into experiment metadata (L535-556).
- **Result feedback**: pass/fail is automatic feedback `key="pass"`, score 1/0/None for skipped (`submit_result` L628-665). Extra metrics come through `t.log_feedback` (L1310), `t.log_inputs/log_outputs/log_reference_outputs`, and `expect(...)` matchers that log `key="expectation"` (`_expect.py` L109-120).
- **Caching**: `cache=`/`cached_hosts=` enable vcrpy cassettes per suite (L335-359, L1086-1089).
- **Vitest/Jest** (`js/src/utils/jestlike/index.ts`): same model. `ls.describe` = dataset, `ls.test` = example. Pass is logged as feedback `{key: "pass", score: true/false}` (L602, L622), commits are tagged (L444-447), and a custom reporter prints a table `[doc]` (vitest-jest.md).
- openevals judges detect the pytest context via `_TEST_CASE.get()` and auto-log feedback inside `t.trace_feedback()` (`OE/utils.py:_run_evaluator_untyped` L197, L265-296).

## Judge implementation

**`create_llm_as_judge`** (`OE/llm.py` L547) → `_create_llm_as_judge_scorer` (L145).

Prompt:
- `prompt` may be an f-string template, a LangChain prompt/`StructuredPrompt`, or a callable returning messages.
- Template variables are `{inputs}`, `{outputs}`, `{reference_outputs}` plus any extra kwargs. Dicts are stringified first (`_stringify_prompt_param`). Variables that are `None` are dropped, so `str.format` raises `KeyError` if the template needs them.
- An optional `system` message is prepended.
- Attachments are spliced at `{attachments}` as multimodal blocks (L207-223).

Prebuilt prompts:
- They live in `OE/prompts/` as string constants: quality, rag, safety, security, trajectory, conversation, image and voice. Conversation prompts include `TASK_COMPLETION_PROMPT`, `USER_SATISFACTION_PROMPT`, `KNOWLEDGE_RETENTION_PROMPT`, `PERCEIVED_ERROR_PROMPT` and `AGENT_TONE_PROMPT`.
- Style is XML-sectioned `<Rubric>/<Instructions>/<input>/<output>/<reference_outputs>` (e.g. `OE/prompts/quality/correctness.py`).
- Conversation prompts take the whole transcript as `{outputs}` and are boolean. `TASK_COMPLETION_PROMPT` returns TRUE only if every human request was completed without a re-ask.

Few-shot:
- `_append_few_shot_examples` (L37) appends `<example><input>…</input><output>…</output><reasoning>…</reasoning><score>…</score></example>` blocks to the *last user message*. It raises if there is none.
- `FewShotExample = {inputs, outputs, score, reasoning?}` (`OE/types.py`).
- The examples are static and hand-supplied. There is no retrieval of similar examples and no link to human-corrected feedback in the SDK.

Score modes (`_construct_default_output_json_schema`, L70):
- default → `{"type": "boolean"}`
- `continuous=True` → `{"type": "number"}` with the 0-1 range stated *only in the description*. No `minimum`/`maximum`, so out-of-range values pass through.
- `choices=[…]` → `{"type": "number", "enum": choices}`
- `use_reasoning=True` (default) adds a `reasoning` string field placed *before* `score`, with the instruction to end with "Thus, the score should be: X". Reasoning becomes the feedback `comment`.
- `output_schema=` replaces all of this and returns the raw dict (L601-604, L636-643).

Parsing:
- With a LangChain model: `judge.with_structured_output(schema)` (L274).
- With a raw OpenAI client: `response_format={"type":"json_schema", strict: True}` then `json.loads(content)` (L286-339). The model string can be `"openai:..."`.
- Neither path has retry, parse-failure fallback, temperature pinning, or multiple-sample voting. A parse failure raises into the caller. Inside `evaluate()` that becomes an `extra.error` feedback row.

Calibration: none in code. Scores are not calibrated or aggregated. There is no position-bias control in single-output judges. openevals issue #212 measured roughly 30 points of presentation bias on near-tied candidates (https://github.com/langchain-ai/openevals/issues/212). Judge-vs-human alignment is a server feature ("align evaluators" / annotation queues) `[doc]`, not in these libs.

**Trajectory judges**:
- `agentevals.trajectory.llm.create_trajectory_llm_as_judge` (`AE/trajectory/llm.py` L98) reuses openevals' private `_create_llm_as_judge_scorer`. It formats the message list to a string with `_chat_completion_messages_to_string` and uses `TRAJECTORY_ACCURACY_PROMPT(_WITH_REFERENCE)`. The rubric is "logical progression, relatively efficient, semantically equivalent to reference".
- Unlike `create_llm_as_judge`, it has no `system` parameter even though its docstring documents one (agentevals #64).
- `create_graph_trajectory_llm_as_judge` (`AE/graph_trajectory/llm.py` L85) renders `<input>/<trajectory>/<result>` per graph invocation (`_format_thread` L45). It explains `__start__` and `__interrupt__` in the prompt.

## Multi-turn simulation and trajectory model (openevals)

- **Trajectory model**: a flat `list[ChatCompletionMessage]` in OpenAI format. `_trajectory_reducer` (`OE/simulators/multiturn.py` L51) coerces every update to OpenAI messages, assigns uuid `id`s, and appends messages whose `id` is new (dedup by id, LangGraph `add_messages`-style). No per-turn structure is kept. Turn boundaries are implicit, and `turn_counter` is removed before returning (L244).
- **App contract**: `app(inputs: ChatCompletionMessage, *, thread_id)` receives *only the latest user message*. The app must keep its own conversation state keyed by `thread_id`. It returns one message, coerced by `_coerce_and_assign_id_to_message` (L42). The tests return `res["messages"][-1]` (`python/tests/simulators/test_multiturn.py` L102). So tool calls and tool results inside a turn are **not** in the simulated trajectory unless the app returns them some other way. You score the conversation surface, not the agent's internal trajectory.
- **Simulated user**: `create_llm_simulated_user(system, model|client, fixed_responses)` (`OE/simulators/prebuilts.py` L18). It uses `fixed_responses[turn]` while available, then calls the LLM.
  - Role flip: user↔assistant are swapped so the simulator LLM "plays" the user. Tool messages and assistant messages with tool_calls are filtered out (`_is_internal_message` L11).
  - If there are no messages yet, it sends a bootstrap "Generate an initial query…" prompt.
  - A static list of strings also works as `user`. That case raises if turns exceed the list length (L114-117).
- **Stopping**: `max_turns` and/or `stopping_condition(trajectory, turn_counter)`. At least one is required (L198-201). There is no built-in "user is satisfied / said goodbye" detector. You write that as a stopping condition, often an LLM call.
- **Scoring**: only at the end, over the final trajectory. There is no per-turn scoring hook. Results come back in `MultiturnSimulationResult{evaluator_results, trajectory}` (`OE/types.py` L90). Running it inside `evaluate()` means wrapping the simulation as the target and returning the trajectory as `outputs` `[inferred]`.

## Trajectory match (agentevals, mirrored in openevals)

`create_trajectory_match_evaluator(trajectory_match_mode, tool_args_match_mode, tool_args_match_overrides)` (`AE/trajectory/match.py` L24; openevals copy at `OE/trajectory/match.py` L24). The feedback key is `trajectory_<mode>_match` and the score is boolean.

Modes:
- **strict** (`AE/trajectory/strict.py:_scorer` L20): same length, and pairwise per position:
  - same `role`
  - both-or-neither have `tool_calls`
  - same number of tool calls, matched by name + arg matcher *order-independently within one message* (a `seen[]` array, L53-70)
  - message `content` is **not** compared
- **unordered** (`unordered.py:_scorer`): `superset(out, ref) and superset(ref, out)`, over the *flattened list of all tool calls* (`_extract_tool_calls` L78). Message roles, content and non-tool messages are ignored.
- **superset** (`superset.py`): output tool calls ⊇ reference. Extra calls are allowed.
- **subset** (`subset.py`): output tool calls ⊆ reference, i.e. no calls outside the reference.
- `_is_trajectory_superset` (`AE/trajectory/utils.py` L90) is **greedy first-fit**. For each reference call it takes the first unused output call with the same name whose args match. Call multiplicity is respected, but a non-exact arg matcher (subset/superset/custom) can greedily consume the "wrong" output call and yield a false negative that a bipartite matching would avoid.

Tool-arg matching (`_get_matcher_for_tool_name` L194):
- Global `tool_args_match_mode`: `exact` (dict `==`), `ignore`, `subset` (output args ⊆ ref), `superset` (output args ⊇ ref).
- Per-tool `tool_args_match_overrides[tool_name]` can be a mode string, a list of dotted key paths to compare (`_get_partial_matcher_on_keys` L174), or a callable `(out_args, ref_args) -> bool`.
- Divergence bug: in agentevals the key-path matcher treats a key missing on both sides as `None == None`, which is a match. openevals' copy fixes this with a `missing` sentinel (`OE/trajectory/utils.py` L174-195). The two packages now disagree on the same input.

Graph trajectories (LangGraph-specific):
- `extract_langgraph_trajectory_from_snapshots` / `extract_langgraph_trajectory_from_thread(graph, config)` (`AE/graph_trajectory/utils.py` L17, ~L85) walk `graph.get_state_history()`. They build `steps` (node names per invocation, with subgraph prefix `ns:` and `__interrupt__` markers), `results` (the last message only, "to reduce context size", L39-46) and `inputs` (`__resuming__` after interrupts).
- `graph_trajectory_strict_match` (`AE/graph_trajectory/strict.py:_scorer` L9) is exact list equality of `steps`.
- There is no unordered or subset mode for graphs, and no partial or edit-distance score.

## Online / production path

Not in these repos except as SDK calls. Online evaluators run server-side on sampled runs and on *threads* (multi-turn conversations grouped by thread/session id), re-evaluated after an idle timeout `[doc]` (evaluation-concepts; issue #2603 asks to re-evaluate threads when new messages arrive after the timeout). Production traces become offline examples through "add to dataset". `Example.source_run_id` and `DatasetTransformation` types (`remove_system_messages`, `convert_to_openai_message`, `extract_tools_from_run`, `schemas.py` L241) hint at server-side run→example transforms `[inferred]`.

## Regression comparison (docs)

`[doc]` compare-experiment-results.md:
- Select ≥2 experiments on a dataset and click Compare.
- One experiment is the **source** (by default the first selected). Each run cell is red if it *regressed* on any feedback key vs the source's run for the same example, and green if it improved.
- Each feedback column header shows counts of better/worse. Buttons filter to only regressed or only improved rows.
- "Higher is better" is configurable per feedback key and persisted.
- There is a text Diff view (2 experiments only), side-by-side traces, and charts with x-axis labels from experiment metadata.
- With repetitions, cells show the mean with drill-down to individual scores and stddev (repetition.md).

The docs describe no significance test, no noise threshold, and no pairing statistics. A regression appears to be any per-example score decrease `[inferred from doc wording]`. The SDK builds the comparison URL as `<dataset>/compare?selectedSessions=<id>` (`testing/_internal.py` L1011-1015). `ExperimentResults.comparison_url` (`_runner.py` L605) exposes it. The SDK has no programmatic regression API. You'd pull `get_experiment_results` / feedback and diff yourself.

## Extension model

- Evaluators are any callable whose parameter names match the supported set, or a `RunEvaluator` subclass with `evaluate_run(run, example)` (`evaluator.py:RunEvaluator` L124). `@run_evaluator` / `@comparison_evaluator` decorators exist.
- Targets can be any callable, a LangChain `Runnable`, or `@traceable` functions.
- openevals: every evaluator is a factory returning a `SimpleEvaluator`. Custom judges use `prompt=callable` or `output_schema`. agentevals depends on openevals' *private* `_create_llm_as_judge_scorer`.
- No plugin registry or entry points. Composition is function-level.

## Strengths

- **Named-parameter evaluator binding** (`_normalize_evaluator_func`, `evaluator.py` L663) plus return-type coercion (`_format_evaluator_result` L932). One-line evaluators (`def exact(outputs, reference_outputs): return outputs == reference_outputs`) work, and the same functions run in `evaluate()`, pytest and online.
- **Evaluator errors become first-class feedback** (`_run_evaluators` L1748-1784, `extra={"error": True}`). They aren't silently dropped, so error rates are visible per key.
- **Streaming pipeline for sync and async**. The async path puts target+evaluators in one task under one semaphore (`_arunner.py` L824-892), so `max_concurrency` is a true upper bound and results stream.
- **Re-scoring existing experiments** (`evaluate(experiment_id, evaluators=...)` → `evaluate_existing`). Judges can be iterated on without re-running the app.
- **Tool-arg matching granularity** (`_get_matcher_for_tool_name`, `AE/trajectory/utils.py` L194). Global mode plus per-tool mode, key-path list or callable is a practical answer to "args contain timestamps/ids".
- **pytest integration that ties examples to git**: dataset version tags `git:commit:<sha>` (`_end_tests` L545-556), deterministic experiment naming across xdist workers (L423-432), and automatic `pass` feedback.
- **Judge prompt hygiene**: reasoning-before-score ordering in the JSON schema (`llm.py` L101-110), `choices` as an enum, and strict JSON-schema mode for OpenAI.

## Weaknesses

- **No retries, timeouts, or rate limiting in the eval loop** (`_forward` L2029-2078; `aiter_with_concurrency` has no timeout). A hung target stalls the experiment, and a transient 429 becomes a failed row.
- **Dataset version recorded loosely**: `max(modified_at)` of the examples used (`_get_dataset_version` L1897). Comparative eval uses the first experiment's version without checking the second (`_runner.py` L933-941). Issue #2430 reports snapshot staleness.
- **Comparative eval race**: `random.shuffle(runs_list)` mutates a list shared across concurrently running comparators (L960-961, L994-1004). Scores can be attributed to the wrong run (open issue #3582).
- **Repetitions are flat reruns** (`examples` L1523-1532) with no repetition index in the result row. Comparative eval then hands the comparator all K×2 runs for an example (`runs_dict` L943-947). Statistics are UI-only means/stddev.
- **pytest example-id hashing includes the Python test's return value** (`_TestCase.end_run` L918-925). This differs from JS, which hashes reference outputs. Returning LLM output from a test silently forks examples and dataset versions.
- **Judge**: continuous score has no schema bounds (`llm.py` L88-93), there is no parse-failure retry, no multi-sample or majority vote, and no presentation-bias mitigation (openevals #212).
- **Multi-turn simulator**: synchronous, one conversation per call, evaluator errors `print`ed (`multiturn.py` L255-256). The app returns one message per turn, so intra-turn tool calls are lost from the trajectory. There is no per-turn scoring and no turn structure in the output.
- **Trajectory match**: greedy first-fit matching (`_is_trajectory_superset` L90). The strict mode ignores content. Unordered/subset/superset ignore everything except tool calls. Scores are boolean only, with no partial credit (precision/recall, edit distance).
- **Code duplication and drift**: the trajectory evaluators exist in both agentevals and openevals, and they already diverge (missing-key matcher fix only in openevals, `OE/trajectory/utils.py` L174-195). agentevals imports openevals *private* functions (`AE/trajectory/llm.py` L3-16).
- **Graph trajectories are LangGraph-coupled** (`get_state_history`, `StateSnapshot.tasks`), and `results` keeps only the last message (`graph_trajectory/utils.py` L39-46).

## Worth adapting

- **Evaluator = plain function with named-argument injection + permissive return coercion.** It gives the lowest friction to write scorers and the same function works offline, in tests and online.
- **Error-as-feedback with an `error` flag** per declared key. You can compute "judge failure rate" and distinguish "scored 0" from "couldn't score".
- **Re-evaluate an existing experiment** as a first-class entry point. It decouples expensive generation from cheap scoring iteration.
- **Per-tool arg-matching overrides** (mode | key paths | callable). This is a small surface that covers most real tool-call comparison needs.
- **Trajectory match modes as set relations** (strict/unordered/subset/superset). This is a clear vocabulary. Adopt it with a bipartite matcher and optional partial scores.
- **Simulated user with `fixed_responses` prefix then LLM**, plus role flipping and filtering of internal/tool messages. Scripted openings become reproducible without giving up open-ended continuations.
- **Tag dataset versions with the git commit** at the end of a test session. Every CI run can be traced back to exactly which examples it saw.
- **Reasoning-before-score structured output**, with `choices` as an enum for discrete rubrics.

## Don't copy

- **Implicit, timestamp-derived dataset version stored in free-form experiment metadata** (`_get_dataset_version`, `_end`). Comparisons can't guarantee they are apples-to-apples. Record an explicit immutable version id per experiment and refuse or warn on mismatched comparisons.
- **Tests that mutate the dataset as a side effect of running** (`sync_example` in `_end_run`), combined with content-hash example ids that include outputs. Dataset identity then depends on test behavior, and dataset versions churn on every CI run.
- **Regression = any per-example score drop vs a source experiment** with no noise model `[doc]`. With LLM non-determinism this flags noise as regressions. Pair it with repetitions and a significance or threshold rule.
- **Shared mutable state across concurrent evaluator tasks** (`random.shuffle` on a shared list, #3582). Position randomization must be per-call on a copy, with the mapping kept explicitly.
- **Magic parameter-name binding with a silent legacy fallback** (any 2 positional args → `(Run, Example)`, `evaluator.py` L686-707). A typo changes semantics without erroring. If you adopt name binding, fail loudly on unknown names.
- **Two packages owning the same evaluator code** (agentevals vs openevals trajectory), with cross-package private imports. Pick one owner.
- **Framework-specific trajectory extraction as the core graph evaluator** (LangGraph `StateSnapshot`). Normalize to a framework-neutral step and tool-call trace first.

## Open questions

- Whether the server stores a real version id for experiments beyond `metadata.dataset_version`, and whether the compare view warns on version mismatch. The docs don't say.
- How the regression view decides "regressed" with repetitions (per-run, or on the mean?) and for non-numeric/categorical feedback. It's not documented.
- Whether online thread evaluators and offline multi-turn experiments share a data model (threads vs examples with message-list inputs). There is no SDK type for threads in the eval path I read.
- Whether the Python pytest example-id hashing of the return value (`end_run` L921-925) is intended. Nothing in the unit tests (`python/tests/unit_tests/test_testing.py` L30-80) covers a non-None return.
- The exact behavior of sync evaluators under `aevaluate` (thread offload vs blocking the loop). I did not open `_arun_evaluators` in full.
- Whether agentevals is formally deprecated in favor of `openevals.trajectory`. The README only calls openevals a "companion"; the code duplication suggests migration.
