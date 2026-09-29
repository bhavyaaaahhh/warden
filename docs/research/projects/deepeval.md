# DeepEval
Repo: https://github.com/confident-ai/deepeval @ `a7519f5cd1b178459d4b30601aec9546811e6ee6` · License: Apache-2.0 · Lang: Python (plus a separate TS SDK) · Last commit: 2026-09-28 · Activity: package `__version__` 4.2.6; releases python-v4.2.0 (2026-08-24) and v4.2.4 (2026-09-22), all within weeks of each other; about 18.5k stars and 685 open issues (gh api, 2026-09-29).

Permalink prefix used below: `P = https://github.com/confident-ai/deepeval/blob/a7519f5cd1b178459d4b30601aec9546811e6ee6/deepeval`

## Purpose
DeepEval is a pytest-style framework for evaluating LLM apps offline. It ships 50+ LLM-as-judge metrics covering RAG, agents, multi-turn, safety, multimodal and voice. It also ships a synthetic data generator (`Synthesizer`), a user simulator for multi-turn tests (`ConversationSimulator`), and tracing (`@observe`), which lets metrics attach to individual spans. It is mostly offline. The online, regression and dataset-versioning pieces live in the commercial Confident AI backend, which the OSS client talks to through `deepeval.confident.api`. It targets Python app developers who want `assert_test` in CI.

## Core abstractions

**Unit of data (single-turn): `LLMTestCase`** (`test_case/llm_test_case.py:LLMTestCase`, [L351](P/test_case/llm_test_case.py#L351)). A flat pydantic model with the fields `input`, `actual_output`, `expected_output`, `context`, `retrieval_context`, `tools_called: List[ToolCall]`, `expected_tools: List[ToolCall]`, token cost and counts, `completion_time`, `flaky`, `tags`, MCP fields, and a private `_trace_dict`. Metrics declare which fields they need through `SingleTurnParams` ([L180](P/test_case/llm_test_case.py#L180)).
- **`ToolCall`** ([L248](P/test_case/llm_test_case.py#L248)) has `name`, `type` (FUNCTION or MCP), `description`, `reasoning`, `input_parameters: Dict`, and `output: Any`. `__eq__` and `__hash__` look only at `(name, input_parameters, output)`. The hash is made recursive through `_make_hashable`, which lets ToolCorrectness use set intersection.
- Trade-off: tools called and tools expected are flat lists. Nothing records call order beyond list order, nothing represents parent/child or parallel calls, and nothing links a call to the turn or step that made it. Anything richer has to come from the trace (`_trace_dict`).

**Unit of data (multi-turn): `ConversationalTestCase` + `Turn`** (`test_case/conversational_test_case.py`, [Turn L62](P/test_case/conversational_test_case.py#L62), [ConversationalTestCase L213](P/test_case/conversational_test_case.py#L213)).
```python
class ConversationalTestCase(BaseModel):
    turns: List[Turn]
    scenario: Optional[str]; context: Optional[List[str]]
    user_description: Optional[str]; expected_outcome: Optional[str]; chatbot_role: Optional[str]
    metadata, comments, tags, flaky, expected_labels: Dict[str,str], mcp_servers, multimodal, voice
```
- `Turn` has `role` (user or assistant), `content`, `retrieval_context`, `tools_called: List[ToolCall]`, MCP calls, `metadata`, and voice fields (`audio`, `latency_ms`, `interrupted`).
- Tools called are stored per turn, but **no `Turn` or `ConversationalTestCase` field holds expected tools**. `MultiTurnParams` ([L33](P/test_case/conversational_test_case.py#L33)) has `TOOLS_CALLED` and no `EXPECTED_TOOLS`. Expected tool behaviour in multi-turn tests can only be written in natural language, as `expected_outcome` or a G-Eval criterion.
- The two test case classes are separate types with no shared base, and the metric hierarchies are split the same way (`BaseMetric` vs `BaseConversationalMetric`, `metrics/base_metric.py` [L111](P/metrics/base_metric.py#L111), [L206](P/metrics/base_metric.py#L206)). The executor routes each case with `isinstance`.

**Goldens and the dataset model**: `Golden`, `ConversationalGolden`, `Persona`, `EvaluationDataset` (`dataset/golden.py`, `dataset/dataset.py`).
- A golden is a test case *before* the app has run. `Golden` ([L175](P/dataset/golden.py#L175)) holds `input`, `expected_output`, `context`, `expected_tools`, and so on.
- `ConversationalGolden` ([L316](P/dataset/golden.py#L316)) holds `scenario` (required), `expected_outcome`, `persona`, optional seed `turns`, and `context`.
- `Persona` ([L72](P/dataset/golden.py#L72)) separates *who* the user is (`name`, free-text `characteristics`, `metadata`) from *what* they want (the golden's scenario and outcome). All other persona fields are voice-only: interruption behaviour, `speaks_first`, TTS voice, background noise, `hold_timeout`. `Persona.prompt_block()` renders the persona into a `<persona>` tag block.
- `EvaluationDataset` is a single-type container. `_multi_turn` is inferred from `isinstance(goldens[0], ...)`. It loads from CSV, JSON and JSONL. `push`, `pull`, `create_version` and `get_versions` ([L928-1080](P/dataset/dataset.py#L928)) are HTTP calls to Confident AI. **The OSS package does not version datasets locally.**

**Unit of scoring: `BaseMetric` / `BaseConversationalMetric`** (`metrics/base_metric.py`).
- A metric is a *stateful object*. `measure(test_case)` writes `self.score`, `self.reason`, `self.success`, `self.error`, `self.evaluation_cost` and `self.verbose_logs` onto the instance.
- `is_successful()` computes `score >= threshold`, and any error counts as a failure.
- Because of this state, the executor must `copy_metrics(...)` before each test case (`evaluate/execute/e2e.py` [L614](P/evaluate/execute/e2e.py#L614)).
- `__init_subclass__` wraps every metric method in tracing (`observe_methods`).
- Prompts are Jinja2 templates in one 213 KB `templates/metrics/templates.json`, keyed `[ClassName][method]` and rendered by `templates/resolver.py:resolve_template`. `PromptMixin._get_prompt` lets a user pass an `evaluation_template` class to override any method.
- Trade-off: a single file makes prompts easy to override and audit. Because the score lives on the metric instance, metrics are not reentrant: one instance cannot score two cases concurrently.

**Execution record: tracing.** `@observe` spans (`tracing/tracing.py`) can carry `metrics=[...]` and set a per-span `LLMTestCase` through `update_current_span()`. Metrics with `requires_trace=True` get the nested span tree as JSON in `test_case._trace_dict` (`evaluate/execute/agentic.py:_a_execute_span_test_case` [L330](P/evaluate/execute/agentic.py#L330), `_a_execute_trace_test_case` [L441](P/evaluate/execute/agentic.py#L441)). Five metrics set that flag: TaskCompletion, StepEfficiency, PlanAdherence, PlanQuality and AgentLoopDetection.

## Conversation simulator
`simulator/conversation_simulator.py:ConversationSimulator` ([L232](P/simulator/conversation_simulator.py#L232)).
- **App under test:** `model_callback`, sync or async. The simulator inspects the callback's signature and passes only the kwargs it accepts from `{input, turns, thread_id}` (`generate_turn_from_callback` [L1209](P/simulator/conversation_simulator.py#L1209)). The callback must return a `Turn`, so the app itself decides whether to fill in `tools_called` and `retrieval_context` on each assistant turn. `thread_id` is a fresh UUID per conversation. Voice agents use a `voice_config` connector instead (TTS, then the agent, then STT). You must pass exactly one of `model_callback` or `voice_config`.
- **Simulated user:** a `SimulationNode` graph (`simulator/simulation_graph/node.py`).
  - The default is a single node, `default_simulation_node()`. It calls `SimulationTemplate.simulate_first_user_turn` or `simulate_user_turn` (`simulator/template.py`) with scenario, persona block, language and the full serialized history. It expects the JSON `{"simulated_input": ...}` back.
  - Custom graphs can mix scripted actions (callables that return a str or Turn) with LLM-routed edges. After the assistant replies, `_SimulationGraphRunner.advance` ([L100](P/simulator/simulation_graph/runner.py#L100)) asks the simulator LLM which `when=` edge description matches the reply. Nodes can set `terminal=True` and `max_visits`.
  - Seed `golden.turns` are prepended. If the last seed turn is a user turn, the first callback call replies to it.
- **Stopping criteria**, from `_simulate_single_conversation` ([L463](P/simulator/conversation_simulator.py#L463)). Before every user turn the loop checks, in order:
  - (a) `simulation_counter >= max_user_simulations`. The default is 10 simulated user turns.
  - (b) The `stopping_controller`. The default is `expected_outcome_controller` (`simulator/controller/controller.py` [L230](P/simulator/controller/controller.py#L230)). It sends the whole history plus `expected_outcome` to the simulator LLM, which returns `{is_complete, reason}`. It is skipped when `expected_outcome` is None or there are no turns yet.
  - (c) A graph node's `max_visits` is exhausted, or a `terminal` node fired.
  - A custom controller is any callable that takes a subset of `Context` fields (`turns`, `golden`, `last_assistant_turn`, `simulated_user_turns`, ...) and returns `Decision(should_end, reason)` (`controller/types.py`). Returning anything else, including `None`, means continue.
- **Concurrency:** one `asyncio.Semaphore(max_concurrent=5)` across conversations, then `asyncio.gather` ([L438-461](P/simulator/conversation_simulator.py#L438)). A single conversation runs sequentially. `simulate()` also accepts `on_turn` and `on_simulation_complete` hooks.
- **Output:** `ConversationalTestCase` carrying scenario, expected outcome and user description, with persona characteristics copied into metadata.
- **Cost:** each simulated turn makes one user-generation LLM call and one completion-check LLM call (plus one routing call if the node has edges). The completion check reads the full history each time, so judge tokens grow roughly quadratically with conversation length.
- **Failure handling:** none specific to the simulator. A callback exception or bad JSON propagates up through `gather` and aborts the batch (see issue #2299, where a JSON bug in a prompt aborted simulations).

## Multi-turn metrics
All of them follow the "QAG" pattern in `metrics/utils/qag.py`: the judge returns yes/no verdicts per item, and the score is the fraction of verdicts that pass.
- **`score_qag_verdicts`** ([L113](P/metrics/utils/qag.py#L113)) sets polarity through `passing=`. It drops `None` verdicts (out-of-vocabulary replies) silently, returns 1.0 when there are no verdicts, and in strict mode maps any score below threshold to 0.
- **`normalize_qag_verdict`** ([L58](P/metrics/utils/qag.py#L58)) takes the leading `[a-z]+` token and maps the legacy `idk` to borderline.

| Metric | Unit judged | Context passed | Score |
|---|---|---|---|
| `TurnRelevancyMetric` ([L41](P/metrics/turn_relevancy/turn_relevancy.py#L41)) | last assistant message of each window | **sliding window**: `get_unit_interactions` groups turns into user→assistant exchanges, then `get_turns_in_sliding_window(units, window_size=10)` yields `units[max(0,i-9):i+1]` for every i (`metrics/utils/turns.py` [L51](P/metrics/utils/turns.py#L51)). One LLM call per exchange, `asyncio.gather`'d. | fraction "yes" |
| `KnowledgeRetentionMetric` ([L38](P/metrics/knowledge_retention/knowledge_retention.py#L38)) | each assistant turn | Pass 1: for each user turn, extract "knowledge" given `turns[:i]` (full prefix). Pass 2: for each assistant turn, a verdict against all knowledge accumulated before it. No window. | fraction "no" (no forgetting) ([L358](P/metrics/knowledge_retention/knowledge_retention.py#L358)) |
| `ConversationCompletenessMetric` ([L43](P/metrics/conversation_completeness/conversation_completeness.py#L43)) | each extracted user intention | Extract intentions from all turns, then one verdict per intention over **all turns**. Takes a `window_size=3` argument, stores it, and never reads it. | fraction "yes" |
| `RoleAdherenceMetric` ([L39](P/metrics/role_adherence/role_adherence.py#L39)) | whole conversation, one call | All turns plus `chatbot_role`. The judge returns the indices of out-of-character assistant turns. | `(n_assistant - n_ooc)/n_assistant` ([L332](P/metrics/role_adherence/role_adherence.py#L332)) |
| `ConversationalGEval` ([L46](P/metrics/conversational_g_eval/conversational_g_eval.py#L46)) | whole conversation, one call | All turns, filtered to `evaluation_params` (`convert_turn_to_dict`) plus conversation-level fields. No window. | G-Eval (below) |
| `ConversationalDAGMetric` | per node | Nodes may set `turn_window=(start,end)` to slice turns (`metrics/conversational_dag/nodes.py:_resolve_text` [L45](P/metrics/conversational_dag/nodes.py#L45)). | DAG leaf |

Other metrics that take `window_size` are TurnFaithfulness and TurnContextual{Precision,Recall,Relevancy}. Goal accuracy, MCP-use, TopicAdherence and ToolUse also exist; I did not read them in depth.

## Judge implementation

**Model layer and retries.**
- Every call goes through `generate_with_schema_and_extract` (`metrics/utils/generation.py` [L71](P/metrics/utils/generation.py#L71)). Native models are asked for structured output via `schema=`.
- Custom models that return a string go through `trimAndLoadJson`, which slices from the first `{` to the last `}`, strips trailing commas, and on failure sets `metric.error` and raises.
- The judge does not re-prompt when parsing fails. Retries happen only at the transport layer: tenacity with exponential jitter, `DEEPEVAL_RETRY_MAX_ATTEMPTS` default **2** (`config/settings.py` [L905](P/config/settings.py#L905), `models/retry_policy.py:create_retry_decorator`).
- The OpenAI judge defaults to `temperature=0.0` (`models/llms/openai_model.py` ~L97).

**G-Eval** (`metrics/g_eval/g_eval.py:GEval` [L48](P/metrics/g_eval/g_eval.py#L48)).
- **Prompt construction:** `criteria` goes to an LLM, which generates `evaluation_steps` once per metric instance and caches them on `self.evaluation_steps`. You can pass steps directly to skip this.
- `construct_test_case_string` then prints just the selected `evaluation_params` fields as `Label:\nvalue` (`g_eval/utils.py` [L325](P/metrics/g_eval/utils.py#L325)).
- The `GEval.generate_evaluation_results` template asks for JSON `{reason, score}`, with the score an integer in `score_range`. That range defaults to 0-10, or comes from `Rubric` score ranges, which are sorted and checked for overlap ([L210](P/metrics/g_eval/utils.py#L210)).
- Strict mode swaps in a binary template.
- **Logprob weighting:** `generate_rubric_score` (`metrics/utils/decision.py` [L610](P/metrics/utils/decision.py#L610)) calls `model.generate_raw_response(prompt, top_logprobs=20)`. `calculate_weighted_summed_score` ([L337](P/metrics/g_eval/utils.py#L337)) scans generated tokens *backwards* for the last token equal to the parsed integer score. From that token's top-20 alternatives it drops any with probability below 1% or that are not decimal, then returns `Σ score·p / Σ p`. That is a renormalized expectation over the candidate score tokens.
- **Fallback:** if the model has no logprobs (checked explicitly only for OpenAI and Azure in `no_log_prob_support`; any other provider raises `AttributeError` in `generate_raw_response`), it falls back to the plain structured call and uses the raw integer. If logprob parsing throws `KeyError`, `TypeError` or `ValueError`, it also falls back to the raw integer ([L590-607](P/metrics/utils/decision.py#L590)). **Nothing records which of the two paths produced a score.** See issue #1029.
- Scores of 10 and above tokenize as multi-digit strings, so the token match can be fragile. I did not verify this for rubric ranges above 9.
- **Threshold:** `score = (g - min)/(max - min)`, and success is `score >= threshold` with a default threshold of 0.5. `strict_mode` forces the threshold to 1 and uses the int score.
- There is no calibration, no repeated sampling, and no variance estimate. Issue #3110 asks what the canonical verdict is when reruns straddle the threshold, and it is unresolved.

**DAG** (`metrics/dag/`).
- A user-built decision tree of `TaskNode`s (an LLM extracts or transforms text into `output_label`), `BinaryJudgementNode`s (exactly 2 `VerdictNode` children), `NonBinaryJudgementNode`s, and leaf `VerdictNode`s carrying `score` in 0..10 or a child metric (GEval or any BaseMetric).
- `DAGMetric.__init__` rejects cycles (`is_valid_dag_from_roots`).
- `_DeepAcyclicGraphRunner` (`dag/runner.py` [L43](P/metrics/dag/runner.py#L43)) visits roots, tracks indegree so a node with several parents runs only after all of them finish (`_check_remaining`), and follows the verdict node whose value matches the parent's judgement.
- The leaf sets `metric.score = node.score / 10` ([L125](P/metrics/dag/runner.py#L125)), or copies the child metric's score, cost and reason.
- The design gives deterministic score *mapping* on top of LLM classification, so each judge call is a narrow classification instead of a holistic 1-10 score. It is the most distinctive piece of DeepEval's judge design.

## Agent metrics
**`ToolCorrectnessMetric`** (`metrics/tool_correctness/tool_correctness.py` [L43](P/metrics/tool_correctness/tool_correctness.py#L43)) is mostly deterministic.
- `evaluation_params` chooses whether `input_parameters` and/or `output` are compared. Tool name and type are always compared.
- **Default mode:** greedy best match per expected tool; score = Σ best match / len(expected) ([L518](P/metrics/tool_correctness/tool_correctness.py#L518)). Parameters are compared with a recursive key-overlap ratio, `_compare_dicts` ([L606](P/metrics/tool_correctness/tool_correctness.py#L606)). Extra calls go unpenalized (recall-only).
- **`should_consider_ordering`:** weighted LCS DP ([L561](P/metrics/tool_correctness/tool_correctness.py#L561)).
- **`should_exact_match`:** same length and positional equality, scored 0 or 1 ([L493](P/metrics/tool_correctness/tool_correctness.py#L493)).
- **Optional LLM part:** if `available_tools` is given, an LLM also gives a "tool selection" score, and the final score is `min(deterministic, selection)`.
- It is single-turn only. It reads `LLMTestCase.tools_called` and `expected_tools`.

**`TaskCompletionMetric`** ([L47](P/metrics/task_completion/task_completion.py#L47)) sets `requires_trace=True`.
- If `_trace_dict` exists, an LLM extracts `(task, outcome)` from the serialized trace JSON. Otherwise it extracts them from input, actual_output and tools_called.
- A second call returns a 0-1 `verdict` for whether the outcome achieved the task. The raw verdict is the score; strict mode maps sub-threshold scores to 0.
- `task=` can be fixed by the user.

**Trace-based evaluation.** `dataset.evals_iterator()` ([L1610](P/dataset/dataset.py#L1610)) yields goldens. User code calls the traced app inside the loop, and the resulting trace is scored by `_a_execute_agentic_test_case` ([L74](P/evaluate/execute/agentic.py#L74)).
- That function builds a trace-level `LLMTestCase` from `trace.output`, `trace.tools_called` and `trace.expected_tools`.
- It then walks the DFS over every root span, running each span's attached metrics against the span's own test case. Spans that errored skip their metrics (`_skip_metrics_for_error`).
- On cancellation or timeout, unfinished metrics are marked `success=False` with an explanatory `error`.

## Execution model
- **Entry points:**
  - `evaluate(test_cases, metrics, async_config, display_config, cache_config, error_config)` (`evaluate/evaluate.py`).
  - `assert_test(test_case, metrics)` ([L84](P/evaluate/evaluate.py#L84)).
  - `dataset.evals_iterator()`.
  - `deepeval test run` (a pytest wrapper).
- **Concurrency:** `a_execute_test_cases` ([L471](P/evaluate/execute/e2e.py#L471)).
  - One `asyncio.Semaphore(max_concurrent=20)` over test cases, with optional `throttle_value` sleeps between task creation.
  - Each task is wrapped in `_await_with_outer_deadline` with a per-task timeout (`_common.py` [L228](P/evaluate/execute/_common.py#L228)).
  - The whole batch runs under `asyncio.wait_for(gather(*tasks), timeout=get_gather_timeout())`. On timeout, pending tasks are cancelled and the error is re-raised unless `ignore_errors` is set.
  - Inside a test case, metrics run concurrently. Inside a metric, verdicts use `asyncio.gather` with no separate limit, so a single TurnRelevancy call on a 50-turn conversation fans out to 50 judge calls at once. Only the outer semaphore bounds anything.
  - Sync metrics call `loop.run_until_complete` on a shared event loop. Issue #2952 reports failures inside already-running loops such as Jupyter and FastAPI.
- **Error handling:** `_execute_metric` ([L244](P/evaluate/execute/_common.py#L244)).
  - `MissingTestCaseParamsError` with `skip_on_missing_params` sets `metric.skipped` and leaves success as None.
  - Otherwise, with `ignore_errors`, any exception becomes `metric.error` and `success=False`. Without it, the exception propagates.
  - An odd `except TypeError` branch retries `a_measure` without kwargs, for older custom metrics.
- **Caching:** `test_run/cache.py`.
  - The cache is a JSON file, `.deepeval/.deepeval-cache.json`, locked with portalocker. It is written by default; it is only *read* with `use_cache=True` or `deepeval test run --use-cache`, and is disabled with `--repeat`.
  - Key: `serialize({input, actual_output, expected_output, context, retrieval_context, hyperparameters})` ([L145-160](P/test_run/cache.py#L145)).
  - A hit also requires the metric name plus a config match on `threshold, evaluation_model, strict_mode, include_reason, n, language, embeddings, evaluation_params, assessment_questions, evaluation_steps/criteria` (`Cache.same_metric_configs` [L357](P/test_run/cache.py#L357)).
  - **The key leaves out `tools_called`, `expected_tools`, the trace, and the judge prompt template.** A cached ToolCorrectness or TaskCompletion score can go stale when only the tool calls change, and editing a prompt template does not invalidate anything.
  - Conversational test cases skip the cache entirely: `_a_execute_conversational_test_cases` gets no `use_cache`.
- **pytest integration:**
  - `deepeval test run file.py` (`cli/test/command.py` [L47](P/cli/test/command.py#L47)) builds pytest args: `-n` for xdist, `--count` for repeats via pytest-repeat, `-m` marks, `--identifier`, and `-p deepeval`.
  - The plugin (`plugins/plugin.py`) creates a global test run at session start and wraps each test in a trace observer, so `assert_test(golden=...)` can score the spans the test produced. At session end the run is uploaded or saved.
  - `assert_test` raises `AssertionError` listing every failed metric's score, threshold and reason. A test case with `flaky=True` only emits `warnings.warn` ([L201-209](P/evaluate/evaluate.py#L201)).
  - Per its own comment, for conversations "test_result right now is just the result for the last message".

## Data model & storage
- **`TestRun`** (`test_run/test_run.py` [L148](P/test_run/test_run.py#L148)) holds `test_cases`, `conversational_test_cases` (API forms with per-metric `MetricData`), `metrics_scores`, `trace_metrics_scores`, `hyperparameters`, `prompts`, pass and fail counts, cost, `dataset_alias`/`dataset_id`, `identifier`, and `official: bool`.
- **Local persistence (OSS)** has three forms:
  1. `.deepeval/.latest_run_full.json` is a rolling snapshot, overwritten on every run.
  2. With `DisplayConfig.results_folder` or `DEEPEVAL_RESULTS_FOLDER`, each run is written as `test_run_<YYYYMMDD_HHMMSS>.json` under a lock (`evaluate/local_store.py:write_test_run` [L143](P/evaluate/local_store.py#L143)).
  3. Opt-in SQLite (`DEEPEVAL_LOCAL_STORE=sqlite`, `sqlite_store/store.py` [L58](P/sqlite_store/store.py#L58)) with tables `test_runs`, `test_cases`, `traces`, `spans` and `metric_data`, plus `SCHEMA_VERSION = 1` and a full `payload_json` per run. Its docstring suggests plain SQL (`avg(score) GROUP BY run`) for comparing runs.
- `deepeval inspect` is a Textual TUI over one saved run (`inspect/loader.py`).
- **Comparing runs (OSS vs Confident AI):** the OSS code has no run-vs-run diff, baseline selection or significance test.
  - `evaluate/compare.py:compare()` is something else: an *arena* in which `ArenaGEval` picks a winner per `ArenaTestCase` across contestants within one run. It counts wins and optionally posts an "experiment" to Confident AI.
  - Regression tracking exists only as a flag: `deepeval test run --official` sets `TestRun.official = True` to "mark this test run as the official baseline on Confident AI" (`cli/test/command.py` [L114](P/cli/test/command.py#L114)).
  - Per the CLI login prompt in `constants.py`, the regression comparison itself happens server-side [inferred]. Dataset versions are server-side too (`create_version`, `get_versions`).

## Synthesizer
`synthesizer/synthesizer.py:Synthesizer`.
- Four ways to generate goldens: `generate_goldens_from_docs` (chunk, embed, build contexts by similarity, with a quality filter via `ContextConstructionConfig`), `generate_goldens_from_contexts`, `generate_goldens_from_scratch`, and `generate_goldens_from_goldens`. There are `*_conversational_goldens_*` variants of each that emit `ConversationalGolden(scenario, expected_outcome)`.
- **Pipeline per context** (`_a_generate_from_context` [L966](P/synthesizer/synthesizer.py#L966)):
  1. Generate N inputs.
  2. Critic-filter them: a critic model scores each input, and inputs below `synthetic_input_quality_threshold=0.5` are rewritten up to `max_quality_retries=3` (`_a_rewrite_inputs` [L1504](P/synthesizer/synthesizer.py#L1504)).
  3. Evolve Evol-Instruct style: `num_evolutions` steps, each type drawn with `random.choices` over weighted `Evolution` types, with **no seed** (`_a_evolve_input` [L1643](P/synthesizer/synthesizer.py#L1643)).
  4. Optionally restyle, then optionally generate an expected output.
- Cost is tracked. The expected output is written by the same generator model, with no check against the context beyond the prompt.

## Online / production path
Out of scope for OSS. Traces can be exported to Confident AI (`tracing/otel`, `metric_collection` on spans), and online evaluation runs server-side [inferred].

## Extension model
- **Custom metric:** subclass `BaseMetric` or `BaseConversationalMetric`, implement `measure`, `a_measure` and `is_successful`.
- **Low-code metrics:** `GEval`, `ConversationalGEval`, `DAGMetric` and `ArenaGEval`, which can be uploaded and pulled through `upload()` / `pull()`.
- **Prompt override:** per-metric `evaluation_template` classes.
- **Custom judge:** subclass `DeepEvalBaseLLM` (`generate`, `a_generate`, optional `generate_raw_response` for logprobs).
- **Simulator:** custom `stopping_controller` callables and `SimulationNode` graphs.
- **Integrations:** LangChain, LlamaIndex, pydantic-ai, OpenAI Agents and CrewAI (`integrations/`).

## Strengths
- **Clean split between simulated user and stopping logic.** Persona (who) is separate from scenario and expected_outcome (what). `stopping_controller` is a small typed hook that receives `Context` and returns `Decision`, and scripted and LLM-driven user behaviour mix in one graph (`simulator/controller/types.py`, `simulation_graph/node.py`).
- **G-Eval logprob expectation is implemented carefully.** It picks the *last* matching score token so the reasoning text cannot hijack it, filters out tokens below 1% and non-numeric tokens, and renormalizes (`g_eval/utils.py:calculate_weighted_summed_score`).
- **DAG metrics** turn holistic grading into chains of narrow classifications with deterministic score leaves, and validate cycles and indegree (`dag/runner.py`).
- **Shared QAG scoring helper.** One `score_qag_verdicts` with an explicit `passing=` polarity and `Verdict` enum, instead of per-metric string compares (`metrics/utils/qag.py`).
- **ToolCorrectness is deterministic by default,** with ordering-aware weighted LCS and partial credit for parameters (`tool_correctness.py:_compute_weighted_lcs`).
- **Local SQLite store** is stdlib-only, keeps a lossless payload column, and uses a normalized metric table that SQL can query (`sqlite_store/store.py`).

## Weaknesses
- **No multi-turn expected tools.** `Turn` has `tools_called` only, and `MultiTurnParams` has no `EXPECTED_TOOLS`. Tool-use correctness in conversations cannot be checked deterministically (`conversational_test_case.py:MultiTurnParams`).
- **Dead parameter:** `ConversationCompletenessMetric(window_size=3)` stores the value and never reads it. Verdicts always see the full conversation (`conversation_completeness.py` L58/L80 vs L111-118).
- **Silent score-path switching:** G-Eval quietly falls back from the logprob-weighted score to the raw integer when logprobs are missing or parsing fails, and nothing records which one was used (`decision.py:_score_from_raw_response`; issue #1029).
- **Silent verdict dropping:** `score_qag_verdicts` discards `None` (unparseable) verdicts from the denominator and returns 1.0 when every verdict is dropped (`qag.py` L128-130). Judge failures therefore inflate scores.
- **Cache key is incomplete:** it leaves out tools_called, expected_tools, the trace and the prompt template, and conversational cases are never cached (`test_run/cache.py` L145-160; `e2e.py:_a_execute_conversational_test_cases`).
- **No repeated trials or variance.** A threshold compares a single stochastic judge sample (issue #3110). `--repeat` just reruns pytest and disables the cache.
- **Stateful metric instances** need a deep copy per test case and cannot be shared across concurrent tasks (`e2e.py` L614 `copy_metrics`).
- **Simulator is fail-fast:** a callback or JSON error in one conversation fails the whole `gather`, and each turn makes an extra full-history LLM completion check.
- **Unbounded fan-out inside a metric:** verdict `gather` calls have no inner semaphore, and a single outer semaphore of 20 counts test cases, not LLM calls (`e2e.py` L488).
- **Run comparison and dataset versioning are server-only.** OSS offers `--official` as a flag and no diff logic.

## Worth adapting
- **Golden → simulator → test case pipeline with `Persona` separate from `scenario` and `expected_outcome`.** Scenarios can be reused across personas, and the simulated conversation keeps the outcome as a grading target.
- **Stopping hook shaped as `(Context) -> Decision{should_end, reason}`**, with a turn cap plus an optional LLM outcome check. The small interface fits both scripted and judged termination.
- **Callback signature introspection** (`{input, turns, thread_id}` subset). The app under test can be stateless (take full turns) or stateful (take a thread_id) without adapters.
- **Unit-interaction sliding window** (`get_unit_interactions` + window over exchanges rather than raw turns). This keeps per-turn metrics O(n) calls with bounded context and never splits a user/assistant pair.
- **DAG metric** as a way to make rubric grading reproducible: LLMs only classify, and code assigns the score.
- **Last-token logprob expectation** for rubric scores, but record the score path.
- **Normalized results tables plus a lossless payload blob** for local run history.

## Don't copy
- **Score, reason and error as mutable attributes on the metric object.** It forces deep copies, blocks reentrancy, and ties configuration to results. Return an immutable result value instead.
- **A cache key hand-picked from a few fields.** It leaves out tool calls, the trace and the prompt version, so stale scores come back silently. Hash the full scored input and the full judge config, including the template.
- **Dropping unparseable verdicts from the denominator and returning 1.0 when nothing is left.** Parse failure should be an explicit error or abstain state that shows in the result.
- **Silently falling back from weighted to raw scores.** It mixes two estimators in one column, so score distributions shift when the provider changes.
- **Two parallel type hierarchies (single vs conversational)** with duplicated sync and async code paths. Most metric files contain both `measure` and `a_measure` bodies that are near-identical copies, which roughly doubles maintenance and lets the two drift.
- **Unused or no-op parameters** such as `window_size` on ConversationCompleteness.
- **Putting comparison and versioning behind the hosted backend.** The OSS data model has `official` and `dataset_id`, but nothing that uses them locally.

## Open questions
- How Confident AI actually computes regressions for an `official` baseline: per-case diffs, thresholds, any statistics. The server is closed [doc/inferred only].
- Whether `generate_raw_response` is implemented for non-OpenAI providers (Anthropic, Gemini, local). If not, G-Eval there is always the raw integer. I did not check every `models/llms/*`.
- Logprob matching for rubric ranges with scores ≥ 10, where the score may tokenize as multiple tokens. Not tested.
- KnowledgeRetention indexes `self.knowledges[:i]` by turn index. I did not verify that the knowledge list is aligned one-to-one with turns.
- The internals of GoalAccuracy, TopicAdherence, ToolUse and the MCP multi-turn metrics (whether they window, and how they score) were not read.
- The experimental "System One / Jev" eval mode (`metrics/utils/system_one.py`, `EXPERIMENTAL.md`) replaces some LLM verdicts with a probability model. Its calibration was not examined.
