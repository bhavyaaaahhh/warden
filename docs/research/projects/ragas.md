# Ragas
Repo: https://github.com/explodinggradients/ragas (now redirects to `vibrantlabsai/ragas`) @ `298b68274234c060deacab3cf5fb52aa3a20e885` · License: Apache-2.0 · Lang: Python · Last commit: 2026-02-24 · Activity: ~15.9k stars, 619 open issues, roughly one release every 2-3 weeks (v0.3.8 2025-10-28 to v0.4.3 2026-01-13)

Permalink base: `https://github.com/explodinggradients/ragas/blob/298b68274234c060deacab3cf5fb52aa3a20e885/`. All `file:line` refs below are relative to that base, under `src/ragas/`.

## Purpose
An offline evaluation library for LLM apps. It started with RAG metrics (faithfulness, context precision/recall) and later added agent/multi-turn metrics, synthetic testset generation from a knowledge graph, and an "experiments" API for running a user function over a dataset. It is a library, not a server: there is no online path, no trace ingestion and no UI. It is aimed at Python developers running evals from notebooks or scripts. The codebase is mid-migration. `evaluate()`/`aevaluate()` and every metric class under `ragas.metrics.*` emit `DeprecationWarning` in favour of `@experiment` plus `ragas.metrics.collections` (`evaluation.py:105-111`, `metrics/__init__.py:126-199`). **So two parallel APIs exist, with duplicated metric implementations.**

## Core abstractions

### Unit of data: `SingleTurnSample` / `MultiTurnSample` (legacy path)
`dataset_schema.py:54` and `dataset_schema.py:100`, both pydantic `BaseSample`.
```python
class SingleTurnSample(BaseSample):
    user_input, response, reference: Optional[str]
    retrieved_contexts, reference_contexts: Optional[List[str]]
    retrieved_context_ids, reference_context_ids, multi_responses, rubrics, persona_name, ...
class MultiTurnSample(BaseSample):
    user_input: List[Union[HumanMessage, AIMessage, ToolMessage]]
    reference: Optional[str] = None
    reference_tool_calls: Optional[List[ToolCall]] = None
    rubrics: Optional[Dict[str, str]] = None
    reference_topics: Optional[List[str]] = None
```
- `MultiTurnSample` is one flat bag of optional fields. Every multi-turn metric reads the field it needs (`reference_tool_calls`, `reference_topics`, `reference`), so the schema grows with each new metric. RAG-oriented fields (`retrieved_contexts`) don't exist in the multi-turn sample at all.
- `validate_user_input` (`dataset_schema.py:126-167`) checks message order: a `ToolMessage` must follow an `AIMessage` that has `tool_calls`, or another `ToolMessage`. It does **not** check that the number of tool results matches the number of calls, and it has no id-based linkage.

### Message types
`messages.py:6-134`:
```python
class Message(BaseModel): content: str; metadata: Optional[Dict]
class ToolCall(BaseModel): name: str; args: Dict[str, Any]
class HumanMessage(Message): type = "human"
class ToolMessage(Message):  type = "tool"
class AIMessage(Message):    type = "ai"; tool_calls: Optional[List[ToolCall]]
```
- There is no `SystemMessage`, no `tool_call_id` on `ToolCall` or `ToolMessage`, and no tool name on `ToolMessage`. `content` must be `str`, so multimodal or structured tool outputs have to be stringified. The LangGraph converter drops system messages and raises `TypeError` on non-string content (`integrations/langgraph.py:9-110`).
- Consequence: a tool result cannot be tied to the call that produced it. No metric can score tool-output handling per call, or parallel calls to the same tool.
- Judges see a conversation as `pretty_repr()` text: `Human: ...`, `AI: ...`, `Tools:\n  name: {args}`, `ToolOutput: ...` (`messages.py:56-134`).

### Dataset container: `EvaluationDataset` (legacy)
`dataset_schema.py:318`, built on `RagasDataset(ABC, Generic[Sample])` (`:187`). All samples must share one type (`validate_samples`, `:204`), so single-turn and multi-turn rows can't be mixed. It supports HF/pandas/CSV/JSONL conversion and has no versioning.

### Unit of data (new path): `Dataset` / `DataTable`
`dataset.py:31` (`DataTable`) and `:458` (`Dataset`). This is a list-like wrapper with an optional pydantic `data_model` and a pluggable `backend`. Rows are dicts or models, with no fixed schema. `load`/`save`/`reload` go through the backend (`dataset.py:156-289`).

### Unit of scoring (legacy): `Metric` hierarchy
`metrics/base.py`: `Metric(ABC)` (`:75`) holds `name` and `_required_columns: Dict[MetricType, Set[str]]`. Column names can carry `:optional` or `:ignored` suffixes (`:96-140`). Mixins:
- `SingleTurnMetric` (`:392`) / `MultiTurnMetric` (`:516`). The public `multi_turn_ascore(sample, callbacks, timeout)` wraps `_multi_turn_ascore` in `asyncio.wait_for` plus a callback span (`:584-628`).
- `MetricWithLLM(Metric, PromptMixin)` (`:155`) with `llm` and `output_type ∈ {BINARY, DISCRETE, CONTINUOUS, RANKING}` (`:67`).
- Return type is a bare `float`, with no reason attached. The reason is only reachable through callback traces.

Trade-off: capability is expressed through class mixins, and the runner dispatches on `isinstance(metric, MultiTurnMetric)` (`evaluation.py:265-276`). It is simple, but a metric's input contract is a set of strings validated against `VALID_COLUMNS`, not a typed signature.

### Unit of scoring (new): `SimpleBaseMetric` / `BaseMetric` / `MetricResult`
- `SimpleBaseMetric` (`metrics/base.py:689`): `score(**kwargs)` / `ascore(**kwargs)` return a `MetricResult`. `abatch_score` is an unbounded `asyncio.gather` (`:773-796`).
- `MetricResult` (`metrics/result.py:11-40`) holds `value`, `reason` and `traces`, where `traces` may only have the keys `input` and `output`. It proxies arithmetic and comparisons to `value`.
- `collections.BaseMetric` (`metrics/collections/base.py:13`) is the base for the rewritten metrics. Inputs are plain keyword arguments, for example `ToolCallAccuracy.ascore(user_input, reference_tool_calls)`. There is no sample object.
- `SimpleLLMMetric` (`metrics/base.py:846`), `DiscreteMetric` (`metrics/discrete.py:19`) and `NumericMetric` are user-defined judges built from a format-string prompt. The response model is generated automatically as `{reason: str, value: Literal[allowed_values]}` (`discrete.py:63-73`, `create_auto_response_model` at `base.py:799`). The `llm` is passed in `kwargs` at score time (`base.py:874-911`).

### Batch execution: `@experiment` / `ExperimentWrapper`
`experiment.py:116-232`. It wraps any async function `row -> result`. `arun(dataset, name, backend)` creates an `Experiment(DataTable)` (`:17`), fans out every row, appends non-`None` results, then saves. The experiment has no built-in metric concept: the user function calls metrics itself and returns a dict or model that includes the scores. `version_experiment` (`:21-100`) runs a git commit and creates a `ragas/<name>` branch to snapshot code state.

## Execution model

**Legacy `evaluate()` / `aevaluate()`** (`evaluation.py:59-345`):
- One job per (row, metric) pair, submitted to an `Executor` with `timeout=run_config.timeout` (`:243-278`).
- `Executor` (`executor.py:19`) uses `as_completed` with an `asyncio.Semaphore(max_workers)` (`async_utils.py:58-96`). The default `max_workers` is 16 (`run_config.py:54`). Results are re-sorted by submission index (`executor.py:193-202`).
- **Failure handling:** `wrap_callable_with_index` catches every exception. If `raise_exceptions=False` (the default for `evaluate()`), it logs and returns `np.nan` (`executor.py:64-86`). With `raise_exceptions=True`, the first exception propagates (`:173-174`, `:187-189`). `EvaluationResult` aggregates with `safe_nanmean` (`dataset_schema.py:445`). **Failed rows are silently dropped from the denominator.** Issue #3028 notes that the reported score can rise as the system degrades.
- **Timeout:** per metric call, `asyncio.wait_for` with a default of 180 s (`metrics/base.py:605`, `run_config.py:51`). A timeout becomes `NaN` like any other error.
- **Retries:** tenacity `AsyncRetrying` with `wait_random_exponential(max=60)`, `stop_after_attempt(10)`, retrying on `(Exception,)` by default (`run_config.py:97-115`). It is applied around `BaseRagasLLM.agenerate_text` only (`llms/base.py:113-122`). The instructor-based `InstructorLLM.agenerate` (`llms/base.py:1117-1163`) has no Ragas-level retry wrapper; it relies on client or instructor behaviour, which is unverified. Because the retry covers all exceptions, programming errors are retried 10 times.
- **Parse retries:** separate from transport retries (see Judge implementation).
- **Caching:** optional `DiskCacheBackend` wraps `generate_text`/`agenerate_text` (`cache.py:70`, `llms/base.py:65-66`). The cache key comes from hashed arguments (`cache.py:135`).
- **Cancellation:** `Executor.cancel()` sets a `threading.Event` that `as_completed` checks between futures (`executor.py:56-62`, `async_utils.py:86-96`).
- **Sync entry:** `evaluate()` uses `nest_asyncio` by default so it works in Jupyter (`evaluation.py:475-484`).

**New `ExperimentWrapper.arun`** (`experiment.py:141-198`):
- Uses `asyncio.as_completed` over all rows with **no semaphore**, so concurrency is unbounded and the user must throttle.
- Per-row exceptions are `print`ed and the row is **dropped** from the experiment (`:185-187`). No error record is persisted.
- Results are appended in completion order, not dataset order.
- There is a single `save()` at the end (`:196`), so a crash mid-run loses everything.
- There are no retries or timeouts at the framework level. Retries only happen inside the LLM wrapper, if the user uses one.

## Data model & storage
- Backends implement `BaseBackend` (`backends/base.py:9`): `load_dataset`, `load_experiment`, `save_dataset`, `save_experiment`, `list_datasets`, `list_experiments`, all over `List[Dict]`.
- They are discovered via the `ragas.backends` entry-point group (`backends/registry.py:112-136`). Built-in backends are `local/csv`, `local/jsonl`, `inmemory` and `gdrive` (pyproject `[project.entry-points."ragas.backends"]`).
- Layout: `<root>/datasets/<name>.csv|jsonl` and `<root>/experiments/<name>.csv|jsonl`.
- `LocalCSVBackend._save` rewrites the whole file with `csv.DictWriter` (`backends/local_csv.py:69-94`). Nested values such as message lists or tool calls get stringified, which is lossy for multi-turn data. JSONL handles nesting and datetimes (`backends/local_jsonl.py:61-154`).
- **No run metadata** is stored: model, prompt hash, config, timestamps or per-row errors. An experiment is just a named table of whatever rows the user returned. Versioning is left to git (`version_experiment`).
- Legacy `EvaluationResult` (`dataset_schema.py:412`) keeps scores, the dataset, cost callback and parsed traces in memory only.

**Run comparison:**
- `ragas evals <file> --dataset --metrics --baseline` (`cli.py:371-457`, `run_experiments` at `:264`) takes a per-field mean for numeric fields or a `Counter` for categorical ones (`calculate_aggregated_metrics`, `:173-188`). It prints current, baseline and delta.
- There are no significance tests, no paired per-row diff and no variance.
- **The CLI looks broken at this SHA.** It looks for an object with `run_async` and a `project` with `get_dataset`/`get_experiment` (`cli.py:402-410`). `ExperimentWrapper` only defines `__call__` and `arun` (`experiment.py:134-198`), and the code carries `# TODO: Project class not implemented yet` (`cli.py:404`).

## Judge implementation

**Legacy `PydanticPrompt`** (`prompt/pydantic_prompt.py:82`):
- A prompt is a class with `instruction`, `input_model`, `output_model` and few-shot `examples: List[(InputModel, OutputModel)]`.
- `to_string` builds: instruction, then the JSON Schema of `output_model` with a "use double quotes" note, then the examples as JSON, then `input: <model_dump_json>` and `Output:` (`:89-134`).
- The prompt hashes with SHA-256 over instruction, examples and language (`:414-437`). It saves and loads to JSON with a version-mismatch warning (`:451-507`). It can be translated to other languages via `adapt()` (`:357-394`).
- **Parsing:**
  - `RagasOutputParser.parse_output_string` runs `extract_json` and then pydantic parsing.
  - On `OutputParserException` it calls a second LLM prompt, `FixOutputFormat`, with the bad output and the original prompt, and parses again. `retries_left` counts down from 3 (`:525-558`).
  - When retries run out it raises `RagasOutputParserException`, which the Executor turns into `NaN`.
  - For LangChain and instructor LLMs the fixer is skipped and it is plain `model_validate_json` (`:317-319`).
- `generate_multiple(n)` requests n samples (`:188-349`). This is the only hook for self-consistency.

**New collections metrics:**
- These use `llm.agenerate(prompt_str, ResponseModel)` through `instructor` (for example `metrics/collections/agent_goal_accuracy/metric.py:116-129`).
- JSON mode is the default for OpenAI, chosen over tool mode because of pydantic `Dict` issues (`llms/base.py:574-585`). Structured-output validation and repair is delegated to instructor.
- Prompts are `BasePrompt` subclasses in each metric's `util.py`.

**Calibration:**
- `SimpleLLMMetric.align(train_dataset, embedding_model)` turns the prompt into a `DynamicFewShotPrompt`: retrieval-based few-shot from human-annotated rows (`metrics/base.py:1310-1393`).
- `validate_alignment` computes Cohen's kappa against gold labels (`:1395`, `discrete.py:75-89`).
- Legacy `MetricWithLLM.train()` optimises instructions and demonstrations from `MetricAnnotation` (`metrics/base.py:307-377`; optimisers in `optimizers/genetic.py`, `optimizers/dspy_optimizer.py`).
- This is the most mature judge-alignment loop among the library-style tools I've read.

### Multi-turn / agent metrics: how they are computed

| Metric | File:symbol | Inputs | Algorithm |
|---|---|---|---|
| ToolCallAccuracy | `metrics/_tool_call_accuracy.py:ToolCallAccuracy` (L17-181); collections `metrics/collections/tool_call_accuracy/metric.py:ToolCallAccuracy.ascore` (L89-170) | `user_input` messages, `reference_tool_calls` | See below |
| ToolCallF1 | `metrics/_tool_call_f1.py:ToolCallF1._multi_turn_ascore` (L42-68); collections `tool_call_f1/metric.py` (L69-110) | same | See below |
| AgentGoalAccuracyWithReference | `metrics/_goal_accuracy.py:104-144` | `user_input`, `reference` (desired end state) | See below |
| AgentGoalAccuracyWithoutReference | `metrics/_goal_accuracy.py:148-185` | `user_input` | See below |
| TopicAdherenceScore / TopicAdherence | `metrics/_topic_adherence.py:136-248`; collections `topic_adherence/metric.py` | `user_input`, `reference_topics`, `mode` | See below |
| AspectCritic (multi-turn) | `metrics/_aspect_critic.py:AspectCritic._multi_turn_ascore` (L198-212) | `user_input` (+ optional reference) | See below |

**ToolCallAccuracy** (rule-based)
- Flattens all `AIMessage.tool_calls` in order.
- Gets a sequence-aligned flag: the name lists must be equal. With `strict_order=False`, both lists are sorted by `(name, sorted args)` first.
- Zips the reference and predicted calls pairwise. If the names match, it adds the argument score: the fraction of reference arguments whose `str(pred) == str(ref)`. The legacy version compares through a pluggable `arg_comparison_metric`, `ExactMatch` by default (L58-83).
- Divides by `len(ref)`, applies a coverage penalty, and multiplies by the aligned flag.
- Both empty → 1.0.

**ToolCallF1** (rule-based)
- Builds sets of `(name, hashable(args))` for expected and actual calls.
- Computes precision, recall and F1 with exact argument equality.
- Both empty → 0.0.

**AgentGoalAccuracyWithReference** (two LLM calls)
- `InferGoalOutcomePrompt` over `pretty_repr()` produces `{user_goal, end_state}`.
- `CompareOutcomePrompt(desired=reference, arrived=end_state)` produces `verdict ∈ {"0","1"}`.
- Score is `float(verdict)`.

**AgentGoalAccuracyWithoutReference** (two LLM calls)
- Same as the with-reference version, except that `desired = inferred user_goal`.
- The same LLM infers both the goal and the end state from the same transcript, then compares them.

**TopicAdherenceScore / TopicAdherence** (1 + N + 1 LLM calls)
- Extracts the topics from the Human turns.
- For each topic, asks "did the AI refuse?" and produces `answered = not refused`.
- One call classifies each topic against `reference_topics`, producing `in_scope`. If the list length mismatches it is padded with False or truncated (L228-235).
- Counts:
  - TP = answered & in_scope
  - FP = answered & out_of_scope
  - FN = refused & in_scope
- Returns precision, recall or F1 depending on `mode`, with an epsilon of 1e-10.

**AspectCritic (multi-turn)**
- The instruction is `"Evaluate the Input based on the criteria defined... Criteria Definition: {definition}"` (L130).
- It runs one LLM call over `pretty_repr()` and returns `verdict` 0/1.
- There is no collections equivalent. The replacement is `DiscreteMetric` with a user prompt.

## Online / production path
None in core. The integrations convert other formats into `MultiTurnSample` messages:
- LangGraph/LangChain (`integrations/langgraph.py:convert_to_ragas_messages`)
- AG-UI SSE events (`integrations/ag_ui.py:convert_to_ragas_messages` L691, `run_ag_ui_row` L1266)
- Swarm, LlamaIndex, Bedrock
- `integrations/tracing/*` for Langfuse and MLflow (not read in detail)

`run_ag_ui_row` is the closest thing to "run the agent live, then score". It calls an HTTP endpoint for a **single** user input and turns failures into placeholder rows (L1347-1388).

## Testset generation
- `TestsetGenerator.generate_with_langchain_docs` / `generate` (`testset/synthesizers/generate.py:414-625`) builds a `KnowledgeGraph`: `Node` and `Relationship` pydantic models (`testset/graph.py:35`, `:92`, `:146`) with `NodeType.CHUNK/DOCUMENT` nodes.
- It runs `default_transforms` (`testset/transforms/default.py:34-212`), which choose a pipeline by document-length bins:
  - HeadlinesExtractor and HeadlineSplitter
  - Summary, Themes and NER extractors (LLM)
  - Embedding extractor
  - Cosine-similarity and entity-overlap relationship builders
  - LLM `CustomNodeFilter`
- Generation then runs in two Executor passes: scenarios first, then samples (`generate.py:537-600`). `raise_exceptions=True` is the default here, unlike `evaluate()`.
- Scenarios are node clusters × persona × `QueryStyle` × `QueryLength` (`synthesizers/base.py:34-78`).
- The default distribution is equal weight across `SingleHopSpecificQuerySynthesizer`, `MultiHopAbstractQuerySynthesizer` and `MultiHopSpecificQuerySynthesizer`. A synthesizer is dropped if the graph has no suitable clusters (`synthesizers/__init__.py:24-56`).
- **"Multi-hop" means multi-document, not multi-turn.** Every built-in synthesizer returns a `SingleTurnSample` (`single_hop/base.py:117-134`, `multi_hop/base.py:156-173`). `TestsetSample` can hold a `MultiTurnSample` (`synthesizers/testset_schema.py:20-32`), but at this SHA no synthesizer produces conversations, tool-use scenarios, reference tool calls or user simulations. The only match for "simulat" is a docstring calling synthesizers "scenario simulators" (`generate.py:435`).

## Extension model
- New metric: subclass `MultiTurnMetric`/`MetricWithLLM` and implement `_multi_turn_ascore` (legacy), subclass `collections.BaseMetric` and implement `ascore(**kwargs)`, or use the `@discrete_metric` / `@numeric_metric` / `@ranking_metric` decorators on plain functions (`metrics/decorator.py`, `discrete.py:128-178`).
- Prompts: swap `metric.<x>_prompt` (dataclass fields such as `workflow_prompt`, `topic_extraction_prompt`), or use `PromptMixin.get_prompts`/`set_prompts`.
- Backends: through the `ragas.backends` entry points.
- LLMs: `BaseRagasLLM`, `InstructorBaseRagasLLM`, `llm_factory(model, client=...)`, or the LiteLLM adapter.
- Tracing: LangChain callback groups (`callbacks.new_group`) at the evaluation, row, metric and prompt levels.

## Strengths
- **Readable, pure-function tool-call metrics.** `ToolCallAccuracy` and `ToolCallF1` are deterministic and cheap, with order and exact-match semantics spelled out (`collections/tool_call_accuracy/metric.py:89-170`, `_tool_call_f1.py:42-68`). The legacy `arg_comparison_metric` hook lets you swap exact match for semantic or LLM comparison per argument (`_tool_call_accuracy.py:58-83`).
- **Typed prompt objects.** `PydanticPrompt` gives each judge prompt typed I/O models, typed few-shot examples, a content hash and save/load. Prompts become diffable artifacts (`prompt/pydantic_prompt.py:82-507`).
- **Parse-failure repair is separate from transport retries.** The `FixOutputFormat` prompt handles bad output, and tenacity handles transport errors (`pydantic_prompt.py:525-558`, `run_config.py:97-115`).
- **Judge-alignment loop.** Annotate, build a dynamic few-shot prompt, check Cohen's kappa (`metrics/base.py:1310-1445`, `discrete.py:75-89`). Plus instruction and demonstration optimisers (`optimizers/`).
- **Useful error decomposition.** `TopicAdherence` splits "answered vs refused" from "in scope vs out of scope" into a TP/FP/FN confusion, so precision measures over-answering and recall measures over-refusing (`_topic_adherence.py:237-248`).
- **Structural validation of transcripts** before scoring (`dataset_schema.py:126-167`).
- **Well-built KG testset pipeline.** Composable `Transform`s with `Parallel` stages (`transforms/default.py:123-130`), and personas plus style and length as explicit scenario axes.

## Weaknesses
- **The ToolCallAccuracy coverage penalty is dead code.** `is_sequence_aligned` compares whole name lists, so any length mismatch (a retry, a redundant lookup, or a missing call) sets `sequence_aligned=0` and the score to 0. The warning says "only the first N will be compared", but nothing partial is ever returned (`collections/tool_call_accuracy/metric.py:133-168`). If the penalty did apply, it would be applied twice: the score is divided by `len(ref)` and then multiplied by `compared/len(ref)`. Related issues: #1893, and #2079 (open, "returns score higher than 1.0").
- **Inconsistent empty-case semantics.** Both lists empty gives 1.0 in ToolCallAccuracy (`:117-118`) but 0.0 in ToolCallF1, because precision and recall are both 0 (`_tool_call_f1.py:56-66`).
- **Inconsistent argument equality.** ToolCallAccuracy uses `str(value)` equality, so nested dicts with different key order don't match (`collections/tool_call_accuracy/util.py:exact_match_args`). ToolCallF1 uses order-insensitive `frozenset` hashing (`_tool_call_f1.py:_make_hashable`). Neither handles numeric/string coercion ("5" vs 5 match only in Accuracy) or optional or default arguments.
- **ToolCallF1 collapses duplicates.** Set semantics mean calling the same tool twice with the same args counts once.
- **No tool-call identity.** `ToolCall` and `ToolMessage` have no ids, so tool results can't be attributed to calls. There is no system prompt, and there are no per-turn metadata such as latency or tokens (`messages.py`).
- **AspectCritic `strictness` is ignored.** It is stored and forced odd (L134-138), but both `_ascore` and `_multi_turn_ascore` make one `generate` call and pass a one-element list to the majority vote (`_aspect_critic.py:190-212`).
- **AgentGoalAccuracyWithoutReference is circular.** One LLM pass infers both goal and end state from the transcript, and a second checks that they agree. There is no rubric, no partial credit, and binary output only (`_goal_accuracy.py:168-185`). Issue #2122 reports scores that are always 0. Issue #3014 reports that the collections version never sets `output_type`, so `train()` fails.
- **Silent NaN on failure.** Errors become `NaN` and `nanmean` drops them from the denominator. There is no count of failed rows in the summary (`executor.py:64-86`, `dataset_schema.py:445`). See #3028, #2893 and #2980.
- **Experiments are fragile.** Concurrency is unbounded, failures are printed and dropped, results come in completion order, and a crash loses everything because the only save is at the end. No run metadata is stored (`experiment.py:141-198`).
- **Baseline comparison is only a delta of means,** and the CLI path appears broken at this SHA (`cli.py:402-410` vs `experiment.py:116-198`).
- **Two coexisting metric APIs** with duplicated implementations (`metrics/_*.py` vs `metrics/collections/*`). Deprecated names are re-exported via `__getattr__`. Several multi-turn metrics are marked deprecated with no drop-in successor (AspectCritic).
- **Telemetry is on by default** (`_analytics.py:41-49`, opt out with `RAGAS_DO_NOT_TRACK=true`). The legacy `evaluate()` also silently defaults to `gpt-4o-mini` via `OpenAI()` when no LLM is set (`evaluation.py:174-181`).

## Worth adapting
- **Split tool-call scoring.** Use a strict sequence metric (sequence equality times per-call argument accuracy) alongside a set-based F1. They answer different questions: "did it follow the plan?" versus "did it make the right calls at all?". Keep both, but fix the empty-case, duplicate and argument-normalisation semantics, and make argument comparison pluggable per argument.
- **Typed judge prompts.** Pydantic input model, output model and examples, rendered with the output JSON Schema and hashed. This makes prompt identity part of result provenance.
- **Two-layer failure handling for judges.** Transport retries with jittered exponential backoff, separate from a bounded "repair the output" LLM call on parse failure.
- **A confusion matrix for topic or scope adherence.** Answered or refused, crossed with in scope or out of scope, gives precision (over-answering) and recall (over-refusing) instead of one opaque score.
- **Structural validation of transcripts before scoring,** covering tool-message ordering.
- **An alignment loop for LLM judges.** Human labels feed dynamic few-shot examples, validated by Cohen's kappa on a held-out split.
- **Entry-point-discovered storage backends,** for pluggable dataset and experiment stores.

## Don't copy
- **Converting errors to NaN and averaging with nanmean without reporting the failure count.** It hides regressions and biases scores upward as failures grow. Failed rows should be explicit results with an error, counted in every aggregate.
- **Unbounded `as_completed` over the whole dataset, a single save at the end, and dropped failed rows** (`ExperimentWrapper.arun`). There is no backpressure, no partial results, no per-row error record, and ordering is nondeterministic.
- **One flat optional-field sample schema that grows per metric** (`reference_tool_calls`, `reference_topics`, `rubrics` on `MultiTurnSample`). It couples the dataset schema to the metric catalogue. Expectations belong in a typed, metric-agnostic `expected`/`reference` payload, or in metric config.
- **Messages without ids** for tool calls and results, and **no system message**. These can't express parallel calls or attribute tool outputs.
- **Retrying on `(Exception,)` ten times by default.** Programming and validation errors get retried with up to 60 s waits.
- **Binary goal verdicts where the same model infers both the goal and the outcome.** It is self-referential with no reference anchor or rubric.
- **A comparison that is only a difference of means,** with no per-row pairing, variance or significance.
- **Keeping two parallel metric APIs alive for long,** and silently defaulting to a paid model when none is configured.

## Open questions
- Whether `InstructorLLM` calls get any retry or timeout beyond the per-metric `asyncio.wait_for`. It depends on the instructor/client `max_retries` defaults, which I didn't verify.
- Whether issue #2079 (ToolCallAccuracy > 1.0) still reproduces at this SHA. From the code read, the score is bounded by 1, so the issue may be stale.
- How `integrations/tracing/langfuse.py` and the MLflow integration map traces to samples. Not read.
- Whether any example or notebook outside `src/` does multi-turn synthetic generation or user simulation. I only searched `src/`.
- How the `gdrive` backend handles concurrency and conflicts. Not read.
- `validate_supported_metrics` behaviour when a metric list mixes single-turn-only metrics with multi-turn samples. The results indexing `results[len(metrics)*i + j]` (`evaluation.py:299`) assumes every metric was submitted for every row. Not verified.
