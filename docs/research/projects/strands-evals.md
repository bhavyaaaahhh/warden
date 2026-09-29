# Strands Evals (AWS)

Repo: https://github.com/strands-agents/evals @ `8ad35adbf8037849167845f2aa05e67eaf47639e` · License: Apache-2.0 · Lang: Python · Last commit: 2026-09-28 · Activity: active. Frequent minor releases (v1.2.0 2026-08-21, v1.3.0 2026-09-15, v1.4.0 2026-09-22), about 220 stars. Open issues are mostly feature requests, with a few evaluator correctness bugs (#334, #342, #355).

Permalink prefix: `SE` = `https://github.com/strands-agents/evals/blob/8ad35adbf8037849167845f2aa05e67eaf47639e/src/strands_evals`

## Purpose

Strands Evals is an offline evaluation library tied closely to the Strands Agents SDK. It covers:
- An `Experiment` made of `Case`s, a user-supplied task function and a list of `Evaluator`s (mostly LLM judges, plus a few deterministic ones).
- Session mappers that turn OTel spans from Strands, LangChain, ADK, OpenInference and others into a common `Session`/trace model for trajectory evaluators.
- Simulation (`ActorSimulator` for users, `ToolSimulator` for tools).
- Chaos testing (`ChaosCase`, `ChaosPlugin`), plus experimental red-teaming, a CLI and remote trace providers (CloudWatch, Langfuse, OpenSearch).

The important difference from every other project in `docs/research/projects/` is where multi-turn control lives. **The library does not own the conversation loop.** The user writes the `while simulator.has_next()` loop inside the task function (README "Multi-turn Conversation Simulation"). The library supplies the simulator, the tool fakes and the fault injectors as parts. Contrast LangWatch Scenario, tau-bench and DeepEval, where the framework drives the turns.

## Core abstractions

### Test case: `Case`
`SE/case.py#L9-L56`:
```python
class Case(BaseModel, Generic[InputT, OutputT]):
    name: str | None = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: InputT
    expected_output: OutputT | None = None
    expected_assertion: str | None = None
    expected_trajectory: list[Any] | None = None
    expected_interactions: list[Interaction] | None = None
    expected_environment_state: list[EnvironmentState] | None = None
    metadata: dict[str, Any] | None = None
```
- **A flat, serializable Pydantic record** with separate expected slots for output, trajectory, interactions (multi-agent node hand-offs) and **named environment state** (`EnvironmentState{name, state}`, `SE/types/evaluation.py#L36-L45`).
- **No first-class persona, goal or hidden-information field.** For multi-turn runs, `input` is the opening user message, and `metadata["task_description"]` optionally feeds persona generation (`SE/simulation/actor_simulator.py#L119-L140`).
- **Contrast.** tau-bench stores `instruction` and gold actions. Scenario stores prose `description` and `criteria`. Here a case is data but thin on simulation intent. The persona is generated rather than authored, unless the user builds an `ActorProfile` by hand.

### Evaluation record: `EvaluationData` / `EvaluationOutput`
`SE/types/evaluation.py#L77-L138`. `EvaluationData` pairs every `expected_*` with an `actual_*` (output, trajectory as a list or `Session`, interactions, environment state). The task function returns either a bare output or a dict `{output, trajectory, interactions, environment_state, input}` (`SE/experiment.py#L223-L269`). `EvaluationOutput{score: float, test_pass: bool, reason, label}` is the per-evaluator verdict.

### User simulator: `ActorSimulator`
`SE/simulation/actor_simulator.py#L38-L325`.
- **Profile.** An `ActorProfile{traits: dict, context: str, actor_goal: str}` (`SE/types/simulation/actor.py#L7-L22`). `from_case_for_user_simulator` **generates** the profile with a separate structured-output LLM call from `case.input` and `metadata.task_description` (`#L62-L140`).
- **System prompt.** `DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE` (`SE/simulation/prompt_templates/actor_system_prompt.py`) holds the profile as a dict dump, 2 to 3 sentence replies, "never give solutions", no meta-references and exit conditions.
- **Seeding the conversation.** The history is seeded with a **randomly chosen greeting** (`random.choice(INITIAL_GREETINGS)`, `#L249-L256`, unseeded) and the initial query in the simulator's own voice.
- **One call per turn.** `act(agent_message)` makes one Strands `Agent` call with `structured_output_model=ActorResponse{reasoning, stop, message, stop_reason}` (`#L266-L321`).
- **Goal check.** The simulator has a tool, `get_conversation_goal_completion` (`SE/simulation/tools/goal_completion.py#L12-L38`), which spins up **another** LLM agent to assess whether the goal is met.
- **Stop conditions.** Either the simulator sets `stop=true` (`stop_reason="goal_completed"`), or `_turn_count >= max_turns` (default 10, `stop_reason="max_turns"`). `has_next()` is just `not self.stop` (`#L323`).
- **Trade-off.** The simulator decides when the goal is met (as in tau-bench's `###STOP###`). No judge sits in the loop and there is no mid-conversation assertion hook. Assertions happen afterwards, over the whole session.

### Tool simulator with shared state: `ToolSimulator` + `StateRegistry`
`SE/simulation/tool_simulator.py#L21-L412`.
- **`@simulator.tool(output_schema=, name=, share_state_id=, initial_state_description=)`** (`#L281-L345`). It extracts the tool spec from the function's signature and docstring and registers a `RegisteredTool`.
- **The function body is never executed.** `_create_tool_wrapper` (`#L191-L232`) replaces it with a wrapper that calls `_call_tool`.
- **`_call_tool`** (`#L252-L280`) builds `TOOL_RESPONSE_PROMPT_TEMPLATE` from:
  - the tool name and input JSON schema
  - the output schema (a Pydantic model, default `DefaultToolResponse`)
  - the call payload
  - `state_context`, the JSON of the current shared state

  A fresh Strands `Agent` then generates a response with `structured_output_model=output_schema` (`_simulate_tool_call`, `#L234-L241`). If the response is not JSON, the parser falls back to `{"result": text}` (`#L243-L250`).
- **"Shared state" is a prompt-context log, not a database.** `StateRegistry` (`#L21-L136`) keeps, per `state_key` (`share_state_id or tool_name`, `#L211`), an `initial_state` **description string** plus a `deque(maxlen=20)` of `{tool_name, response, timestamp, parameters}` records (`cache_tool_call`, `#L93-L118`).
  - Tools with the same `share_state_id` see each other's recent calls. That is how `create_booking` followed by `get_booking` stays consistent.
  - The consistency is only as good as the LLM's reading of the log, and older calls fall off after 20.
- **Contrast.**
  - tau-bench tools are real Python functions over a JSON DB, whose final state is hashed against gold.
  - Here state is never materialized, so `expected_environment_state` cannot be checked against simulated tools. `StateEquals` (`SE/evaluators/deterministic/environment_state.py#L15`) compares only what the task function puts in `environment_state`.

### Trajectory matching
There are two layers.
1. **Deterministic.** `ToolCalled(tool_name)` (`SE/evaluators/deterministic/trajectory.py#L6-L47`) gives 1.0 if any `ToolExecutionSpan` in the `Session` has that name. It checks name only, with no argument or order checks. Siblings: `Equals`, `Contains`, `StartsWith`, `StateEquals` and `SkillInvoked`.
2. **Judge-mediated.** `TrajectoryEvaluator(rubric)` (`SE/evaluators/trajectory_evaluator.py#L14-L104`) gives an LLM judge three scorer **tools** (`SE/tools/evaluation_tools.py`). The judge "chooses the most appropriate scoring tool based on rubric requirements... use the tool's output as an initial score, adjust based on other evaluation criteria" (`prompt_templates.py#L92-L122`).

The three scorers:

| Scorer | Lines | Behavior |
|---|---|---|
| `exact_match_scorer` | `#L4-L22` | Positional equality over `zip(actual, expected)` divided by `len(expected)`. No guard for an empty `expected` (ZeroDivisionError), and extra trailing actions are not penalized. |
| `in_order_match_scorer` | `#L25-L45` | Subsequence match. Extra actions are allowed. |
| `any_order_match_scorer` | `#L48-L67` | Set intersection divided by `len(expected)`. Duplicates in `expected` cap the score below 1.0, and counts are ignored. |

Matching compares whole list elements with `==`. Whether arguments count depends on what the user puts in the list (usually tool names). Separate LLM evaluators handle `ToolSelectionAccuracyEvaluator` and `ToolParameterAccuracyEvaluator`. Because the judge picks the mode and adjusts the number, the **match mode is not a declared property of the case**. The same case can be scored under different modes on different runs.

### Chaos / fault injection: `ChaosCase`, effects, `ChaosPlugin`
- **`ChaosCase(Case)`** (`SE/chaos/case.py#L19-L201`) adds `effects: {"tool_effects": {tool_name: [ToolEffect]}, "model_effects": {"*": [ModelEffect]}}`.
- **Validation.** At most one effect per tool (`_validate_tool_effects`, `#L93`) and at most one pre-hook model effect.
- **`ChaosCase.expand(cases, effect_maps, include_no_effect_baseline)`** (`#L114`) builds the Cartesian product of cases and named effect maps, with composite names such as `flight_search|search_timeout` and a fresh `session_id` each.

**Effects** (`SE/chaos/effects.py`) form a Pydantic hierarchy with a `hook: "pre"|"post"` ClassVar and an `effect_type` discriminator. That gives serialization round-trips via `ToolEffectUnion` and `ModelEffectUnion` (`#L298-L316`).

| Effect | Category | Hook | Behavior |
|---|---|---|---|
| `Timeout`, `NetworkError`, `ExecutionError`, `ValidationError` | Tool | pre | Return a fixed error string used as the cancel message |
| `TruncateFields`, `RemoveFields`, `CorruptValues` | Tool | post | Mutate the parsed JSON tool result |
| `EmptyResponse`, `FullRefusal` | Model | pre | Cancel the model call |
| `MalformedJson`, `Confabulation`, `SuccessFraming` | Model | post | Rewrite the assistant text |

**Injection** (`SE/chaos/plugin.py#L48-L289`) is a Strands `Plugin` using SDK hooks:
- `BeforeToolCallEvent` sets `event.cancel_tool = effect.apply()` (`#L91-L116`).
- `AfterToolCallEvent` rewrites `event.result["content"]` blocks (`#L146-L174`, `_apply_to_tool_blocks` `#L272-L289`).
- `BeforeModelCallEvent` sets `event.cancel` (`#L192-L198`).
- `MessageAddedEvent` rewrites non-tool-use assistant content (`#L201-L213`).
- Structured-output `MalformedJson` fires **once per invocation**, tracked in `invocation_state` (`#L118-L143`) and cleared in `after_invocation`. The agent's corrected retry then passes.

**Scoping.** A `ContextVar` holds the active case (`SE/chaos/_context.py`). `ChaosExperiment` wraps the user task so that it sets and resets the var around each case (`SE/chaos/experiment.py#L100-L121`). The task body "contains zero chaos concepts".

**Determinism:**
- **Which fault fires, and where, is fully deterministic.** No probability or rate parameter exists. An effect attached to a tool fires on **every** call of that tool in that case. The case, not a random draw, decides.
- **Payload content is not seeded.** `RemoveFields`/`CorruptValues` choose keys with the module-global `random.sample`/`random.choice` (`#L237-L238`, `#L288-L294`). `Confabulation`, `FullRefusal` and `SuccessFraming` pick templates with `random.choice` (`#L422-L473`).
- **So a rerun of the same `ChaosCase` hits the same tool with the same fault class, but not necessarily the same corrupted keys or text.** No seed parameter exists anywhere in the package (grep for `seed` finds only a docstring).

Chaos-specific LLM evaluators (`FailureCommunicationEvaluator`, `RecoveryStrategyEvaluator` and `PartialCompletionEvaluator`, in `SE/evaluators/chaos/`) grade how the agent handled the injected fault.

## Execution model

`Experiment.run_evaluations_async(task, max_workers=10, evaluation_data_store=None)` (`SE/experiment.py#L620-L718`) runs an `asyncio.Queue` with N worker tasks (`_worker`, `#L499-L593`). Sync tasks run in `asyncio.to_thread` (`#L243-L247`). `run_evaluations` is the sequential wrapper with `max_workers=1` (`#L595-L618`).
- **Retries.** Task and evaluator calls retry only **throttling** errors, via tenacity: 6 attempts with exponential backoff (`#L59`, `#L271-L328`, `#L330-L468`).
- **Failure isolation.**
  - A task exception marks every evaluator `test_pass=False, score=0, reason="An error occurred: ..."` (`#L569-L590`). An infrastructure error is indistinguishable from an agent failure in the report.
  - Evaluator exceptions are recorded as failures (`#L440-L468`).
- **Tracing.** Each case gets an OTel span with `gen_ai.evaluation.*` attributes.
- **Caching (task-output level only).** When an `EvaluationDataStore` (a Protocol with `load`/`save` by `case.name`, `SE/evaluation_data_store.py`) is passed, a stored `EvaluationData` is reused and the task is skipped (`#L533-L543`). `LocalFileTaskResultStore` writes `<dir>/<case_name>.json` (`SE/local_file_task_result_store.py`).
  - The key is the **case name only**. Changing the agent, prompt or model does not invalidate it.
  - This is "replay the transcript, re-score it" caching. It is useful for iterating on evaluators, and dangerous if mistaken for an agent-regression run.
  - Judge calls themselves are never cached.
- **Repeats.** None built in. Open issue [#349](https://github.com/strands-agents/evals/issues/349) asks for a "risk classification utility for multi-run results (bug vs flaky vs pass)".

## Data model and storage

- **Experiment definitions.** `Experiment.to_file` and `from_file` (`#L732`, `#L851`) serialize cases and evaluator configs to JSON. Custom evaluator classes must be registered on load. Chaos effects round-trip thanks to the discriminated unions.
- **Results.** `EvaluationReport{overall_score, scores[], cases[], test_passes[], reasons[], detailed_results[][], diagnoses[], recommendations[]}` (`SE/types/evaluation_report.py#L10-L30`) uses **parallel arrays**, one report per evaluator. It has `to_file`/`from_file` (`#L295`, `#L328`) and `flatten` for merging chaos sub-reports (`#L65`).
- **No run id, no baseline, no diff.** The CLI `report` command only renders or dumps a saved report (`SE/cli/commands/report.py`). `run --fail-on` gives a CI exit code based on `test_passes` or `overall_score` (`SE/cli/commands/run.py#L204-L216`). Comparing across runs is left to the user (or to CloudWatch, via `telemetry/_cloudwatch_logger.py`).
- **Test runner.** There is no pytest plugin. Integration is "call `run_evaluations` in a test and assert on the report", or the `strands-evals run` CLI.

## Judge implementation

Every LLM evaluator constructs a Strands `Agent` and calls it with `structured_output_model=` (for example `TrajectoryEvaluator.evaluate`, `#L70-L83`). Parsing is therefore Pydantic validation through the SDK's structured output. There is no custom repair path. Prompts are versioned template modules (`*_v0.py`). Open issue [#342](https://github.com/strands-agents/evals/issues/342) reports that judges silently score 0 when a trace exceeds the context window. No calibration or human-agreement tooling exists.

## Online / production path

`providers/` (CloudWatch, Langfuse, OpenSearch) and `mappers/` pull production traces into a `Session` for offline evaluation of real sessions. `detectors/` (failure detection, root-cause analysis) run LLM diagnosis on failing cases. This is batch re-scoring of traces, not streaming.

## Extension model

- Subclass `Evaluator` (sync and async `evaluate`).
- Implement `EvaluationDataStore`.
- Add a session mapper for a new framework's OTel dialect.
- Custom `structured_output_model` for the actor (it must have `message` and `stop` fields, `#L258-L264`).
- Extra tools for the actor or for the trajectory judge.

## Strengths

- **Chaos as data.** Effects are typed, serializable and discriminated. `ChaosCase.expand` builds the case-by-fault matrix with an explicit baseline (`chaos/case.py#L114`). Faults are attached by tool name, which gives deterministic placement without probabilistic sampling.
- **Hook-based injection keeps the agent code clean.** The ContextVar scoping and SDK hooks (`chaos/plugin.py`, `chaos/experiment.py#L100-L121`) inject faults without touching the agent. The once-per-invocation malformed structured output is a careful touch that tests recovery without breaking typed callers.
- **Expected-vs-actual pairs cover four dimensions**: output, trajectory, interactions and environment state (`types/evaluation.py#L77-L109`).
- **Framework-neutral trajectory input.** Session mappers normalize OTel from several agent frameworks into one `Session` model.
- **Shared-state tool simulation** lets multi-tool flows stay roughly consistent without writing fakes (`tool_simulator.py#L211`, `#L93-L118`).

## Weaknesses

- **The user writes the multi-turn loop** (README). There are no turn-level assertions, no in-loop judge and no standard transcript shape unless the user collects spans correctly (`memory_exporter.clear()` per turn in the README example).
- **Simulation is nondeterministic by construction:**
  - an unseeded random greeting (`actor_simulator.py#L251`)
  - an LLM-generated persona (`#L138`)
  - a nested LLM goal-completion call (`goal_completion.py#L35`)
  - LLM-generated tool responses

  None of these is cached, and there are no seeds.
- **Simulated tool state is a 20-entry prompt log**, not a state machine (`tool_simulator.py#L48`). The final state cannot be verified, and consistency depends on the LLM.
- **The trajectory match mode is chosen by the judge at runtime**, and the scorers have edge-case bugs: division by zero on an empty expected list, and duplicate handling (`tools/evaluation_tools.py#L22`, `#L63-L67`).
- **The task-output cache is keyed by case name only** (`local_file_task_result_store.py`). Stale results are silent.
- **Task errors are scored as agent failures** (`experiment.py#L569-L590`).
- **Results use parallel arrays** with no run identity or comparison support (`evaluation_report.py#L23-L30`).
- **Tight coupling to the Strands SDK.** The simulator, judges, tool simulator and chaos plugin all instantiate `strands.Agent` or use Strands hooks. Chaos cannot target an agent built on another framework.

## Worth adapting

- **Typed, discriminated fault effects** with a `pre`/`post` hook class (cancel the call versus mutate the result). Tool-scoped attachment in the case, plus a `expand(cases × effect_maps, baseline=True)` matrix generator.
- **Deterministic fault *placement*** (every call to tool X in this case), with the case as the unit of variation instead of an injection probability. Add a per-case seed for payload mutation, which Strands lacks.
- **Expected/actual slots per dimension** (output, trajectory, interactions, env state) on one record.
- **A structured-output stop signal for the simulator**, with `stop_reason` in `{goal_completed, max_turns}` stamped by the harness rather than by the LLM (`actor_simulator.py#L281-L321`).
- **An explicit baseline variant** alongside each chaos variant, so resilience is measured as a delta.

## Don't copy

- **LLM-generated tool responses as the default "environment."** Without materialized state, outcome checks are impossible and runs are not reproducible. Prefer real or fake stateful tools, as in tau-bench, and use LLM fakes only for the long tail.
- **Letting the judge choose the trajectory match mode.** The mode (exact, in-order, any-order, with or without args) should be declared on the case and computed deterministically.
- **A result cache keyed only by case name.** Key on case content plus agent and config fingerprint, or label it clearly as "re-score stored transcript".
- **Unseeded module-global `random` in fault payloads and simulator setup.**
- **Collapsing task exceptions into `score=0` failures.** Keep `error` as a separate status.
- **A parallel-array report schema.** Use row-per-(case, evaluator) records with stable ids.

## Open questions

- Whether `ToolSimulator`-backed tools get chaos effects applied. Both go through Strands tool hooks, so probably yes [inferred], but I did not trace it.
- How `experiment_generator.py` builds cases (topic planning, persona diversity). It is out of scope and was not read.
- Whether the Strands SDK exposes a seed for model calls that users could thread through. The evals package itself has none.
