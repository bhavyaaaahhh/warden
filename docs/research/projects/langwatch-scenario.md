# LangWatch Scenario

Repo: https://github.com/langwatch/scenario @ `0698d00d0fee261b3fa05e1131044be686f484dc` · License: Apache-2.0 · Lang: Python + TypeScript (two SDKs in one repo, `python/` and `javascript/`) · Last commit: 2026-09-29 · Activity: very active. Separate release trains (python v1.5.0 and javascript v1.6.0, both 2026-09-06), about 980 stars, about 190 open issues. Many recent issues deal with judge-verdict semantics and voice.

Permalink prefixes used below:
- `PY` = `https://github.com/langwatch/scenario/blob/0698d00d0fee261b3fa05e1131044be686f484dc/python/scenario`
- `JS` = `https://github.com/langwatch/scenario/blob/0698d00d0fee261b3fa05e1131044be686f484dc/javascript/src`

All refs are Python unless marked `JS`. The two SDKs mirror each other on purpose (comments such as "The JavaScript SDK uses the same text; keep them in sync" in `_judge/criterion_verdicts.py`). Divergences are called out where I checked them.

## Purpose

Scenario is an offline testing library for multi-turn agents. It runs inside an ordinary test runner (pytest or vitest). A test declares three agents: the agent under test, an LLM user simulator and an LLM judge. The executor makes them take turns. An optional **script** lets the author mix fixed turns, free simulated turns, plain-code assertions and judge checkpoints. Results print to the console and are optionally streamed as events to the LangWatch platform. The library also has voice and red-team modes, which this dossier ignores.

Compared with the other dossiers:
- **tau-bench.** An episode there is a benchmark with gold DB state. In Scenario, an episode is a unit test with free-text criteria.
- **DeepEval and promptfoo.** They simulate first and score afterwards. Scenario lets the judge run inside the loop, and the judge can stop the conversation.

## Core abstractions

### Test case: `scenario.run(...)` arguments (not a data type)
`PY/scenario_executor.py#L2325-L2345` `run(name, description, agents, max_turns, min_turns, cache_key, script, set_id, metadata, parameters, fields, evaluators, ...)`.
- **No test-case record exists.** A scenario is the argument list of a function call inside a test function, so there is no dataset. This fits code-first unit tests. It is poor for managing hundreds of cases, diffing them, or generating them.
- **Persona.** `UserSimulatorAgent(persona=...)` (`PY/user_simulator_agent.py#L160`) is injected as a `<persona>` block (`#L381-L385`).
- **Goal and hidden information.** Both are only free text in `description`. No field separates what the user knows from what the agent may see. The simulator gets `description` inside `<scenario>` (`#L392-L430`).
  - The judge also gets `description` (`PY/judge_agent.py#L761-L808`). So the "hidden" user facts are visible to the judge. They are not visible to the agent under test.
  - Contrast: tau-bench keeps `instruction` for the user only, and the evaluation is a separate gold record.
- **Criteria.** `JudgeAgent(criteria=[str])` holds natural-language rubric lines. `judge(criteria=[...])` script steps can override them per call (`PY/types.py#L104-L137` `JudgmentRequest`).
- **Structured fields.** `fields`, `parameters` and `metadata` are free dicts. `fields` is readable from state (`PY/scenario_state.py` `ScenarioState.fields`/`field`).

### Unit of work: `AgentAdapter`, used for every role
`PY/agent_adapter.py#L16-L124`:
```python
class AgentAdapter(ABC):
    name: Optional[str] = None
    role: ClassVar[AgentRole] = AgentRole.AGENT
    @abstractmethod
    async def call(self, input: AgentInput) -> AgentReturnTypes: ...
```
- **One interface covers the agent under test, the simulator and the judge.** The role enum is `USER | AGENT | JUDGE` (`PY/types.py#L85-L101`).
- **`AgentInput`** (`PY/types.py#L139-L185`) contains:
  - `thread_id`
  - `messages`: full history, OpenAI chat format
  - `new_messages`: messages since this agent last spoke
  - `judgment_request`
  - `scenario_state`
  - `propagation_headers`
- **Return types.** `str`, one OpenAI message, a list of messages, or a `ScenarioResult`. Only the judge is expected to return a result, but nothing enforces that.
- **Trade-off.** Any agent can be wrapped in about five lines, and adapters can be stateless (replay `messages`) or stateful (key on `thread_id`). The cost is that the history format is OpenAI chat messages in Python and Vercel AI SDK `ModelMessage` in JS (`JS/domain/agents/index.ts#L1,L62-L66`). Transcripts are therefore not portable between the SDKs.

### Tool calls inside a turn
Tool calls are captured in two ways, and both are merged:
1. **Returned messages.** The adapter returns assistant messages with `tool_calls` plus `role:"tool"` result messages, which go into history.
2. **OpenTelemetry spans.** A `JudgeSpanCollector` span processor (`PY/_tracing/judge_span_collector.py#L14`) collects spans emitted in the process during the run.

`ScenarioState.tool_calls(name)` merges "the tool calls of the assistant messages and the tool spans of the traces" (`PY/scenario_state.py#L214-L232`, `_state_views.py#L489`). The judge sees both: a transcript with `[tool_call: name(args)]` lines (`PY/_judge/judge_utils.py#L140-L217`) and a digest of the spans (`judge_agent.py` `_build_trace_digest`). A black-box agent that returns only text still gets tool-level judging if it emits OTel spans. Nothing in the adapter contract requires a tool trace.

### Script DSL: `user() / agent() / judge() / proceed() / succeed() / fail()` plus plain functions
`PY/script.py#L20-L388`. Every step is `lambda state: state._executor.<step>(...)`. A script step is any `Callable[[ScenarioState], None | ScenarioResult | Awaitable]` (`PY/types.py#L440-L488`).

| Step | Behavior |
|---|---|
| `user("text")` / `agent("text")` | Injects a fixed message with no LLM call (`_script_call_agent`, `PY/scenario_executor.py#L1815-L1935`, content branch) |
| `user()` / `agent()` | Calls the simulator or the agent for one generated turn |
| `judge(criteria=[...], additional_context=...)` | A **checkpoint** (`#L1883-L1912`). A pass records the result and the script continues. A fail ends the run with the failed criteria accumulated so far. |
| `proceed(turns=N, on_turn=, on_step=)` | Free-running loop of user, agent and judge turns, optionally capped at N turns, with per-turn hooks (`#L1745-L1789`) |
| a plain function raising `AssertionError` | A deterministic check. It fails the run, emits a finished event and **re-raises** so the test runner sees a normal assertion failure (`#L724-L726`, `#L755-L798`) |
| `succeed()` / `fail()` | Hard stop with a verdict (`#L1791-L1806`) |

The default script is `[proceed()]` (`#L262`). A fully free conversation is simply the degenerate script.
- **Why it is useful.** One construct expresses scripted openers, mid-conversation assertions ("after turn 2 the agent must have called `get_order`"), judge checkpoints at specific turns and free continuation.
- **Why it is fragile.** Steps are closures over a private executor (`state._executor`). A script is code, not data, so it cannot be serialized, stored, diffed or rerun from a UI.

### Result: `ScenarioResult` and `CriterionResult`
`PY/types.py#L250-L328`. Fields:
- `success`, `messages` and `reasoning`
- `passed_criteria`, `failed_criteria` and `inconclusive_criteria` (lists of strings)
- `criteria: List[CriterionResult{criterion, requirement, status, reasoning}]`
- `total_time`, `agent_time` and `evaluations`

The verdict has three states per criterion, but the run is still boolean: an inconclusive criterion "still fails the run" (`#L240-L248`). The JS field names differ (`metCriteria`/`unmetCriteria`, `JS/domain/core/execution.ts#L74-L93`). That is naming drift between two SDKs that claim parity.

## Execution model

### Turn loop
`_new_turn` (`PY/scenario_executor.py#L528-L562`) resets the pending roles to `[USER, AGENT, JUDGE]` and opens a LangWatch trace per turn. `_step` (`#L588-L618`) pops the next role, finds the next pending agent with that role and calls it. When all roles are consumed it starts a new turn and checks `max_turns` (default 10, `#L607`).
- **The judge runs every turn by default.** Each free turn therefore makes three LLM calls (simulator, agent, judge decision). `min_turns` suppresses the judge's decision call below a floor (`judge_agent.py#L668-L674`).
- **Multiple agents per role are allowed.** `_next_agent_for_role` iterates in list order (`#L619-L629`), which gives simple multi-agent or multi-simulator turns.

### Concurrency
`run()` runs every scenario in a fresh thread with its own event loop (`#L2473-L2492`), so blocking user code does not stall parallel scenarios. The comment warns that adapters bound to the caller's loop must use `arun`. Parallelism across scenarios is left to the test runner (pytest-xdist or vitest concurrency). The library has no worker pool.

### Failure handling
- Adapter exceptions are wrapped as `RuntimeError("[AgentName] ...")` and propagate (`#L1227-L1232`). There is no retry or timeout in the executor.
- Every exit path emits exactly one "run finished" event, `BaseException` included (`#L849-L863`, `_emit_finished_once`).

### Stop conditions, in precedence order
1. A script step returns a `ScenarioResult`: `succeed()`, `fail()`, a failed checkpoint, or a verdict from the final judge.
2. A script function raises `AssertionError`.
3. The judge's decision phase calls `make_verdict`. This leads to the verdict phase. A **voluntary** verdict that comes back inconclusive keeps the conversation going (`judge_agent.py#L691-L701`).
4. The last turn (`current_turn >= max_turns - 1`) forces a verdict (`#L638-L641`).
5. The script ends with checkpoint results. The run succeeds if no checkpoint failed (`#L800-L845`).
6. Otherwise `_reached_max_turns` returns failure **without consulting the judge** (`#L631-L665`, open issue [#934](https://github.com/langwatch/scenario/issues/934)).

### Caching and determinism (Python only)
`scenario_cache` (`PY/cache.py#L60-L146`) is a `wrapt` decorator backed by `joblib.Memory` at `~/.scenario/cache` (`#L30-L34`). It is applied to `UserSimulatorAgent._generate_text` (`user_simulator_agent.py#L350`) and `JudgeAgent.call` (`judge_agent.py#L597`). Users can apply it to their own adapter as `@scenario.cache()`.
- **When it is active.** Only when `run(cache_key=...)` is set (`#L109-L110`).
- **Cache key.** `json.dumps({cache_key, scenario config minus agents, all_args})`, where an `AgentInput` is dumped minus `thread_id` (`#L126-L139`).
- **Known flaws:**
  - Only positional arguments enter the key: `zip(parameters, args)` (`#L120`), open issue [#926](https://github.com/langwatch/scenario/issues/926).
  - [inferred] Every message gets a per-turn random `trace_id` (`scenario_executor.py#L409`). That field is not excluded, so cache keys after the first turn probably differ between runs. See Open questions.
- **Seeds.** No seed parameter exists. Determinism comes from `temperature=0.0` defaults (simulator `#L216`, judge `#L405`) plus this cache.
- **TS.** The JS SDK has no equivalent cache. A grep of `javascript/src` for `cacheKey` finds nothing.

### Repeats
None. `pytest-repeat` or a loop is the only option. No pass^k, flake rate or variance calculation exists anywhere. That is notable next to tau-bench's pass^k.

## Data model and storage

- **In memory.** `ScenarioState` (`PY/scenario_state.py#L39`) holds the messages, the turn stamps and a span provider. It also offers read helpers (`tool_calls`, `traces`, `turns`, `transcript`, `last_agent_message`), which are what assertions use.
- **Events.** There are three: `ScenarioRunStarted`, `MessageSnapshot` and `ScenarioRunFinished` (`PY/_events/events.py#L124-L201`). They go through an Rx event bus to `EventReporter`, which POSTs them to LangWatch `/api/scenario-events` when an API key is present (`PY/_events/event_reporter.py#L69-L133`). The schemas are generated from the LangWatch OpenAPI (`_generated/langwatch_api_client`).
- **Grouping.** A `batch_run_id` groups one process run (env `SCENARIO_BATCH_RUN_ID`, `PY/_utils/ids.py#L43-L58`). A `set_id` groups scenarios into a suite.
- **Local persistence is minimal:**
  - Python: the pytest plugin keeps results in memory and prints a summary.
  - JS: the vitest setup appends every event as JSONL to `.scenario/<test_id>.log` (`JS/integrations/vitest/setup.ts`).
  - Red-team runs write JSON reports (`PY/pytest_plugin.py#L22-L64`).
- **Comparing across runs.** This happens only in the LangWatch platform (closed server behavior, [doc]). The OSS library has no baseline or diff.

## Judge implementation

The judge works in two phases (`PY/judge_agent.py#L598-L711`).

**Decision phase** (`_build_decision_system_prompt` `#L761-L808`, tools `#L810-L850`, `_run_decision_phase` `#L852`):
- Forced tool choice (`tool_choice="required"`, `#L907`) between two argument-free tools, `continue_test` and `make_verdict`.
- The decision tools have no reasoning field, "so nothing in this call can pre-commit it to an outcome."
- Rules: end early if a "must not" criterion is already violated, and lean toward continuing while the conversation is short.

**Verdict phase** (`_run_judgment_phase` `#L1015`):
- A single `finish_test` tool call is pinned with `tool_choice`. Its schema has one object per criterion, `{requirement, reasoning, status: passed|failed|inconclusive}`, with reasoning required before status (`PY/_judge/criterion_verdicts.py#L72-L131`).
- `parse_criterion_verdicts` (`#L162-L205`) fails closed: a missing criterion becomes `failed`, and bare `true`/`false` values are accepted as legacy.

**Large transcripts.** Above a token threshold the judge gets a skeleton plus `expand_transcript` and `grep_transcript` discovery tools. It runs a bounded discovery loop before the forced verdict (`#L1443-L1626`, `_force_verdict` `#L1628`).

**Remote traces.** The judge can poll LangWatch for the agent's server-side trace and wait for it to settle, with a timeout and one extension (`#L1015-L1075`, `_extend_wait_and_rejudge` `#L1277`).

**Criterion keys.** They are derived as `re.sub(...)[:70]` of the criterion text (`criterion_verdicts.py#L60-L69`). [inferred] Two criteria sharing a 70-character prefix collide in the schema.

**Calibration.** None: no judge-vs-human validation and no repeated judging. Open issue [#887](https://github.com/langwatch/scenario/issues/887) lists verdict-pipeline gaps: an unreachable INCONCLUSIVE state, forced checkpoints and JS/Python divergence.

## User simulator implementation

`PY/user_simulator_agent.py#L351-L466`:
- **System prompt.** A fixed template: `<role>` ("very short inputs, few words, all lowercase... like when they google"), `<goal>`, `<scenario>{description}</scenario>`, `<rules>` and the `<persona>` block. A custom `system_prompt` replaces the template but still gets the persona appended (`#L392-L430`).
- **Role reversal.** It prepends a fake assistant greeting and **reverses roles** so the simulator LLM plays "assistant" (`reverse_roles`, `PY/_utils/utils.py#L442-L470`). It calls the model with `tools=[]` (`#L440-L452`).
- **Tool-call leak.** `reverse_roles` keeps assistant `tool_calls` messages un-reversed and passes `role:"tool"` results through. The simulator therefore sees the agent's internal tool calls and results, which a real user never sees. [inferred] This can leak internal data into simulated user turns.
- **No stop signal.** Unlike tau-bench's `###STOP###` or DeepEval's expected-outcome check, the simulator cannot end the conversation. Only the judge, the script or `max_turns` can.

## Online / production path

The library has no online path. The platform imports traces into scenarios (open spike [#779](https://github.com/langwatch/scenario/issues/779) covers creating a scenario from a real transcript). Scenario turns are tagged `langwatch.origin=simulation` (`scenario_executor.py#L541-L548`) so the platform can separate them from production traffic.

## Extension model

- Subclass `AgentAdapter` for any role (custom simulator, custom judge).
- Script steps are arbitrary callables.
- `evaluators=[ScenarioEvaluator]` run after the conversation (`_run_evaluators`, `#L896`). They come with mapping helpers such as `ToolCallsMapping` (`PY/evaluators.py#L196-L224`).
- Provider access goes through LiteLLM (Python) and the AI SDK (JS).

## Strengths

- **One script mixes deterministic and LLM steps.** Fixed turns, generated turns, code assertions and judge checkpoints at specific turns all share one ordered list (`script.py`). Assertion failures surface as native test-runner failures (`scenario_executor.py#L798`). This is the clearest model I have seen for "assert X at turn N, then let it run."
- **The judge stops the conversation.** A cheap, reasoning-free decision call separates "enough evidence?" from the verdict (`judge_agent.py#L761-L850`). Early termination on a violation saves turns.
- **Per-criterion structured verdicts.** Each criterion gets a restated requirement, reasoning before status, a three-way status and fail-closed parsing (`criterion_verdicts.py`). Negative criteria ("must not X") are explicitly normalized.
- **Tool calls come from messages and OTel spans.** The judge can check internal tool behavior without the adapter serializing it (`scenario_state.py#L214`, `judge_span_collector.py`).
- **Tiny adapter contract.** One async method and a role (`agent_adapter.py#L81-L82`).

## Weaknesses

- **No test-case data model.** Cases live in test code. There is no dataset, versioning or generation, and no way to diff or share a suite outside the code.
- **The hidden-information boundary is implicit.** One `description` string is shared by the simulator and the judge. There is no field for "user knows but the agent must elicit".
- **The judge is called every turn by default.** That multiplies cost, and early termination was a source of bugs (open issues [#980](https://github.com/langwatch/scenario/issues/980) "Judge ends the run at the first turn" and [#934](https://github.com/langwatch/scenario/issues/934)).
- **Weak determinism story.** Caching is Python-only, ignores kwargs ([#926](https://github.com/langwatch/scenario/issues/926)) and likely misses on per-turn `trace_id` [inferred]. There are no seeds and no repeat or pass^k support.
- **The simulator sees agent internals.** Tool calls and tool results pass through `reverse_roles` (`utils.py#L447-L453`).
- **The pytest integration monkeypatches `ScenarioExecutor.run`** at configure time (`pytest_plugin.py#L246-L331`) instead of using fixtures or hooks. The report is in-memory console output.
- **Cross-run comparison needs the hosted platform.** The OSS package has no baseline, diff or local run store.
- **SDK drift.** Message formats and result field names differ between Python and JS (`types.py#L318-L325` vs `JS/domain/core/execution.ts#L74-L93`), and caching exists on one side only.

## Worth adapting

- **The script as an ordered list of steps** (fixed message, generated turn, assertion, judge checkpoint, free-run-N). Make it **data**, not closures. It solves "mid-conversation assertions" cleanly.
- **The two-phase judge.** A binary continue/verdict gate with no reasoning, followed by a verdict call with per-criterion schema and forced tool choice. It works as a stop condition and as a scorer.
- **Per-criterion `requirement / reasoning / status(passed|failed|inconclusive)`.** Missing answers fail closed, and negative criteria are restated as positive requirements.
- **Two tool-call sources merged:** the agent's returned messages plus OTel spans from the same run, keyed by thread or run id.
- **The `langwatch.origin=simulation` tag** on simulated traffic, so simulated traces never pollute production analytics.

## Don't copy

- **Scenarios as function-call arguments, scripts as closures over a private executor** (`state._executor`). They cannot be stored, versioned, diffed or rerun, and no UI can author them.
- **Monkeypatching the runner's execute method** for reporting (`pytest_plugin.py#L331`). It is fragile under concurrency and plugin ordering.
- **A cache key built from full serialized inputs that include volatile ids.** Keys must be built from an explicit, stable, documented tuple (model, prompt version, normalized messages, seed). If the key silently includes volatile fields, "deterministic mode" becomes a false promise.
- **Leaving hidden information and persona as prose in one shared field.** Use explicit fields with explicit visibility (user-only, judge-only).
- **Giving the user simulator raw tool traffic.** Filter the history to what a user would observe.
- **Running the judge's decision call every turn with no budget control** beyond `min_turns`.

## Open questions

- Does the joblib cache actually hit on multi-turn replays, given the `trace_id` on each message? The only test (`python/tests/test_arun_cache.py`) exercises a trivial argument, not `AgentInput` with history.
- What the LangWatch platform does with `set_id` and `batch_run_id` for comparison across runs (server-side, [doc] only).
- Whether the JS judge is exactly the same two-phase algorithm. I read the prompt-parity comment and the file layout, not the full JS judge.
- The exact behavior when an adapter returns both text and a `ScenarioResult` in multi-agent settings.
