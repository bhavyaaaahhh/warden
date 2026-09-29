# tau-bench / tau2-bench (Sierra)

Repo (tau1): https://github.com/sierra-research/tau-bench @ `59a200c6d575d595120f1cb70fea53cef0632f6b` · License: MIT · Lang: Python · Last commit: 2026-03-18 · Activity: effectively frozen. No releases, 55 open issues (mostly ground-truth bug reports), about 1.45k stars.

Repo (tau2): https://github.com/sierra-research/tau2-bench @ `5bfa7e37b36656b37dc6d022156be6563c1007f3` · License: MIT · Lang: Python · Last commit: 2026-09-28 · Activity: active. Releases v0.1.2 (2025-07), v0.2.0 (2025-10, leaderboard), v1.0.0 "τ³" (2026-03, voice, knowledge, task quality), v1.0.1 (2026-07, banking grading fixes). 239 open issues, about 2.1k stars.

Papers: tau-bench, Yao et al. 2024, [arXiv:2406.12045](https://arxiv.org/abs/2406.12045). tau2-bench, Barres et al. 2025, [arXiv:2506.07982](https://arxiv.org/abs/2506.07982).

Permalink prefixes used below:
- `T1` = `https://github.com/sierra-research/tau-bench/blob/59a200c6d575d595120f1cb70fea53cef0632f6b`
- `T2` = `https://github.com/sierra-research/tau2-bench/blob/5bfa7e37b36656b37dc6d022156be6563c1007f3`

## Purpose

This is an offline benchmark for multi-turn, tool-using customer-service agents. The two roles in a conversation are:
- **The agent under test.** It has domain tools and a policy document.
- **An LLM-simulated user.** It gets hidden instructions describing its identity, intent and preferences.

The episode ends when the user emits a stop token, the agent transfers to a human, or a step or error cap is hit. Scoring is rule-based. It compares the final environment state against a gold state, and it checks that required facts appear in the agent's messages.

tau2 adds three things:
- **A dual-control domain (telecom).** The *user* also has tools that change a shared world state.
- **A reward basis you can compose per task.** Components are DB hash, env assertions, communicate, NL assertions and action matching.
- **More infrastructure.** A batch runner, retries, resume, LLM reviewers of the simulator, voice and full-duplex modes, and a leaderboard.

The headline metric in both is **pass^k**: the probability that *all* k i.i.d. trials of a task succeed. It is used for reliability, not capability. The benchmark is aimed at model and agent builders who compare agents under fixed tasks and domains.

## Core abstractions

### Test case: `Task`

**tau1** `T1/tau_bench/types.py#L15-L19`. The model is flat:
```python
class Task(BaseModel):
    user_id: str
    actions: List[Action]      # gold action sequence (name + kwargs)
    instruction: str           # hidden user instruction (persona + goal + preferences)
    outputs: List[str]         # strings the agent must say (answers to info questions)
```
- The persona is baked into free text. For example: *"You are Yusuf Rossi in 19122 … You are detail-oriented and want to make sure everything is addressed in one go."* (`T1/tau_bench/envs/retail/tasks_test.py#L5-L8`).
- Task files pass an `annotator=` kwarg that the model does not declare. Pydantic silently drops it.
- Tasks are Python literals, split into `tasks_test/train/dev.py`.
- The paper's authoring rule is the load-bearing part: the instruction "sets up user identity, intent, preferences in a way that guarantees only one possible outcome under the domain policy" (paper §3, Fig. 2d). The whole scoring scheme depends on that uniqueness.

**tau2** `T2/src/tau2/data_model/tasks.py#L560-L640`. The model is structured and stored as JSON:
- `user_scenario: UserScenario` (`#L52`). This holds `persona` plus `instructions`. `instructions` is either a free string or a `StructuredUserInstructions` (`#L15-L32`) with the fields `domain`, `reason_for_call`, `known_info`, `unknown_info` and `task_instructions`. The known/unknown split names what the user may disclose and what it must claim not to know.
- `initial_state: InitialState` (`#L506`). This holds `initialization_data` (DB patches for agent and user), `initialization_actions: list[EnvFunctionCall]` and `message_history` (a pre-seeded conversation prefix).
- `evaluation_criteria: EvaluationCriteria` (`#L366-L489`). This holds `actions`, `env_assertions`, `communicate_info`, `nl_assertions` and `reward_basis`.
- Other fields: `ticket` (the task as a ticket, for solo/no-user mode), `user_tools` (a per-task allowlist of user tools), `issues: list[TaskIssue]` (`#L278`, per-task bug tracking with status open/resolved/wont_fix), and `description` (purpose and relevant policies, visible to the evaluator only).
- Trade-off: one record carries the simulator prompt, the env setup and the grading spec. Tasks are therefore self-contained and diffable. The price is that the three concerns version together.

### Expected action: `Action`

**tau1** `Action(name, kwargs)` (`T1/tau_bench/types.py#L10-L12`) is used only to *replay* onto a fresh DB.

**tau2** `Action` (`T2/src/tau2/data_model/tasks.py#L116-L196`) adds `action_id`, `requestor` ("assistant" or "user") and `compare_args`.
- `compare_with_tool_call` (`#L175-L196`) matches name plus a chosen subset of arguments.
- When `compare_args is None`, it compares only the keys **the agent's call** supplied (`compare_args = tool_call.arguments.keys()`). A call that omits an argument therefore still matches. This is open issue [#568](https://github.com/sierra-research/tau2-bench/issues/568).

### Expected state: `EnvAssertion`

`EnvAssertion(env_type, func_name, arguments, assert_value, message)` (`T2/src/tau2/data_model/tasks.py#L198-L235`).
- It names a Python function on the agent toolkit or the user toolkit that must return a bool.
- Example (telecom): `assert_internet_speed(expected_speed=200, expected_desc="excellent")` on the user's device.
- Trade-off: it checks the property that matters instead of a whole-DB hash. The catch is that every assertion is code in the domain toolkit.

### Scorer: evaluators

These live in `T2/src/tau2/evaluator/`. Each is a class with a classmethod `calculate_reward(task, full_trajectory, …) -> RewardInfo`, subclassing `EvaluatorBase[Message|Tick]`.
- `EnvironmentEvaluator` (`evaluator_env.py#L17-L165`)
- `ActionEvaluator` (`evaluator_action.py#L62-L109`)
- `CommunicateEvaluator` (`evaluator_communicate.py#L7-L88`)
- `NLAssertionsEvaluator` (`evaluator_nl_assertions.py#L16-L135`)
- An orchestrating function, `evaluate_simulation` (`evaluator.py#L88`)

Every evaluator is binary per component. `RewardInfo` (`T2/src/tau2/data_model/simulation.py#L1100`) keeps the per-check detail (`db_check`, `action_checks`, `communicate_checks`, `nl_assertions`, `env_assertions`) plus `reward_breakdown` keyed by `RewardType`.

### Unit of execution: `Env.step` (tau1) and `Orchestrator` (tau2)

- **tau1** is gym-like. `Env.reset/step(Action) -> EnvResponse(observation, reward, done, info)` (`T1/tau_bench/envs/base.py#L78-L119`). An `Action` named `respond` goes to the user simulator. Any other name is dispatched to `tools_map[name].invoke(data=self.data, **kwargs)`, and exceptions become `"Error: …"` observations.
- **tau2** is a message router with three roles, AGENT, USER and ENV. See `Orchestrator.step` (`T2/src/tau2/orchestrator/orchestrator.py#L819-L896`). Both AGENT and USER can send tool calls to ENV, and the result is routed back to whichever role asked.

### Environment: `Environment`

`Environment(domain_name, policy, tools: ToolKitBase, user_tools: ToolKitBase, solo_mode)` (`T2/src/tau2/environment/environment.py#L34-L63`).
- **Tools.** Tools are methods decorated with `@is_tool(ToolType.READ|WRITE|THINK|GENERIC, mutates_state=…)` (`T2/src/tau2/environment/toolkit.py#L64`) over a pydantic `DB` (`T2/src/tau2/environment/db.py#L7`).
- **Routing.** `make_tool_call(tool_name, requestor, **kwargs)` (`environment.py#L158-L185`) sends each call to the agent or user toolkit by requestor.
- **Policy.** The policy is a Markdown string. Telecom concatenates `<main_policy>` and `<tech_support_policy>` (manual or workflow variant) (`T2/src/tau2/domains/telecom/environment.py#L96-L140`). tau1 also defines `rules` (`T1/tau_bench/envs/retail/rules.py`), but no agent code reads `env.rules`. Only `wiki` becomes the system prompt (`T1/tau_bench/agents/tool_calling_agent.py#L35-L38`).

## Execution model

### tau1

`run()` (`T1/tau_bench/run.py#L20-L121`) works like this:
- It loops over `num_trials`. Inside each trial, it runs `ThreadPoolExecutor(max_workers=max_concurrency).map(_run, idxs)`.
- Each task gets a fresh `get_env(...)` (`#L66-L74`) so that DB state is isolated.
- An exception inside `agent.solve` becomes `reward=0.0` with the traceback in `info` (`#L89-L96`). **Crashes, including API errors, count as agent failures.**
- There are no retries.
- Results are appended to one JSON checkpoint by re-reading and re-writing the whole file under a lock (`#L103-L109`). That is O(n²) and has no resume.
- Agents loop to `max_num_steps=30` (`T1/tau_bench/agents/tool_calling_agent.py#L28-L39`). They keep only the first tool call per message (`#L54`), which enforces the "one tool call at a time" policy by truncation.

### tau2

- **Batch runner.** `T2/src/tau2/runner/batch.py`. Per-trial seeds come from `random.seed(config.seed)` followed by `randint` per trial (`#L742-L743`). Work is spread over a `ThreadPoolExecutor` and checkpoints support auto-resume (`runner/checkpoint.py`).
- **Retry.** `run_with_retry` (`T2/src/tau2/runner/progress.py#L19-L60`) retries *any exception* up to `max_retries`, default 3 (`config.py#L40`). After that it emits a placeholder `SimulationRun` with `TerminationReason.INFRASTRUCTURE_ERROR` rather than raising.
- **Termination reasons.** These are explicit and typed (`T2/src/tau2/data_model/simulation.py#L1281-L1291`): `USER_STOP`, `AGENT_STOP`, `MAX_STEPS`, `TIMEOUT`, `TOO_MANY_ERRORS`, `AGENT_ERROR`, `USER_ERROR`, `INFRASTRUCTURE_ERROR`, `CONTEXT_WINDOW_EXCEEDED` and `UNEXPECTED_ERROR`.
- **Caps.** `max_steps` defaults to 200 in the CLI config (`config.py#L4`) and 100 in the Orchestrator constructor. `max_errors` defaults to 10, where each tool result with `error=True` increments `num_errors` (`orchestrator.py#L313-L329`, `#L734-L750`).
- **Premature termination scores 0.** `evaluate_simulation` returns reward 0 for anything other than AGENT_STOP or USER_STOP (`evaluator.py#L119-L129`). Infra errors are then *excluded* from metrics (see pass^k below).
- **Hallucination retry.** This is full-duplex only (`batch.py#L585-L665`). An LLM fact-checker (`evaluator/hallucination_reviewer.py`) looks for user-simulator fabrications. If it finds any, the simulation is re-run with feedback and a new seed (`seed + n*1000`).
- **LLM caching.** It is configurable (`DEFAULT_LLM_CACHE_TYPE = "redis"`, `config.py#L48`). All roles default to `gpt-4.1-2025-04-14` at temperature 0 (`config.py#L17-L30`).
- **Known concurrency bug.** Telecom user-tool state is a class attribute, so concurrent sims share it (issue [#154](https://github.com/sierra-research/tau2-bench/issues/154), open).

## The user simulator

### tau1

The code is in `T1/tau_bench/envs/user.py`. The base is `BaseUserSimulationEnv` with `reset(instruction)`, `step(content)` and `get_total_cost()`. `load_user` picks the strategy from `UserStrategy` (`#L312-L317`), which has five values: human, llm, react, verify and reflection.

**Prompt** (`LLMUserSimulationEnv.build_system_prompt`, `#L55-L68`). The system prompt is "You are a user interacting with an agent. Instruction: …" followed by these rules:
- one line at a time
- don't give away all instructions at once
- don't hallucinate (say you don't remember)
- emit `'###STOP###'` as a standalone message when the goal is satisfied
- don't repeat the instruction verbatim
- stick to the persona

The conversation is seeded with a fake agent turn, `"Hi! How can I help you today?"` (`#L76`). The simulator's own messages sit in the assistant role.

**Strategies:**
- **`llm`** (`#L37-L85`). One completion per turn.
- **`react`** (`#L88-L153`). The prompt asks for a `Thought:` line and then a `User Response:` line, and `parse_response` extracts the reply (`#L136-L146`). **Bug:** the `elif "Thought:" in response` branch is checked *before* `"User Response:"`. It returns everything after `Thought:`, so the thought text and the literal "User Response:" label leak to the agent. Any other format raises `ValueError`, which kills the episode and scores it 0.
- **`verify`** (`#L156-L194`). The simulator samples up to `max_attempts=3` times. After each sample, a second LLM call, `verify()` (`#L206-L232`), asks whether the response is "satisfactory" and parses the reply with `"true" in content.lower()`. If no sample passes, the last candidate is returned.
- **`reflection`** (`#L270-L309`). On a failed verify, `reflect()` (`#L235-L267`) produces "Reflection: … Response: …", which is parsed by `split("Response:")`. It then regenerates, up to `max_attempts=2`. Every candidate goes through `super().generate_next_message`, which appends to `self.messages`. **Rejected candidates therefore stay in the simulator's own history.**
- **`human`**. `input()` for manual play.

**Cost accounting:** `self.total_cost = res._hidden_params["response_cost"]` overwrites the value instead of adding to it (`#L52`, `#L123`). The `user_cost` reported by tau1 is therefore only the last call's cost.

**End signal:** `done = "###STOP###" in observation` (`T1/tau_bench/envs/base.py#L99`), or a terminate tool (`transfer_to_human_agents`, `T1/tau_bench/envs/retail/env.py#L41`), or the agent's 30-step cap. The step cap does not set `done`, so the reward stays 0.

### tau2

There is **one strategy**: `UserSimulator` (`T2/src/tau2/user/user_simulator.py#L99-L267`), plus `DummyUser` for solo mode and voice/streaming variants (`registry.py#L283-L289`). The react, verify and reflection strategies were dropped.

The prompt comes from three layers (per the class docstring):
1. **Global guidelines.** `data/tau2/user_simulator/simulation_guidelines{,_tools,_voice,_voice_tools}.md`, loaded by `get_global_user_sim_guidelines` (`#L49-L65`).
2. **Task persona and instructions.** These go into `SYSTEM_PROMPT = "{guidelines}\n<scenario>\n{instructions}\n</scenario>"` (`#L86-L92`).
3. **Runtime `PersonaConfig`.** Verbosity and interrupt tendency, spliced in at `<PERSONA_GUIDELINES>` (`T2/src/tau2/data_model/persona.py#L40-L70`, `user_simulator.py#L139-L162`).

The tool-guidelines variant (`T2/data/tau2/user_simulator/simulation_guidelines_tools.md`) tells the user to:
- either message the agent or make a tool call per turn, never both
- never invent tool results
- fix and retry failed tool calls
- ask the agent for one action at a time
- disclose information progressively

**Mechanics:**
- **Role flipping.** `UserState.flip_roles()` (`user_simulator_base.py#L62`) swaps roles so that the simulator LLM sees the agent as "user".
- **Tool calls.** The user's tool calls are re-stamped `requestor="user"` (`user_simulator.py#L255-L266`) and routed to ENV. User tool calls are not shown to the agent.
- **End signals** (`user_simulator_base.py#L51-L53`, `UserSimulator.is_stop` `#L183-L196`). There are three tokens: `###STOP###` (goal satisfied), `###TRANSFER###` (agent transferred) and `###OUT-OF-SCOPE###` (the scenario lacks the information to continue). All three map to `TerminationReason.USER_STOP` (`orchestrator.py#L840-L844`).
  - OUT-OF-SCOPE therefore bypasses the "premature termination = 0" rule and is graded normally. This is issue [#517](https://github.com/sierra-research/tau2-bench/issues/517), open.

### World-state coupling (the tau2 paper's main simulator-fidelity lever)

In telecom, the user's device is a separate `TelecomUserDB`. `TelecomEnvironment.sync_tools` (`T2/src/tau2/domains/telecom/environment.py#L39-L93`) runs after every step (`orchestrator.py#L896`). It propagates backend state (line status, roaming, data quota, bills) into the user's "surroundings".

User tools are things like `check_status_bar`, `toggle_airplane_mode`, `run_speed_test`, `reseat_sim_card` and `make_payment` (`T2/src/tau2/domains/telecom/user_tools.py`). The user's answers are therefore grounded in tool observations rather than in the prompt text.

## The environment and dual control

- **DB.** tau1 uses JSON dicts mutated in place by `Tool.invoke(data, **kwargs)` static methods (`T1/tau_bench/envs/tool.py`, for example `T1/tau_bench/envs/retail/tools/cancel_pending_order.py`). tau2 uses pydantic `DB` models (`BaseModelNoExtra`), with TOML or JSON data files per domain.
- **Domains.** tau1 has retail and airline. tau2 has airline, retail, telecom, banking_knowledge and mock (`T2/data/tau2/domains/`).
- **Dual control.** This is telecom only in the bundled data. The environment has both `tools` (agent: CRM/billing) and `user_tools` (device). Grading hashes **both** DBs: `agent_db_match and user_db_match` (`evaluator_env.py#L118-L129`).
  - Gold `Action`s can carry `requestor="user"`. The telecom task above expects the user to call `toggle_roaming`.
  - Initialization can break the user's device via `initialization_actions` with `env_type: "user"`, for example `set_user_location(abroad=true)` and `turn_roaming_off` (`T2/data/tau2/domains/telecom/tasks.json`, first task).
- **Solo / no-user mode.** `solo_mode=True` gives the agent both toolkits and asserts no tool-name overlap (`environment.py#L443-L463`). It uses the `ticket` field and `DummyUser`. In solo mode, an agent text message that isn't a stop is an `AGENT_ERROR` (`orchestrator.py#L869-L877`).
- **Oracle Plan mode.** Agent variants `llm_agent_gt` and `llm_agent_solo_gt` (`registry.py`) give the agent the gold plan. The paper uses the no-user, oracle and default modes to separate reasoning errors from communication and coordination errors.
- **Compositional task generation.** Telecom tasks are composed from atomic scenarios (init, solution and assertion functions), for example `src/tau2/domains/telecom/tasks/mobile_data_issues.py` and `service_issues.py`. Task ids encode the composition, for example `[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]`.

## Scoring

### tau1: `Env.calculate_reward`

`T1/tau_bench/envs/base.py#L124-L164`:
1. Hash the current (agent-produced) data using `consistent_hash(to_hashable(data))`. This is SHA-256 over `str()` of a sorted-key tuple tree (`#L27-L41`). Dict keys are sorted, list order is kept, and sets are sorted.
2. Reload a fresh DB and replay `task.actions` through **`self.step`**, skipping terminate tools. Then hash again. `r_actions = (hash == gt_hash)`.
3. If `task.outputs` is non-empty, require every output string to be a case-insensitive substring of some `respond` message after commas are stripped. The reward is `r_actions × r_outputs`, which is either 0 or 1.

**Side effect:** replaying through `self.step` appends the gold actions to `self.actions`. This is issue [#92](https://github.com/sierra-research/tau-bench/issues/92), open. It is harmless today because no gold task contains a `respond` action (verified: zero occurrences in all `tasks*.py`), but it corrupts `self.actions` for any later reader.

Note that tau1's "action" reward is **not** action matching. It is final-DB-state comparison, where the gold state is derived by replay. Reads don't matter. Only the write set and its arguments do.

### tau2: `evaluate_simulation`

`T2/src/tau2/evaluator/evaluator.py#L88-L351`:
1. If termination was not AGENT_STOP or USER_STOP, the reward is 0 (`#L119-L129`).
2. **DB and ENV_ASSERTION** (`EnvironmentEvaluator.calculate_reward`, `evaluator_env.py#L23-L165`):
   - Build a *predicted* env by replaying the recorded trajectory's tool calls with `Environment.set_state` (`environment.py#L293-L410`). Non-mutating tools are skipped during replay, and hallucinated tool names are no-ops. With `strict=True`, a replayed output that differs from the recorded one raises.
   - Build a *gold* env by replaying `evaluation_criteria.actions` (errors are logged, not raised) (`#L97-L115`).
   - Compare `get_db_hash()`, which is `sha256(json.dumps(db.model_dump(), sort_keys=True, default=str))` (`toolkit.py#L242-L244`, `utils/utils.py#L39-L45`), for both the agent DB and the user DB.
   - Then run each `EnvAssertion` against the predicted env.
3. **COMMUNICATE** (`evaluator_communicate.py#L50-L88`). The same substring check as tau1, carrying a `# TODO: This could be improved!` comment (`#L69-L71`).
4. **ACTION** (`evaluator_action.py#L16-L59`). Every gold action must match *some* agent or user tool call. Order and extra calls are ignored.
5. **NL_ASSERTION** (`evaluator_nl_assertions.py#L60-L135`). One LLM call judges all assertions and returns JSON `{results:[{expectedOutcome, reasoning, metExpectation}]}`, parsed with bare `json.loads` (`#L127`). A malformed reply raises.
   - The reward is `all(...)` over the returned results, so **an empty `results` list passes**. This is issue [#554](https://github.com/sierra-research/tau2-bench/issues/554), open.
   - The docs label this component "experimental / WIP".
6. **Combine** (`#L222-L275`). The final reward is the **product** of the components in `task.evaluation_criteria.reward_basis`. The default basis is `[DB, COMMUNICATE]`, which gives tau1 parity (`tasks.py#L237-L268`, `#L456-L465`). Components outside the basis still run and are stored as diagnostics.
   - `EvaluationType.ALL_IGNORE_BASIS` multiplies everything together, which gives a reference-path similarity score.
   - `docs/evaluation.md` states that airline, retail and telecom never gate on ACTION. Only about 9 of about 100 banking_knowledge tasks do.

**Design intent** (`T2/docs/evaluation.md`; issues [#224](https://github.com/sierra-research/tau2-bench/issues/224), [#129](https://github.com/sierra-research/tau2-bench/issues/129)): `actions` is *one* reference trajectory used to derive the target state, not a checklist. An agent that correctly refuses gets full reward without calling any tool.

## pass^k

**Definition** (tau1 paper §3): pass^k is "the chance that all k i.i.d. task trials are successful, averaged across tasks".

It is the unbiased estimator from n trials with c successes, averaged over tasks:

`pass^k = E_task[ C(c, k) / C(n, k) ]`

Contrast this with pass@k (at least one of k succeeds), which rewards search. pass^k penalizes inconsistency and falls as k grows. The paper reports gpt-4o at under 50% pass^1 and under 25% pass^8 on retail.

**Implementation:**
- **tau1** is `display_metrics` (`T1/tau_bench/run.py#L180-L203`). Success means `|reward − 1| ≤ 1e-6`. It counts `c` per task_id and averages `comb(c,k)/comb(num_trials,k)` over tasks. It uses the *global* `num_trials`, so a task with missing trials is silently penalized.
- **tau2** is `pass_hat_k` (`T2/src/tau2/metrics/agent_metrics.py#L113-L126`), applied per task in `get_tasks_pass_hat_k` (`#L169-L191`) with `len(df)` (that task's *surviving* trial count) as n.
  - `get_metrics_df` (`#L129-L166`) first **drops INFRASTRUCTURE_ERROR rows**. It then caps max_k at the minimum per-task trial count, with a warning.
  - Issue [#497](https://github.com/sierra-research/tau2-bench/issues/497) points out that the per-task n therefore varies: 2 of 2 survivors gives pass^2 = 1.0, while 2 of 5 gives 0.1.
  - It also points out that RELEASE_NOTES says the *leaderboard* instead counts infra errors as failures. That makes two conventions.
  - Issue [#493](https://github.com/sierra-research/tau2-bench/issues/493): pass^1 can read 100% from a single surviving sim.

**Why it matters for reliability.** A deployed agent sees the same intent many times. The per-task success rate p compounds: pass^k ≈ p^k per task. Two agents with equal average reward can differ a lot at k=8, depending on whether their failures sit on a few hard tasks or are spread thinly as flakiness. Averaging over tasks rather than over trials keeps each task weighted equally.

**Noise floor.** Issue [#540](https://github.com/sierra-research/tau2-bench/issues/540) recomputes the spread of published baselines from 4-trial trajectory files. Single-run suite scores move about 5–11 points on airline and about 7–9 points on retail between trials. One retail submission ranged 64–77%. That spread includes user-simulator sampling. The leaderboard reports no interval on pass^1.

## Simulator noise, errors and validity

**Handling in code:**
- The default is temperature 0 for every role (tau2 `config.py#L19-L20`). tau1 also defaults the agent to 0.0. The user model runs at the provider default because no temperature is passed in `user.py`.
- The verify and reflection strategies (tau1) are an LLM self-check on each user turn. They were dropped in tau2.
- tau2 constrains the user through tools and observable state (the telecom design) instead of through prompting.
- tau2 has LLM reviewers:
  - `ConversationReviewer` (`T2/src/tau2/evaluator/review_llm_judge.py#L410`). It labels agent errors as minor or critical, and user-simulator errors as minor, `critical_helped` or `critical_hindered` (`#L60-L120`, parsing at `#L349-L402`). The `critical_helped` label marks a simulator error that made the agent *pass* by mistake. This is post-hoc diagnostic metadata and does not change the reward.
  - The hallucination fact-checker gates reruns, but only in full-duplex mode.
- tau1 has `auto_error_identification.py` (`#L115-L135`). It does LLM fault assignment for failures (user, agent or environment) and then fault typing (wrong tool, wrong argument, goal partially completed, other).
- `TaskIssue` records on tasks track known ground-truth problems in the data itself.

**Measured simulator error rates** (tau2 paper, Table 2). These come from gpt-4.1 as the simulator, with two annotators per conversation:

| Domain | Conversations | Critical errors | Benign errors | Total errors |
|---|---|---|---|---|
| airline | 100 | 13% | 34% | 47% |
| retail | 50 | 12% | 28% | 40% |
| telecom | 50 | 6% | 10% | 16% |

The paper credits telecom's lower rate to environment-constrained behavior. The tau1 paper's limitations section lists three simulator problems: instruction typos or ambiguity, users lacking domain knowledge, and limited LM reasoning or memory (for example, authorizing an agent-recommended item the instruction didn't cover).

**Known validity issues** (GitHub, open unless noted):
- **Ground-truth errors or ambiguous tasks.** Many are reported against tau1 airline and retail:
  - tau1 issues [#65](https://github.com/sierra-research/tau-bench/issues/65), [#41](https://github.com/sierra-research/tau-bench/issues/41), [#37](https://github.com/sierra-research/tau-bench/issues/37), [#35](https://github.com/sierra-research/tau-bench/issues/35), [#70](https://github.com/sierra-research/tau-bench/issues/70) and [#85](https://github.com/sierra-research/tau-bench/issues/85) (closed: tasks 26 and 33 apply the same policy oppositely).
  - tau2 issues [#84](https://github.com/sierra-research/tau2-bench/issues/84), [#89](https://github.com/sierra-research/tau2-bench/issues/89), [#156](https://github.com/sierra-research/tau2-bench/issues/156), [#321](https://github.com/sierra-research/tau2-bench/issues/321) and [#496](https://github.com/sierra-research/tau2-bench/issues/496).
  - The tau2 repo explicitly claims "verified" versions of the tau1 domains.
- **False positives from outcome-only grading.**
  - [#384](https://github.com/sierra-research/tau2-bench/issues/384): no-op and missing checks.
  - [#320](https://github.com/sierra-research/tau2-bench/issues/320): a required human handoff is missing but still rewarded.
  - [#298](https://github.com/sierra-research/tau2-bench/issues/298): trajectories that violate confirm-before-mutate still pass. Policy procedure is not graded, only end state.
  - [#15](https://github.com/sierra-research/tau2-bench/issues/15): spurious airline rewards.
- **Brittleness of hash equality.**
  - [#514](https://github.com/sierra-research/tau2-bench/issues/514): `sort_keys` does not sort lists, so an equivalent `payment_history` in a different order fails. tau1 has the same property, since `to_hashable` keeps list order.
  - [#485](https://github.com/sierra-research/tau2-bench/issues/485): a free-text `reason` argument enters the hashed state, which makes tasks unsolvable.
  - [#387](https://github.com/sierra-research/tau2-bench/issues/387) (closed): time-dependent tool output broke replay.
- **Simulator behavior.**
  - [#182](https://github.com/sierra-research/tau2-bench/issues/182): the user sends STOP right after confirming, before the agent writes. A reporter says a prompt tweak moved gpt-5.2 pass^1 from 63.1% to 78.9%, which shows **the scores are very sensitive to the simulator prompt**.
  - [#172](https://github.com/sierra-research/tau2-bench/issues/172): the simulator invents a passenger.
  - [#57](https://github.com/sierra-research/tau2-bench/issues/57): telecom location confusion.
- **Reproducibility.**
  - tau1 [#13](https://github.com/sierra-research/tau-bench/issues/13): which user model and strategy produced the reported numbers?
  - tau2 [#540](https://github.com/sierra-research/tau2-bench/issues/540): leaderboard entries were scored by different evaluator commits, two have `git_commit: unknown`, and trajectory files don't name the NL-assertion judge model.

## Judge implementation

Rule-based grading dominates. The LLM judges are:
- **NL assertions.** Prompt inline at `T2/src/tau2/evaluator/evaluator_nl_assertions.py#L83-L114`. Parsed with bare `json.loads`, with no retry or repair. Uses gpt-4.1 at temperature 0 (`config.py#L24-L26`). No calibration against humans in code.
- **Conversation review and hallucination review.** In `review_llm_judge.py` and `hallucination_reviewer.py`, defaulting to `claude-opus-4-5` (`config.py#L32`). They parse through `extract_json_from_llm_response`. These are diagnostic, except for the full-duplex rerun gate.
- **tau1 verify/reflect.** Parse with `"true" in text` or `split("Response:")`, which is fragile.

## Data model and storage

- **tau1:** one JSON file per run holding a list of `EnvRunResult(task_id, reward, info, traj, trial)` (`T1/tau_bench/types.py#L64-L69`). The filename encodes the config. There is no dataset versioning beyond git.
- **tau2:** `Results`/`SimulationRun` (`T2/src/tau2/data_model/simulation.py#L1294`, `#L1423`). Each run records messages or ticks, termination_reason, seed, costs, `RewardInfo` with a per-check breakdown, review results and a git commit.
  - Tasks are JSON with `split_tasks.json` splits (for example `small` and `full` in telecom).
  - `evaluate-trajs` can re-grade stored trajectories under new evaluator code using lenient replay (`strict_replay=False`). Issue [#502](https://github.com/sierra-research/tau2-bench/issues/502) shows lenient replay can change scores sharply (0.72 to 0.10) with no error.

## Online / production path

None. The project is purely offline. There is a gym wrapper (`src/tau2/gym/gym_agent.py`) for RL training, but nothing ingests production traces.

## Extension model

- **tau2 `Registry`** (`T2/src/tau2/registry.py#L84`, `#L283-L300`) registers user simulators, agent factories, domains (env constructor plus task loader) and task sets by name.
- **New domains** supply:
  - a `DB` model
  - a `ToolKitBase` subclass with `@is_tool` methods (and optionally user tools)
  - a policy .md file
  - tasks JSON
  - optionally `sync_tools` for cross-DB coupling
- **New scorers** are new `RewardType`s plus an evaluator class. Combination is hard-coded in `evaluate_simulation`, so a new component means editing that function. There is no plugin registry for evaluators.
- **tau1** uses subclasses of `Env`, `Tool` and `Agent`, and `get_env` dispatch.

## Strengths

- **Outcome-based grading via gold-trajectory replay.** An author writes one valid action sequence. The target state is *derived*, and any path reaching it passes (`evaluator_env.py#L97-L131`, `docs/evaluation.md`). This is cheap to author, deterministic, and robust to the many ways a conversation can go.
- **pass^k as a first-class reliability metric**, with the unbiased combinatorial estimator (`agent_metrics.py#L113-L126`). It shows flakiness that mean reward hides.
- **Reward basis per task, with diagnostics kept apart from gates** (`tasks.py#L237-L268`, `evaluator.py#L222-L275`). Every check is always computed and stored. Only the declared ones gate the score, so you can inspect partial action matching without changing the headline metric.
- **Typed termination reasons plus an infra-error bucket** (`simulation.py#L1281`, `progress.py#L19`). These separate "agent failed" from "API fell over", which tau1 conflated (`T1/tau_bench/run.py#L89-L96`).
- **Environment-grounded user simulation** (telecom `sync_tools`, user tools). This is the single measured lever that cut critical simulator errors roughly in half (paper Table 2).
- **Structured user instructions** with an explicit `unknown_info` field (`tasks.py#L15-L32`). These give the simulator, and any hallucination checker, a precise contract for what the user may not reveal.
- **Mode ablations** (solo, oracle plan, default). They attribute failures to reasoning versus user coordination without new tasks.
- **Per-task `issues` field** (`tasks.py#L278`). It is an in-data audit trail for ground-truth disputes.

## Weaknesses

- **The uniqueness assumption carries the whole scheme.** Hash equality needs every valid solution to produce a byte-identical DB. Violations show up as list-order sensitivity (#514), free-text arguments in state (#485), time-dependent outputs (#387) and many ground-truth disputes (tau1 #35/#37/#41/#65/#85).
- **Outcome-only grading misses procedure.** Confirm-before-write, authentication order and required handoffs are not checked unless an author adds an assertion (#298, #320, #384).
- **COMMUNICATE is a bare substring match** (`evaluator_communicate.py#L69`, with its own TODO). It is gamed by dumping numbers and fails on formatting differences ("$54.04" vs "54.04" works only because of the comma strip).
- **Simulator variance is inside the metric.** Temperature-0 simulator runs still produce a noise floor of 5–11 points (#540), and prompt wording alone moves scores by about 15 points (#182). Results depend on the simulator model, the prompt version and the evaluator commit, and these are not always recorded (#540, tau1 #13).
- **pass^k bookkeeping is inconsistent.** Infra errors are excluded, which gives per-task n that differ. The leaderboard's convention is different (#497, #493). tau1 uses a global n (`run.py#L198`).
- **Weak parsing in LLM components.** tau1 ReAct leaks thoughts (`user.py#L139-L141`). tau1 verify uses `"true" in` and keeps rejected candidates in history (`user.py#L232`, `#L278-L292`). The tau2 NL judge has an `all([])` pass (#554) and no JSON repair (`evaluator_nl_assertions.py#L127`).
- **Loose ACTION matching.** With `compare_args=None`, the comparison uses the *agent's* keys, so omitted arguments pass (#568).
- **OUT-OF-SCOPE and TRANSFER both map to USER_STOP** (`orchestrator.py#L840-L844`, #517). This loses information about why the conversation ended.
- **Hard-coded evaluator composition** in a long if/elif (`evaluator.py#L171-L351`). New scorers need core edits.
- **tau1 plumbing bugs:** user cost is overwritten instead of accumulated (`user.py#L52`), gold replay mutates `self.actions` (#92), and `Env.__init__`/`reset` use `random.randint(0, len(tasks))`, which is inclusive and can go out of range (`base.py#L69`, `#L80`).

## Worth adapting

- **A test case = (hidden user scenario, initial state, evaluation criteria), with the scenario never shown to the agent.** This is a general shape for any multi-turn agent eval. Add a structured `known_info`/`unknown_info` split so that simulator honesty can be checked.
- **Deriving the expected state by replaying a reference trajectory**, instead of hand-specifying it, and then comparing *outcomes*. This generalizes to any agent with a sandboxable, resettable backend (DB, filesystem, mock APIs). Also store the reference path for diagnostic action-overlap scores.
- **A declared gating basis versus always-on diagnostics.** A product of binary components per test case, with the declared set stored on the case and every component's detail kept in the result.
- **pass^k from n ≥ k trials using C(c,k)/C(n,k), averaged per task.** Report it next to mean and pass@k.
  - Fix n per task and count infra failures explicitly: either retry until n valid trials exist, or report the exclusion.
  - Report an interval on pass^1. The #540 analysis shows why that is necessary.
- **Typed termination reasons, with infra errors kept apart from agent failures.** Retry the infra ones and score the rest.
- **Simulator reliability as a measured quantity.** Review simulator turns with humans or an LLM, and tag `critical_helped` versus `critical_hindered`, so that false passes and false fails caused by the simulator can be counted and filtered.
- **Grounding the simulator in tools and state instead of prompt text** wherever the user has observable state.
- **Mode ablations** (no-user/solo with a ticket, oracle plan). These are cheap ways to break a failure down into reasoning, planning and communication.
- **Recording provenance on every run:** simulator model, prompt version, judge model and evaluator commit. The tau2 issues show what goes wrong when this is missing.

## Don't copy

- **Whole-DB hash equality as the only state check.** Order-sensitive and cosmetic differences flip the verdict, and nothing tells you *which* field differed. Prefer a structural diff with normalization (order-insensitive collections, ignored fields) plus targeted assertions (the `EnvAssertion` idea), and report the diff.
- **Substring match for "communicated info".** Use normalized value extraction or an LLM check with a calibrated rubric, and keep the rule check as a fast pre-filter.
- **Magic stop tokens inside free text** (`###STOP###` substring). Use a structured "end conversation" tool or field with a reason enum on the simulator, and don't collapse distinct reasons.
- **LLM self-verification of simulator turns with `"true" in text` parsing** (tau1 verify/reflection). It doubles or triples cost, has no measured benefit in code, and pollutes history. tau2 dropped it in favor of environment grounding and post-hoc review.
- **Excluding failed trials from pass^k while keeping per-task n variable.** It biases toward tasks where infrastructure was flaky.
- **A hard-coded if/elif evaluator composition.** Make scorers pluggable, with a declared combination rule.
- **Tasks as Python literals** (tau1). They can't be diffed or validated like data. tau2's JSON plus pydantic is the better model.

## What generalizes vs what is benchmark-specific

**Generalizes to a general eval platform:**
- the hidden-scenario simulated user with a structured persona and instructions and an explicit end signal
- multi-trial runs and pass^k
- outcome-based grading against a derived gold state
- a composable per-case reward basis with diagnostic components
- typed termination and infra-error separation
- simulator error review and attribution (critical_helped/hindered)
- recording seeds, models and evaluator versions
- mode ablations

**Benchmark-specific:**
- the customer-service domains, their policies and tools
- the "exactly one valid outcome" authoring discipline. It is feasible for curated benchmark tasks but rarely true of arbitrary production tasks, where the state check has to tolerate several valid end states.
- DB-hash grading, which needs a fully mockable, deterministic backend
- the `sync_tools` coupling between telecom DBs
- the compositional telecom task generator
- the leaderboard submission and verification tooling
- voice and full-duplex tick linearization

## Open questions

- Which pass^k convention governs published taubench.com numbers, library exclusion or leaderboard counting infra errors as failures? The maintainers had not answered #497 at the time of reading.
- Did tau1's published numbers use the `llm` user strategy with gpt-4o at the provider-default temperature (tau1 #13)? The code does not pin the user temperature.
- Is there any calibration of the NL-assertion judge or the conversation reviewer against human labels? None was found in code. The tau2 paper's simulator-error study was manual annotation, not judge validation.
- How do the compositional telecom generator functions guarantee "provable correctness" (the paper's claim)? I read the task JSON, not the generator internals in `domains/telecom/tasks/*.py`.
- Did the "verified" tau2 airline and retail task fixes resolve the tau1 ground-truth issues listed above? I did not diff the task sets.
