# Proposed architecture: offline evaluation

Target design, derived from `gap-analysis.md`, `reusable-components.md`, `capability-matrix.md` and `evaluation-methodology.md`. Every element carries a provenance tag. The mapping onto the current repo is in `migration-from-current.md`, written after this design was frozen.

## 1. Problem

**User:** an engineer who owns an LLM agent (single-turn or conversational, with tools).

**What they need:** after changing a prompt, model or tool, run their test suite on demand and get a verdict they can trust: *did the agent get worse, better, or is the difference noise?* Plus: which conversations broke, and why.

**Why current tools make this hard** (`gap-analysis.md`):
- Nobody tests run-vs-run significance; tools that repeat trials don't diff, and tools that diff don't repeat (gap 1).
- Judge failures inflate scores in 4 of 9 tools (gap 3).
- Multi-turn cases have no home for expected tool calls per turn, and scripted conversations can't be stored as data (gap 5, Scenario).
- Row identity is position- or input-based in several tools, so diffs misalign.

## 2. Options considered

**A. Adopt Inspect AI as the runner, build comparison on its logs.** Inspect has the best scoring model (unscored, epochs, reducers, stderr). But the agent under test must run inside Inspect's solver model, there is no user simulator, and there's no run diff (`inspect-ai.md`). Users would adopt a whole framework to get a comparison feature. The `.eval` zip log is also a poor query store for cross-run pairing.

**B. Adopt promptfoo.** Strong on execution and trace-based trajectory assertions, but row identity is positional, multi-turn is split across three mechanisms (one forces concurrency 1), and judge parsing is lenient (`promptfoo.md`). The gaps we'd fix are in its core.

**C. Build a thin native runner and store, composing adapted primitives; integrate formats and formulas.** The pieces that matter are small: a case model, a harness loop, a scorer contract, and a statistics module. Every piece has prior art to adapt. The one novel part is the trial-aware paired comparison, which is the gap.

**Recommendation: C.** `reusable-components.md` §1 finds no eval library suitable as an as-is core dependency, each having a defect in the area it would own. The differentiator (a sound regression verdict) can't be bolted onto A or B without replacing their result model.

## 3. Components

```
 suite (JSONL, versioned)          agent under test (user callable)
        │                                   ▲
        ▼                                   │ messages in, turn out
   ┌──────────┐   per case × trial    ┌─────┴──────┐   spans
   │  runner  │──────────────────────▶│  harness   │──────────▶ trace store
   └──────────┘                       │ (loop,     │
        │                             │  script,   │
        │  transcript + trace         │  simulator)│
        ▼                             └────────────┘
   ┌──────────┐   score records
   │ scorers  │──────────────────▶ results store
   └──────────┘                          │
                                         ▼
                                ┌─────────────────┐
                        CLI ◀───│ compare (pure)  │───▶ viewer
                                └─────────────────┘
```

### 3.1 Case model

A **suite** is a JSONL file of cases. Each case:

| Field | Meaning | Provenance |
|---|---|---|
| `id` | user-assigned, stable across edits | [adapted: Braintrust `origin` row id] — user-assigned, never position or input |
| content hash | hash of the case's canonical JSON | [adapted: Inspect `task_identifier`] — per case, used for pairing |
| `input` | single-turn input | [adopted: common to all 9] |
| `turns` | scripted conversation: an ordered list of steps | [adapted: Scenario script DSL] — **stored as data**, not closures |
| `scenario` | simulated user: persona, goal, known/unknown info, max turns | [adapted: τ2 `user_scenario` + DeepEval persona] |
| `expected` | case-level expectations (contains, exact, tools, …) | [adapted: Strands expected/actual slots] |

A step in `turns` is one of:
- `{"user": "..."}` — a fixed user message
- `{"expect": {...}}` — checks on the agent's reply to the *preceding* user message (contains, tool calls made in that turn)

A case has exactly one of `input`, `turns`, `scenario` [adapted: Scenario, simplified — agent turns are implicit after every user step].

### 3.2 Messages and tool calls

Transcripts use the **OpenAI chat message shape with `tool_calls[].id` / `tool_call_id`** [adopted: OpenAI chat format, per `reusable-components.md` §1]. Tool calls made *within* a turn are kept in the transcript [adapted: openevals, fixing its one-message app contract].

### 3.3 Agent interface

- Single-turn: `agent(input) -> output` [adopted: common].
- Multi-turn: `agent(messages) -> str | message | list[message]`. The harness passes the full conversation so far and appends what the agent returns (tool calls and tool results included). Stateless by design, so trials and turns are independent [adapted: DeepEval callback, minus signature introspection — one explicit contract].
- Tool calls are also captured from the agent's trace spans (`tool_call` spans), so agents that don't return tool messages still get trajectory checks [adapted: promptfoo `trajectory:*` over OTel spans]. "No trace" never counts as "no tool calls" [adapted: fixes Braintrust JS].

### 3.4 Harness

The harness owns the conversation loop [adapted: Scenario/τ2 — harness owns the loop]:
- **scripted**: for each user step, call the agent, record the turn, evaluate that turn's `expect`.
- **simulated**: a `UserSimulator` produces the next user message and a structured stop decision `{stop: bool, reason}` [adapted: Strands `ActorResponse.stop`, replacing τ `###STOP###`]. The simulator sees only user-visible messages, never tool traffic [adapted: openevals internal-message filter]. Simulator model/prompt version is recorded [novel — gap 4: simulator validity unmeasured in products].
- **limits** (max turns) end the case as a scored outcome, not an error [adapted: Inspect `EvalSampleLimit`].
- **termination reason** is typed: `completed`, `max_turns`, `simulator_stop`, `agent_error`, `infra_error` [adapted: τ2 `TerminationReason`, reasons not collapsed].

### 3.5 Trials

Each case runs **k** trials (default 3) [adapted: Inspect epochs]. Infra errors are retried (2×) and then left out with `n_i < k` recorded [adapted: τ2 `run_with_retry`, fixing its pass^k bias]. Each trial has a deterministic id from (run, case, trial) [adapted: Braintrust `uuid5(...:trial:i)`].

### 3.6 Scorer contract

```
Scorer.name, Scorer.version, Scorer.kind ∈ {check, metric}, Scorer.direction ∈ {higher, lower}
score(ctx) -> Score | list[Score] | None       # None = n/a
ctx: case, output, transcript, turns (per-turn records), trace, trial_index
Score: outcome ∈ {pass, fail, unscored}, value: float | None, reason, criterion: str | None
```

- `unscored` excluded from denominators, counted as coverage [adapted: Inspect `Score.unscored`].
- A scorer that raises produces `unscored` with the exception as reason, never a dropped row [adapted: LangSmith `extra={"error": True}`].
- Multiple criteria per scorer via a list of scores [adapted: Inspect dict-valued score / Scenario per-criterion verdicts].
- Metric values aren't forced into [0,1] [adapted: Inspect; fixes Braintrust].
- Judge scorers add provenance (model, prompt hash) to `version` [adapted: Ragas `PydanticPrompt` hash, stored per score].

Built-in scorers:
- **checks:** `contains`, `exact_match`, `no_errors`, `tool_calls` (trajectory: strict / unordered / subset / superset over tool names, with optional arg subset match) [adapted: agentevals vocabulary, bipartite rather than greedy for unordered], `turn_expectations` (per-turn `expect` results, one criterion per turn) [novel — gap 5].
- **metrics:** latency, cost, tokens, turns [adopted: operational metrics common to all].

### 3.7 Storage

Row per (run, case, trial) and per (trial, scorer, criterion) [adapted: DeepEval/promptfoo SQL stores, not parallel arrays]. The run records suite hash, per-case hashes, k, git SHA, version tag, agent ref, simulator version. Comparisons are **computed, not stored** — a pure function over two runs [novel wrt stored diffs; keeps CLI and viewer consistent].

### 3.8 Comparison

Implements `evaluation-methodology.md` §§2–7 exactly:
- pair by case id + content hash; exclude changed cases [adapted: Braintrust `origin` pairing];
- per-case summaries (stable pass/fail/flaky, pass^k on full-trial cases) [adapted: Inspect reducers, τ-bench pass^k];
- per-scorer verdict: exact McNemar (k=1) or paired sign-flip permutation (k>1), paired bootstrap CI, Benjamini–Hochberg across scorers [novel — gap 1; methods adopted from literature];
- per-case transitions (broke/fixed/degraded/stabilised) [novel — gap 1];
- metrics: paired bootstrap on per-case means, CI-vs-±10% rule [adapted: Inspect `bootstrap_stderr`];
- `incomparable` when a scorer's version differs; `inconclusive` below 90% coverage or < 5 pairs [novel — gap 3].
- Deterministic: the RNG is seeded from the two run ids, so the same comparison always prints the same p-value [novel].

**Baseline selection:** explicit run id or version tag; else the stored baseline for (suite, agent). Git merge-base selection is Phase 2 [adapted: Braintrust, deferred].

### 3.9 Commands

`run`, `check` (run + compare against baseline, exit 0/1/2), `compare`, `baseline`, `runs`, and `calibrate` (A/A: run the same version twice, report flip rate and false-positive rate) [adapted: methodology §7]. All on demand; no default CI gate (product constraint).

## 4. Methodology check

| Methodology requirement | Where the design meets it |
|---|---|
| outcomes pass/fail/unscored/error/n/a (§2) | Score outcome + trial termination reason + `None` |
| k trials, retry infra errors, fixed-n pass^k (§3) | 3.5, 3.8 |
| paired by stable id + hash (§4) | 3.1, 3.8 |
| McNemar / permutation / bootstrap / BH / verdicts (§5) | 3.8 |
| metric bootstrap, p95 rule (§6) | 3.8 |
| A/A noise floor (§7) | `calibrate` |
| judge provenance, unscored on parse failure (§8) | 3.6 (judges themselves are Phase 2) |
| simulator provenance, scripted mode (§9) | 3.4 |
| exit codes (§10) | 3.9 |

## 5. Use cases walked through

**UC1: prompt change on a support agent (scripted multi-turn).** Suite has 30 cases with `turns`. User runs `check --trials 3`. Each case runs 3 times; per-turn `expect` produces a `turn_expectations` criterion per turn. Compare pairs 30 cases, sign-flip test on `turn_expectations` pass rates gives p = 0.004, Δ = −0.13 [−0.21, −0.05] → `regression`, exit 1. Transitions list 4 cases as **broke**, each with the failing turn and reason. ✔

**UC2: flaky agent, no real change.** User runs `calibrate`. A/A shows 6 of 30 cases flaky and no scorer with a verdict other than "no detectable change". Later a `check` with one case flipping pass→fail prints "no detectable change (1 discordant case; at least 6 needed at k=1)". ✔ — the design avoids the false alarm current tools raise.

**UC3: judge prompt edited between runs.** Scorer version differs → that scorer's verdict is `incomparable`, exit 2, other scorers still compared. ✔ (Judges ship in Phase 2, but the contract handles it now.)

**UC4: tool-call regression.** Case expects `lookup_order` then `refund` (strict). Candidate calls `refund` directly. `tool_calls` fails with reason "missing lookup_order at position 0". Captured from spans even though the agent returns only text. ✔
