# Offline evaluation taxonomy

## Dataset types

| Type | Purpose | Built from | Refreshed |
|---|---|---|---|
| **Golden** | defines "correct" for core tasks | hand-written by owners | when requirements change; versioned |
| **Regression** | locks in fixed bugs | a failing production trace or bug report, turned into a case | append-only; a case is never edited, only retired |
| **Adversarial / safety** | injection, jailbreak, out-of-scope requests | public suites (garak, PyRIT), plus owner-written cases | per suite release |
| **Synthetic** | coverage at scale | generated scenarios (IntellAgent-style policy graphs, persona × goal grids) | regenerated; generated cases need review before they gate anything |
| **Fault** | robustness under tool failure | an existing case × a typed fault (Strands `ChaosCase`) | derived automatically from the other types |

## Case shapes

| Shape | Contents | Run by |
|---|---|---|
| **Single-turn** | input, expectations | one agent call |
| **Scripted multi-turn** | ordered steps: fixed user turns, agent turns, checks at specific turns (Scenario script, stored as data) | the harness, deterministically |
| **Simulated multi-turn** | persona, scenario (goal, what the user knows, what the user doesn't know), max turns, stop reasons, expectations (τ2, DeepEval) | the harness, with an LLM user |

Simulated cases can start with a scripted prefix of fixed turns and then hand over to the simulator (openevals `fixed_responses`).

## Versioning rules

- Every case has a stable id and a content hash. Editing the content changes the hash; the id stays.
- A run records the dataset version (content hash of the whole file) and each case's hash, so comparisons pair only unchanged cases (methodology §4).
- Cases record where they came from (hand-written, trace id, generator + seed).

## Execution dimensions

- **Trials** per case (methodology §3).
- **Limits** (turns, tokens, time, cost) end the case as a scored outcome.
- **Environment**: none, mocked tools (fixed responses per tool call), or stateful fakes. LLM-invented tool responses (Strands `ToolSimulator`) aren't used where end-state is checked.
