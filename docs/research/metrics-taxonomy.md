# Metrics taxonomy (offline)

For each family: what it measures, what the scorer needs as input, how it fails, and its cost. The last column defines what the scorer interface must support.

| Family | Examples | Needs | Failure modes | Cost | Interface requirement |
|---|---|---|---|---|---|
| **Deterministic check** | exact match, regex, JSON-schema valid, forbidden-string | output | brittle on paraphrase; substring passes on negation | ~0 | output + case expectations |
| **Reference-based** | contains-all, F1 / token overlap, JSON diff (Braintrust `JSONDiff`) | output + reference | reference drift; many valid answers | ~0 | reference on the case |
| **Tool-trajectory** | tool used / not used, sequence (strict, unordered, subset, superset), arg match per tool (agentevals, promptfoo `trajectory:*`) | full trace with tool calls and their ids | lenient arg defaults let omitted args pass (τ2 #568); greedy matching (LangSmith) | ~0 | **trace**, expected calls (per case, optionally per turn) |
| **End-state** | final environment state equals gold state (τ2) | environment snapshot + gold state | order-sensitive hashing (τ2 #514); misses procedure | ~0 once env exists | **environment state** before/after |
| **Semantic** | embedding similarity | output + reference | hidden rescaling (Braintrust 0.7 floor); weak on correctness | low | reference |
| **Judge, single** | per-criterion yes/no, forced enum, DAG (DeepEval) | transcript, optional reference/rubric | bias, parse failure, self-preference; unvalidated | medium | **async**, **provenance**, `unscored` outcome |
| **Judge, pairwise** | A vs B with ties | two outputs for the same case | position bias (Battle, select-best have none) | 2× judge | **pairwise inputs** from two runs |
| **Human** | annotation, preference | transcript | rater disagreement; slow | high | **labels** stored by case + version |
| **Operational** | latency, cost, tokens, turns, tool-call count | trace | small-n percentiles | ~0 | trace; numeric value, direction |
| **Reliability** | pass@k, pass^k, flip rate | several trials | biased when trials drop (τ2) | k× run cost | **repeats** grouped by case |
| **Safety** | injection resisted, refusal where required, no data exfiltration (garak, PyRIT suites) | transcript + trace | judge-dependent | medium | same as judge + trajectory |
| **Simulator fidelity** | simulator stayed in role, didn't leak hidden goal (τ2 reviewer) | transcript + scenario | only τ2 measures it | medium | scenario visible to this scorer only |

## What this means for the scorer contract

A scorer must be able to receive: the case (input, expectations, scenario), the output, the **full trace** (tool calls with ids, per-turn messages), the environment state if any, and optionally a paired output from another run. It returns, per criterion: a pass/fail or numeric value, a reason, and an outcome that can be `unscored`. Judge scorers carry provenance. Scorers can be async. Repeats are grouped by the runner, not by the scorer.
