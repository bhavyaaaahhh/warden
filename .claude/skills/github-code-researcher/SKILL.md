---
name: github-code-researcher
description: Use when you need to know how an open-source eval, tracing, or observability project actually works internally - its data model, scorer/evaluator abstraction, execution engine, storage, or extension points - rather than what its README or marketing says. Also use when a capability-matrix cell or architecture decision depends on an unverified claim about another project.
---

# GitHub Code Researcher

## Overview

You produce implementation-level dossiers on other projects by reading their source. Every finding is pinned to a file, a symbol and a commit SHA.

**Core principle:** if you didn't open the file, you don't know. READMEs describe intent; code describes behavior.

## Setup

Clone shallowly **outside this repo**, so third-party code never lands in Warden's git tree:

```bash
mkdir -p ~/.cache/warden-research
git clone --depth 1 https://github.com/<org>/<repo> ~/.cache/warden-research/<repo>
git -C ~/.cache/warden-research/<repo> rev-parse HEAD   # record this SHA
```

Reuse an existing clone if one is present (`git pull` first). Use `gh` for issues, PRs, release cadence and contributor activity. Check the `LICENSE` file directly: license affects whether Warden can reuse the code or only the idea.

## Reading order

Don't read linearly. Find the load-bearing abstractions:

1. **Entry point.** The public API from the quickstart (`evaluate()`, `Task`, `@scorer`, the CLI main). Trace one call end to end.
2. **Core types.** grep for `class .*(Base|Protocol|ABC)`, `@dataclass`, pydantic models, and TypedDicts in the evaluator, dataset, trace and result modules.
3. **Execution.** How items run: concurrency (asyncio, threads, pools), retries, timeouts, caching, rate limits, and what happens when one item fails.
4. **Judges.** Where LLM-judge prompts live, how the output is parsed, how parse failures are handled, and whether scores are calibrated or aggregated.
5. **Persistence.** What gets stored where, the schema or migrations, and whether runs and datasets are versioned.
6. **Online path.** If one exists, how production traces become eval inputs: sampling, async queues, OTel ingestion.
7. **Extension points.** Plugin registries, entry points, subclass hooks.
8. **Tests.** They show intended behavior and edge cases the docs skip.
9. **Issues.** Search open issues for "slow", "scale", "flaky", "judge" and "breaking". Pain points reveal limits the docs hide.

## Judge on the merits, not against Warden

Don't read Warden's own source or compare against it. Its current abstractions would anchor what you count as "good". Evaluate each design against the problem it solves and against how other projects solve the same concern. Use a neutral vocabulary so dossiers line up:

| Concern | Names you'll see |
|---|---|
| Unit of scoring | evaluator, metric, scorer, grader, assertion |
| Unit of data | test case, example, sample, datapoint, row |
| A batch execution | experiment, eval run, task run, job |
| Execution record | trace, span, observation, run tree, log |
| Comparing executions | comparison, regression, diff, A/B |

## Dossier format

Write `docs/research/projects/<name>.md`:

```markdown
# <Project>
Repo: <url> @ <sha> · License: <spdx> · Lang: <lang> · Last commit: <date> · Activity: <releases/contributors, brief>

## Purpose
One paragraph. Offline / online / both. Who it's for.

## Core abstractions
For each: neutral concern (from the table above), their name, file:symbol, fields/signature (short excerpt ≤15 lines), the design trade-off it implies.

## Execution model
Concurrency, retries, caching, failure handling — with file:symbol refs.

## Data model & storage
What's persisted, schema, versioning.

## Judge implementation (if any)
Prompt location, output parsing, failure handling, calibration.

## Online / production path (if any)

## Extension model

## Strengths   — concrete, each with a code ref
## Weaknesses  — concrete, each with a code ref or issue link
## Worth adapting   — pattern + the problem it solves well
## Don't copy  — decision + why it's a poor design (coupling, scale limit, leaky abstraction…)
## Open questions   — things you couldn't verify
```

Link to exact lines with permalinks: `https://github.com/<org>/<repo>/blob/<sha>/<path>#L<a>-L<b>`.

## Evidence rules

- Every claim in Core abstractions, Execution and Data model has a file:symbol ref at the recorded SHA.
- If you didn't verify something, put it in Open questions. Don't state it.
- Quote code only as much as needed to show the shape. A dossier is not a mirror of the repo.
- Closed-source platforms (LangSmith, Braintrust and the like): analyze the open client SDKs and API reference. Label server-side behavior `[doc]` or `[inferred]`, never `[code]`.

## Common mistakes

| Mistake | Fix |
|---|---|
| Reading README + docs site only | Open the entry-point function and follow it |
| "Supports async evaluation" with no ref | Show the executor/gather call, or move it to Open questions |
| Reading the whole repo | Follow the reading order; stop once the abstractions are clear |
| Refs to `main` branch | Pin to SHA; `main` links rot |
| Listing features | Explain *how* and the trade-off it implies |
| Cloning into the Warden repo | `~/.cache/warden-research/` only |
| "Similar to what Warden does" | Don't compare to Warden's code; compare across projects |

## Report back

When dispatched as a subagent, return the dossier path plus five bullets or fewer: the findings that matter most to the question you were asked. Give the answer; the dossier holds the detail.
