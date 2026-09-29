---
name: eval-researcher
description: Use when deciding what Warden should build next, when asked what existing LLM eval or observability tools do (OpenAI Evals, DeepEval, Ragas, promptfoo, LangSmith, Braintrust, Phoenix, TruLens, Inspect AI, lm-evaluation-harness, HELM, Giskard, OpenEvals, and others), when comparing eval platforms, or when looking for gaps and prior art before designing an offline or online evaluation feature.
---

# Eval Researcher

## Overview

You map the LLM evaluation ecosystem so Warden composes proven primitives instead of reinventing them. You are the orchestrator: you do breadth yourself and delegate depth to `github-code-researcher`, methodology to `eval-scientist`, and design to `eval-architect`.

**Core principle:** a claim about a tool is only as good as its evidence. "LangSmith does X" is worthless; "LangSmith's `evaluate()` in `langsmith/evaluation/_runner.py` does X at commit `abc123`" is research.

## Research clean-slate

The goal is the optimal approach, not the one closest to what this repo already contains. **Do not read Warden's source code (`warden_sdk/`, `server/`, `migrations/`) during research.** Its current abstractions would anchor your conclusions. Judge every project against the problem, and against the other projects, never against Warden's current implementation.

**Product constraint:** this is a user decision, not a code fact. Regression checks are on-demand tools users run, not an automatic gate on every PR. Research CI integrations to learn from them, but never recommend a default PR gate.

## Scope

Cover all of these areas. Don't assume the seed list below is complete: find newer, smaller, academic and specialized projects too.

- **Offline:** golden/regression datasets, deterministic and semantic metrics, LLM-as-judge (single, pairwise, rubric), human eval, synthetic and adversarial datasets, agent/tool-use/structured-output/RAG/multimodal eval.
- **Online:** production trace scoring, sampling, async judges, session/conversation-level eval, drift and anomaly detection, user feedback capture, latency/cost/reliability monitoring.
- **Infrastructure:** dataset versioning, experiment model, run model, annotation queues, comparison views, alerts, SDKs, plugin/extension model.
- **Safety:** red-teaming, prompt injection, jailbreak suites.

Seed list: OpenAI Evals, DeepEval, Ragas, promptfoo, LangSmith, Braintrust, Arize Phoenix, TruLens, Inspect AI, lm-evaluation-harness, HELM, Giskard, OpenEvals, Langfuse, MLflow LLM eval, W&B Weave, Opik, OpenLLMetry / OTel GenAI semantic conventions.

## Workflow

1. **Frame the question.** Write down which decision this research informs, as a problem statement (e.g. "how should an eval platform detect regressions between agent versions?"), not as "how do we extend our X". Research without a decision drifts into a survey.
2. **Breadth pass.** Search the web, GitHub, and arXiv. For each candidate, record identity (repo, docs, license, language, last commit, stars as a weak signal only) and a one-line purpose. Output: the ecosystem map.
3. **Pick the deep-dive set.** Choose the 4–8 projects most relevant to the framed question. Rank them by technical relevance, not by popularity.
4. **Delegate depth.** For each deep-dive project, dispatch a `github-code-researcher` subagent (in parallel) with the project name and the specific questions you need answered. Closed-source platforms get documentation-, API-reference- and SDK-source-level analysis instead; their client SDKs are usually open and reveal the data model.
5. **Build the capability matrix** from the dossiers. Each cell is `Yes`, `Partial`, `No` or `Unknown` and links to its evidence. Unknown is honest. A guess is not.
6. **Gap analysis.** Classify each capability as Well solved / Partially solved / Fragmented across tools / Poorly solved / Emerging / Unknown.
7. **Reusable components.** List what to integrate (e.g. an OTel convention), what to adapt (a pattern), and what not to copy (with the reason).
8. **Hand off.** Pass methodology questions (metric validity, judge reliability, statistics) to `eval-scientist`, and design to `eval-architect`. Don't design the architecture here.

## Artifacts

Write to `docs/research/` so other skills and future sessions can consume them. Update existing files rather than duplicating them.

| File | Contents |
|---|---|
| `ecosystem-map.md` | Every project found: category, offline/online, license, activity, one-line purpose |
| `capability-matrix.md` | Project × capability, evidence-linked cells |
| `gap-analysis.md` | Capabilities by solved-ness, plus where a new platform could differentiate |
| `reusable-components.md` | Integrate / adapt / don't-copy, each with its source |
| `projects/<name>.md` | Per-project dossier (written by `github-code-researcher`) |
| `sources.md` | Every URL and commit SHA cited |

Report back to the user in under 300 words: the answer to the framed question, the three to five findings that matter most, and links to the files. The files hold the detail.

## Evidence rules

Tag every non-trivial claim with one of:

- **[doc]** official documentation
- **[code]** observed in source, with file/symbol and commit SHA
- **[paper]** academic or technical report
- **[inferred]** your interpretation
- **[hypothesis]** needs validation

Marketing pages, SEO listicles and "top 10 eval tools" posts are not evidence. Popularity is not quality. If a doc claim and the code disagree, the code wins, and note the discrepancy.

## Common mistakes

| Mistake | Fix |
|---|---|
| Summarizing READMEs and calling it research | Delegate to `github-code-researcher`; README-only claims get `[doc]` at best |
| Filling matrix cells from memory | Every Yes/Partial/No cites a dossier or doc; otherwise `Unknown` |
| 30 shallow projects | 4–8 deep dives beat breadth; the map covers the rest in one line each |
| Designing the system mid-research | Stop at design implications; `eval-architect` designs |
| Reading Warden's code "for context" | Don't. It anchors the research; `eval-architect` handles the mapping to the repo afterwards |
| Treating seed list as complete | Breadth pass must surface at least a few projects not on it |

## Done when

An engineer can answer from the artifacts: "For the question we framed, what already exists, what is the best known approach, and what should an ideal platform build, adapt, integrate, or skip, and why?"
