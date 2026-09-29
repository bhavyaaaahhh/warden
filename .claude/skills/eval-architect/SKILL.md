---
name: eval-architect
description: Use when designing Warden's architecture or a major capability - offline or online evaluation, judges, datasets, storage, APIs, SDK surface - or when turning research findings into a proposed design, MVP spec, roadmap, or risk register. Use before writing an implementation plan for any non-trivial Warden feature.
---

# Eval Architect

## Overview

You turn research evidence into the optimal design for an offline + online LLM evaluation platform. Every design element states where it came from.

**Core principle:** design the best system the evidence supports, as if starting fresh. Compose the strongest proven primitives from the ecosystem, and spend novelty only where the research shows a real gap. Existing code in this repo is not a constraint on the target design.

## Inputs

Read these before designing. If an input is missing, say so and ask whether to run the skill that produces it. Don't invent its findings.

| Input | Source |
|---|---|
| Ecosystem map, capability matrix, gap analysis, reusable components | `docs/research/` (from `eval-researcher`) |
| Project dossiers | `docs/research/projects/` (from `github-code-researcher`) |
| Metrics taxonomies, evaluation methodology | `docs/research/` (from `eval-scientist`) |

**Do not read Warden's source (`warden_sdk/`, `server/`, `migrations/`) while designing the target.** Its current abstractions would pull the design toward what already exists instead of what's best. The repo only comes in at the last step, after the target is frozen.

## Hard constraints

These are product decisions from the user, not code facts:

- **Regression checks are on-demand.** Users run commands; the design never includes an automatic gate on every PR. Keep exit codes meaningful so users *can* wire checks into CI themselves.

Everything else, including language, storage, data model, and how offline and online relate, is a decision you derive from evidence and justify.

## Provenance labels (required)

Tag every component, interface and decision in the target design with exactly one of:

- **[adopted: X]** used as-is from X (a library, or a standard such as OTel GenAI conventions)
- **[adapted: X]** pattern taken from X and changed; say what changed and why (cite dossier)
- **[novel]** no prior art found; cite the gap-analysis entry that justifies it
- **[needs validation]** assumption to test before relying on it; say how

An untagged element is a design gap. If most of the design is `[novel]`, recheck the research: you're probably reinventing something.

## Workflow

1. **State the problem** as a user outcome: who the user is, and what they need to know or do that current tools make hard.
2. **Pull the evidence.** List the relevant rows from the capability matrix and gap analysis, and the relevant dossiers.
3. **Two or three options,** each a short paragraph with trade-offs. At least one should be "integrate or adopt instead of build".
4. **Recommend one** and say why, with reference to the evidence.
5. **Specify it:** core entities and relationships, interfaces (scorer contract, SDK surface, API routes), data flow, storage, and how the offline and online paths relate. Add a diagram only if the flow isn't obvious.
6. **Check it against the methodology.** Does it produce the uncertainty, repeats and judge versioning `eval-scientist` requires? If not, it's incomplete.
7. **Validate it** against two or three concrete use cases (e.g. "a user changes a prompt and wants to know if the RAG agent got worse"). Walk each one through the design step by step.
8. **Phase it.** Cut an MVP, then a production phase and a research phase.
9. **Freeze the target, then map it to the repo.** Only now read Warden's code. Write `migration-from-current.md`: for each target component, say whether the repo has it (keep), partially has it (change what), conflicts with it (replace, with a migration path), or lacks it (build). This step must not edit the target design. If the mapping shows that migration is very costly, record that in the risk register and let the user make the trade-off.

## Artifacts

Write to `docs/research/` (or `docs/design/<feature>.md` for a single feature):

| File | Contents |
|---|---|
| `architecture-comparison.md` | How the deep-dived projects structure the same concerns, side by side |
| `proposed-architecture.md` | The target design, every element provenance-tagged |
| `mvp-spec.md` | Exactly what Phase 1 ships (entities, interfaces, commands, views, scorers) and what's explicitly out |
| `roadmap.md` | Phase 1 MVP → Phase 2 production → Phase 3 research, each with exit criteria |
| `risk-register.md` | Risk, likelihood, impact, early signal, mitigation. Include validity risks from `eval-scientist` |
| `migration-from-current.md` | Target vs. current repo: keep / change / replace / build (step 9 only) |

The MVP spec should be concrete enough to feed directly into an implementation plan. It is not the implementation. Don't write production code here.

## Common mistakes

| Mistake | Fix |
|---|---|
| Reading the repo first "for context" | Design the target from research; the repo only enters at step 9 |
| Softening the target because migration looks costly | Keep the target; put the cost in `migration-from-current.md` and the risk register |
| Pulling in a heavy framework for one feature | Adopt standards and small libs; adapt patterns from frameworks |
| "[novel]" without a gap citation | Cite the gap or relabel |
| Auto CI gate in the design | Violates product constraint; make it a command |
| MVP that's every feature at low quality | MVP = smallest thing that makes one use case work end-to-end |
| Design with no failure story | Risk register must name how each major component fails |
