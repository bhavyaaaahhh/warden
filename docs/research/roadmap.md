# Roadmap

| Phase | Ships | Exit criteria | Status |
|---|---|---|---|
| **1. MVP** | `mvp-spec.md`: scripted multi-turn cases, trials, scorer contract with `unscored`, trial-aware paired comparison, `calibrate`, exit codes | Acceptance list in `mvp-spec.md` passes; A/A false-positive rate ≈ α on the demo suite | Built (#5) |
| **2. Production** | Simulated users (scenario cases, structured stop, visibility filter, simulator provenance); LLM judges (forced-enum per-criterion verdicts, format-repair retry, provenance, pairwise with order swap); judge validation command (κ vs human labels); git merge-base baseline; re-score without re-run; trajectory arg matching per tool | A judge validated to κ ≥ 0.6 on ≥ 50 labelled cases; simulated suite A/A flip rate measured and reported | Built (#6–#9, #11). **Exit criteria not yet met:** needs real Anthropic credentials and human labels |
| **3. Research** | End-state grading via structural diff; mocked stateful tools; typed fault injection with seeded payloads; synthetic scenario generation; online scoring of sampled production traces sharing the same scorer contract | Each feature validated against τ2 domains as a conformance fixture set | End state, mocks, faults, generation, traces → cases built (#10, #11). **Not built:** simulator-fidelity scorer, τ2 conformance fixtures, online scoring |

## Still open for offline evals

- **Validate on a real agent.** Everything has run only against the fake demo agents and stand-in models. The judge, simulated user and generator have never called Claude.
- **Calibrate the conventions.** The thresholds (α = 0.05, 90% coverage, ±10% on metrics, k = 3, κ ≥ 0.6) come from convention and need checking with `calibrate` and `validate-judge` on real suites.
- **Simulator fidelity.** Measure how often the simulated user breaks character or leaks its goal (τ2's reviewer), as a scorer.
- **Conformance fixtures.** Run τ2-bench domains through the harness and compare with published behaviour.
- **Labelling in the viewer.** Human labels are JSONL files today; there's no annotation queue in the UI.
