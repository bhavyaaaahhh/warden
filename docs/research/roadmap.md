# Roadmap

| Phase | Ships | Exit criteria |
|---|---|---|
| **1. MVP** | `mvp-spec.md`: scripted multi-turn cases, trials, scorer contract with `unscored`, trial-aware paired comparison, `calibrate`, exit codes | Acceptance list in `mvp-spec.md` passes; A/A false-positive rate ≈ α on the demo suite |
| **2. Production** | Simulated users (scenario cases, structured stop, visibility filter, simulator provenance); LLM judges (forced-enum per-criterion verdicts, format-repair retry, provenance, pairwise with order swap); judge validation command (κ vs human labels); git merge-base baseline; re-score without re-run; trajectory arg matching per tool | A judge validated to κ ≥ 0.6 on ≥ 50 labelled cases; simulated suite A/A flip rate measured and reported |
| **3. Research** | End-state grading via reference-trajectory replay and structural diff; mocked stateful tools; typed fault injection with seeded payloads; simulator fidelity scorer; synthetic scenario generation from policy graphs; online scoring of sampled production traces sharing the same scorer contract | Each feature validated against τ2 domains as a conformance fixture set |
