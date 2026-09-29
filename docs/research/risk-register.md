# Risk register

| Risk | Likelihood | Impact | Early signal | Mitigation |
|---|---|---|---|---|
| Small suites never reach significance, so users see only "no detectable change" and stop trusting the tool | High | High | Most `check` outputs say "no detectable change" with wide CIs | Print minimum detectable effect and the discordant-pair floor; list **broke** cases regardless of the suite verdict; recommend more cases or trials |
| k trials multiply cost and time | High | Medium | Users set `--trials 1` everywhere | Default 3 only for `check`; show cost per run; Phase 2 re-score without re-run |
| Permutation/bootstrap p-values differ run to run | Medium | Medium | Same comparison prints different p | Seed RNG from the two run ids |
| Conventions (α, 90% coverage, ±10%, k = 3) are wrong for real suites | Medium | Medium | A/A `calibrate` shows false positives above α | Treat as [convention] in the methodology; calibrate on real data; make them flags |
| Case edits silently exclude many cases from comparison | Medium | High | Warning about > 10% excluded | Warn loudly; show which cases changed |
| Stateless `agent(messages)` contract doesn't fit agents with server-side session state | Medium | Medium | Users ask for thread ids | Phase 2: optional `thread_id` in a context argument |
| Tool calls missed when agents neither return tool messages nor trace them | Medium | High | `tool_calls` fails with "no tool calls observed" on agents known to use tools | Distinguish "no trace" from "no calls" (→ `unscored`), per the design |
| Simulated users (Phase 2) add noise larger than the effects measured | High | High | A/A flip rate on simulated suites far above scripted | Keep scripted mode first-class; record simulator provenance; measure simulator error |
| LLM judges (Phase 2) disagree with humans | High | High | κ < 0.6 in validation | Judge validation gate; label unvalidated judges; `unscored` on parse failure |
| Schema migration of existing stored runs | Low | Medium | Old runs fail to load in compare | Additive columns with defaults; old runs treated as k = 1 |
| Runs list computes each verdict in Python, fetching baseline runs | Medium | Low | `/eval_runs` gets slow with many large runs | Cache baseline fetches per request; later, persist the verdict when `check` runs |
