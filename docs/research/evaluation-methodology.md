# Evaluation methodology: regression, improvement, or noise

This is the spec `eval-architect` builds to. It covers **offline** evaluation of single- and multi-turn agents. Online evaluation is out of scope for this document.

Sources are tagged: **[paper]** literature, **[code]** observed in a dossier in `projects/`, **[convention]** a chosen default with no empirical basis yet, **[measure]** to be calibrated on real data.

## 1. The question every comparison answers

"Did the candidate get better or worse than the baseline, on the same test cases, by more than run-to-run noise?"

Every verdict the platform shows states: what it was compared to, n (cases and trials), the effect size with an interval, and whether that interval rules out zero.

## 2. The unit of measurement

A **score record** is one (case, trial, scorer) observation. Its outcome is exactly one of:

| Outcome | Meaning | Counts in pass-rate denominator? |
|---|---|---|
| `pass` | scorer ran and the check held | yes |
| `fail` | scorer ran and the check did not hold | yes |
| `unscored` | scorer could not decide: judge parse failure, judge API error, missing input (Inspect `Score.unscored`) [code] | **no**; reported as coverage |
| `error` | the agent run itself hit an infrastructure error (timeout of the harness, provider 5xx) (promptfoo tri-state, τ2 typed termination) [code] | **no**; retried, then reported |
| `n/a` | scorer does not apply to this case | no, and not reported as missing coverage |

Two rules fix the score-inflation bug found in 4 of 9 projects (`gap-analysis.md`, gap 3):

- A judge that fails to produce a parseable verdict yields `unscored`. It is **never** `pass`, and never silently dropped from a count that then reads as a higher rate.
- An **agent-caused** failure (the agent crashed, exceeded a turn/token/cost limit, called a forbidden tool) is a `fail`, not an `error`. Limits are outcomes (Inspect) [code]. Only failures of the harness or provider are `error`.

**Coverage** = scored / (scored + unscored + error) per scorer. If coverage for a scorer is below **90%** [convention] in either run, that scorer's verdict is `inconclusive`, whatever the numbers say.

Metric scorers (latency, cost, tokens, a judge's 1–5 rating) carry a numeric value instead of pass/fail and follow §6.

## 3. Trials

Agents are nondeterministic, so a single run per case cannot separate a regression from a flaky case (tau-bench issue #540 reports 5–11 point run-to-run swings [code]).

- Each case runs **k trials** per run. Default **k = 3** for `check` [convention]; k = 1 is allowed but the output says the result cannot distinguish flakiness.
- Every trial of a case uses the same inputs; caching is keyed per trial, so trials are never collapsed into one cached answer (Inspect keys by epoch [code]).
- Trials that end in `error` are retried up to **2** times [convention]. If a trial still errors, it is left out, and the case's valid-trial count `n_i < k` is recorded.

### Per-case summary

For case *i* with `c_i` passes among `n_i` valid trials:

- pass rate `p_i = c_i / n_i`
- **stable pass** if `c_i = n_i`; **stable fail** if `c_i = 0`; otherwise **flaky**
- `pass@k` and `pass^k` use the unbiased estimators `1 − C(n−c, k)/C(n, k)` and `C(c, k)/C(n, k)` (Chen et al. 2021, arXiv:2107.03374; τ-bench arXiv:2406.12045) [paper]

**pass^k is computed only on cases with `n_i = k`.** τ2 drops errored trials and then computes pass^k on fewer trials, which raises the score of cases that errored [code]. Cases with fewer valid trials are excluded from pass^k and counted separately.

## 4. Pairing

Comparisons are **paired** by a stable case id. Position-based matching (promptfoo `testIdx`) and matching on input text (Braintrust default) are not used [code].

A case is compared only if it exists in both runs with the **same case content hash**. Cases that were added, removed or edited are listed but excluded. If more than **10%** [convention] of baseline cases are excluded, the whole comparison carries a warning.

A comparison across different **judge versions** (model, prompt, rubric) for the same scorer is marked `incomparable` for that scorer. The score moved because the ruler changed.

## 5. Pass/fail verdict per scorer

Let `d_i = p_i(candidate) − p_i(baseline)` over the paired cases, and let `Δ = mean(d_i)`.

**Test.**
- k = 1: `d_i ∈ {−1, 0, 1}`. Use the **exact McNemar test** on discordant pairs: b = pass→fail count, c = fail→pass count, a two-sided binomial test of b vs. b + c at 0.5 [paper: McNemar 1947].
- k > 1: use a **paired sign-flip permutation test** on `d_i` (flip the sign of each nonzero `d_i` at random, 10,000 resamples, two-sided). It is exact for small n, needs no normality, and reduces to McNemar when k = 1 [paper: Good, *Permutation Tests*, 2005].
- **Interval**: a 95% **paired bootstrap** CI on Δ, resampling cases (so trials stay clustered within their case), 10,000 resamples (Miller 2024, "Adding Error Bars to Evals", arXiv:2411.00640) [paper].

**Verdict.**

| Verdict | Rule |
|---|---|
| `regression` | adjusted p < α **and** Δ < 0 |
| `improvement` | adjusted p < α **and** Δ > 0 |
| `no detectable change` | otherwise. Also report the **minimum detectable effect** for this n |
| `inconclusive` | coverage < 90% in either run, or fewer than 5 paired cases |
| `incomparable` | judge version changed |

- **α = 0.05** [convention].
- **Multiple scorers**: p-values across the scorers of one comparison are adjusted with **Benjamini–Hochberg** at q = 0.05 [paper: Benjamini & Hochberg 1995], counting only the scorers whose smallest attainable p-value (2 / 2^m for m cases that changed) is below α [paper: Tarone 1990, *Biometrics* 46:515]. A scorer where nothing changed can't be significant, and including it only dilutes the others. (Found in the first end-to-end check: 6 of 6 refund cases broke in every trial, raw p = 0.031, but BH over a second scorer with no changes pushed it to 0.062.)
- **No effect-size floor by default.** A floor such as "ignore |Δ| < 2 points" is a product decision the user sets per suite [convention].

**What small suites can detect.** With k = 1, the smallest possible two-sided exact McNemar p is `2 × 0.5^(b+c)`. So it takes at least **6 discordant cases, all in the same direction**, to reach p < 0.05 (p = 0.031); 5 gives p = 0.0625. The output states this, so a 20-case suite with one flip is shown as "no detectable change", not as a regression.

### Per-case diagnostics (not verdicts)

Every paired case gets a transition label, independent of the suite-level test, so users can see *where* things moved:

| Baseline → candidate | Label |
|---|---|
| stable pass → stable fail | **broke** |
| stable fail → stable pass | **fixed** |
| stable pass → flaky, or flaky → stable fail | degraded |
| flaky → stable pass | stabilised |
| stable fail → flaky | improved |
| flaky → flaky, or unchanged | unchanged |

"Broke" and "fixed" are listed first. With k ≥ 3, **broke** is strong evidence for that case even when the suite-level test isn't significant.

## 6. Numeric metrics (latency, cost, tokens, graded scores)

- Per case, average the valid trials. Compare with a **paired bootstrap** over cases on the difference of means (or medians), with a 95% CI [paper: Efron & Tibshirani 1993].
- **p95 needs n ≥ 50 cases** [convention; with fewer, the p95 is close to the max]. Under 50, show the median and label p95 as unstable.
- Verdict: `worse` or `better` only if the **whole CI** of the relative change lies beyond ±**10%** [convention] in that direction; otherwise `no detectable change`. A fixed percentage with no interval is not used.
- Cost and tokens also get a suite **total** as a plain number.
- Direction ("higher is better") is declared per metric; latency, cost and tokens are lower-is-better.

## 7. The noise floor (A/A calibration)

Before a suite's verdicts are trusted, run the **same version against itself**: two runs, k trials each [measure].

- Report the per-case flip rate and the share of cases that are flaky.
- The false-positive rate of §5 on A/A runs should be ≈ α. If an A/A comparison reports a regression, the suite's trials are too few or its scorers too noisy, and the output says so.
- Cases flaky in more than **50%** [convention] of A/A comparisons are marked **quarantined**. They are still run and shown but excluded from the verdict, and listed.

A/A runs are a command users choose to run, like every other check.

## 8. LLM judges

A judge scorer carries **provenance**: judge model id, prompt/rubric hash and version, and temperature. It is stored on every score record (§4 uses it).

Output rules:
- Binary or small-enum verdicts per criterion, with reasoning before the verdict. The verdict is forced through structured output or a function call with an enum (Braintrust `select_choice`) [code].
- Several criteria each get their own verdict, and the case fails if any required criterion fails (Scenario per-criterion verdicts) [code].
- Parse failure gets **one** format-repair retry (Ragas) [code]; if that fails, the result is `unscored`.
- Pairwise judges run both orders and report a tie when the two orders disagree (position bias; Zheng et al. 2023, arXiv:2306.05685) [paper].

**Validation before a judge gates anything** (a judge that hasn't been validated can still be shown, labelled "unvalidated"):
- 50–200 human-labelled cases; judge vs. human **Cohen's κ ≥ 0.6** [convention; "substantial" band in Landis & Koch 1977]. Report the confusion matrix.
- Re-judge the same cases three times and report the self-flip rate.
- Store the validation result with the judge version. Changing the version clears it.

## 9. Multi-turn specifics

- The **case** is the conversation. All scorers score a conversation (possibly at specific turns). Trials repeat the whole conversation.
- A **simulated user** is a source of noise and bias of its own (τ2 paper: simulator errors in 16–47% of conversations, arXiv:2506.07982) [paper]. So:
  - The simulator's model, prompt and version are provenance, like a judge's. A change makes the comparison `incomparable`.
  - A **scripted** mode (fixed user turns, no simulator) is available for cases where exact reproducibility matters more than coverage.
  - Simulator-caused endings (the simulator gave up or broke character) are recorded as a distinct termination reason, so they can be excluded.
- **Tool-call trajectory** checks are deterministic scorers with a declared match mode (strict, unordered, subset, superset) (agentevals) [code]. They are preferred over judges where they apply, because they don't add judge noise.

## 10. Exit codes for `check`

| Code | Meaning |
|---|---|
| 0 | no scorer has a `regression` verdict |
| 1 | at least one `regression` |
| 2 | could not decide: `inconclusive` or `incomparable` on a scorer, or the run failed |

Users can wire these into CI themselves. The platform does not install a gate by default.

## 11. Open questions (need data)

- Is k = 3 enough for typical suites, or does the A/A flip rate call for 5? [measure]
- Is 90% coverage the right cut-off for `inconclusive`? [measure]
- Is the sign-flip test well enough powered at 20–50 cases, or should small suites get a Bayesian estimate with a credible interval? [measure]
