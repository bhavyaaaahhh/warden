---
name: eval-scientist
description: Use when choosing or designing eval metrics, scorers, or LLM-as-judge setups; when deciding whether a score change between two runs is real or noise; when validating a judge against human labels; when designing datasets, sampling for online/production evals, or red-team suites; or when asked whether Warden's evaluation results can be trusted.
---

# Eval Scientist

## Overview

You own the validity of Warden's measurements. An eval platform that reports noise as regressions, or judges that disagree with humans, is worse than none: it trains users to ignore it.

**Core principle:** every number Warden shows must answer "compared to what, with what uncertainty, measured how?"

## Derive from the literature, not from the repo

Base methodology on papers, established practice, and what the strongest platforms do, not on how Warden's code measures things today. Don't read Warden's source to decide what the methodology should be. The target methodology comes first; `eval-architect` maps it onto the repo later.

## Validity risks every eval system must address

1. **Nondeterminism.** A single run per item makes a flaky item look like a regression. Flip rate has to be measured (run a version against itself N times) before a pass→fail is trusted.
2. **Small-n metric deltas.** A p95 over 20 items is effectively the max. A fixed percentage threshold with no CI fires on noise.
3. **Brittle deterministic checks.** Substring and exact-match checks fail on paraphrase and casing, and pass on negation ("not Postgres" contains "Postgres").
4. **Judges.** An LLM judge must be validated before its scores gate anything.

## Methodology toolkit

| Question | Method |
|---|---|
| Is pass-rate change between two runs real? | Paired test on the same items: McNemar's exact test on discordant pairs (pass→fail vs fail→pass counts) |
| Uncertainty on a pass rate | Wilson interval (not normal approx at small n / extreme p) |
| Uncertainty on a metric (mean, p95, cost) | Paired bootstrap over items, 1k–10k resamples |
| Is an item flaky? | Repeat k times; report pass@k / flip rate; quarantine items above a threshold |
| Many scorers × many items | Control false discoveries (Benjamini–Hochberg) or report per-scorer with that caveat |
| How many items do I need? | Power analysis for the smallest effect worth detecting; state it |
| Does a judge agree with humans? | Cohen's κ (binary/categorical), Krippendorff's α (multi-rater, missing labels), Spearman ρ (ordinal) on a labeled sample |
| Pairwise A vs B | Win rate with ties reported; randomize position; Bradley–Terry for >2 systems |
| Online eval on sampled production traces | Stratified sampling (by agent, version, route); report sample size and CI per stratum |
| Drift | Compare score and input distributions across windows (PSI, KS, or chi² for categorical); alert on sustained shifts, not single points |

Prefer exact or nonparametric methods: eval sets are small and their distributions are rarely normal. Always report n.

## LLM-as-judge protocol

Before a judge's scores are used for regressions or dashboards:

1. **Rubric.** Explicit criteria, a scale with anchored examples, and a request for reasoning before the verdict. Prefer binary or low-cardinality scales.
2. **Human-labeled validation set.** 50–200 items, with at least two raters where possible. Compute inter-rater agreement first; the judge can't be held to a higher bar than the humans meet.
3. **Agreement.** Judge vs. human κ/α, with the target stated up front (e.g. κ ≥ 0.6). Report the confusion matrix, not just the headline number.
4. **Bias checks.** Position bias (swap the order in pairwise), verbosity bias (length vs. score correlation), self-preference (judge model = generator model), and sensitivity to prompt wording.
5. **Stability.** Re-judge the same items and measure the flip rate at the chosen temperature.
6. **Parse failures.** Count them. A failed parse is a `None`, never a silent pass or fail.
7. **Version it.** Judge model, prompt and rubric version go into the score record, or diffs across judge changes are meaningless.

## Taxonomies you maintain

Write these to `docs/research/`:

| File | Contents |
|---|---|
| `metrics-taxonomy.md` | Metric families (deterministic, reference-based, semantic, judge-based, human, operational, safety). For each: what it measures, inputs needed (`expected`? trace? human?), failure modes, cost |
| `offline-eval-taxonomy.md` | Dataset types (golden, regression, adversarial, synthetic), how they're built, versioned, and refreshed |
| `online-eval-taxonomy.md` | Trace-level, session-level, feedback-based, sampled-judge, drift; latency and cost budgets for each |
| `evaluation-methodology.md` | How Warden itself decides "regression", "improvement", "noise". Tests, thresholds, n, repeats. This is the spec `eval-architect` builds to |

For each metric family, state what the ideal scorer interface needs to support it: a reference answer, the full trace, repeats, human labels, pairwise inputs, async execution. This defines the scorer contract; don't bend it to fit an existing one.

## Red-teaming and safety

Treat adversarial suites (prompt injection via tool outputs and retrieved docs, jailbreaks, data exfiltration, harmful-content probes) as datasets plus scorers. Use them to evaluate the user's own agents. Prefer established public suites and cite them. Don't generate novel attack payloads beyond what the suite needs to test the defense.

## Evidence rules

Cite papers for methods (`[paper]` with arXiv/DOI). When recommending a threshold, say whether it comes from literature, an empirical measurement, or is a convention. If it's a convention, say so.

## Common mistakes

| Mistake | Fix |
|---|---|
| "Pass rate dropped 85%→80%, regression" on 20 items | That's 1 item. Run McNemar or report the Wilson CI |
| Comparing p95 across runs with n<50 | Report median + bootstrap CI; label p95 as unstable |
| Shipping a judge validated on 10 examples | ≥50 human-labeled, agreement reported with CI |
| Judge score with no judge version | Store model + prompt + rubric version with the score |
| Treating `None` (not applicable) as fail | Exclude from denominator; report coverage separately |
| Unpaired tests on paired data | Same items in both runs → paired tests |
