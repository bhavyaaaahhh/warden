"""Statistics for comparing two eval runs. Pure functions, standard library only.

The methods and thresholds are specified in docs/research/evaluation-methodology.md.
Every function that resamples takes a seeded Random, so the same comparison
always prints the same numbers.
"""

import math
import random
from collections.abc import Callable, Sequence
from itertools import product

RESAMPLES = 10_000
# Enumerate every sign flip exactly up to this many nonzero differences (2^16 = 65k).
EXACT_FLIP_LIMIT = 16


def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test of k successes in n trials at p = 0.5."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar_exact(broke: int, fixed: int) -> float:
    """Exact McNemar test on the discordant pairs of a paired pass/fail comparison."""
    return binomial_two_sided(broke, broke + fixed)


def sign_flip_test(diffs: Sequence[float], rng: random.Random) -> float:
    """Paired permutation test: two-sided p-value for mean(diffs) != 0.

    Under the null each paired difference is equally likely to have either sign.
    Exact when there are few nonzero differences, Monte Carlo otherwise.
    """
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    # Tolerance so ties with the observed statistic count as "at least as extreme".
    eps = 1e-12
    if len(nonzero) <= EXACT_FLIP_LIMIT:
        hits = sum(
            abs(sum(s * d for s, d in zip(signs, nonzero))) >= observed - eps
            for signs in product((1, -1), repeat=len(nonzero))
        )
        return hits / 2 ** len(nonzero)
    hits = sum(
        abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed - eps
        for _ in range(RESAMPLES)
    )
    # +1 so a Monte Carlo p-value is never exactly 0.
    return (hits + 1) / (RESAMPLES + 1)


def bootstrap_ci(
    n: int,
    statistic: Callable[[list[int]], float | None],
    rng: random.Random,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI, resampling case indices 0..n-1 with replacement.

    `statistic` receives the resampled indices. Resampling whole cases keeps
    each case's trials together (a clustered bootstrap).
    """
    if n == 0:
        return None
    values = []
    for _ in range(RESAMPLES):
        v = statistic([rng.randrange(n) for _ in range(n)])
        if v is not None:
            values.append(v)
    if not values:
        return None
    values.sort()
    lo = values[int((1 - level) / 2 * (len(values) - 1))]
    hi = values[int((1 + level) / 2 * (len(values) - 1))]
    return lo, hi


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        adjusted[i] = running
    return adjusted


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k trials passes, from c passes in n trials."""
    if n - c < k:
        return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Chance that all k trials pass (tau-bench's pass^k), from c passes in n trials."""
    return math.comb(c, k) / math.comb(n, k)


def min_attainable_p(nonzero: int) -> float:
    """Smallest p-value a two-sided sign-flip or exact McNemar test can give with this many nonzero pairs."""
    return min(1.0, 2 / 2**nonzero)


def min_discordant_for_significance(alpha: float = 0.05) -> int:
    """Fewest discordant pairs, all in one direction, that exact McNemar can call significant."""
    m = 1
    while binomial_two_sided(0, m) >= alpha:
        m += 1
    return m


def percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Agreement between two raters beyond chance, from (rater_a, rater_b) label pairs."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({x for pair in pairs for x in pair})
    observed = sum(a == b for a, b in pairs) / n
    expected = sum(
        (sum(a == lab for a, _ in pairs) / n) * (sum(b == lab for _, b in pairs) / n) for lab in labels
    )
    if expected == 1:
        return 1.0 if observed == 1 else 0.0  # everyone gave the same single label
    return (observed - expected) / (1 - expected)
