import random

import pytest

from warden_sdk.evals import stats


def test_mcnemar_exact_matches_hand_computed_values():
    # All discordant pairs one way: p = 2 * 0.5^n.
    assert stats.mcnemar_exact(6, 0) == pytest.approx(0.03125)
    assert stats.mcnemar_exact(5, 0) == pytest.approx(0.0625)
    # 1 vs 1 is as balanced as it gets.
    assert stats.mcnemar_exact(1, 1) == 1.0
    assert stats.mcnemar_exact(0, 0) == 1.0
    # 8 broke, 1 fixed: 2 * (1 + 9) / 512.
    assert stats.mcnemar_exact(8, 1) == pytest.approx(20 / 512)


def test_min_discordant_is_six_at_five_percent():
    assert stats.min_discordant_for_significance(0.05) == 6


def test_sign_flip_reduces_to_mcnemar_for_binary_diffs():
    rng = random.Random(0)
    diffs = [-1] * 6 + [0] * 14
    assert stats.sign_flip_test(diffs, rng) == pytest.approx(stats.mcnemar_exact(6, 0))
    diffs = [-1] * 8 + [1] + [0] * 5
    assert stats.sign_flip_test(diffs, rng) == pytest.approx(stats.mcnemar_exact(8, 1))


def test_sign_flip_no_change():
    assert stats.sign_flip_test([0, 0, 0], random.Random(0)) == 1.0


def test_sign_flip_monte_carlo_is_seeded_and_small_for_a_large_shift():
    diffs = [-0.5] * 30
    a = stats.sign_flip_test(diffs, random.Random(1))
    b = stats.sign_flip_test(diffs, random.Random(1))
    assert a == b
    assert a < 0.001


def test_benjamini_hochberg():
    adjusted = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    # Sorted: 0.01·4/1, 0.03·4/2, 0.04·4/3, 0.2·4/4, then a running minimum from the top.
    assert adjusted == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.2])
    # Monotone in the raw p-value and never below it.
    raw = [0.001, 0.5, 0.02, 0.03]
    for p, q in zip(raw, stats.benjamini_hochberg(raw)):
        assert q >= p


def test_pass_k_estimators():
    # 3 trials, 2 passed.
    assert stats.pass_hat_k(3, 2, 1) == pytest.approx(2 / 3)
    assert stats.pass_hat_k(3, 2, 3) == 0.0
    assert stats.pass_hat_k(3, 3, 3) == 1.0
    assert stats.pass_at_k(3, 2, 1) == pytest.approx(2 / 3)
    assert stats.pass_at_k(3, 1, 2) == pytest.approx(2 / 3)
    assert stats.pass_at_k(3, 0, 3) == 0.0


def test_bootstrap_ci_contains_the_estimate_and_is_seeded():
    values = [0.0, 1.0] * 20
    mean = lambda idx: sum(values[i] for i in idx) / len(idx)  # noqa: E731
    ci = stats.bootstrap_ci(len(values), mean, random.Random(3))
    assert ci == stats.bootstrap_ci(len(values), mean, random.Random(3))
    lo, hi = ci
    assert lo < 0.5 < hi
    assert 0.25 < lo and hi < 0.75
