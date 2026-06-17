"""Known-answer tests for the MR estimators and diagnostics."""

from __future__ import annotations

import numpy as np

from post_gwas_causal.mr.egger import mr_egger
from post_gwas_causal.mr.harmonize import harmonize
from post_gwas_causal.mr.heterogeneity import cochran_q
from post_gwas_causal.mr.ivw import mr_ivw
from post_gwas_causal.mr.weighted_median import mr_weighted_median
from post_gwas_causal.simulate import MRConfig, simulate_mr


def test_ivw_recovers_true_effect() -> None:
    data = simulate_mr(MRConfig(causal_effect=0.3, n_instruments=40, seed=1))
    res = mr_ivw(data.bx, data.by, data.byse)
    assert abs(res.beta - 0.3) < 0.05
    assert res.ci_low < 0.3 < res.ci_high


def test_ivw_equals_closed_form_weighted_slope() -> None:
    data = simulate_mr(MRConfig(causal_effect=0.25, seed=2))
    res = mr_ivw(data.bx, data.by, data.byse, method="fixed")
    w = 1.0 / data.byse**2
    closed_form = np.sum(w * data.bx * data.by) / np.sum(w * data.bx**2)
    assert np.isclose(res.beta, closed_form)


def test_egger_intercept_zero_under_no_pleiotropy() -> None:
    data = simulate_mr(MRConfig(causal_effect=0.3, pleiotropy=0.0, n_instruments=40, seed=3))
    res = mr_egger(data.bx, data.by, data.byse)
    # Intercept CI should cover zero when there is no directional pleiotropy.
    assert res.intercept_ci_low < 0.0 < res.intercept_ci_high
    # And the slope should still recover the causal effect.
    assert abs(res.slope - 0.3) < 0.1


def test_egger_intercept_nonzero_under_directional_pleiotropy() -> None:
    # Positive mean exposure effect keeps instruments co-oriented so the
    # injected directional pleiotropy does not cancel under MR-Egger's
    # positive-exposure coding.
    data = simulate_mr(
        MRConfig(
            causal_effect=0.3,
            pleiotropy=0.05,
            n_instruments=40,
            gamma_mean=0.15,
            gamma_sd=0.05,
            seed=4,
        )
    )
    res = mr_egger(data.bx, data.by, data.byse)
    # Injected directional pleiotropy should push the intercept off zero.
    assert res.intercept > 0.0
    assert res.intercept_pvalue < 0.05


def test_weighted_median_robust_to_one_invalid_instrument() -> None:
    cfg = MRConfig(
        causal_effect=0.3,
        n_instruments=30,
        invalid_idx=[0],
        invalid_effect=0.6,
        seed=5,
    )
    data = simulate_mr(cfg)
    ivw = mr_ivw(data.bx, data.by, data.byse)
    wm = mr_weighted_median(data.bx, data.by, data.bxse, data.byse, n_bootstrap=300)
    # The single strong invalid instrument biases IVW more than the median.
    assert abs(wm.beta - 0.3) < abs(ivw.beta - 0.3)
    assert abs(wm.beta - 0.3) < 0.1


def test_cochran_q_low_heterogeneity_under_valid_instruments() -> None:
    data = simulate_mr(MRConfig(causal_effect=0.3, pleiotropy=0.0, n_instruments=40, seed=6))
    q = cochran_q(data.bx, data.by, data.byse)
    assert q.pvalue > 0.05
    assert q.i_squared < 0.3


def test_cochran_q_high_heterogeneity_with_pleiotropy() -> None:
    data = simulate_mr(
        MRConfig(
            causal_effect=0.3, n_instruments=40, invalid_idx=[0, 1, 2], invalid_effect=0.5, seed=7
        )
    )
    q = cochran_q(data.bx, data.by, data.byse)
    assert q.pvalue < 0.05


def test_harmonize_flips_and_drops() -> None:
    res = harmonize(
        rsid=["rs1", "rs2", "rs3", "rs4"],
        beta_exposure=np.array([0.1, 0.2, 0.3, 0.4]),
        se_exposure=np.array([0.01, 0.01, 0.01, 0.01]),
        effect_allele_exposure=["A", "C", "G", "A"],
        other_allele_exposure=["G", "T", "A", "T"],  # rs4 is palindromic A/T
        beta_outcome=np.array([0.1, -0.2, 0.3, 0.4]),
        se_outcome=np.array([0.01, 0.01, 0.01, 0.01]),
        effect_allele_outcome=["A", "T", "G", "A"],  # rs2 outcome allele flipped
        other_allele_outcome=["G", "C", "A", "T"],
    )
    # rs4 (palindromic) dropped.
    assert res.n_dropped == 1
    assert "rs4" not in res.rsid
    # rs2 outcome sign flipped (effect allele was the other allele).
    idx = res.rsid.index("rs2")
    assert res.by[idx] > 0  # original -0.2 flipped to +0.2
    assert res.n_flipped == 1
