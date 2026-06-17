"""Known-answer tests for ABF and SuSiE fine-mapping."""

from __future__ import annotations

import numpy as np

from post_gwas_causal.finemap.abf import finemap_abf, log_abf
from post_gwas_causal.finemap.susie import finemap_susie
from post_gwas_causal.simulate import LocusConfig, simulate_locus


def test_log_abf_increases_with_z() -> None:
    se = np.array([0.02, 0.02])
    beta_small = np.array([0.01, 0.01])
    beta_large = np.array([0.2, 0.2])
    assert (log_abf(beta_large, se) > log_abf(beta_small, se)).all()


def test_abf_pip_sums_to_one_and_finds_causal() -> None:
    causal = 25
    stats, _ = simulate_locus(
        LocusConfig(n_snps=60, causal_idx=[causal], causal_effects=[0.2], seed=7)
    )
    res = finemap_abf(stats.beta_arr, stats.se_arr)
    assert np.isclose(sum(res.pip), 1.0)
    # 95% credible set should contain the true causal SNP.
    assert causal in res.credible_set
    # And the top SNP should be at or very near the causal one.
    assert abs(res.top_snp - causal) <= 2


def test_abf_credible_set_coverage() -> None:
    stats, _ = simulate_locus(
        LocusConfig(n_snps=50, causal_idx=[25], causal_effects=[0.18], seed=11)
    )
    res = finemap_abf(stats.beta_arr, stats.se_arr, coverage=0.95)
    pip = res.pip_arr
    assert pip[res.credible_set].sum() >= 0.95


def test_susie_single_effect_finds_causal() -> None:
    causal = 30
    stats, ld = simulate_locus(
        LocusConfig(n_snps=60, n_samples=20000, causal_idx=[causal], causal_effects=[0.2], seed=5)
    )
    res = finemap_susie(stats.z_arr, ld, stats.n, n_effects=3)
    assert res.converged
    # Highest-PIP SNP should be at/near the causal SNP.
    assert abs(int(np.argmax(res.pip_arr)) - causal) <= 2
    # At least one credible set should contain the causal SNP.
    assert any(causal in cs for cs in res.credible_sets)


def test_susie_two_causal_variants() -> None:
    stats, ld = simulate_locus(
        LocusConfig(
            n_snps=80,
            n_samples=40000,
            causal_idx=[15, 60],
            causal_effects=[0.2, 0.2],
            rho=0.4,
            seed=9,
        )
    )
    res = finemap_susie(stats.z_arr, ld, stats.n, n_effects=5)
    pip = res.pip_arr
    # Both causal regions should carry substantial posterior mass.
    assert pip[12:19].max() > 0.3
    assert pip[57:64].max() > 0.3
