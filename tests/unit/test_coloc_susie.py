"""Tests for SuSiE-based colocalization (coloc-SuSiE)."""

from __future__ import annotations

import numpy as np

from post_gwas_causal.coloc.abf import coloc_abf
from post_gwas_causal.coloc.susie import coloc_susie
from post_gwas_causal.simulate import ColocConfig, ar1_ld_matrix, simulate_coloc


def _synthesize_z(
    ld: np.ndarray, n: int, causal: dict[int, float], rng: np.random.Generator
) -> np.ndarray:
    """Marginal Z-scores under the RSS model ``z ~ N(sqrt(n) R b, R)``."""
    p = ld.shape[0]
    b = np.zeros(p)
    for idx, eff in causal.items():
        b[idx] = eff
    mean = np.sqrt(n) * (ld @ b)
    chol = np.linalg.cholesky(ld + 1e-6 * np.eye(p))
    return mean + chol @ rng.standard_normal(p)


def test_shared_single_signal_matches_coloc_abf() -> None:
    """On a one-causal-variant shared locus, coloc-SuSiE agrees with coloc.abf."""
    d = simulate_coloc(
        ColocConfig(
            n_snps=60,
            n_samples=30000,
            rho=0.6,
            shared_causal=True,
            causal_idx_1=25,
            effect_1=0.12,
            effect_2=0.12,
            seed=1,
        )
    )
    cs = coloc_susie(d.trait1.z_arr, d.trait2.z_arr, d.ld_arr, d.trait1.n, d.trait2.n)
    ab = coloc_abf(d.trait1.beta_arr, d.trait1.se_arr, d.trait2.beta_arr, d.trait2.se_arr)
    assert cs.best_pp_h4 > 0.8
    assert abs(cs.best_pp_h4 - ab.pp_h4) < 0.1
    assert cs.n_cs1 == 1 and cs.n_cs2 == 1


def test_distinct_single_signal_low_pp_h4() -> None:
    """Distinct causal variants give low colocalization probability."""
    d = simulate_coloc(
        ColocConfig(
            n_snps=60,
            n_samples=40000,
            rho=0.4,
            shared_causal=False,
            causal_idx_1=10,
            causal_idx_2=45,
            effect_1=0.18,
            effect_2=0.18,
            seed=6,
        )
    )
    cs = coloc_susie(d.trait1.z_arr, d.trait2.z_arr, d.ld_arr, d.trait1.n, d.trait2.n)
    assert cs.best_pp_h4 < 0.5


def test_two_signal_locus_recovers_shared_pair() -> None:
    """With two independent signals in trait 1, coloc-SuSiE isolates the shared one.

    Trait 1 has causal variants at SNP 12 and SNP 45; trait 2 is causal only at
    SNP 45. The headline colocalization should be high (the shared 45 signal),
    and the best pair should point at a variant near SNP 45 -- something a
    single-causal locus-wide test cannot resolve cleanly.
    """
    rng = np.random.default_rng(7)
    n = 50000
    ld = ar1_ld_matrix(60, rho=0.45)  # signals at 12 and 45 are ~independent
    z1 = _synthesize_z(ld, n, {12: 0.10, 45: 0.10}, rng)
    z2 = _synthesize_z(ld, n, {45: 0.11}, rng)

    cs = coloc_susie(z1, z2, ld, n, n, n_effects=5)
    assert cs.n_cs1 >= 2  # SuSiE separates the two trait-1 signals
    assert cs.best_pp_h4 > 0.7
    top = cs.pairs[0]
    assert abs(top.best_snp - 45) <= 3


def test_pairs_sorted_and_bounded() -> None:
    """Pairs are sorted by descending PP.H4 and posteriors stay in [0, 1]."""
    d = simulate_coloc(ColocConfig(n_snps=50, n_samples=25000, shared_causal=True, seed=3))
    cs = coloc_susie(d.trait1.z_arr, d.trait2.z_arr, d.ld_arr, d.trait1.n, d.trait2.n)
    h4s = [p.pp_h4 for p in cs.pairs]
    assert h4s == sorted(h4s, reverse=True)
    for pair in cs.pairs:
        assert 0.0 <= pair.pp_h4 <= 1.0
        assert 0.0 <= pair.pp_h3 <= 1.0


def test_no_signal_gives_no_pairs() -> None:
    """A locus with no real signal yields no credible sets and best_pp_h4 == 0."""
    rng = np.random.default_rng(0)
    ld = ar1_ld_matrix(40, rho=0.3)
    z1 = np.linalg.cholesky(ld + 1e-6 * np.eye(40)) @ rng.standard_normal(40)
    z2 = np.linalg.cholesky(ld + 1e-6 * np.eye(40)) @ rng.standard_normal(40)
    cs = coloc_susie(z1, z2, ld, 20000, 20000)
    assert cs.best_pp_h4 == 0.0 or len(cs.pairs) == 0


def test_susie_exposes_lbf() -> None:
    """finemap_susie now exposes per-effect log Bayes factors aligned to alpha."""
    from post_gwas_causal.finemap.susie import finemap_susie

    rng = np.random.default_rng(1)
    ld = ar1_ld_matrix(40, rho=0.5)
    z = _synthesize_z(ld, 30000, {20: 0.12}, rng)
    res = finemap_susie(z, ld, 30000, n_effects=5)
    assert res.lbf_arr.shape == (5, 40)
    assert len(res.cs_effects) == len(res.credible_sets)
