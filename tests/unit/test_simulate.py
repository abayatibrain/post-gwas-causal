"""Tests for the locus / coloc / MR simulators."""

from __future__ import annotations

import numpy as np

from post_gwas_causal.simulate import (
    ColocConfig,
    LocusConfig,
    MRConfig,
    ar1_ld_matrix,
    block_ld_matrix,
    simulate_coloc,
    simulate_locus,
    simulate_mr,
)


def test_ar1_ld_matrix_structure() -> None:
    r = ar1_ld_matrix(5, 0.5)
    assert r.shape == (5, 5)
    assert np.allclose(np.diag(r), 1.0)
    assert np.isclose(r[0, 1], 0.5)
    assert np.isclose(r[0, 2], 0.25)
    # Positive semi-definite.
    assert np.linalg.eigvalsh(r).min() > -1e-9


def test_block_ld_matrix_is_pd() -> None:
    r = block_ld_matrix(10, block_size=5, block_rho=0.8)
    assert r.shape == (10, 10)
    assert np.allclose(np.diag(r), 1.0)
    assert np.linalg.eigvalsh(r).min() > 0


def test_simulate_locus_shapes_and_top_snp() -> None:
    stats, ld = simulate_locus(LocusConfig(n_snps=40, causal_idx=[20], seed=1))
    assert len(stats.beta) == 40
    assert len(stats.rsid) == 40
    assert ld.shape == (40, 40)
    # The strongest signal should be near the causal SNP.
    top = int(np.argmax(np.abs(stats.z_arr)))
    assert abs(top - 20) <= 3


def test_simulate_coloc_shared_vs_distinct() -> None:
    shared = simulate_coloc(ColocConfig(shared_causal=True, seed=2))
    distinct = simulate_coloc(ColocConfig(shared_causal=False, seed=2))
    assert shared.causal_idx_1 == shared.causal_idx_2
    assert distinct.causal_idx_1 != distinct.causal_idx_2


def test_simulate_mr_recovers_signal() -> None:
    data = simulate_mr(MRConfig(causal_effect=0.3, seed=3))
    assert data.true_effect == 0.3
    # Mean Wald ratio should be in the right ballpark.
    ratio = data.by / data.bx
    assert abs(np.median(ratio) - 0.3) < 0.25
