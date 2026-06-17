"""Known-answer tests for Giambartolomei coloc.abf."""

from __future__ import annotations

import numpy as np

from post_gwas_causal.coloc.abf import coloc_abf
from post_gwas_causal.simulate import ColocConfig, simulate_coloc


def test_shared_causal_gives_high_pp_h4() -> None:
    data = simulate_coloc(
        ColocConfig(shared_causal=True, effect_1=0.15, effect_2=0.15, n_samples=30000, seed=4)
    )
    res = coloc_abf(
        data.trait1.beta_arr,
        data.trait1.se_arr,
        data.trait2.beta_arr,
        data.trait2.se_arr,
    )
    pp = res.as_dict()
    assert pp["PP.H4"] > 0.8
    assert res.best_snp_h4 == data.causal_idx_1 or abs(res.best_snp_h4 - data.causal_idx_1) <= 2


def test_distinct_causal_gives_high_pp_h3() -> None:
    data = simulate_coloc(
        ColocConfig(
            shared_causal=False,
            causal_idx_1=10,
            causal_idx_2=45,
            effect_1=0.18,
            effect_2=0.18,
            n_samples=40000,
            rho=0.4,
            seed=6,
        )
    )
    res = coloc_abf(
        data.trait1.beta_arr,
        data.trait1.se_arr,
        data.trait2.beta_arr,
        data.trait2.se_arr,
    )
    pp = res.as_dict()
    assert pp["PP.H3"] > pp["PP.H4"]
    assert pp["PP.H3"] > 0.5


def test_posteriors_sum_to_one() -> None:
    data = simulate_coloc(ColocConfig(shared_causal=True, seed=1))
    res = coloc_abf(
        data.trait1.beta_arr,
        data.trait1.se_arr,
        data.trait2.beta_arr,
        data.trait2.se_arr,
    )
    assert np.isclose(sum(res.as_dict().values()), 1.0)
