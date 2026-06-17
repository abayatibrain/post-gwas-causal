"""Tests for C+T and shrinkage polygenic scores."""

from __future__ import annotations

import numpy as np

from post_gwas_causal.prs.clump_threshold import clump, clump_and_threshold
from post_gwas_causal.prs.evaluate import evaluate_prs, roc_auc
from post_gwas_causal.prs.shrinkage import ldpred_inf, score_individuals


def _simulate_target(seed: int = 0, n: int = 800, p: int = 40):
    rng = np.random.default_rng(seed)
    causal = np.zeros(p)
    causal[[5, 15, 25, 35]] = [0.5, -0.4, 0.45, 0.3]
    geno = rng.binomial(2, 0.3, size=(n, p)).astype(float)
    pheno = geno @ causal + rng.standard_normal(n)
    # Marginal GWAS estimates.
    beta = np.empty(p)
    pval = np.empty(p)
    from scipy import stats as sps

    for j in range(p):
        g = geno[:, j] - geno[:, j].mean()
        denom = float(g @ g)
        b = float(g @ (pheno - pheno.mean()) / denom)
        resid = pheno - b * g
        se = float(np.sqrt(resid.var(ddof=2) / denom))
        beta[j] = b
        pval[j] = float(2.0 * sps.norm.sf(abs(b / se)))
    ld = np.corrcoef(geno, rowvar=False)
    return geno, pheno, beta, pval, ld, causal


def test_clump_returns_independent_index_snps() -> None:
    pvals = np.array([1e-8, 1e-2, 1e-7, 0.5])
    # SNP 0 and 2 correlated; SNP 0 wins, removes 2.
    ld = np.array(
        [
            [1.0, 0.0, 0.9, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    idx = clump(pvals, ld, clump_r2=0.5)
    assert 0 in idx
    assert 2 not in idx  # removed as satellite of SNP 0


def test_ct_prs_correlates_with_phenotype() -> None:
    geno, pheno, beta, pval, ld, _ = _simulate_target(seed=1)
    res = clump_and_threshold(beta, pval, ld, geno, clump_r2=0.1, p_threshold=0.05)
    ev = evaluate_prs(res.scores_arr, pheno)
    assert res.n_selected >= 2
    assert ev.pearson_r > 0.3


def test_ldpred_inf_shrinks_and_scores() -> None:
    geno, pheno, beta, _, ld, _ = _simulate_target(seed=2)
    shrunk = ldpred_inf(beta, ld, n_samples=geno.shape[0], heritability=0.4)
    assert shrunk.shape == beta.shape
    scores = score_individuals(geno, shrunk)
    ev = evaluate_prs(scores, pheno)
    assert ev.pearson_r > 0.3


def test_roc_auc_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert np.isclose(roc_auc(scores, labels), 1.0)


def test_evaluate_binary_reports_auc() -> None:
    rng = np.random.default_rng(3)
    scores = rng.standard_normal(200)
    labels = (scores + 0.5 * rng.standard_normal(200) > 0).astype(int)
    ev = evaluate_prs(scores, labels)
    assert ev.auc is not None
    assert ev.auc > 0.7
