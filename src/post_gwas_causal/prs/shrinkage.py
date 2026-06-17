"""LDpred-inf-style infinitesimal Bayesian shrinkage of GWAS effect sizes.

Under the infinitesimal prior every SNP has a small normally distributed effect
``beta_j ~ N(0, h2 / M)`` where ``h2`` is the SNP heritability and ``M`` the
number of SNPs. Conditioning on the marginal (LD-correlated) effect estimates
yields posterior-mean joint effects

.. math::

    \\tilde\\beta = \\left(R + \\frac{M}{N\\, h^2} I\\right)^{-1} \\hat\\beta_{\\mathrm{marg}},

where ``R`` is the LD matrix, ``N`` the GWAS sample size, and
``hat_beta_marg`` the marginal (standardized) effect estimates. The shrinkage
term ``M / (N h2)`` down-weights noise and de-correlates the LD-smeared signal,
typically giving better-calibrated polygenic scores than raw C+T when the trait
is highly polygenic.

References
----------
Vilhjálmsson, B.J. et al. (2015). Modeling linkage disequilibrium increases
accuracy of polygenic risk scores. *American Journal of Human Genetics*
97(4):576-592.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ldpred_inf", "score_individuals"]


def ldpred_inf(
    marginal_beta: np.ndarray,
    ld: np.ndarray,
    n_samples: int,
    heritability: float,
    *,
    n_snps: int | None = None,
) -> np.ndarray:
    r"""Infinitesimal-model posterior-mean (joint) effect sizes.

    Parameters
    ----------
    marginal_beta
        Marginal (per-SNP, standardized) effect estimates from the GWAS.
    ld
        LD correlation matrix ``R`` for the same SNPs.
    n_samples
        GWAS sample size ``N``.
    heritability
        SNP heritability ``h^2`` explained by the locus / SNP set.
    n_snps
        Number of SNPs ``M`` for the prior. Defaults to ``len(marginal_beta)``.

    Returns
    -------
    numpy.ndarray
        Shrunk joint effect sizes ``(R + (M / (N h2)) I)^{-1} beta_marg``.
    """
    marginal_beta = np.asarray(marginal_beta, dtype=float)
    ld = np.asarray(ld, dtype=float)
    m = n_snps if n_snps is not None else marginal_beta.shape[0]
    if heritability <= 0.0:
        raise ValueError("heritability must be positive")

    shrinkage = m / (n_samples * heritability)
    a = ld + shrinkage * np.eye(ld.shape[0])
    return np.linalg.solve(a, marginal_beta)


def score_individuals(genotypes: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Polygenic score per individual = genotype-weighted sum of effects.

    Parameters
    ----------
    genotypes
        Dosage matrix, shape ``(n_individuals, n_snps)``.
    beta
        Per-SNP effect sizes.

    Returns
    -------
    numpy.ndarray
        Per-individual scores, length ``n_individuals``.
    """
    genotypes = np.asarray(genotypes, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if genotypes.shape[1] != beta.shape[0]:
        raise ValueError("genotypes second dimension must equal len(beta)")
    return genotypes @ beta
