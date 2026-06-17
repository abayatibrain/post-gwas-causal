"""Simplified SuSiE-style fine-mapping from summary statistics (SuSiE-RSS).

SuSiE ("Sum of Single Effects") models the vector of joint effects as a sum of
``L`` *single-effect* vectors, each of which puts all its mass on one SNP. It
fits the model by Iterative Bayesian Stepwise Selection (IBSS): cycle over the
``L`` effects, and for each one compute single-SNP Bayes factors on the
*residualized* signal (the marginal Z-scores minus the contribution of the
other ``L-1`` effects), giving a posterior distribution over which SNP that
effect lands on.

This implementation is the summary-statistics ("RSS") form. It consumes:

* ``z`` — marginal Z-scores,
* ``R`` — the LD (correlation) matrix,
* ``n`` — the GWAS sample size.

For each single effect ``l`` it produces a categorical posterior ``alpha_l``
over SNPs; the per-SNP posterior inclusion probability is
``PIP_j = 1 - prod_l (1 - alpha_{l,j})``. Credible sets are formed per single
effect.

References
----------
Wang, G., Sarkar, A., Carbonetto, P., & Stephens, M. (2020). A simple new
approach to variable selection in regression, with application to genetic
fine mapping. *JRSS-B* 82(5):1273-1300.

Zou, Y., Carbonetto, P., Wang, G., & Stephens, M. (2022). Fine-mapping from
summary data with the "Sum of Single Effects" model. *PLoS Genetics* 18(7).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

__all__ = ["SusieResult", "finemap_susie"]


class SusieResult(BaseModel):
    """Result of SuSiE-style multi-effect fine-mapping.

    Attributes
    ----------
    pip
        Per-SNP posterior inclusion probability, ``1 - prod_l (1 - alpha_l)``.
    alpha
        ``L x p`` posterior assignment matrix; ``alpha[l]`` is the categorical
        posterior over SNPs for single effect ``l``.
    credible_sets
        One credible set (list of SNP indices) per single effect that is
        "in play". Purity-filtered sets are returned.
    n_iter
        Number of IBSS iterations run before convergence.
    converged
        Whether the ELBO-surrogate (max change in alpha) fell below the
        tolerance before ``max_iter``.
    """

    pip: list[float]
    alpha: list[list[float]]
    credible_sets: list[list[int]]
    n_iter: int
    converged: bool

    @property
    def pip_arr(self) -> np.ndarray:
        """Posterior inclusion probabilities as a NumPy array."""
        return np.asarray(self.pip, dtype=float)


def _single_effect_posterior(
    xtr: np.ndarray,
    n: int,
    prior_variance: float,
    residual_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Posterior for one single effect given a residualized association vector.

    ``xtr`` plays the role of ``X' r`` (the inner product between each
    standardized genotype column and the current residual). Under standardized
    genotypes ``X'X`` has unit diagonal, so each SNP's univariate effect
    estimate is ``b_hat_j = xtr_j / n`` with variance ``sigma2 / n``.

    Parameters
    ----------
    xtr
        Length-``p`` vector of genotype-residual inner products.
    n
        Sample size.
    prior_variance
        Prior variance on the single effect.
    residual_variance
        Current residual variance ``sigma2``.

    Returns
    -------
    tuple
        ``(alpha, post_mean)`` — the categorical posterior over SNPs and the
        posterior mean effect *conditional on* each SNP being the causal one.
    """
    shat2 = residual_variance / n  # variance of each univariate estimate
    bhat = xtr / n

    # Posterior variance / mean of the effect given SNP j is the one.
    post_var = 1.0 / (1.0 / prior_variance + 1.0 / shat2)
    post_mean = post_var * bhat / shat2

    # Log Bayes factor per SNP (Wakefield form).
    z2 = bhat**2 / shat2
    r = prior_variance / (prior_variance + shat2)
    log_bf = 0.5 * np.log(1.0 - r) + 0.5 * z2 * r

    m = np.max(log_bf)
    w = np.exp(log_bf - m)
    alpha = w / w.sum()
    return alpha, post_mean


def _purity(cs: list[int], abs_r: np.ndarray) -> float:
    """Minimum absolute correlation among SNPs in a credible set."""
    if len(cs) <= 1:
        return 1.0
    sub = abs_r[np.ix_(cs, cs)]
    return float(sub[np.triu_indices(len(cs), k=1)].min())


def finemap_susie(
    z: np.ndarray,
    ld: np.ndarray,
    n: int,
    *,
    n_effects: int = 5,
    prior_variance: float = 0.1,
    residual_variance: float = 1.0,
    coverage: float = 0.95,
    max_iter: int = 100,
    tol: float = 1e-4,
    min_purity: float = 0.5,
) -> SusieResult:
    """Fine-map a locus with a simplified SuSiE-RSS model.

    Parameters
    ----------
    z
        Marginal GWAS Z-scores, length ``p``.
    ld
        ``p x p`` LD correlation matrix.
    n
        GWAS sample size.
    n_effects
        Number of single effects ``L`` to fit (upper bound on causal variants).
    prior_variance
        Prior variance on each single effect.
    residual_variance
        Residual variance ``sigma2`` (held fixed in this implementation).
    coverage
        Target coverage of each per-effect credible set.
    max_iter
        Maximum IBSS iterations.
    tol
        Convergence tolerance on the max change in any ``alpha`` entry.
    min_purity
        Minimum within-set absolute LD for a credible set to be retained.

    Returns
    -------
    SusieResult
        PIPs, posterior assignment matrix, and purity-filtered credible sets.

    Notes
    -----
    The model works on the scaled association statistic ``X'y ≈ sqrt(n) * z``
    (standardized genotypes and phenotype), with ``X'X = n R``. The residual
    inner product for effect ``l`` is

    ``X'r_l = sqrt(n) z - n R (mu_total - mu_l)``

    where ``mu_l = alpha_l * post_mean_l`` is the expected effect contributed
    by single effect ``l``.
    """
    z = np.asarray(z, dtype=float)
    ld = np.asarray(ld, dtype=float)
    p = z.shape[0]
    if ld.shape != (p, p):
        raise ValueError("ld must be a (p, p) matrix matching z")

    xty = np.sqrt(n) * z  # X'y under standardized genotypes/phenotype

    alpha = np.zeros((n_effects, p))
    mu = np.zeros((n_effects, p))  # posterior mean conditional on each SNP
    b = np.zeros((n_effects, p))  # expected effect = alpha * mu per effect

    converged = False
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        max_change = 0.0
        for eff in range(n_effects):
            # Fitted contribution of all *other* effects.
            b_other = b.sum(axis=0) - b[eff]
            # Residual inner product X'(y - X b_other) = X'y - (X'X) b_other.
            xtr = xty - n * (ld @ b_other)
            new_alpha, new_mu = _single_effect_posterior(xtr, n, prior_variance, residual_variance)
            max_change = max(max_change, float(np.max(np.abs(new_alpha - alpha[eff]))))
            alpha[eff] = new_alpha
            mu[eff] = new_mu
            b[eff] = new_alpha * new_mu
        if max_change < tol:
            converged = True
            break

    pip = 1.0 - np.prod(1.0 - alpha, axis=0)

    abs_r = np.abs(ld)
    credible_sets: list[list[int]] = []
    for eff in range(n_effects):
        order = np.argsort(alpha[eff])[::-1]
        cumulative = np.cumsum(alpha[eff][order])
        k = int(np.searchsorted(cumulative, coverage)) + 1
        k = min(k, p)
        cs = order[:k].astype(int).tolist()
        # Keep only effects that actually concentrate mass and are pure.
        if alpha[eff].max() < 1.0 / p * 2:
            continue
        if _purity(cs, abs_r) >= min_purity:
            credible_sets.append(cs)

    return SusieResult(
        pip=pip.tolist(),
        alpha=alpha.tolist(),
        credible_sets=credible_sets,
        n_iter=n_iter,
        converged=converged,
    )
