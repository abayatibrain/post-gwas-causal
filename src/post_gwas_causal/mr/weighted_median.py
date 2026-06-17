"""Weighted-median Mendelian-randomization estimator.

The weighted-median estimator orders the per-instrument Wald ratios
``by_j / bx_j`` and takes the value at which the cumulative *weight* reaches
50%. Weights are the inverse-variance weights of the ratios. The estimator is
consistent as long as instruments carrying at least half of the total weight
are valid, making it robust to a minority of invalid (pleiotropic) instruments
— unlike IVW, which a single strong outlier can drag off.

The standard error is obtained by parametric bootstrap: resample each
instrument's exposure and outcome effects from their sampling distributions and
recompute the weighted median.

References
----------
Bowden, J., Davey Smith, G., Haycock, P.C., & Burgess, S. (2016). Consistent
estimation in Mendelian randomization with some invalid instruments using a
weighted median estimator. *Genetic Epidemiology* 40(4):304-314.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats

__all__ = ["WeightedMedianResult", "mr_weighted_median"]


class WeightedMedianResult(BaseModel):
    """Weighted-median MR estimate.

    Attributes
    ----------
    beta
        Weighted-median causal-effect estimate.
    se
        Bootstrap standard error.
    z, pvalue
        Wald Z-statistic and two-sided p-value.
    ci_low, ci_high
        95% confidence interval.
    n_snps
        Number of instruments.
    n_bootstrap
        Number of bootstrap replicates used for the SE.
    """

    beta: float
    se: float
    z: float
    pvalue: float
    ci_low: float
    ci_high: float
    n_snps: int
    n_bootstrap: int


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of ``values`` using cumulative ``weights``.

    Uses the standard interpolated definition (Bowden 2016): order the values,
    accumulate normalized weights, and linearly interpolate to the point where
    the cumulative weight (shifted by half the current weight) crosses 0.5.
    """
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    p = np.cumsum(w) - 0.5 * w
    p /= np.sum(w)
    # Locate the bracket around 0.5.
    below = np.where(p < 0.5)[0]
    if below.size == 0:
        return float(v[0])
    k = below[-1]
    if k == len(v) - 1:
        return float(v[-1])
    frac = (0.5 - p[k]) / (p[k + 1] - p[k])
    return float(v[k] + frac * (v[k + 1] - v[k]))


def mr_weighted_median(
    bx: np.ndarray,
    by: np.ndarray,
    bxse: np.ndarray,
    byse: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> WeightedMedianResult:
    """Weighted-median MR estimate with a bootstrap standard error.

    Parameters
    ----------
    bx
        Per-instrument exposure effects.
    by
        Per-instrument outcome effects.
    bxse
        Standard errors of the exposure effects.
    byse
        Standard errors of the outcome effects.
    n_bootstrap
        Number of parametric-bootstrap replicates for the SE.
    seed
        RNG seed for the bootstrap.

    Returns
    -------
    WeightedMedianResult
        Estimate, bootstrap SE, Z, p-value, and 95% CI.

    Notes
    -----
    The per-instrument Wald ratio is ``by_j / bx_j`` and its inverse-variance
    weight (delta method, leading term) is ``bx_j**2 / byse_j**2``.
    """
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    bxse = np.asarray(bxse, dtype=float)
    byse = np.asarray(byse, dtype=float)
    if not (bx.shape == by.shape == bxse.shape == byse.shape):
        raise ValueError("bx, by, bxse, byse must share the same shape")
    m = bx.shape[0]
    if m < 3:
        raise ValueError("weighted median requires at least three instruments")

    ratio = by / bx
    weights = bx**2 / byse**2
    beta = _weighted_median(ratio, weights)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        bx_b = bx + bxse * rng.standard_normal(m)
        by_b = by + byse * rng.standard_normal(m)
        ratio_b = by_b / bx_b
        weights_b = bx_b**2 / byse**2
        boot[b] = _weighted_median(ratio_b, weights_b)

    se = float(np.std(boot, ddof=1))
    z = beta / se if se > 0 else np.inf
    pvalue = float(2.0 * stats.norm.sf(abs(z)))
    return WeightedMedianResult(
        beta=float(beta),
        se=se,
        z=float(z),
        pvalue=pvalue,
        ci_low=float(beta - 1.96 * se),
        ci_high=float(beta + 1.96 * se),
        n_snps=m,
        n_bootstrap=n_bootstrap,
    )
