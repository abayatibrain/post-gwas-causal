"""Inverse-variance-weighted (IVW) Mendelian-randomization estimator.

The IVW estimate is the slope of a weighted regression of the per-instrument
outcome effects ``by`` on the exposure effects ``bx`` through the origin, with
weights ``1 / byse**2``. Algebraically it equals

.. math::

    \\hat\\beta_{IVW}
        = \\frac{\\sum_j w_j\\, bx_j\\, by_j}{\\sum_j w_j\\, bx_j^2},
    \\qquad w_j = 1 / byse_j^2 .

This is also the inverse-variance-weighted mean of the per-instrument Wald
ratios ``by_j / bx_j``. The *fixed-effect* SE assumes no heterogeneity; the
*random-effect* SE inflates the fixed SE by the (over-dispersion) residual
standard error when Cochran's Q exceeds its degrees of freedom.

References
----------
Burgess, S., Butterworth, A., & Thompson, S.G. (2013). Mendelian randomization
analysis with multiple genetic variants using summarized data.
*Genetic Epidemiology* 37(7):658-665.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats

__all__ = ["IVWResult", "mr_ivw"]


class IVWResult(BaseModel):
    """Inverse-variance-weighted MR estimate.

    Attributes
    ----------
    beta
        Estimated causal effect (the weighted regression slope).
    se
        Standard error (fixed- or random-effects per ``method``).
    se_fixed
        Fixed-effect standard error.
    se_random
        Random-effect (multiplicative over-dispersion) standard error.
    z, pvalue
        Wald Z-statistic and two-sided p-value for ``beta``.
    ci_low, ci_high
        95% confidence-interval bounds.
    n_snps
        Number of instruments.
    method
        ``"fixed"`` or ``"random"``.
    """

    beta: float
    se: float
    se_fixed: float
    se_random: float
    z: float
    pvalue: float
    ci_low: float
    ci_high: float
    n_snps: int
    method: str


def mr_ivw(
    bx: np.ndarray,
    by: np.ndarray,
    byse: np.ndarray,
    *,
    method: str = "random",
) -> IVWResult:
    """Inverse-variance-weighted MR estimate.

    Parameters
    ----------
    bx
        Per-instrument exposure effect sizes.
    by
        Per-instrument outcome effect sizes.
    byse
        Standard errors of the outcome effects.
    method
        ``"fixed"`` for the fixed-effect SE, ``"random"`` (default) for the
        multiplicative random-effects SE that never shrinks below the fixed SE.

    Returns
    -------
    IVWResult
        Point estimate, standard errors, Z, p-value, and 95% CI.
    """
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    byse = np.asarray(byse, dtype=float)
    if not (bx.shape == by.shape == byse.shape):
        raise ValueError("bx, by, byse must share the same shape")
    m = bx.shape[0]
    if m < 2:
        raise ValueError("IVW requires at least two instruments")
    if method not in {"fixed", "random"}:
        raise ValueError("method must be 'fixed' or 'random'")

    w = 1.0 / byse**2
    beta = float(np.sum(w * bx * by) / np.sum(w * bx**2))
    se_fixed = float(np.sqrt(1.0 / np.sum(w * bx**2)))

    # Multiplicative over-dispersion: residual SD of the weighted regression.
    resid = by - beta * bx
    q = float(np.sum(w * resid**2))
    dof = m - 1
    overdispersion = max(1.0, q / dof)
    se_random = se_fixed * np.sqrt(overdispersion)

    se = se_fixed if method == "fixed" else se_random
    z = beta / se
    pvalue = float(2.0 * stats.norm.sf(abs(z)))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    return IVWResult(
        beta=beta,
        se=float(se),
        se_fixed=se_fixed,
        se_random=float(se_random),
        z=float(z),
        pvalue=pvalue,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_snps=m,
        method=method,
    )
