"""Cochran's Q heterogeneity test for Mendelian randomization.

Heterogeneity among the per-instrument Wald ratios is a hallmark of horizontal
pleiotropy. Cochran's Q is the weighted sum of squared residuals of the IVW
fit:

.. math::

    Q = \\sum_j \\frac{(by_j - \\hat\\beta_{IVW}\\, bx_j)^2}{byse_j^2},

which is asymptotically ``chi-square`` with ``n_snps - 1`` degrees of freedom
under the no-heterogeneity null. The ``I^2`` statistic summarizes the fraction
of the total variation attributable to heterogeneity rather than chance.

References
----------
Bowden, J. et al. (2017). Improving the visualization, interpretation and
analysis of two-sample summary data Mendelian randomization via the radial
plot and radial regression. *International Journal of Epidemiology*.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats

__all__ = ["HeterogeneityResult", "cochran_q"]


class HeterogeneityResult(BaseModel):
    """Cochran's Q heterogeneity statistics.

    Attributes
    ----------
    q
        Cochran's Q statistic.
    dof
        Degrees of freedom (``n_snps - 1``).
    pvalue
        Upper-tail chi-square p-value; small values indicate heterogeneity.
    i_squared
        ``I^2 = max(0, (Q - dof) / Q)``, the proportion of variation due to
        heterogeneity.
    n_snps
        Number of instruments.
    """

    q: float
    dof: int
    pvalue: float
    i_squared: float
    n_snps: int


def cochran_q(bx: np.ndarray, by: np.ndarray, byse: np.ndarray) -> HeterogeneityResult:
    """Compute Cochran's Q heterogeneity statistic for an IVW fit.

    Parameters
    ----------
    bx
        Per-instrument exposure effects.
    by
        Per-instrument outcome effects.
    byse
        Standard errors of the outcome effects.

    Returns
    -------
    HeterogeneityResult
        Q, degrees of freedom, p-value, and ``I^2``.
    """
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    byse = np.asarray(byse, dtype=float)
    if not (bx.shape == by.shape == byse.shape):
        raise ValueError("bx, by, byse must share the same shape")
    m = bx.shape[0]
    if m < 2:
        raise ValueError("Cochran's Q requires at least two instruments")

    w = 1.0 / byse**2
    beta = float(np.sum(w * bx * by) / np.sum(w * bx**2))
    q = float(np.sum(w * (by - beta * bx) ** 2))
    dof = m - 1
    pvalue = float(stats.chi2.sf(q, dof))
    i_squared = float(max(0.0, (q - dof) / q)) if q > 0 else 0.0
    return HeterogeneityResult(q=q, dof=dof, pvalue=pvalue, i_squared=i_squared, n_snps=m)
