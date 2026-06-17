"""MR-Egger regression — causal slope plus a directional-pleiotropy test.

IVW forces the regression of outcome effects on exposure effects through the
origin. MR-Egger relaxes that, fitting

.. math::

    by_j = \\beta_0 + \\beta_1\\, bx_j + \\varepsilon_j,
    \\qquad \\varepsilon_j \\sim N(0, byse_j^2),

by weighted least squares with weights ``1 / byse**2``. The *slope* ``β1`` is
the MR-Egger causal estimate; the *intercept* ``β0`` estimates the average
directional (horizontal) pleiotropic effect. An intercept significantly
different from zero is evidence that the InSIDE assumption is violated and that
IVW is biased.

To respect MR-Egger's orientation convention, all instruments are coded so the
exposure effect is positive before fitting.

References
----------
Bowden, J., Davey Smith, G., & Burgess, S. (2015). Mendelian randomization with
invalid instruments: effect estimation and bias detection through Egger
regression. *International Journal of Epidemiology* 44(2):512-525.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from pydantic import BaseModel
from scipy import stats

__all__ = ["EggerResult", "mr_egger"]


class EggerResult(BaseModel):
    """MR-Egger regression result.

    Attributes
    ----------
    slope
        Causal-effect estimate (the pleiotropy-adjusted slope ``β1``).
    slope_se, slope_pvalue
        Standard error and two-sided p-value of the slope.
    slope_ci_low, slope_ci_high
        95% CI for the slope.
    intercept
        Estimated average directional pleiotropy ``β0``.
    intercept_se, intercept_pvalue
        SE and two-sided p-value of the intercept (the pleiotropy test).
    intercept_ci_low, intercept_ci_high
        95% CI for the intercept; covering zero is consistent with no
        directional pleiotropy.
    n_snps
        Number of instruments.
    """

    slope: float
    slope_se: float
    slope_pvalue: float
    slope_ci_low: float
    slope_ci_high: float
    intercept: float
    intercept_se: float
    intercept_pvalue: float
    intercept_ci_low: float
    intercept_ci_high: float
    n_snps: int


def mr_egger(bx: np.ndarray, by: np.ndarray, byse: np.ndarray) -> EggerResult:
    """Fit MR-Egger regression.

    Parameters
    ----------
    bx
        Per-instrument exposure effects.
    by
        Per-instrument outcome effects.
    byse
        Standard errors of the outcome effects (used as inverse weights).

    Returns
    -------
    EggerResult
        Slope (causal effect), intercept (pleiotropy), their SEs, p-values and
        95% CIs.

    Notes
    -----
    Instruments are first oriented so every exposure effect is positive (flip
    the sign of both ``bx`` and ``by`` where ``bx < 0``); this is the standard
    MR-Egger coding that makes the intercept interpretable as net directional
    pleiotropy. A t-distribution with ``n_snps - 2`` degrees of freedom is used
    for inference, matching the WLS fit.
    """
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    byse = np.asarray(byse, dtype=float)
    if not (bx.shape == by.shape == byse.shape):
        raise ValueError("bx, by, byse must share the same shape")
    m = bx.shape[0]
    if m < 3:
        raise ValueError("MR-Egger requires at least three instruments")

    # Orient so exposure effects are positive.
    sign = np.where(bx < 0, -1.0, 1.0)
    bx_o = bx * sign
    by_o = by * sign

    weights = 1.0 / byse**2
    design = sm.add_constant(bx_o)
    model = sm.WLS(by_o, design, weights=weights).fit()

    intercept, slope = model.params
    intercept_se, slope_se = model.bse
    dof = m - 2

    def _t_pvalue(estimate: float, se: float) -> float:
        return float(2.0 * stats.t.sf(abs(estimate / se), df=dof))

    t_crit = float(stats.t.ppf(0.975, df=dof))

    return EggerResult(
        slope=float(slope),
        slope_se=float(slope_se),
        slope_pvalue=_t_pvalue(slope, slope_se),
        slope_ci_low=float(slope - t_crit * slope_se),
        slope_ci_high=float(slope + t_crit * slope_se),
        intercept=float(intercept),
        intercept_se=float(intercept_se),
        intercept_pvalue=_t_pvalue(intercept, intercept_se),
        intercept_ci_low=float(intercept - t_crit * intercept_se),
        intercept_ci_high=float(intercept + t_crit * intercept_se),
        n_snps=m,
    )
