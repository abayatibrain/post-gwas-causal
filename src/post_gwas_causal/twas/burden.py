"""FUSION-style TWAS burden test from summary statistics.

A TWAS asks whether *genetically predicted expression* of a gene is associated
with a trait. Given a set of eQTL weights ``w`` (the SNP-to-expression
prediction model), the GWAS Z-scores ``Z`` at those SNPs, and the SNP LD matrix
``R``, the FUSION burden statistic is

.. math::

    Z_{TWAS} = \\frac{w^\\top Z}{\\sqrt{w^\\top R\\, w}} .

Under the null of no expression-trait association, ``Z_TWAS`` is standard
normal. The numerator is the LD-aware weighted sum of GWAS signal; the
denominator is the standard deviation of that weighted sum induced by LD among
the model SNPs (since ``Var(w'Z) = w'Rw`` when ``Cov(Z) = R``).

References
----------
Gusev, A. et al. (2016). Integrative approaches for large-scale transcriptome-
wide association studies. *Nature Genetics* 48(3):245-252.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats

__all__ = ["TwasResult", "twas_burden"]


class TwasResult(BaseModel):
    """FUSION-style TWAS burden-test result.

    Attributes
    ----------
    z
        TWAS Z-statistic ``w'Z / sqrt(w'Rw)``.
    pvalue
        Two-sided standard-normal p-value.
    n_snps
        Number of SNPs in the expression-prediction model.
    """

    z: float
    pvalue: float
    n_snps: int


def twas_burden(
    weights: np.ndarray,
    gwas_z: np.ndarray,
    ld: np.ndarray | None = None,
) -> TwasResult:
    """Compute the FUSION TWAS burden Z-statistic and p-value.

    Parameters
    ----------
    weights
        eQTL / expression-prediction weights ``w``, one per model SNP.
    gwas_z
        GWAS Z-scores at the model SNPs, aligned to ``weights``.
    ld
        SNP LD correlation matrix ``R``. If ``None``, the identity matrix is
        assumed (independent SNPs).

    Returns
    -------
    TwasResult
        TWAS Z-statistic and p-value.

    Raises
    ------
    ValueError
        If shapes are inconsistent or the LD-weighted variance is non-positive
        (e.g. all-zero weights).
    """
    weights = np.asarray(weights, dtype=float)
    gwas_z = np.asarray(gwas_z, dtype=float)
    if weights.shape != gwas_z.shape:
        raise ValueError("weights and gwas_z must share the same shape")
    p = weights.shape[0]

    if ld is None:
        ld = np.eye(p)
    else:
        ld = np.asarray(ld, dtype=float)
        if ld.shape != (p, p):
            raise ValueError("ld must be a (p, p) matrix matching the weights")

    numerator = float(weights @ gwas_z)
    variance = float(weights @ ld @ weights)
    if variance <= 0.0:
        raise ValueError("w'Rw must be positive; check weights and LD matrix")

    z = numerator / np.sqrt(variance)
    pvalue = float(2.0 * stats.norm.sf(abs(z)))
    return TwasResult(z=z, pvalue=pvalue, n_snps=p)
