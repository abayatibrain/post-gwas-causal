"""Wakefield approximate Bayes factor fine-mapping (single causal variant).

Under the assumption that a locus harbours exactly one causal variant, the
posterior probability that SNP ``j`` is the causal one is proportional to its
approximate Bayes factor (ABF) times its prior. With a flat prior the
posterior inclusion probabilities (PIPs) are simply the normalized ABFs, and a
95% credible set is the smallest set of SNPs whose PIPs sum to 0.95.

The Wakefield (2009) ABF for a single SNP with effect estimate ``beta``,
standard error ``se``, and prior variance ``W`` on the effect is

.. math::

    \\mathrm{ABF} = \\sqrt{\\frac{V}{V + W}}\\,
        \\exp\\!\\left(\\frac{Z^2}{2}\\cdot\\frac{W}{V + W}\\right),

where ``V = se**2`` and ``Z = beta / se``. This is the Bayes factor comparing
the alternative (effect ~ N(0, W)) to the null (no effect).

References
----------
Wakefield, J. (2009). Bayes factors for genome-wide association studies:
comparison with p-values. *Genetic Epidemiology* 33(1):79-86.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

__all__ = ["AbfResult", "approximate_bayes_factor", "finemap_abf", "log_abf"]


class AbfResult(BaseModel):
    """Result of single-causal-variant ABF fine-mapping.

    Attributes
    ----------
    pip
        Posterior inclusion probabilities, one per SNP, summing to 1.
    log_abf
        Per-SNP log approximate Bayes factors.
    credible_set
        Indices of SNPs in the 95% credible set (smallest set with cumulative
        PIP >= ``coverage``), ordered by descending PIP.
    coverage
        Target coverage of the credible set (e.g. 0.95).
    top_snp
        Index of the SNP with the highest PIP.
    """

    pip: list[float]
    log_abf: list[float]
    credible_set: list[int]
    coverage: float
    top_snp: int

    @property
    def pip_arr(self) -> np.ndarray:
        """Posterior inclusion probabilities as a NumPy array."""
        return np.asarray(self.pip, dtype=float)


def log_abf(beta: np.ndarray, se: np.ndarray, prior_variance: float = 0.04) -> np.ndarray:
    r"""Per-SNP log approximate Bayes factor (Wakefield 2009).

    Parameters
    ----------
    beta
        Effect-size estimates.
    se
        Standard errors of ``beta``.
    prior_variance
        Prior variance ``W`` on the per-allele effect under the alternative.
        The default ``0.04`` corresponds to a prior SD of ``0.2`` on a
        standardized effect, a common choice for quantitative traits.

    Returns
    -------
    numpy.ndarray
        Natural-log ABF for each SNP. Larger values favour the alternative.
    """
    beta = np.asarray(beta, dtype=float)
    se = np.asarray(se, dtype=float)
    v = se**2
    z = beta / se
    r = prior_variance / (v + prior_variance)
    return 0.5 * np.log(1.0 - r) + 0.5 * z**2 * r


def approximate_bayes_factor(
    beta: np.ndarray, se: np.ndarray, prior_variance: float = 0.04
) -> np.ndarray:
    """Per-SNP approximate Bayes factor on the natural scale.

    Thin wrapper over :func:`log_abf` exponentiating the result. Prefer
    :func:`log_abf` for numerical stability when SNPs have very large Z-scores.

    Parameters
    ----------
    beta, se, prior_variance
        See :func:`log_abf`.

    Returns
    -------
    numpy.ndarray
        Approximate Bayes factors.
    """
    return np.exp(log_abf(beta, se, prior_variance))


def _softmax(log_weights: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over log-scale weights."""
    m = np.max(log_weights)
    w = np.exp(log_weights - m)
    return w / w.sum()


def _credible_set(pip: np.ndarray, coverage: float) -> list[int]:
    """Smallest set of SNP indices with cumulative PIP >= ``coverage``."""
    order = np.argsort(pip)[::-1]
    cumulative = np.cumsum(pip[order])
    # Index of the first SNP at which cumulative coverage is reached.
    k = int(np.searchsorted(cumulative, coverage)) + 1
    k = min(k, len(order))
    return order[:k].astype(int).tolist()


def finemap_abf(
    beta: np.ndarray,
    se: np.ndarray,
    *,
    prior_variance: float = 0.04,
    coverage: float = 0.95,
    prior: np.ndarray | None = None,
) -> AbfResult:
    """Single-causal-variant fine-mapping by approximate Bayes factors.

    Computes per-SNP ABFs, converts them to posterior inclusion probabilities
    (PIPs) under a (by default flat) prior, and constructs a credible set.

    Parameters
    ----------
    beta
        Per-SNP effect estimates.
    se
        Per-SNP standard errors.
    prior_variance
        Prior variance ``W`` on the effect (see :func:`log_abf`).
    coverage
        Target credible-set coverage in ``(0, 1]``.
    prior
        Optional per-SNP prior probability of being causal (need not be
        normalized). Defaults to a flat prior.

    Returns
    -------
    AbfResult
        PIPs, log-ABFs, credible set, and the top SNP.
    """
    beta = np.asarray(beta, dtype=float)
    se = np.asarray(se, dtype=float)
    if beta.shape != se.shape:
        raise ValueError("beta and se must have the same shape")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")

    labf = log_abf(beta, se, prior_variance)
    if prior is None:
        log_prior = np.zeros_like(labf)
    else:
        prior = np.asarray(prior, dtype=float)
        log_prior = np.log(prior / prior.sum())

    pip = _softmax(labf + log_prior)
    cs = _credible_set(pip, coverage)
    return AbfResult(
        pip=pip.tolist(),
        log_abf=labf.tolist(),
        credible_set=cs,
        coverage=coverage,
        top_snp=int(np.argmax(pip)),
    )
