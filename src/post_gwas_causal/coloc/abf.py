"""Giambartolomei (2014) Bayesian colocalization — ``coloc.abf``.

Given summary statistics for two traits over a shared set of SNPs, decide
whether they share a single causal variant. The method enumerates five
mutually exclusive hypotheses for the locus and returns their posterior
probabilities:

==========  ====================================================
Hypothesis  Meaning
==========  ====================================================
H0          No causal variant for either trait
H1          A causal variant for trait 1 only
H2          A causal variant for trait 2 only
H3          Distinct causal variants for the two traits
H4          A *shared* causal variant (colocalization)
==========  ====================================================

Each SNP contributes a Wakefield approximate Bayes factor (ABF) per trait.
Configurable prior probabilities ``p1`` (causal for trait 1), ``p2`` (causal
for trait 2), and ``p12`` (causal for both) weight the hypotheses.

References
----------
Giambartolomei, C. et al. (2014). Bayesian test for colocalisation between
pairs of genetic association studies using summary statistics.
*PLoS Genetics* 10(5):e1004383.

Wakefield, J. (2009). Bayes factors for genome-wide association studies.
*Genetic Epidemiology* 33(1):79-86.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy.special import logsumexp

from post_gwas_causal.finemap.abf import log_abf

__all__ = ["ColocResult", "coloc_abf"]


class ColocResult(BaseModel):
    """Posterior probabilities of the five coloc hypotheses.

    Attributes
    ----------
    pp_h0, pp_h1, pp_h2, pp_h3, pp_h4
        Posterior probabilities of H0..H4. They sum to 1.
    n_snps
        Number of SNPs analysed (intersection of the two traits).
    priors
        The ``(p1, p2, p12)`` priors used.
    best_snp_h4
        Index of the SNP most likely to be the shared causal variant under H4
        (the SNP whose combined per-trait ABF is largest).
    """

    pp_h0: float
    pp_h1: float
    pp_h2: float
    pp_h3: float
    pp_h4: float
    n_snps: int
    priors: tuple[float, float, float]
    best_snp_h4: int

    def as_dict(self) -> dict[str, float]:
        """Return the posteriors keyed by hypothesis name."""
        return {
            "PP.H0": self.pp_h0,
            "PP.H1": self.pp_h1,
            "PP.H2": self.pp_h2,
            "PP.H3": self.pp_h3,
            "PP.H4": self.pp_h4,
        }


def coloc_abf(
    beta1: np.ndarray,
    se1: np.ndarray,
    beta2: np.ndarray,
    se2: np.ndarray,
    *,
    prior_variance1: float = 0.04,
    prior_variance2: float = 0.04,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 1e-5,
) -> ColocResult:
    r"""Run the ``coloc.abf`` colocalization test for two traits.

    Parameters
    ----------
    beta1, se1
        Effect sizes and standard errors for trait 1.
    beta2, se2
        Effect sizes and standard errors for trait 2. Must be SNP-aligned with
        trait 1 (same SNP order).
    prior_variance1, prior_variance2
        Wakefield prior variances ``W`` for the two traits.
    p1, p2, p12
        Prior probability that a given SNP is causal for trait 1 only, trait 2
        only, and *both* traits, respectively. Defaults follow Giambartolomei
        (2014): ``1e-4``, ``1e-4``, ``1e-5``.

    Returns
    -------
    ColocResult
        Posterior probabilities PP.H0..PP.H4.

    Notes
    -----
    The hypothesis-level (unnormalized log) posteriors are built from per-SNP
    log ABFs ``l1_i`` (trait 1) and ``l2_i`` (trait 2):

    * ``log L(H0) = 0``
    * ``log L(H1) = log p1 + logsumexp_i l1_i``
    * ``log L(H2) = log p2 + logsumexp_i l2_i``
    * ``log L(H3) = log(p1 p2) + logsumexp_{i != j} (l1_i + l2_j)``
    * ``log L(H4) = log p12 + logsumexp_i (l1_i + l2_i)``

    The H3 term excludes the diagonal (which is exactly the H4 / shared-SNP
    configuration), computed as ``S1 * S2 - S_diag`` in linear space, evaluated
    stably in log space.
    """
    beta1 = np.asarray(beta1, dtype=float)
    se1 = np.asarray(se1, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)
    se2 = np.asarray(se2, dtype=float)
    if not (beta1.shape == se1.shape == beta2.shape == se2.shape):
        raise ValueError("all four input arrays must share the same shape")

    l1 = log_abf(beta1, se1, prior_variance1)
    l2 = log_abf(beta2, se2, prior_variance2)

    log_p1 = np.log(p1)
    log_p2 = np.log(p2)
    log_p12 = np.log(p12)

    # Per-hypothesis log-evidence (log of sum over SNP configurations,
    # already weighted by priors).
    log_l_h0 = 0.0
    log_l_h1 = log_p1 + logsumexp(l1)
    log_l_h2 = log_p2 + logsumexp(l2)
    log_l_h4 = log_p12 + logsumexp(l1 + l2)

    # H3: sum over all (i, j) with i != j of exp(l1_i + l2_j).
    #   sum_{i,j} = (sum_i e^{l1_i}) (sum_j e^{l2_j})
    #   diagonal  = sum_i e^{l1_i + l2_i}
    s1 = logsumexp(l1)
    s2 = logsumexp(l2)
    log_full = s1 + s2
    log_diag = logsumexp(l1 + l2)
    # log(e^log_full - e^log_diag), stable since log_full >= log_diag. When a
    # single SNP dominates both traits the off-diagonal mass underflows to ~0;
    # clip the argument away from 1 so the log is a large finite negative rather
    # than -inf (H3 is then negligible, which is the correct behaviour).
    diff = np.minimum(np.exp(log_diag - log_full), 1.0 - 1e-15)
    log_offdiag = log_full + np.log1p(-diff)
    log_l_h3 = log_p1 + log_p2 + log_offdiag

    logs = np.array([log_l_h0, log_l_h1, log_l_h2, log_l_h3, log_l_h4])
    pp = np.exp(logs - logsumexp(logs))

    return ColocResult(
        pp_h0=float(pp[0]),
        pp_h1=float(pp[1]),
        pp_h2=float(pp[2]),
        pp_h3=float(pp[3]),
        pp_h4=float(pp[4]),
        n_snps=int(beta1.shape[0]),
        priors=(p1, p2, p12),
        best_snp_h4=int(np.argmax(l1 + l2)),
    )
