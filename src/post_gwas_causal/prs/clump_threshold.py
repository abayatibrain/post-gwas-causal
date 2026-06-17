"""Clumping + p-value thresholding (C+T) polygenic risk scores.

The classic PRS recipe:

1. **Clump** — greedily keep the most significant SNP, then remove every SNP in
   high LD (``r^2 > clump_r2``) with it; repeat on the remaining SNPs. This
   yields an approximately independent set of index SNPs.
2. **Threshold** — keep only index SNPs with ``p <= p_threshold``.
3. **Score** — for each individual, sum the retained effect sizes weighted by
   their (dosage) genotypes:
   ``PRS_i = sum_j beta_j * G_ij``.

References
----------
Choi, S.W., Mak, T.S.H., & O'Reilly, P.F. (2020). Tutorial: a guide to
performing polygenic risk score analyses. *Nature Protocols* 15:2759-2772.

International Schizophrenia Consortium (2009). Common polygenic variation
contributes to risk of schizophrenia and bipolar disorder. *Nature* 460.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

__all__ = ["ClumpThresholdResult", "clump", "clump_and_threshold"]


class ClumpThresholdResult(BaseModel):
    """C+T scoring result.

    Attributes
    ----------
    scores
        Per-individual polygenic scores.
    selected_idx
        Indices of the SNPs retained after clumping and thresholding.
    n_selected
        Number of retained SNPs.
    """

    scores: list[float]
    selected_idx: list[int]
    n_selected: int

    @property
    def scores_arr(self) -> np.ndarray:
        """Per-individual scores as a NumPy array."""
        return np.asarray(self.scores, dtype=float)


def clump(
    pvalues: np.ndarray,
    ld: np.ndarray,
    *,
    clump_r2: float = 0.1,
) -> list[int]:
    """Greedy LD clumping; return the retained index-SNP positions.

    Parameters
    ----------
    pvalues
        Per-SNP association p-values.
    ld
        SNP LD correlation matrix.
    clump_r2
        ``r^2`` threshold above which a SNP is removed as a satellite of a more
        significant index SNP.

    Returns
    -------
    list of int
        Positions of the retained (approximately independent) index SNPs,
        ordered from most to least significant.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    ld = np.asarray(ld, dtype=float)
    r2 = ld**2
    order = np.argsort(pvalues)
    remaining = np.ones(pvalues.shape[0], dtype=bool)
    index_snps: list[int] = []
    for j in order:
        if not remaining[j]:
            continue
        index_snps.append(int(j))
        remaining[j] = False
        satellites = (r2[j] > clump_r2) & remaining
        remaining[satellites] = False
    return index_snps


def clump_and_threshold(
    beta: np.ndarray,
    pvalues: np.ndarray,
    ld: np.ndarray,
    genotypes: np.ndarray,
    *,
    clump_r2: float = 0.1,
    p_threshold: float = 5e-8,
) -> ClumpThresholdResult:
    """Compute C+T polygenic scores for a target sample.

    Parameters
    ----------
    beta
        Per-SNP effect sizes from the discovery GWAS.
    pvalues
        Per-SNP p-values from the discovery GWAS.
    ld
        Reference LD correlation matrix (SNP x SNP).
    genotypes
        Target-sample dosage matrix, shape ``(n_individuals, n_snps)``.
    clump_r2
        LD ``r^2`` clumping threshold.
    p_threshold
        Keep only clumped SNPs with ``p <= p_threshold``.

    Returns
    -------
    ClumpThresholdResult
        Per-individual scores and the retained SNP indices.
    """
    beta = np.asarray(beta, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    genotypes = np.asarray(genotypes, dtype=float)
    if genotypes.shape[1] != beta.shape[0]:
        raise ValueError("genotypes second dimension must equal the number of SNPs")

    index_snps = clump(pvalues, ld, clump_r2=clump_r2)
    selected = [j for j in index_snps if pvalues[j] <= p_threshold]

    scores = genotypes[:, selected] @ beta[selected] if selected else np.zeros(genotypes.shape[0])

    return ClumpThresholdResult(
        scores=scores.tolist(),
        selected_idx=selected,
        n_selected=len(selected),
    )
