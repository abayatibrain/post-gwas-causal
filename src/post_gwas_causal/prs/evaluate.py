"""Evaluate polygenic scores against an observed phenotype.

For quantitative phenotypes the natural metric is the squared Pearson
correlation ``R^2`` between the score and the phenotype. For binary
phenotypes the area under the ROC curve (AUC) measures discrimination. Both are
computed here from first principles (no sklearn dependency).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

__all__ = ["PrsEvaluation", "evaluate_prs", "roc_auc"]


class PrsEvaluation(BaseModel):
    """Polygenic-score evaluation metrics.

    Attributes
    ----------
    r2
        Squared Pearson correlation between score and phenotype.
    pearson_r
        Pearson correlation (signed).
    auc
        ROC AUC when the phenotype is binary, else ``None``.
    n
        Number of individuals.
    """

    r2: float
    pearson_r: float
    auc: float | None
    n: int


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U identity.

    Parameters
    ----------
    scores
        Continuous scores (higher = more likely positive).
    labels
        Binary 0/1 labels.

    Returns
    -------
    float
        Area under the ROC curve in ``[0, 1]``.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUC requires both positive and negative labels")

    # Rank-based AUC = (sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos n_neg).
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.shape[0], dtype=float)
    ranks[order] = np.arange(1, order.shape[0] + 1)
    # Average ranks for ties.
    combined = np.concatenate([pos, neg])
    _assign_tied_ranks(combined, ranks)
    n_pos = pos.size
    sum_pos_ranks = ranks[:n_pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * neg.size)
    return float(auc)


def _assign_tied_ranks(values: np.ndarray, ranks: np.ndarray) -> None:
    """Replace ranks of tied values with their average rank (in place)."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    i = 0
    n = values.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def evaluate_prs(scores: np.ndarray, phenotype: np.ndarray) -> PrsEvaluation:
    """Evaluate a polygenic score against a phenotype.

    Parameters
    ----------
    scores
        Per-individual polygenic scores.
    phenotype
        Per-individual phenotype. If it contains only ``{0, 1}`` it is treated
        as binary and an AUC is reported.

    Returns
    -------
    PrsEvaluation
        ``R^2``, Pearson correlation, and (for binary phenotypes) AUC.
    """
    scores = np.asarray(scores, dtype=float)
    phenotype = np.asarray(phenotype, dtype=float)
    if scores.shape != phenotype.shape:
        raise ValueError("scores and phenotype must share the same shape")

    pearson = 0.0 if np.std(scores) == 0.0 else float(np.corrcoef(scores, phenotype)[0, 1])

    is_binary = set(np.unique(phenotype)).issubset({0.0, 1.0})
    auc = roc_auc(scores, phenotype) if is_binary else None

    return PrsEvaluation(
        r2=pearson**2,
        pearson_r=pearson,
        auc=auc,
        n=int(scores.shape[0]),
    )
