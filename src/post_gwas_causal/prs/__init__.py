"""Polygenic risk scoring.

* :mod:`~post_gwas_causal.prs.clump_threshold` — clumping + p-value
  thresholding (C+T).
* :mod:`~post_gwas_causal.prs.shrinkage` — LDpred-inf-style infinitesimal
  Bayesian shrinkage of effect sizes given LD.
* :mod:`~post_gwas_causal.prs.evaluate` — score evaluation (R^2, AUC).
"""

from __future__ import annotations

from post_gwas_causal.prs.clump_threshold import ClumpThresholdResult, clump_and_threshold
from post_gwas_causal.prs.evaluate import PrsEvaluation, evaluate_prs
from post_gwas_causal.prs.shrinkage import ldpred_inf, score_individuals

__all__ = [
    "ClumpThresholdResult",
    "PrsEvaluation",
    "clump_and_threshold",
    "evaluate_prs",
    "ldpred_inf",
    "score_individuals",
]
