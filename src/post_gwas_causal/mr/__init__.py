"""Two-sample Mendelian randomization estimators and diagnostics.

* :mod:`~post_gwas_causal.mr.ivw` — inverse-variance-weighted estimator
  (fixed- and random-effects).
* :mod:`~post_gwas_causal.mr.egger` — MR-Egger slope plus the intercept test
  for directional pleiotropy.
* :mod:`~post_gwas_causal.mr.weighted_median` — weighted-median estimator with
  a bootstrap standard error (robust to up to 50% invalid instruments).
* :mod:`~post_gwas_causal.mr.heterogeneity` — Cochran's Q heterogeneity test.
* :func:`~post_gwas_causal.mr.harmonize.harmonize` — align effect alleles
  between the exposure and outcome GWAS.
"""

from __future__ import annotations

from post_gwas_causal.mr.egger import EggerResult, mr_egger
from post_gwas_causal.mr.harmonize import HarmonizedData, harmonize
from post_gwas_causal.mr.heterogeneity import HeterogeneityResult, cochran_q
from post_gwas_causal.mr.ivw import IVWResult, mr_ivw
from post_gwas_causal.mr.weighted_median import WeightedMedianResult, mr_weighted_median

__all__ = [
    "EggerResult",
    "HarmonizedData",
    "HeterogeneityResult",
    "IVWResult",
    "WeightedMedianResult",
    "cochran_q",
    "harmonize",
    "mr_egger",
    "mr_ivw",
    "mr_weighted_median",
]
