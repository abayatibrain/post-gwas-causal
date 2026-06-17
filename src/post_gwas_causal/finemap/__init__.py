"""Statistical fine-mapping of GWAS loci.

Two complementary methods:

* :mod:`post_gwas_causal.finemap.abf` — Wakefield (2009) approximate Bayes
  factors, posterior inclusion probabilities, and 95% credible sets. Assumes a
  *single* causal variant per locus.
* :mod:`post_gwas_causal.finemap.susie` — a simplified SuSiE-style
  ("Sum of Single Effects") fine-mapper that fits ``L`` single-effect
  components by Iterative Bayesian Stepwise Selection (IBSS), allowing
  *multiple* causal variants.
"""

from __future__ import annotations

from post_gwas_causal.finemap.abf import (
    AbfResult,
    approximate_bayes_factor,
    finemap_abf,
)
from post_gwas_causal.finemap.susie import SusieResult, finemap_susie

__all__ = [
    "AbfResult",
    "SusieResult",
    "approximate_bayes_factor",
    "finemap_abf",
    "finemap_susie",
]
