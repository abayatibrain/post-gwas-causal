"""Bayesian colocalization of two GWAS traits (coloc.abf)."""

from __future__ import annotations

from post_gwas_causal.coloc.abf import ColocResult, coloc_abf, coloc_from_logbf
from post_gwas_causal.coloc.susie import (
    ColocSusiePair,
    ColocSusieResult,
    coloc_susie,
)

__all__ = [
    "ColocResult",
    "ColocSusiePair",
    "ColocSusieResult",
    "coloc_abf",
    "coloc_from_logbf",
    "coloc_susie",
]
