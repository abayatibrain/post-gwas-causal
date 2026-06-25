"""SuSiE-based colocalization (coloc-SuSiE).

Classic ``coloc.abf`` assumes **one** causal variant per trait in the region.
That assumption is wrong often enough to matter: allelic heterogeneity and
overlapping signals at the same locus are common, and when two independent
signals sit in the window the single-causal model can dilute a real
colocalization (drag PP.H4 down) or smear PP.H3/H4 in ways that are hard to
read.

coloc-SuSiE (Wallace 2021) removes the single-causal assumption by first
fine-mapping each trait with SuSiE -- which decomposes the region into
multiple *single effects*, each a credible set -- and then running the standard
coloc hypothesis test for **every pair** of single effects, one from each
trait. The result is a credible-set-by-credible-set table of PP.H4: each entry
asks whether *that* signal in trait 1 colocalizes with *that* signal in trait
2. Two genes sharing one of several signals are no longer hidden behind a
locus-wide average.

The arithmetic is identical to :func:`post_gwas_causal.coloc.abf.coloc_abf`;
the only change is the Bayes factors fed in. Instead of one Wakefield ABF
vector per trait, each single effect contributes its own per-SNP log Bayes
factors (exposed by :class:`~post_gwas_causal.finemap.susie.SusieResult.lbf`).

References
----------
Wallace, C. (2021). A more accurate method for colocalisation analysis allowing
for multiple causal variants. *PLoS Genetics* 17(9):e1009440.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from post_gwas_causal.coloc.abf import coloc_from_logbf
from post_gwas_causal.finemap.susie import SusieResult, finemap_susie

__all__ = ["ColocSusiePair", "ColocSusieResult", "coloc_susie"]


class ColocSusiePair(BaseModel):
    """Colocalization of one trait-1 single effect with one trait-2 single effect.

    Attributes
    ----------
    idx1, idx2
        Indices of the SuSiE single effects (per trait) being compared. These
        index the credible-set lists of ``susie1`` / ``susie2``.
    pp_h4
        Posterior probability that the two signals share a causal variant.
    pp_h3
        Posterior probability that they have distinct causal variants.
    best_snp
        Index (in the locus SNP order) of the most probable shared variant.
    cs1, cs2
        The two credible sets being compared.
    """

    model_config = ConfigDict(frozen=True)

    idx1: int
    idx2: int
    pp_h4: float
    pp_h3: float
    best_snp: int
    cs1: list[int]
    cs2: list[int]


class ColocSusieResult(BaseModel):
    """Result of a coloc-SuSiE analysis over two fine-mapped traits.

    Attributes
    ----------
    pairs
        One :class:`ColocSusiePair` per pair of (purity-filtered) single
        effects, sorted by descending PP.H4.
    n_cs1, n_cs2
        Number of credible sets SuSiE found for each trait.
    best_pp_h4
        Largest PP.H4 across all pairs (``0.0`` if either trait yielded no
        credible set). This is the headline "do these traits colocalize"
        number, now robust to multiple signals.
    """

    model_config = ConfigDict(frozen=True)

    pairs: list[ColocSusiePair]
    n_cs1: int
    n_cs2: int
    best_pp_h4: float


def coloc_susie(
    z1: np.ndarray,
    z2: np.ndarray,
    ld: np.ndarray,
    n1: int,
    n2: int,
    *,
    n_effects: int = 5,
    prior_variance: float = 0.1,
    min_purity: float = 0.5,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 1e-5,
) -> ColocSusieResult:
    """Colocalize two traits at a locus allowing multiple causal variants.

    Parameters
    ----------
    z1, z2
        Marginal GWAS Z-scores for the two traits, SNP-aligned and sharing the
        ``ld`` matrix.
    ld
        ``p x p`` LD correlation matrix for the locus (shared by both traits;
        coloc-SuSiE assumes a common reference panel).
    n1, n2
        Sample sizes for the two traits.
    n_effects
        Maximum number of single effects ``L`` for each SuSiE fit.
    prior_variance
        SuSiE prior variance per single effect.
    min_purity
        Minimum within-set absolute LD for a SuSiE credible set to be kept.
    p1, p2, p12
        Coloc priors passed through to the per-pair hypothesis test.

    Returns
    -------
    ColocSusieResult
        Per-credible-set-pair PP.H4/PP.H3 and the headline ``best_pp_h4``.

    Notes
    -----
    When SuSiE finds a single credible set per trait, the top pair reproduces
    the classic single-variant coloc answer; the value of the method shows up
    when a trait has two or more independent signals in the window.
    """
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    ld = np.asarray(ld, dtype=float)

    susie1 = finemap_susie(
        z1, ld, n1, n_effects=n_effects, prior_variance=prior_variance, min_purity=min_purity
    )
    susie2 = finemap_susie(
        z2, ld, n2, n_effects=n_effects, prior_variance=prior_variance, min_purity=min_purity
    )

    pairs = _pairwise_coloc(susie1, susie2, p1=p1, p2=p2, p12=p12)
    best = max((pair.pp_h4 for pair in pairs), default=0.0)
    return ColocSusieResult(
        pairs=pairs,
        n_cs1=len(susie1.credible_sets),
        n_cs2=len(susie2.credible_sets),
        best_pp_h4=float(best),
    )


def _pairwise_coloc(
    susie1: SusieResult,
    susie2: SusieResult,
    *,
    p1: float,
    p2: float,
    p12: float,
) -> list[ColocSusiePair]:
    """Run coloc on every pair of credible-set single effects."""
    lbf1 = susie1.lbf_arr
    lbf2 = susie2.lbf_arr
    pairs: list[ColocSusiePair] = []
    for ci, eff1 in enumerate(susie1.cs_effects):
        for cj, eff2 in enumerate(susie2.cs_effects):
            res = coloc_from_logbf(lbf1[eff1], lbf2[eff2], p1=p1, p2=p2, p12=p12)
            pairs.append(
                ColocSusiePair(
                    idx1=ci,
                    idx2=cj,
                    pp_h4=res.pp_h4,
                    pp_h3=res.pp_h3,
                    best_snp=res.best_snp_h4,
                    cs1=susie1.credible_sets[ci],
                    cs2=susie2.credible_sets[cj],
                )
            )
    pairs.sort(key=lambda pair: pair.pp_h4, reverse=True)
    return pairs
