"""Harmonize effect alleles between exposure and outcome GWAS.

Two GWAS rarely report effects relative to the same allele. Before any MR
estimator runs, every instrument must be coded so that ``beta_exposure`` and
``beta_outcome`` refer to the *same* effect allele. This module:

* flips the sign of the outcome effect where the outcome's effect allele is the
  exposure's *other* allele,
* drops SNPs whose alleles are inconsistent between studies (after accounting
  for strand-unambiguous flips),
* optionally drops strand-ambiguous palindromic SNPs (A/T, C/G) whose strand
  cannot be resolved from the alleles alone.

References
----------
Hartwig, F.P., Davey Smith, G., & Bowden, J. (2017). Robust inference in
summary data Mendelian randomization. *Int. J. Epidemiology* 46(6).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

__all__ = ["HarmonizedData", "harmonize"]

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
_PALINDROMES = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


class HarmonizedData(BaseModel):
    """Allele-aligned instrument effects ready for MR.

    Attributes
    ----------
    bx, bxse
        Exposure effects and SEs for the retained instruments.
    by, byse
        Outcome effects (sign-aligned to the exposure allele) and SEs.
    rsid
        Retained instrument identifiers.
    n_flipped
        Number of instruments whose outcome effect sign was flipped.
    n_dropped
        Number of instruments dropped (incompatible or ambiguous alleles).
    """

    bx: list[float]
    bxse: list[float]
    by: list[float]
    byse: list[float]
    rsid: list[str]
    n_flipped: int
    n_dropped: int

    @property
    def bx_arr(self) -> np.ndarray:
        """Exposure effects as a NumPy array."""
        return np.asarray(self.bx, dtype=float)

    @property
    def by_arr(self) -> np.ndarray:
        """Outcome effects as a NumPy array."""
        return np.asarray(self.by, dtype=float)


def _is_palindromic(a1: str, a2: str) -> bool:
    return (a1, a2) in _PALINDROMES


def harmonize(
    rsid: list[str],
    beta_exposure: np.ndarray,
    se_exposure: np.ndarray,
    effect_allele_exposure: list[str],
    other_allele_exposure: list[str],
    beta_outcome: np.ndarray,
    se_outcome: np.ndarray,
    effect_allele_outcome: list[str],
    other_allele_outcome: list[str],
    *,
    drop_palindromic: bool = True,
) -> HarmonizedData:
    """Align effect alleles between exposure and outcome GWAS.

    Parameters
    ----------
    rsid
        Instrument identifiers (must be the same SNPs, same order, in both
        studies).
    beta_exposure, se_exposure
        Exposure effects and SEs.
    effect_allele_exposure, other_allele_exposure
        Exposure alleles (single-character, upper-case).
    beta_outcome, se_outcome
        Outcome effects and SEs.
    effect_allele_outcome, other_allele_outcome
        Outcome alleles.
    drop_palindromic
        If ``True`` (default), drop strand-ambiguous palindromic SNPs.

    Returns
    -------
    HarmonizedData
        Retained instruments with outcome effects aligned to the exposure
        effect allele.
    """
    beta_exposure = np.asarray(beta_exposure, dtype=float)
    se_exposure = np.asarray(se_exposure, dtype=float)
    beta_outcome = np.asarray(beta_outcome, dtype=float)
    se_outcome = np.asarray(se_outcome, dtype=float)

    keep_rsid: list[str] = []
    bx: list[float] = []
    bxse: list[float] = []
    by: list[float] = []
    byse: list[float] = []
    n_flipped = 0
    n_dropped = 0

    for i, snp in enumerate(rsid):
        ea_e = effect_allele_exposure[i].upper()
        oa_e = other_allele_exposure[i].upper()
        ea_o = effect_allele_outcome[i].upper()
        oa_o = other_allele_outcome[i].upper()

        if drop_palindromic and _is_palindromic(ea_e, oa_e):
            n_dropped += 1
            continue

        if {ea_e, oa_e} == {ea_o, oa_o}:
            sign = 1.0 if ea_e == ea_o else -1.0
        elif {ea_e, oa_e} == {_COMPLEMENT[ea_o], _COMPLEMENT[oa_o]}:
            # Outcome reported on the opposite strand; flip strand then align.
            sign = 1.0 if ea_e == _COMPLEMENT[ea_o] else -1.0
        else:
            n_dropped += 1
            continue

        if sign < 0:
            n_flipped += 1

        keep_rsid.append(snp)
        bx.append(float(beta_exposure[i]))
        bxse.append(float(se_exposure[i]))
        by.append(float(sign * beta_outcome[i]))
        byse.append(float(se_outcome[i]))

    return HarmonizedData(
        bx=bx,
        bxse=bxse,
        by=by,
        byse=byse,
        rsid=keep_rsid,
        n_flipped=n_flipped,
        n_dropped=n_dropped,
    )
