"""Simulate GWAS-style summary statistics with realistic LD structure.

Everything downstream in this toolkit consumes *summary statistics* — per-SNP
effect sizes, standard errors, Z-scores — together with a reference LD matrix
``R``. This module manufactures those quantities under known ground truth so
the fine-mapping, colocalization, Mendelian-randomization, TWAS, and PRS
implementations can be validated against answers we already know.

The generative model is the standard one used throughout statistical genetics:

* A locus of ``n_snps`` SNPs has a correlation (LD) matrix ``R``.
* A handful of *causal* SNPs carry standardized marginal effects ``b``.
* The vector of *marginal* (observed) standardized effects is ``z_true = R @ b``
  — LD smears each causal signal onto its correlated neighbours.
* Observed Z-scores are ``z ~ MultivariateNormal(sqrt(N) * z_true, R)``: the
  marginal-association estimates at correlated SNPs are themselves correlated
  through ``R``.

References
----------
The "marginal = R @ joint" identity and the ``Z ~ N(sqrt(N) R b, R)`` sampling
model underpin essentially every summary-statistics fine-mapping method
(e.g. Benner 2016 FINEMAP, Wang 2020 SuSiE-RSS).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

__all__ = [
    "ColocConfig",
    "ColocDatasets",
    "LocusConfig",
    "MRConfig",
    "MRDatasets",
    "SummaryStats",
    "ar1_ld_matrix",
    "block_ld_matrix",
    "simulate_coloc",
    "simulate_locus",
    "simulate_mr",
]


class LocusConfig(BaseModel):
    """Configuration for a single-trait GWAS locus simulation.

    Parameters
    ----------
    n_snps
        Number of SNPs in the locus.
    n_samples
        GWAS sample size ``N``.
    rho
        AR(1) correlation decay parameter for the LD matrix (used when
        ``block_size`` is ``None``).
    block_size
        If set, build a block-correlation LD matrix with blocks of this size
        instead of an AR(1) matrix.
    block_rho
        Within-block correlation when ``block_size`` is set.
    causal_idx
        Indices of the causal SNP(s). Defaults to a single causal SNP at the
        centre of the locus.
    causal_effects
        Standardized marginal-per-allele effects of the causal SNP(s). Length
        must match ``causal_idx``.
    maf_low, maf_high
        Range from which minor-allele frequencies are drawn uniformly.
    seed
        RNG seed for reproducibility.
    """

    n_snps: int = Field(default=50, ge=2)
    n_samples: int = Field(default=10_000, ge=10)
    rho: float = Field(default=0.6, ge=0.0, lt=1.0)
    block_size: int | None = Field(default=None, ge=1)
    block_rho: float = Field(default=0.8, ge=0.0, lt=1.0)
    causal_idx: list[int] | None = None
    causal_effects: list[float] | None = None
    maf_low: float = Field(default=0.05, gt=0.0, lt=0.5)
    maf_high: float = Field(default=0.45, gt=0.0, le=0.5)
    seed: int = 0


class ColocConfig(BaseModel):
    """Configuration for a two-trait colocalization simulation.

    Parameters
    ----------
    n_snps
        Number of SNPs in the shared locus.
    n_samples
        Per-trait sample size.
    rho
        AR(1) LD decay parameter.
    shared_causal
        If ``True``, both traits share a single causal SNP (the H4 scenario);
        if ``False`` they have *distinct* causal SNPs (the H3 scenario).
    causal_idx_1, causal_idx_2
        Causal SNP indices for trait 1 / trait 2. When ``shared_causal`` is
        ``True`` only ``causal_idx_1`` is used for both traits.
    effect_1, effect_2
        Standardized causal effects for the two traits.
    seed
        RNG seed.
    """

    n_snps: int = Field(default=50, ge=2)
    n_samples: int = Field(default=20_000, ge=10)
    rho: float = Field(default=0.5, ge=0.0, lt=1.0)
    shared_causal: bool = True
    causal_idx_1: int = 25
    causal_idx_2: int = 40
    effect_1: float = 0.12
    effect_2: float = 0.12
    seed: int = 0


class MRConfig(BaseModel):
    """Configuration for a two-sample Mendelian-randomization simulation.

    Parameters
    ----------
    n_instruments
        Number of (approximately independent) instrument SNPs.
    causal_effect
        True causal effect ``beta`` of the exposure on the outcome.
    n_exposure, n_outcome
        Sample sizes of the exposure and outcome GWAS.
    gamma_mean, gamma_sd
        Mean / SD of the per-instrument exposure effects (drawn once).
    pleiotropy
        Directional (uncorrelated horizontal) pleiotropy added to every
        instrument's outcome effect. Zero means the InSIDE/balanced-pleiotropy
        null holds and MR-Egger's intercept should be ~0.
    invalid_idx
        Indices of instruments to corrupt with a large outlier pleiotropic
        effect (used to test weighted-median robustness).
    invalid_effect
        Magnitude of the corruption applied to ``invalid_idx`` instruments.
    seed
        RNG seed.
    """

    n_instruments: int = Field(default=30, ge=3)
    causal_effect: float = 0.3
    n_exposure: int = Field(default=50_000, ge=10)
    n_outcome: int = Field(default=50_000, ge=10)
    gamma_mean: float = 0.0
    gamma_sd: float = 0.1
    pleiotropy: float = 0.0
    invalid_idx: list[int] | None = None
    invalid_effect: float = 0.2
    seed: int = 0


class SummaryStats(BaseModel):
    """GWAS summary statistics for a single trait at a locus.

    All array-valued fields are stored as plain Python lists so the object is
    JSON/pydantic-friendly; convenience properties return NumPy arrays.
    """

    beta: list[float]
    se: list[float]
    z: list[float]
    maf: list[float]
    n: int
    rsid: list[str]

    model_config = {"frozen": True}

    @property
    def beta_arr(self) -> np.ndarray:
        """Effect sizes as a NumPy array."""
        return np.asarray(self.beta, dtype=float)

    @property
    def se_arr(self) -> np.ndarray:
        """Standard errors as a NumPy array."""
        return np.asarray(self.se, dtype=float)

    @property
    def z_arr(self) -> np.ndarray:
        """Z-scores as a NumPy array."""
        return np.asarray(self.z, dtype=float)

    @property
    def maf_arr(self) -> np.ndarray:
        """Minor-allele frequencies as a NumPy array."""
        return np.asarray(self.maf, dtype=float)


class ColocDatasets(BaseModel):
    """A pair of single-trait :class:`SummaryStats` plus shared LD and truth."""

    trait1: SummaryStats
    trait2: SummaryStats
    ld: list[list[float]]
    causal_idx_1: int
    causal_idx_2: int
    shared_causal: bool

    @property
    def ld_arr(self) -> np.ndarray:
        """LD matrix as a NumPy array."""
        return np.asarray(self.ld, dtype=float)


class MRDatasets(BaseModel):
    """Instrument-level exposure/outcome effects for a two-sample MR analysis."""

    beta_exposure: list[float]
    se_exposure: list[float]
    beta_outcome: list[float]
    se_outcome: list[float]
    effect_allele: list[str]
    other_allele: list[str]
    rsid: list[str]
    true_effect: float

    @property
    def bx(self) -> np.ndarray:
        """Exposure effect sizes as a NumPy array."""
        return np.asarray(self.beta_exposure, dtype=float)

    @property
    def bxse(self) -> np.ndarray:
        """Exposure standard errors as a NumPy array."""
        return np.asarray(self.se_exposure, dtype=float)

    @property
    def by(self) -> np.ndarray:
        """Outcome effect sizes as a NumPy array."""
        return np.asarray(self.beta_outcome, dtype=float)

    @property
    def byse(self) -> np.ndarray:
        """Outcome standard errors as a NumPy array."""
        return np.asarray(self.se_outcome, dtype=float)


def ar1_ld_matrix(n_snps: int, rho: float) -> np.ndarray:
    r"""Build an AR(1) LD correlation matrix.

    Entry ``(i, j)`` equals ``rho ** |i - j|``, the standard first-order
    autoregressive correlation structure. This is always positive
    semi-definite for ``0 <= rho < 1``.

    Parameters
    ----------
    n_snps
        Matrix dimension.
    rho
        Correlation decay parameter in ``[0, 1)``.

    Returns
    -------
    numpy.ndarray
        An ``n_snps x n_snps`` symmetric correlation matrix with unit diagonal.
    """
    idx = np.arange(n_snps)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def block_ld_matrix(n_snps: int, block_size: int, block_rho: float) -> np.ndarray:
    """Build a block-diagonal LD correlation matrix.

    SNPs within the same block of size ``block_size`` share correlation
    ``block_rho``; SNPs in different blocks are uncorrelated. A small ridge is
    added so the matrix is strictly positive definite.

    Parameters
    ----------
    n_snps
        Matrix dimension.
    block_size
        Number of SNPs per block.
    block_rho
        Within-block correlation.

    Returns
    -------
    numpy.ndarray
        Symmetric positive-definite correlation matrix with unit diagonal.
    """
    blocks = np.arange(n_snps) // block_size
    same_block = blocks[:, None] == blocks[None, :]
    r = np.where(same_block, block_rho, 0.0)
    np.fill_diagonal(r, 1.0)
    return _nearest_pd(r)


def _nearest_pd(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Return a symmetric positive-definite version of ``matrix``.

    Clips negative eigenvalues to a small floor, rescales back to unit
    diagonal so the result remains a valid correlation matrix.
    """
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, jitter, None)
    pd = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(pd))
    pd = pd / np.outer(d, d)
    np.fill_diagonal(pd, 1.0)
    return pd


def _maf_to_se(maf: np.ndarray, n: int) -> np.ndarray:
    r"""Standard error of a per-allele effect from a standardized-genotype model.

    For a standardized genotype and (approximately) standardized phenotype the
    sampling SE of a single-SNP marginal effect is ``1 / sqrt(2 f (1-f) N)``
    on the *per-allele* scale, where ``f`` is the allele frequency.
    """
    return 1.0 / np.sqrt(2.0 * maf * (1.0 - maf) * n)


def simulate_locus(config: LocusConfig) -> tuple[SummaryStats, np.ndarray]:
    """Simulate a single-trait GWAS locus with LD.

    Parameters
    ----------
    config
        Locus configuration.

    Returns
    -------
    tuple
        ``(summary_stats, ld_matrix)`` where ``summary_stats`` is a
        :class:`SummaryStats` and ``ld_matrix`` is the ``n_snps x n_snps``
        correlation matrix used to generate it.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_snps

    if config.block_size is not None:
        ld = block_ld_matrix(n, config.block_size, config.block_rho)
    else:
        ld = ar1_ld_matrix(n, config.rho)

    causal_idx = config.causal_idx if config.causal_idx is not None else [n // 2]
    if config.causal_effects is not None:
        effects = config.causal_effects
    else:
        effects = [0.15] * len(causal_idx)
    if len(effects) != len(causal_idx):
        raise ValueError("causal_effects must match causal_idx in length")

    # Joint (true) standardized effects: nonzero only at causal SNPs.
    b = np.zeros(n)
    for i, e in zip(causal_idx, effects, strict=True):
        b[i] = e

    # Marginal standardized effects are the LD-smeared joint effects.
    z_true_per_sd = ld @ b

    # Sample observed Z ~ N(sqrt(N) * R b, R).
    mean_z = np.sqrt(config.n_samples) * z_true_per_sd
    z = rng.multivariate_normal(mean_z, ld)

    maf = rng.uniform(config.maf_low, config.maf_high, size=n)
    se = _maf_to_se(maf, config.n_samples)
    beta = z * se

    rsid = [f"rs{1000 + i}" for i in range(n)]
    stats = SummaryStats(
        beta=beta.tolist(),
        se=se.tolist(),
        z=z.tolist(),
        maf=maf.tolist(),
        n=config.n_samples,
        rsid=rsid,
    )
    return stats, ld


def simulate_coloc(config: ColocConfig) -> ColocDatasets:
    """Simulate two single-trait GWAS over a shared locus for colocalization.

    When ``config.shared_causal`` is ``True`` both traits are driven by the
    same causal SNP (the H4 / colocalization scenario); otherwise each trait
    has its own causal SNP (the H3 / distinct-signal scenario).

    Parameters
    ----------
    config
        Colocalization configuration.

    Returns
    -------
    ColocDatasets
        Both traits' summary statistics, the shared LD matrix, and ground
        truth.
    """
    idx2 = config.causal_idx_1 if config.shared_causal else config.causal_idx_2

    loc1 = LocusConfig(
        n_snps=config.n_snps,
        n_samples=config.n_samples,
        rho=config.rho,
        causal_idx=[config.causal_idx_1],
        causal_effects=[config.effect_1],
        seed=config.seed,
    )
    loc2 = LocusConfig(
        n_snps=config.n_snps,
        n_samples=config.n_samples,
        rho=config.rho,
        causal_idx=[idx2],
        causal_effects=[config.effect_2],
        seed=config.seed + 1,
    )
    s1, ld = simulate_locus(loc1)
    s2, _ = simulate_locus(loc2)

    return ColocDatasets(
        trait1=s1,
        trait2=s2,
        ld=ld.tolist(),
        causal_idx_1=config.causal_idx_1,
        causal_idx_2=idx2,
        shared_causal=config.shared_causal,
    )


def simulate_mr(config: MRConfig) -> MRDatasets:
    """Simulate instrument-level effects for two-sample Mendelian randomization.

    Each instrument SNP has a true exposure effect ``gamma_j``. Its outcome
    effect is ``beta * gamma_j + alpha_j`` where ``beta`` is the causal effect
    and ``alpha_j`` is horizontal pleiotropy. Observed effects add sampling
    noise scaled by the analytic standard errors.

    Parameters
    ----------
    config
        MR configuration.

    Returns
    -------
    MRDatasets
        Harmonized instrument-level exposure and outcome effects.
    """
    rng = np.random.default_rng(config.seed)
    m = config.n_instruments

    # True exposure effects (kept reasonably strong so instruments are valid).
    gamma = config.gamma_mean + config.gamma_sd * rng.standard_normal(m)
    gamma = np.where(np.abs(gamma) < 0.03, np.sign(gamma + 1e-9) * 0.05, gamma)

    # Horizontal pleiotropy: directional component plus optional invalid SNPs.
    alpha = np.full(m, config.pleiotropy)
    if config.invalid_idx:
        for j in config.invalid_idx:
            alpha[j] += config.invalid_effect

    true_outcome = config.causal_effect * gamma + alpha

    # Analytic standard errors for standardized effects.
    maf = rng.uniform(0.1, 0.45, size=m)
    se_x = _maf_to_se(maf, config.n_exposure)
    se_y = _maf_to_se(maf, config.n_outcome)

    bx = gamma + se_x * rng.standard_normal(m)
    by = true_outcome + se_y * rng.standard_normal(m)

    rsid = [f"rs{2000 + j}" for j in range(m)]
    return MRDatasets(
        beta_exposure=bx.tolist(),
        se_exposure=se_x.tolist(),
        beta_outcome=by.tolist(),
        se_outcome=se_y.tolist(),
        effect_allele=["A"] * m,
        other_allele=["G"] * m,
        rsid=rsid,
        true_effect=config.causal_effect,
    )
