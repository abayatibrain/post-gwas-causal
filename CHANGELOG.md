# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-17

### Added — initial release

Five post-GWAS analysis families, implemented as real algorithms (no stubs)
operating on summary statistics, validated by known-answer tests.

- **Simulation** (`simulate.py`) — generative locus model with AR(1) / block
  LD, the `marginal = R @ joint` identity, and `Z ~ N(sqrt(N) R b, R)`
  sampling. Two-trait coloc (shared vs distinct causal) and two-sample MR
  (with optional pleiotropy / invalid instruments) generators.
- **Fine-mapping**:
  - `finemap/abf.py` — Wakefield (2009) approximate Bayes factors, PIPs, and
    95% credible sets (single causal variant).
  - `finemap/susie.py` — SuSiE-RSS sum-of-single-effects fine-mapper via
    Iterative Bayesian Stepwise Selection (multiple causal variants).
- **Colocalization** (`coloc/abf.py`) — Giambartolomei (2014) `coloc.abf`
  returning PP.H0–PP.H4 with configurable `p1/p2/p12` priors. H3 computed via
  the stable off-diagonal log-sum-exp identity.
- **Mendelian randomization**:
  - `mr/ivw.py` — inverse-variance-weighted (fixed + random effects).
  - `mr/egger.py` — MR-Egger slope and pleiotropy-intercept test.
  - `mr/weighted_median.py` — weighted-median estimator with bootstrap SE.
  - `mr/heterogeneity.py` — Cochran's Q and `I^2`.
  - `mr/harmonize.py` — effect-allele alignment with palindrome handling.
- **TWAS** (`twas/burden.py`) — FUSION-style burden statistic
  `z = w'Z / sqrt(w'Rw)`.
- **PRS**:
  - `prs/clump_threshold.py` — clumping + p-value thresholding (C+T).
  - `prs/shrinkage.py` — LDpred-inf infinitesimal Bayesian shrinkage.
  - `prs/evaluate.py` — R², Pearson r, and rank-based AUC.
- **CLI** (`cli.py`) — `typer` commands `simulate`, `finemap`, `coloc`, `mr`,
  `twas`, `prs`, `version`.
- **R bridge** (`rbridge.py`) + `scripts/R/{susie,coloc,mr}.R` — optional
  cross-checks against `susieR` / `coloc` / `TwoSampleMR`. Raises a clean,
  skip-able error when `Rscript` is absent.
- **Docs** — `index`, `architecture`, `biology`, `methods`, and three ADRs
  (Wakefield ABF vs SuSiE; Python coloc.abf vs R coloc; IVW + sensitivity MR
  design).
- **Tests** — unit + integration known-answer tests across all five methods;
  `requires_r` tests skip gracefully without R.
