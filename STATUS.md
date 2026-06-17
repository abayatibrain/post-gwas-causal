# Status — 2026-06-17

Repo: post-gwas-causal
Phase: **v0.1.0 landed** — all five post-GWAS method families implemented as
real algorithms on summary statistics, with known-answer tests passing.

## Completed

- **Simulation core** — LD-aware GWAS summary-statistic generator (AR(1) and
  block LD), two-trait coloc scenarios (shared / distinct causal), and
  two-sample MR generator with tunable pleiotropy and invalid instruments.
- **Fine-mapping** — Wakefield ABF and a working SuSiE-RSS (IBSS) finemapper.
  Credible sets contain the true causal SNP at fixed seeds.
- **Colocalization** — `coloc.abf` with PP.H0–H4. PP.H4 > 0.8 under a shared
  causal variant; PP.H3 dominates under distinct causals.
- **Mendelian randomization** — IVW (matches the closed-form weighted slope),
  MR-Egger (intercept ≈ 0 under no pleiotropy, nonzero under injected
  directional pleiotropy), weighted median (robust to an invalid instrument),
  Cochran's Q, and effect-allele harmonization.
- **TWAS** — FUSION burden Z reproduces the hand-built `w'Z / sqrt(w'Rw)`.
- **PRS** — C+T (correlates with the simulated phenotype) and LDpred-inf
  shrinkage, with R²/AUC evaluation.
- **CLI** — every method reachable via `post-gwas-causal <command>`.
- **R cross-checks** — `scripts/R/{susie,coloc,mr}.R` mirror the Python
  implementations; the Python bridge skips cleanly without `Rscript`.

## Next

- Real-data loaders (OpenGWAS / GWAS Catalog harmonized summary stats).
- Full SuSiE ELBO + residual-variance estimation.
- Susie-coloc (coloc on credible sets) and multi-trait MR (MVMR).
- Demo notebook walking a single locus through all five methods.
