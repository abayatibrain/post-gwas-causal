# Methods

All methods operate on **summary statistics** (per-SNP `beta`, `se`, `Z`,
`MAF`, `N`) plus, where needed, a reference **LD matrix** `R`. No individual-
level genotypes are required.

## Fine-mapping

### Wakefield approximate Bayes factor (`finemap.abf`)

For a SNP with effect `beta`, standard error `se` (so `V = se²`, `Z = beta/se`)
and prior effect variance `W`, the Wakefield ABF is

$$\mathrm{ABF} = \sqrt{\tfrac{V}{V+W}}\,\exp\!\Big(\tfrac{Z^2}{2}\tfrac{W}{V+W}\Big).$$

Under one-causal-variant-per-locus and a flat prior, the posterior inclusion
probability is the normalized ABF, and the 95% credible set is the smallest set
of SNPs whose PIPs sum to 0.95.

> Wakefield, J. (2009). Bayes factors for genome-wide association studies.
> *Genetic Epidemiology* 33(1):79-86.

### SuSiE-RSS (`finemap.susie`)

SuSiE writes the joint effect vector as a sum of `L` single-effect vectors and
fits them by **Iterative Bayesian Stepwise Selection (IBSS)**: for each effect,
residualize out the other `L-1` effects and recompute a single-SNP posterior
over which SNP that effect lands on. Works from `(z, R, N)`; returns per-SNP
PIPs and one credible set per detected signal. Credible sets are purity-filtered
(minimum within-set LD).

> Wang, G. et al. (2020). A simple new approach to variable selection…
> *JRSS-B* 82(5):1273-1300. — Zou, Y. et al. (2022). Fine-mapping from summary
> data with the Sum of Single Effects model. *PLoS Genetics* 18(7).

## Colocalization (`coloc.abf`)

Giambartolomei's `coloc.abf` enumerates five hypotheses for a shared locus and
returns posterior probabilities PP.H0–PP.H4 from per-SNP ABFs and priors
`p1, p2, p12`:

- **H0** no causal variant; **H1**/**H2** causal for trait 1 / trait 2 only;
- **H3** distinct causal variants; **H4** one *shared* causal variant.

H3 is the off-diagonal of the outer product of per-trait ABF sums (computed
stably in log space); H4 is the diagonal. **PP.H4 > 0.8** is the usual
colocalization threshold.

> Giambartolomei, C. et al. (2014). Bayesian test for colocalisation…
> *PLoS Genetics* 10(5):e1004383.

## Mendelian randomization

### IVW (`mr.ivw`)

The inverse-variance-weighted estimate is the through-the-origin weighted slope
of outcome effects on exposure effects,
$\hat\beta = \frac{\sum_j w_j bx_j by_j}{\sum_j w_j bx_j^2}$ with
$w_j = 1/byse_j^2$ — equivalently the IVW mean of the Wald ratios
`by_j/bx_j`. Fixed-effect and multiplicative random-effect SEs are both
reported.

> Burgess, S. et al. (2013). *Genetic Epidemiology* 37(7):658-665.

### MR-Egger (`mr.egger`)

Relaxes the through-the-origin constraint:
$by_j = \beta_0 + \beta_1 bx_j + \varepsilon_j$ by weighted least squares
(instruments oriented to positive exposure effect). The **slope** `β₁` is the
pleiotropy-adjusted causal estimate; a nonzero **intercept** `β₀` flags
directional (horizontal) pleiotropy.

> Bowden, J. et al. (2015). *Int. J. Epidemiology* 44(2):512-525.

### Weighted median (`mr.weighted_median`)

The weighted median of the Wald ratios (weights `bx²/byse²`) is consistent as
long as ≥50% of the weight is on valid instruments — robust to a minority of
pleiotropic outliers. SE by parametric bootstrap.

> Bowden, J. et al. (2016). *Genetic Epidemiology* 40(4):304-314.

### Cochran's Q (`mr.heterogeneity`)

$Q = \sum_j (by_j - \hat\beta bx_j)^2 / byse_j^2 \sim \chi^2_{m-1}$ under no
heterogeneity; `I²` summarizes the excess. Heterogeneity is a pleiotropy
warning sign.

### Harmonization (`mr.harmonize`)

Aligns effect alleles between exposure and outcome GWAS, flipping outcome signs
where alleles are swapped, resolving strand via complement matching, and
dropping ambiguous palindromic (A/T, C/G) SNPs.

## TWAS (`twas.burden`)

The FUSION burden statistic given expression weights `w`, GWAS Z-scores `Z`,
and SNP LD `R`:

$$Z_{\text{TWAS}} = \frac{w^\top Z}{\sqrt{w^\top R\, w}} \sim N(0,1)$$

under the null of no expression–trait association.

> Gusev, A. et al. (2016). Integrative approaches for large-scale TWAS.
> *Nature Genetics* 48(3):245-252.

## Polygenic risk scoring

### Clumping + thresholding (`prs.clump_threshold`)

Greedily keep the most significant SNP, drop everything in LD (`r² > clump_r2`)
with it, repeat; keep clumped SNPs with `p ≤ p_threshold`; score each individual
as `Σ_j beta_j G_ij`.

> Choi, S.W. et al. (2020). Tutorial: PRS analyses. *Nature Protocols*
> 15:2759-2772.

### LDpred-inf shrinkage (`prs.shrinkage`)

Under an infinitesimal prior, posterior-mean joint effects are
$\tilde\beta = (R + \tfrac{M}{N h^2} I)^{-1}\hat\beta_{\text{marg}}$ — LD-aware
shrinkage that decorrelates and denoises the marginal effects.

> Vilhjálmsson, B.J. et al. (2015). Modeling LD increases PRS accuracy.
> *AJHG* 97(4):576-592.

## Evaluation (`prs.evaluate`)

`R²` (squared Pearson correlation) for quantitative phenotypes; rank-based ROC
AUC for binary phenotypes.
