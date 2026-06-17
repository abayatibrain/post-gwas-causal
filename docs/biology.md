# Biology primer

## The problem a GWAS leaves you with

A GWAS scans millions of common variants and flags the ones statistically
associated with a trait. But association is not mechanism. Three facts conspire
to keep a GWAS hit from telling you *what is actually going on*:

1. **Linkage disequilibrium (LD).** Nearby variants are inherited together, so
   they are correlated. The most-significant SNP at a locus is usually *not*
   the causal one — it is just well-correlated with whatever is. Dozens of SNPs
   can share nearly identical p-values.
2. **Most variants are non-coding.** They do not change a protein; they
   presumably change *regulation* — how much of some gene is made, in some cell
   type. A GWAS does not tell you which gene.
3. **Association is directionless.** "High LDL is associated with the same
   variants as heart disease" does not, by itself, prove LDL *causes* heart
   disease. Confounding and reverse causation are always on the table.

The five methods in this toolkit each attack one of these gaps.

## Fine-mapping — *which variant?*

Given the summary statistics at a locus and the LD between its SNPs,
fine-mapping computes, for each SNP, the posterior probability that it is the
causal one (its **posterior inclusion probability**, PIP). It then reports a
**credible set**: the smallest group of SNPs that collectively has, say, 95%
probability of containing the causal variant. A locus that fine-maps to a
single SNP is a gift; one that fine-maps to forty equally-likely SNPs tells you
LD has hidden the truth and you need better data.

We provide two flavours: **Wakefield ABF** (assumes one causal variant per
locus — fast, exact, the workhorse) and a **SuSiE-style** finemapper (allows
several causal variants, returning one credible set per signal).

## Colocalization — *does the trait act through this gene?*

Suppose your disease locus sits next to a gene, and you also have an **eQTL**
study showing which variants control that gene's expression. If the disease
signal and the expression signal are driven by the *same* causal variant, that
is strong evidence the disease acts *through that gene*. If they are driven by
*different* variants that just happen to be neighbours, it is a coincidence of
genomic real estate.

`coloc.abf` formalizes this as five hypotheses (H0–H4) and returns their
posterior probabilities. A high **PP.H4** is the colocalization result you
want: one shared causal variant.

## Mendelian randomization — *is it causal, and which way?*

Genetic variants are randomized at conception and fixed for life, so they are
immune to the confounding and reverse-causation that plague observational
epidemiology. If variants that raise an **exposure** (an LDL-raising allele, a
drug target's activity) also raise disease risk *in proportion to their effect
on the exposure*, that is evidence the exposure **causes** the disease. This is
nature's randomized trial.

The catch is **pleiotropy**: a variant might affect the outcome through some
other pathway, breaking the logic. So MR is never one number — it is an
estimate plus a battery of sensitivity analyses (MR-Egger's pleiotropy test,
the robust weighted-median estimator, Cochran's Q for heterogeneity) that all
have to agree before you believe the causal claim.

## TWAS — *which gene, prioritised?*

A TWAS turns an eQTL model into a gene-level test: using the SNP weights that
predict a gene's expression, it asks whether *genetically predicted expression*
of that gene is associated with the trait. It collapses a locus full of SNPs
into one test per gene, prioritising the gene whose predicted expression tracks
the disease.

## Polygenic risk scoring — *how much risk per person?*

For highly polygenic traits, no single variant matters much, but thousands
together do. A polygenic risk score (PRS) sums a person's risk alleles,
weighted by their GWAS effect sizes, into a single number. PRS can stratify a
population into risk tiers — useful for screening, trial enrichment, and
studying gene-environment interplay. We implement the classic
clumping+thresholding recipe and an LD-aware infinitesimal-shrinkage variant.

## Putting it together

A statistical-genetics scientist rarely runs just one of these. The canonical
post-GWAS workflow is: **fine-map** the locus, **colocalize** it with eQTLs to
nominate a gene, confirm with a **TWAS**, sanity-check the direction of effect
with **MR**, and — separately — build a **PRS** to quantify population-level
risk. Each method covers a blind spot of the others.
