# post-gwas-causal

Post-GWAS causal-mechanism toolkit — fine-mapping, colocalization, Mendelian
randomization, TWAS, and polygenic risk scoring, all from GWAS summary
statistics.

A genome-wide association study (GWAS) localizes disease signal to a region of
the genome. This toolkit takes that signal the rest of the way to a mechanism:
*which variant*, *which gene*, *which direction*, and *how much risk*.

## The five methods

| Method | Question it answers | Module |
| --- | --- | --- |
| Fine-mapping | Which SNP at this locus is causal? | `finemap` |
| Colocalization | Do two traits share the same causal variant? | `coloc` |
| Mendelian randomization | Does the exposure causally affect the outcome? | `mr` |
| TWAS | Is predicted expression of this gene associated? | `twas` |
| Polygenic risk scoring | What is each person's genetic risk? | `prs` |

## Where to go next

- [Biology primer](biology.md) — the plain-language story behind each method.
- [Methods](methods.md) — the statistics and citations.
- [Architecture](architecture.md) — package layout and design.
- [Decision log](adrs/index.md) — the three architecture decision records.

## Install

```bash
uv sync --all-extras --dev
uv run post-gwas-causal --help
```
