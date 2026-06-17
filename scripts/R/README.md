# R cross-checks

The Python implementations in `post_gwas_causal` are the primary, tested code.
These R scripts mirror the same analyses against the **field-standard R
packages** so we can confirm the Python output agrees with the canonical
implementations:

| Script | Mirrors Python | R package |
| --- | --- | --- |
| `susie.R` | `finemap.susie` | [`susieR`](https://stephenslab.github.io/susieR/) |
| `coloc.R` | `coloc.abf` | [`coloc`](https://chr1swallace.github.io/coloc/) |
| `mr.R` | `mr.ivw` / `mr.egger` / `mr.weighted_median` | [`TwoSampleMR`](https://mrcieu.github.io/TwoSampleMR/) |

## Contract

Each script reads a single JSON object from **stdin** and writes a single JSON
object to **stdout**, using `jsonlite`. The Python side (`post_gwas_causal.rbridge`)
shells out with `Rscript --vanilla <script>` and parses the JSON result. This
keeps the Python/R boundary dependency-free beyond the analysis package itself.

## Requirements

```r
install.packages("jsonlite")
install.packages("coloc")
install.packages("susieR")
# TwoSampleMR is on GitHub:
remotes::install_github("MRCIEU/TwoSampleMR")
```

## R is optional

R is **not required** to use the toolkit or to run CI. If `Rscript` is not on
`PATH`, `post_gwas_causal.rbridge.require_rscript()` raises a clear
`RscriptNotFoundError`, and every test that touches R is marked
`@pytest.mark.requires_r` and skips cleanly.

## Running manually

```bash
echo '{"beta1":[...],"se1":[...],"beta2":[...],"se2":[...]}' \
  | Rscript --vanilla scripts/R/coloc.R
```
