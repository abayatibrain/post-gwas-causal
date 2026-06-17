#!/usr/bin/env Rscript
# Mirror of post_gwas_causal.mr.* against the canonical R `TwoSampleMR`.
#
# Reads {bx, bxse, by, byse} from stdin and writes IVW, MR-Egger, and
# weighted-median estimates to stdout.

suppressMessages({
  library(jsonlite)
  library(TwoSampleMR)
})

input <- fromJSON(file("stdin"))

bx <- as.numeric(input$bx)
bxse <- as.numeric(input$bxse)
by <- as.numeric(input$by)
byse <- as.numeric(input$byse)

ivw <- mr_ivw(bx, by, bxse, byse)
egger <- mr_egger_regression(bx, by, bxse, byse)
wmed <- mr_weighted_median(bx, by, bxse, byse, parameters = default_parameters())

out <- list(
  ivw = list(b = ivw$b, se = ivw$se, pval = ivw$pval),
  egger = list(
    b = egger$b,
    se = egger$se,
    pval = egger$pval,
    intercept = egger$b_i,
    intercept_se = egger$se_i,
    intercept_pval = egger$pval_i
  ),
  weighted_median = list(b = wmed$b, se = wmed$se, pval = wmed$pval)
)

cat(toJSON(out, auto_unbox = TRUE, digits = 10))
