#!/usr/bin/env Rscript
# Mirror of post_gwas_causal.coloc.abf against the canonical R `coloc` package.
#
# Reads a JSON object {beta1, se1, beta2, se2} from stdin and writes the
# coloc.abf posterior probabilities {PP.H0.abf .. PP.H4.abf, nsnps} to stdout.

suppressMessages({
  library(jsonlite)
  library(coloc)
})

input <- fromJSON(file("stdin"))

beta1 <- as.numeric(input$beta1)
se1 <- as.numeric(input$se1)
beta2 <- as.numeric(input$beta2)
se2 <- as.numeric(input$se2)
n <- length(beta1)
snp <- paste0("SNP", seq_len(n))

# coloc expects variance (se^2) and an MAF; effect sizes drive the ABF.
dataset1 <- list(
  beta = beta1,
  varbeta = se1^2,
  snp = snp,
  type = "quant",
  sdY = 1,
  N = if (!is.null(input$n1)) input$n1 else 10000
)
dataset2 <- list(
  beta = beta2,
  varbeta = se2^2,
  snp = snp,
  type = "quant",
  sdY = 1,
  N = if (!is.null(input$n2)) input$n2 else 10000
)

res <- coloc.abf(dataset1, dataset2)
summary <- res$summary

out <- list(
  "PP.H0.abf" = unname(summary["PP.H0.abf"]),
  "PP.H1.abf" = unname(summary["PP.H1.abf"]),
  "PP.H2.abf" = unname(summary["PP.H2.abf"]),
  "PP.H3.abf" = unname(summary["PP.H3.abf"]),
  "PP.H4.abf" = unname(summary["PP.H4.abf"]),
  "nsnps" = unname(summary["nsnps"])
)

cat(toJSON(out, auto_unbox = TRUE, digits = 10))
