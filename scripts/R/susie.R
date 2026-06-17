#!/usr/bin/env Rscript
# Mirror of post_gwas_causal.finemap.susie against the canonical R `susieR`.
#
# Reads {z, R, n, L} from stdin (z = marginal Z-scores, R = LD matrix as a
# nested list, n = sample size, L = number of single effects) and writes
# {pip, sets} to stdout.

suppressMessages({
  library(jsonlite)
  library(susieR)
})

input <- fromJSON(file("stdin"))

z <- as.numeric(input$z)
R <- as.matrix(input$R)
n <- as.integer(input$n)
L <- if (!is.null(input$L)) as.integer(input$L) else 5L

# susie_rss fine-maps directly from Z-scores and an LD matrix.
fit <- susie_rss(z = z, R = R, n = n, L = L)

pip <- as.numeric(susie_get_pip(fit))
sets <- susie_get_cs(fit, Xcorr = R)

out <- list(
  pip = pip,
  sets = sets$cs
)

cat(toJSON(out, auto_unbox = TRUE, digits = 10))
