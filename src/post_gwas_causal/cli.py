"""Typer-based CLI for post_gwas_causal.

This is the ``[project.scripts]`` entry point. Each of the five methods —
fine-mapping, colocalization, Mendelian randomization, TWAS, and PRS — is
reachable as a sub-command, plus a ``simulate`` command that writes a synthetic
locus to disk so the pipeline is runnable end-to-end without external data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from post_gwas_causal import __version__
from post_gwas_causal.coloc.abf import coloc_abf
from post_gwas_causal.finemap.abf import finemap_abf
from post_gwas_causal.finemap.susie import finemap_susie
from post_gwas_causal.mr.egger import mr_egger
from post_gwas_causal.mr.heterogeneity import cochran_q
from post_gwas_causal.mr.ivw import mr_ivw
from post_gwas_causal.mr.weighted_median import mr_weighted_median
from post_gwas_causal.prs.clump_threshold import clump_and_threshold
from post_gwas_causal.prs.evaluate import evaluate_prs
from post_gwas_causal.simulate import (
    ColocConfig,
    LocusConfig,
    MRConfig,
    simulate_coloc,
    simulate_locus,
    simulate_mr,
)
from post_gwas_causal.twas.burden import twas_burden

app = typer.Typer(
    name="post-gwas-causal",
    help="post-GWAS causal-mechanism toolkit — see README.md for the biology this answers.",
    no_args_is_help=True,
)
console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def version() -> None:
    """Print the installed version and exit."""
    console.print(f"post-gwas-causal v{__version__}")


@app.command()
def simulate(
    out: Path = typer.Option(..., "--out", "-o", help="Output JSON path for the locus."),
    n_snps: int = typer.Option(50, help="Number of SNPs in the locus."),
    seed: int = typer.Option(0, help="RNG seed."),
) -> None:
    """Simulate a single-trait GWAS locus and write summary stats to JSON."""
    stats, ld = simulate_locus(LocusConfig(n_snps=n_snps, seed=seed))
    payload = {"summary": stats.model_dump(), "ld": ld.tolist()}
    out.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]Wrote {n_snps}-SNP locus to {out}[/green]")


@app.command()
def finemap(
    n_snps: int = typer.Option(50, help="Number of SNPs."),
    seed: int = typer.Option(0, help="RNG seed."),
    method: str = typer.Option("abf", help="Fine-mapping method: 'abf' or 'susie'."),
) -> None:
    """Fine-map a simulated locus and print the credible set."""
    stats, ld = simulate_locus(LocusConfig(n_snps=n_snps, causal_idx=[n_snps // 2], seed=seed))
    if method == "abf":
        res = finemap_abf(stats.beta_arr, stats.se_arr)
        cs = res.credible_set
        top = res.top_snp
    elif method == "susie":
        sus = finemap_susie(stats.z_arr, ld, stats.n)
        cs = sus.credible_sets[0] if sus.credible_sets else []
        top = int(np.argmax(sus.pip_arr))
    else:
        raise typer.BadParameter("method must be 'abf' or 'susie'")

    table = Table(title=f"Fine-mapping ({method})")
    table.add_column("Quantity")
    table.add_column("Value")
    table.add_row("Top SNP", str(top))
    table.add_row("Credible set size", str(len(cs)))
    table.add_row("Credible set", ", ".join(map(str, cs[:10])))
    console.print(table)


@app.command()
def coloc(
    shared: bool = typer.Option(True, help="Simulate a shared causal variant (H4)."),
    seed: int = typer.Option(0, help="RNG seed."),
) -> None:
    """Run colocalization on two simulated traits and print PP.H0-H4."""
    data = simulate_coloc(ColocConfig(shared_causal=shared, seed=seed))
    res = coloc_abf(
        data.trait1.beta_arr,
        data.trait1.se_arr,
        data.trait2.beta_arr,
        data.trait2.se_arr,
    )
    table = Table(title="Colocalization (coloc.abf)")
    table.add_column("Hypothesis")
    table.add_column("Posterior")
    for name, value in res.as_dict().items():
        table.add_row(name, f"{value:.4f}")
    console.print(table)


@app.command()
def mr(
    pleiotropy: float = typer.Option(0.0, help="Directional pleiotropy to inject."),
    seed: int = typer.Option(0, help="RNG seed."),
) -> None:
    """Run IVW, MR-Egger, weighted-median, and Cochran's Q on simulated data."""
    data = simulate_mr(MRConfig(pleiotropy=pleiotropy, seed=seed))
    ivw = mr_ivw(data.bx, data.by, data.byse)
    egger = mr_egger(data.bx, data.by, data.byse)
    wm = mr_weighted_median(data.bx, data.by, data.bxse, data.byse, n_bootstrap=200)
    q = cochran_q(data.bx, data.by, data.byse)

    table = Table(title=f"Mendelian randomization (true beta = {data.true_effect})")
    table.add_column("Method")
    table.add_column("Estimate")
    table.add_column("SE")
    table.add_column("p-value")
    table.add_row("IVW (random)", f"{ivw.beta:.4f}", f"{ivw.se:.4f}", f"{ivw.pvalue:.2e}")
    table.add_row(
        "MR-Egger", f"{egger.slope:.4f}", f"{egger.slope_se:.4f}", f"{egger.slope_pvalue:.2e}"
    )
    table.add_row(
        "  intercept",
        f"{egger.intercept:.4f}",
        f"{egger.intercept_se:.4f}",
        f"{egger.intercept_pvalue:.2e}",
    )
    table.add_row("Weighted median", f"{wm.beta:.4f}", f"{wm.se:.4f}", f"{wm.pvalue:.2e}")
    console.print(table)
    console.print(f"Cochran's Q = {q.q:.2f} (df={q.dof}, p={q.pvalue:.2e}, I^2={q.i_squared:.2f})")


@app.command()
def twas(
    n_snps: int = typer.Option(20, help="Number of model SNPs."),
    seed: int = typer.Option(0, help="RNG seed."),
) -> None:
    """Run a FUSION-style TWAS burden test on a simulated locus."""
    stats, ld = simulate_locus(LocusConfig(n_snps=n_snps, causal_idx=[n_snps // 2], seed=seed))
    rng = np.random.default_rng(seed)
    weights = stats.z_arr / np.sqrt(stats.n) + 0.01 * rng.standard_normal(n_snps)
    res = twas_burden(weights, stats.z_arr, ld)
    console.print(f"TWAS Z = {res.z:.3f}, p = {res.pvalue:.3e} ({res.n_snps} SNPs)")


@app.command()
def prs(
    n_individuals: int = typer.Option(500, help="Target-sample size."),
    n_snps: int = typer.Option(40, help="Number of SNPs."),
    seed: int = typer.Option(0, help="RNG seed."),
) -> None:
    """Build a C+T polygenic score on a simulated target sample and evaluate it."""
    rng = np.random.default_rng(seed)
    causal = np.zeros(n_snps)
    causal[[5, 15, 25]] = [0.4, -0.3, 0.35]
    genotypes = rng.binomial(2, 0.3, size=(n_individuals, n_snps)).astype(float)
    pheno = genotypes @ causal + rng.standard_normal(n_individuals)

    # Discovery GWAS effect estimates and p-values (marginal regressions).
    beta = np.empty(n_snps)
    pval = np.empty(n_snps)
    from scipy import stats as sps

    for j in range(n_snps):
        g = genotypes[:, j]
        gc = g - g.mean()
        denom = float(gc @ gc)
        b = float(gc @ (pheno - pheno.mean()) / denom) if denom > 0 else 0.0
        resid = pheno - b * gc
        se = float(np.sqrt(resid.var(ddof=2) / denom)) if denom > 0 else 1.0
        beta[j] = b
        pval[j] = float(2.0 * sps.norm.sf(abs(b / se))) if se > 0 else 1.0

    ld = np.corrcoef(genotypes, rowvar=False)
    res = clump_and_threshold(beta, pval, ld, genotypes, clump_r2=0.1, p_threshold=0.05)
    ev = evaluate_prs(res.scores_arr, pheno)
    console.print(
        f"C+T selected {res.n_selected} SNPs; PRS R^2 = {ev.r2:.3f} (r = {ev.pearson_r:.3f})"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
