"""Calibration benchmark: coloc-SuSiE vs classic coloc.abf.

Runs both colocalization methods across many simulated loci under two truths --
a *shared* causal variant (H4 is true) and *distinct* causal variants (H3 is
true) -- and reports the distribution of PP.H4. A well-behaved method puts high
PP.H4 on shared loci and low PP.H4 on distinct loci.

On single-causal-variant loci the two methods should agree closely (coloc-SuSiE
reduces to coloc.abf when SuSiE finds one credible set per trait); the point of
the benchmark is to confirm that the SuSiE machinery does not *degrade* the
single-signal answer while buying the ability to handle multiple signals.

Run:

    python examples/benchmark_coloc_susie.py --n 50
"""

from __future__ import annotations

import argparse

import numpy as np

from post_gwas_causal.coloc.abf import coloc_abf
from post_gwas_causal.coloc.susie import coloc_susie
from post_gwas_causal.simulate import ColocConfig, simulate_coloc


def _sweep(shared: bool, n_reps: int, base_seed: int) -> dict[str, float]:
    abf_h4, sus_h4 = [], []
    for i in range(n_reps):
        d = simulate_coloc(
            ColocConfig(
                n_snps=60, n_samples=30000, rho=0.6, shared_causal=shared,
                causal_idx_1=20, causal_idx_2=45, effect_1=0.12, effect_2=0.12,
                seed=base_seed + i,
            )
        )
        ab = coloc_abf(d.trait1.beta_arr, d.trait1.se_arr, d.trait2.beta_arr, d.trait2.se_arr)
        cs = coloc_susie(d.trait1.z_arr, d.trait2.z_arr, d.ld_arr, d.trait1.n, d.trait2.n)
        abf_h4.append(ab.pp_h4)
        sus_h4.append(cs.best_pp_h4)
    return {
        "abf_mean": float(np.mean(abf_h4)),
        "sus_mean": float(np.mean(sus_h4)),
        "abf_frac_gt_0.8": float(np.mean(np.array(abf_h4) > 0.8)),
        "sus_frac_gt_0.8": float(np.mean(np.array(sus_h4) > 0.8)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="Replicates per scenario.")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    shared = _sweep(True, args.n, args.seed)
    distinct = _sweep(False, args.n, args.seed + 10_000)

    print(f"\nPP.H4 over {args.n} replicates per scenario\n" + "-" * 52)
    print(f"{'scenario':<12}{'method':<14}{'mean PP.H4':>12}{'frac>0.8':>12}")
    print(f"{'shared(H4)':<12}{'coloc.abf':<14}{shared['abf_mean']:>12.3f}{shared['abf_frac_gt_0.8']:>12.2f}")
    print(f"{'shared(H4)':<12}{'coloc-SuSiE':<14}{shared['sus_mean']:>12.3f}{shared['sus_frac_gt_0.8']:>12.2f}")
    print(f"{'distinct(H3)':<12}{'coloc.abf':<14}{distinct['abf_mean']:>12.3f}{distinct['abf_frac_gt_0.8']:>12.2f}")
    print(f"{'distinct(H3)':<12}{'coloc-SuSiE':<14}{distinct['sus_mean']:>12.3f}{distinct['sus_frac_gt_0.8']:>12.2f}")
    print(
        "\nExpected: high PP.H4 for shared, low for distinct, and the two methods "
        "close on these single-signal loci."
    )


if __name__ == "__main__":
    main()
