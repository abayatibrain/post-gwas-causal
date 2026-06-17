"""Tests for the optional R bridge.

The R cross-checks are optional; every test that actually invokes ``Rscript``
is marked ``requires_r`` and skips cleanly when R is absent.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from post_gwas_causal.coloc.abf import coloc_abf
from post_gwas_causal.rbridge import (
    RscriptNotFoundError,
    require_rscript,
    rscript_available,
    run_r_script,
)
from post_gwas_causal.simulate import ColocConfig, simulate_coloc

_HAS_R = shutil.which("Rscript") is not None


def test_rscript_available_matches_path() -> None:
    assert rscript_available() == _HAS_R


def test_require_rscript_raises_when_absent() -> None:
    if _HAS_R:
        # When R is present this should not raise.
        assert require_rscript()
    else:
        with pytest.raises(RscriptNotFoundError):
            require_rscript()


@pytest.mark.requires_r
def test_coloc_python_matches_r() -> None:
    if not _HAS_R:
        pytest.skip("Rscript not available")
    data = simulate_coloc(ColocConfig(shared_causal=True, seed=4))
    py = coloc_abf(
        data.trait1.beta_arr,
        data.trait1.se_arr,
        data.trait2.beta_arr,
        data.trait2.se_arr,
    )
    payload = {
        "beta1": data.trait1.beta,
        "se1": data.trait1.se,
        "beta2": data.trait2.beta,
        "se2": data.trait2.se,
    }
    try:
        r_out = run_r_script("coloc.R", payload)
    except RuntimeError as exc:  # R present but coloc package missing
        pytest.skip(f"R coloc package unavailable: {exc}")
    assert np.isclose(py.pp_h4, r_out["PP.H4.abf"], atol=0.05)
