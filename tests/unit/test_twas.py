"""Tests for the FUSION-style TWAS burden statistic."""

from __future__ import annotations

import numpy as np
import pytest

from post_gwas_causal.twas.burden import twas_burden


def test_twas_z_matches_hand_built_example() -> None:
    # Two independent SNPs, equal weights, identical Z.
    weights = np.array([1.0, 1.0])
    z = np.array([3.0, 3.0])
    # w'Z = 6, w'Rw = 2 (identity LD), so Z_twas = 6 / sqrt(2).
    res = twas_burden(weights, z)
    assert np.isclose(res.z, 6.0 / np.sqrt(2.0))


def test_twas_with_ld_uses_quadratic_form() -> None:
    weights = np.array([1.0, 1.0])
    z = np.array([2.0, 2.0])
    ld = np.array([[1.0, 0.5], [0.5, 1.0]])
    # w'Z = 4, w'Rw = 1+1+0.5+0.5 = 3, so Z_twas = 4 / sqrt(3).
    res = twas_burden(weights, z, ld)
    assert np.isclose(res.z, 4.0 / np.sqrt(3.0))


def test_twas_pvalue_two_sided() -> None:
    res = twas_burden(np.array([1.0]), np.array([1.96]))
    assert np.isclose(res.pvalue, 0.05, atol=1e-3)


def test_twas_rejects_degenerate_weights() -> None:
    with pytest.raises(ValueError, match="positive"):
        twas_burden(np.array([0.0, 0.0]), np.array([1.0, 1.0]))


def test_twas_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        twas_burden(np.array([1.0, 1.0]), np.array([1.0]))
