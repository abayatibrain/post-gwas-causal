"""Pytest configuration and shared fixtures for post_gwas_causal."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to checked-in test fixtures (small files only)."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _set_global_seed() -> None:
    """Every test runs with a deterministic seed."""
    random.seed(0)
    np.random.seed(0)
