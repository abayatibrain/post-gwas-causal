"""Integration tests driving the Typer CLI end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from post_gwas_causal.cli import app

runner = CliRunner()


@pytest.mark.integration
def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "post-gwas-causal" in result.stdout


@pytest.mark.integration
def test_cli_simulate_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "locus.json"
    result = runner.invoke(app, ["simulate", "--out", str(out), "--n-snps", "30"])
    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert len(payload["summary"]["beta"]) == 30
    assert len(payload["ld"]) == 30


@pytest.mark.integration
def test_cli_finemap_abf_and_susie() -> None:
    for method in ("abf", "susie"):
        result = runner.invoke(app, ["finemap", "--method", method, "--n-snps", "40"])
        assert result.exit_code == 0, result.stdout
        assert "Fine-mapping" in result.stdout


@pytest.mark.integration
def test_cli_coloc_shared() -> None:
    result = runner.invoke(app, ["coloc", "--shared"])
    assert result.exit_code == 0
    assert "PP.H4" in result.stdout


@pytest.mark.integration
def test_cli_mr() -> None:
    result = runner.invoke(app, ["mr", "--pleiotropy", "0.0"])
    assert result.exit_code == 0
    assert "IVW" in result.stdout
    assert "Cochran" in result.stdout


@pytest.mark.integration
def test_cli_twas_and_prs() -> None:
    twas = runner.invoke(app, ["twas", "--n-snps", "15"])
    assert twas.exit_code == 0
    assert "TWAS Z" in twas.stdout
    prs = runner.invoke(app, ["prs", "--n-individuals", "300", "--n-snps", "30"])
    assert prs.exit_code == 0
    assert "PRS R^2" in prs.stdout
