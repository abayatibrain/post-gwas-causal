"""Thin bridge to the field-standard R packages in ``scripts/R/``.

The Python implementations in this toolkit are validated against the canonical
R packages (``susieR``, ``coloc``, ``TwoSampleMR``) via the scripts in
``scripts/R/``. This module shells out to ``Rscript`` to run them.

R is *optional*. If ``Rscript`` is not on ``PATH``, :func:`require_rscript`
raises :class:`RscriptNotFoundError`, which tests catch to skip cleanly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = ["RscriptNotFoundError", "require_rscript", "rscript_available", "run_r_script"]

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "R"


class RscriptNotFoundError(RuntimeError):
    """Raised when ``Rscript`` is required but not found on ``PATH``."""


def rscript_available() -> bool:
    """Return ``True`` if an ``Rscript`` executable is on ``PATH``."""
    return shutil.which("Rscript") is not None


def require_rscript() -> str:
    """Return the path to ``Rscript`` or raise :class:`RscriptNotFoundError`.

    Returns
    -------
    str
        Absolute path to the ``Rscript`` executable.

    Raises
    ------
    RscriptNotFoundError
        If no ``Rscript`` is found.
    """
    path = shutil.which("Rscript")
    if path is None:
        raise RscriptNotFoundError(
            "Rscript not found on PATH. The R cross-checks are optional; "
            "install R and the susieR/coloc/TwoSampleMR packages to enable them. "
            "See scripts/R/README.md."
        )
    return path


def run_r_script(script_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run an R script in ``scripts/R/`` with a JSON payload on stdin.

    The R scripts read a single JSON object from stdin and print a single JSON
    object to stdout. This keeps the Python/R contract simple and dependency-free
    on the R side beyond the analysis package and ``jsonlite``.

    Parameters
    ----------
    script_name
        File name within ``scripts/R/`` (e.g. ``"coloc.R"``).
    payload
        JSON-serializable inputs for the script.

    Returns
    -------
    dict
        Parsed JSON output from the R script.

    Raises
    ------
    RscriptNotFoundError
        If ``Rscript`` is unavailable.
    FileNotFoundError
        If the requested script does not exist.
    RuntimeError
        If the R process exits non-zero or returns unparseable output.
    """
    rscript = require_rscript()
    script_path = _SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"R script not found: {script_path}")

    logger.debug("Running R script {} via {}", script_path, rscript)
    proc = subprocess.run(
        [rscript, "--vanilla", str(script_path)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"R script {script_name} failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not parse JSON from {script_name}:\n{proc.stdout}") from exc
