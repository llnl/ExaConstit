"""
Shared pytest fixtures for the workflow_common test suite.

What's here
-----------
Fixtures that are reused across multiple test files. Two broad
categories:

1. **Workspace fixtures** - provide throwaway directories for tests
   to write into. pytest's built-in ``tmp_path`` handles the basics;
   the ``workspace`` fixture below adds the repo path to sys.path
   and configures logging so individual tests do not have to.

2. **Fake simulation fixtures** - produce a small executable Python
   script that mimics just enough of ExaConstit's output conventions
   for the integration tests to exercise the full pipeline without
   requiring the real simulation code to be installed.

Why a separate "fake binary" instead of mocking the backend
-----------------------------------------------------------
Integration tests that mock out the backend only verify the Python
code paths. They cannot catch bugs in how environment variables are
set, how working directories are created, how output files are
written and closed, how subprocess stdio is captured, or how timeouts
interact with partially-written files. All of those have bitten real
ExaConstit workflows at least once. Exercising a real subprocess
with a stand-in binary catches those bugs; pure mocks do not.

The fake binary is tiny - about 30 lines of Python with no imports
beyond the stdlib - so tests that use it stay fast (tens of
milliseconds each).
"""
from __future__ import annotations

import os
import stat
import sys
import textwrap
from pathlib import Path
from typing import Callable

import pytest

# Make workflow_common importable without install when running the
# tests directly against a source checkout. The repository root is
# two levels up from this file (``tests/conftest.py``). When
# workflow_common is pip-installed (e.g. `pip install .[test]`)
# the repo-root path is harmless: importing still picks up the
# installed package because site-packages is earlier on sys.path
# than the insert at position 0 would place it... actually, to
# favor the INSTALLED package, append rather than insert. That way
# developers doing "pip install -e ." hit the editable install
# first, and users running pytest in a source checkout without
# install still get a working import chain via the appended path.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


# --- Fake simulation binary ----------------------------------------------
#
# Small Python script that plays the role of ExaConstit for testing.
# It reads a handful of key=value pairs from options.toml via regex
# (not a real TOML parser; we don't want a dependency for tests),
# writes a plausible-looking avg_stress.txt, and optionally fails on
# demand via environment variables. This mirrors what the demo script
# uses but is exposed as a fixture for test files.

FAKE_BINARY_SRC = textwrap.dedent(
    r"""
    #!/usr/bin/env python3
    '''Test stand-in for ExaConstit.

    Reads options.toml for strain_rate and yield_stress (both simple
    numeric fields). Writes results/<basename>/avg_stress.txt and
    avg_def_grad.txt with synthetic stress-strain data that follows
    a saturating exponential (Voce-like) response.

    Environment variables the test suite uses to control behavior:
      FAKE_FAIL=1           -> exit 7 with no output
      FAKE_SLEEP=<seconds>  -> sleep between reading and writing
      FAKE_TRUNCATE=1       -> write partial avg_stress.txt then exit 0
                                (simulates a kill-after-write)
      FAKE_MISSING=<name>   -> skip writing the named output file
    '''
    import math
    import os
    import pathlib
    import re
    import sys
    import time

    options = pathlib.Path("options.toml")
    if not options.exists():
        print("options.toml missing", file=sys.stderr)
        sys.exit(2)
    text = options.read_text()

    def _num(key, default):
        m = re.search(rf"{key}\s*=\s*([-\d.eE+]+)", text)
        return float(m.group(1)) if m else default

    strain_rate = _num("strain_rate", 1e-3)
    yield_stress = _num("yield_stress", 200.0)
    hardening = _num("hardening", 2000.0)
    basename = "options"
    m_base = re.search(r'basename\s*=\s*"([^"]+)"', text)
    if m_base:
        basename = m_base.group(1)

    time.sleep(float(os.environ.get("FAKE_SLEEP", "0.01")))
    if os.environ.get("FAKE_FAIL") == "1":
        print("fake_sim: forced failure", file=sys.stderr)
        sys.exit(7)

    results = pathlib.Path("results") / basename
    results.mkdir(parents=True, exist_ok=True)

    skip = os.environ.get("FAKE_MISSING", "")
    truncate = os.environ.get("FAKE_TRUNCATE") == "1"

    # Generate 50 time steps of saturating response.
    # sigma(eps) = yield_stress + hardening * (1 - exp(-50*eps))
    # eps(t) = strain_rate * t, uniaxial along z (matches ExaConstit's
    # z-axis convention; StressStrainExtractor defaults to Szz/F33).
    n = 50
    t_max = 1.0

    # ExaConstit's volume-averaged output files start with an indented
    # commented header line, followed by indented whitespace-separated
    # data rows. Format matches
    # ExaConstit/src/postprocessing/postprocessing_file_manager.hpp
    # :: GetVolumeAverageHeader. The indentation is load-bearing:
    # pandas' C engine mis-handles the "indented # on line 1 + indented
    # data rows" shape, which is why the framework reader switches to
    # the python engine whenever ``comment`` is set. The fake binary
    # emits the exact shape so that path is exercised end-to-end.
    stress_header = (
        "      # Time            Volume             "
        "Sxx               Syy               Szz               "
        "Sxy               Sxz               Syz"
    )
    defgrad_header = (
        "      # Time            Volume             "
        "F11               F12               F13               "
        "F21               F22               F23               "
        "F31               F32               F33"
    )

    rows_stress = [stress_header]
    rows_F = [defgrad_header]
    for i in range(n):
        t = t_max * i / (n - 1)
        eps = strain_rate * t
        # Load along z so Szz is the active component; Sxx = Syy = 0.
        szz = yield_stress + hardening * (1.0 - math.exp(-50.0 * eps))
        volume = 1.0  # unit cube, held constant for simplicity
        rows_stress.append(
            f"    {t:.8e}    {volume:.8e}    "
            f"{0.0:.8e}    {0.0:.8e}    {szz:.8e}    "
            f"{0.0:.8e}    {0.0:.8e}    {0.0:.8e}"
        )
        # Deformation gradient: axial stretch along z so F33 = 1+eps;
        # transverse (F11, F22) set to 1-0.5*eps for a 0.5 nominal
        # Poisson effect; all off-diagonals zero.
        axial = 1.0 + eps
        lateral = 1.0 - 0.5 * eps
        rows_F.append(
            f"    {t:.8e}    {volume:.8e}    "
            f"{lateral:.8e}    {0.0:.8e}    {0.0:.8e}    "
            f"{0.0:.8e}    {lateral:.8e}    {0.0:.8e}    "
            f"{0.0:.8e}    {0.0:.8e}    {axial:.8e}"
        )

    if truncate:
        # Drop last 10 data rows to simulate a killed write. Keep the
        # header so the file still parses (truncation is about partial
        # data, not a corrupt file).
        rows_stress = rows_stress[:-10]

    if skip != "avg_stress":
        (results / "avg_stress.txt").write_text("\n".join(rows_stress) + "\n")
    if skip != "avg_def_grad":
        (results / "avg_def_grad.txt").write_text("\n".join(rows_F) + "\n")
    """
).lstrip()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A temporary working directory for one test.

    Wraps pytest's ``tmp_path`` so every test that needs a workspace
    gets a fresh one, automatically cleaned up on teardown.

    Returns:
        The workspace path. The directory exists and is empty.
    """
    return tmp_path


@pytest.fixture
def fake_binary(workspace: Path) -> Path:
    """Write the fake simulation script into the workspace and chmod +x.

    The script is written once per test. Tests that need to vary its
    behavior control it through environment variables (``FAKE_FAIL``,
    ``FAKE_SLEEP``, ``FAKE_TRUNCATE``, ``FAKE_MISSING``) rather than
    by editing the script, so we never have to reason about source
    interpolation inside a test.

    Args:
        workspace: The per-test temporary directory.

    Returns:
        Path to the executable script.
    """
    path = workspace / "fake_sim.py"
    path.write_text(FAKE_BINARY_SRC)
    # Give everyone execute permission. The script lives under /tmp
    # (or wherever pytest puts tmp_path) so permissive bits are fine.
    path.chmod(
        path.stat().st_mode
        | stat.S_IEXEC
        | stat.S_IXGRP
        | stat.S_IXOTH
    )
    return path


@pytest.fixture
def master_template(workspace: Path) -> Path:
    """Write a minimal ExaConstit-like master template.

    The template exercises the ``%%key%%`` placeholder syntax and
    includes fields the fake binary looks for (``strain_rate``,
    ``yield_stress``, ``hardening``, ``basename``), plus one the
    fake does not use (``temp_k``) so tests can verify that extra
    keys pass through cleanly.
    """
    content = textwrap.dedent(
        """\
        [Problem]
            name = "sim_%%gene%%_%%obj%%"
            basename = "options"
            strain_rate = %%strain_rate%%
            yield_stress = %%yield_stress%%
            hardening = %%hardening%%
            temperature_k = %%temp_k%%
        """
    )
    path = workspace / "master_options.toml"
    path.write_text(content)
    return path


@pytest.fixture(autouse=True)
def clean_fake_env():
    """Remove any FAKE_* env vars leaked in by prior tests.

    ``autouse=True`` means this runs before every test in the suite,
    without requiring tests to ask for it. Prevents a test that sets
    ``FAKE_FAIL`` from silently corrupting the next test if it
    forgets to clean up.
    """
    keys = [k for k in os.environ if k.startswith("FAKE_")]
    for k in keys:
        os.environ.pop(k, None)
    yield
    keys = [k for k in os.environ if k.startswith("FAKE_")]
    for k in keys:
        os.environ.pop(k, None)
