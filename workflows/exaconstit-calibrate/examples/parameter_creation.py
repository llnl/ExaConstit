from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from workflow_common import (
    ArchiveDB,
    CallablePropertyWriter,
    CaseTemplater,
    InfinityFailureHandler,
    LocalBackend,
    Manifest,
    ObjectiveSpec,
    Problem,
    ProblemConfig,
    SimCase,
    StressStrainExtractor,
    TemplatePathResolver,
    TemplateTarget,
    TextTableReader,
    TextTableSpec,
    configure_logging,
)
from workflows.optimization import (
    Bounds,
    RunConfig,
    build_reference_points,
    derive_population_size,
    run_nsga3,
)

def write_properties(case_dir: Path, gene: Sequence[float],
                     names: Sequence[str], sim_case: SimCase) -> Path:
    """Write material-model properties for one case.

    Produces a `properties.txt` file whose format matches what
    the old `ExaConstit_Problems.py` wrote. Adapt the body to
    whatever format your mechanics binary actually reads — the
    framework doesn't care about the format, only that the file
    exists and is referenced correctly in options.toml.
    """
    path = case_dir / "properties.txt"
    # Zip gene values with their names so the output is both
    # machine-readable AND human-auditable for debugging.
    gene_dict = dict(zip(names, gene))
    lines = [
        "8.920e-6",
        "0.003435984",
        "1.0e-10",
        "168.4e0",
        "121.4e0",
        "75.2e0",
        "44.0e0",
        f"{gene_dict['mprime']:.6g}",
        "1.0e0",
        f"{gene_dict['h0']:.6g}",
        f"{gene_dict['crss0']:.6g}",
        "122.4e-3",
        "0.0",
        "5.0e9",
        f"{gene_dict['crss0']:.6g}",
        "0.0",
        "-1.0307952",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path