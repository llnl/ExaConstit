"""
Guard against the example script rotting silently.

This file does NOT run the example — running it would require a
real ``mechanics`` binary and experimental CSVs. It just proves the
module imports cleanly, which catches:

* Framework imports that were renamed or moved.
* RunConfig / Bounds / Problem signatures that changed.
* The example's custom evaluator classes no longer conforming to
  the framework's "evaluate(results, ctx) -> float" contract in
  a way that would fail on instantiation.

When the API evolves, this test fails at CI time rather than when
a user first tries to run the example.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "nsga3_calibration.py"


@pytest.mark.skipif(
    not EXAMPLE_PATH.is_file(),
    reason="examples/nsga3_calibration.py not present in source tree",
)
def test_example_imports_cleanly():
    """Import the example module without running its main()."""
    spec = importlib.util.spec_from_file_location(
        "nsga3_calibration_example", EXAMPLE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so any circular-ish intra-module refs work
    sys.modules["nsga3_calibration_example"] = module
    try:
        spec.loader.exec_module(module)

        # Sanity-check that the public surface users copy is there.
        assert callable(module.main), "main() missing from example"
        assert hasattr(module, "_StdNormalizedStressEvaluator")
        assert hasattr(module, "_StdNormalizedSlopeEvaluator")

        # Custom evaluators must be constructable with the documented
        # signature - catches drift in the argument list.
        import numpy as np
        import pandas as pd
        from workflow_common import StressStrainExtractor
        df = pd.DataFrame({"strain": [0.0, 0.05, 0.10],
                           "stress": [0.0, 100.0, 150.0]})
        extractor = StressStrainExtractor(
            stress_column="s33",
            strain_source="time_rate", strain_rate=1e-3,
        )
        ev = module._StdNormalizedStressEvaluator(
            experimental=df, extractor=extractor,
        )
        assert hasattr(ev, "evaluate"), \
            "_StdNormalizedStressEvaluator must expose evaluate()"
    finally:
        sys.modules.pop("nsga3_calibration_example", None)
