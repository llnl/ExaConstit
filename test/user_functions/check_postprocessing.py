#!/usr/bin/env python3
"""Check the outputs of the post-processing user function test (postprocessing.toml.in).

Usage: check_postprocessing.py <results directory of one run>

The run has two materials (region names material_A_1 and material_B_2) and these
[[PostProcessing.user_functions]] entries (functions in test_user_functions.cpp):
- unit_volume (volume_integral of 1, all materials + global): equals the Volume column;
- stress_invariants (volume_average of mean and von Mises stress, all materials + global):
  its mean stress equals one third of the trace of the built-in volume-averaged stress
  (both are linear in the stress), with the column names given in the options;
- triaxiality (volume_average, material 1 only, no global file);
- "Stress Triaxiality" (visualization, material 2 + global): a field in material 2's and the
  global visualization files only.
Exits with status 1 and a message for every failed check.
"""

import glob
import os
import sys

import numpy as np

REGIONS = ["material_A_1", "material_B_2"]
NUM_STEPS = 3  # [Time.Fixed] t_final / dt of postprocessing.toml.in


def load(path):
    """Columns of a volume average file (header line skipped), one row per output step."""
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    if data.shape[0] != NUM_STEPS:
        raise SystemExit(f"FAILED: {path} has {data.shape[0]} rows, expected {NUM_STEPS}")
    return data


def header(path):
    with open(path) as f:
        return f.readline().split()


def main(results):
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    def path(stem, region=None):
        suffix = "global" if region is None else "region_" + region
        return os.path.join(results, f"avg_{stem}_{suffix}.txt")

    # Volume integral of 1 = volume, per material and overall
    for region in REGIONS + [None]:
        p = path("unit_volume", region)
        check(os.path.exists(p), f"missing {p}")
        if os.path.exists(p):
            data = load(p)
            check(np.allclose(data[:, 2], data[:, 1], rtol=1e-7, atol=0.0),
                  f"{p}: integral of 1 differs from the volume")
            check(header(p) == ["#", "Time", "Volume", "unit_volume"],
                  f"{p}: unexpected header {header(p)}")

    # Mean stress average = trace of the built-in average stress / 3
    for region in REGIONS + [None]:
        p = path("stress_invariants", region)
        s = path("stress", region)
        check(os.path.exists(p) and os.path.exists(s), f"missing {p} or {s}")
        if os.path.exists(p) and os.path.exists(s):
            data = load(p)
            stress = load(s)
            mean_from_stress = (stress[:, 2] + stress[:, 3] + stress[:, 4]) / 3.0
            scale = max(np.abs(mean_from_stress).max(), 1e-30)
            # The files carry 8 significant digits, so agreement is checked to that precision
            check(np.abs(data[:, 2] - mean_from_stress).max() <= 1e-7 * scale,
                  f"{p}: mean stress differs from the built-in average stress")
            check(np.all(data[:, 3] >= 0.0), f"{p}: negative von Mises stress")
            check(header(p) == ["#", "Time", "Volume", "mean", "von_mises"],
                  f"{p}: unexpected header {header(p)}")

    # Triaxiality average: material 1 only, uniaxial tension gives about 1/3
    p = path("triaxiality", REGIONS[0])
    check(os.path.exists(p), f"missing {p}")
    if os.path.exists(p):
        last = load(p)[-1, 2]
        check(0.25 < last < 0.42, f"{p}: triaxiality {last} is not close to 1/3")
    for unwanted in (path("triaxiality", REGIONS[1]), path("triaxiality")):
        check(not os.path.exists(unwanted), f"{unwanted} should not exist")

    # Visualization field: in material 2's and the global output only
    viz = os.path.join(results, "visualizations")
    for folder, expected in (("region_1", False), ("region_2", True), ("global", True)):
        files = sorted(glob.glob(os.path.join(viz, folder, "**", "*.vtu"), recursive=True))
        check(len(files) > 0, f"no visualization files in {os.path.join(viz, folder)}")
        if files:
            with open(files[-1], errors="ignore") as f:
                has_field = 'Name="Stress Triaxiality"' in f.read()
            check(has_field == expected,
                  f"{folder}: field 'Stress Triaxiality' {'missing' if expected else 'present'}")

    for message in failures:
        print("FAILED:", message)
    if not failures:
        print("All post-processing user function checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
