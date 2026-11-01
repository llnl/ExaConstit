#!/usr/bin/env python3
"""Check the UMAT interface tests (umat.toml.in, umat_two_regions.toml.in).

Usage: check_umat.py <results directory of the tests>
       (holds umat_<case>/ for every case run by test/umat/CMakeLists.txt)

The same hypoelastic material (test_umat_functions.cpp) is run through both UMAT interfaces.
Checks, on the global volume averages of the stress and the deformation gradient:
- standard and batched runs agree, with the default and with the B-bar integrator;
- ExaConstit rotating STRESS / STRAN by DROT (rotate_stress = true) gives the same results as the
  UMAT rotating the stress itself (rotate_stress = false);
- the two-region runs (region 1 batched, region 2 standard, 2 ranks) agree with the one-region
  runs with the same integrator, so the partial-region paths and the rank split give the same
  answer;
- the stress is physical: the axial stress grows every step (the top face is pulled up).
Exits with status 1 and a message for every failed check.
"""

import os
import sys

import numpy as np

NUM_STEPS = 5
FILE_RTOL = 1e-6  # the files carry 8 significant digits; runs differ only by round-off
SZZ_COLUMN = 4  # time, volume, Sxx, Syy, Szz, ...
PAIRS = [
    ("standard_full", "batched_full"),
    ("standard_bbar", "batched_bbar"),
    ("standard_full", "standard_full_selfrot"),
    ("standard_full", "two_regions_full"),
    ("standard_bbar", "two_regions_bbar"),
]
FILES = ["avg_stress_global.txt", "avg_def_grad_global.txt"]


def load(results, case, name):
    path = os.path.join(results, "umat_" + case, name)
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    if data.shape[0] != NUM_STEPS:
        raise SystemExit(f"FAILED: {path} has {data.shape[0]} rows, expected {NUM_STEPS}")
    return data


def main(results):
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    for reference, other in PAIRS:
        for name in FILES:
            a = load(results, reference, name)
            b = load(results, other, name)
            scale = max(np.abs(a).max(), 1.0)
            diff = np.abs(a - b).max()
            check(diff <= FILE_RTOL * scale,
                  f"{other} differs from {reference} in {name}: max diff {diff:.3e}")

    for case in ["standard_full", "standard_bbar", "standard_full_eulerian"]:
        szz = load(results, case, FILES[0])[:, SZZ_COLUMN]
        check(np.all(np.diff(szz) > 0.0) and szz[0] > 0.0,
              f"{case}: axial stress does not grow every step: {szz}")

    for message in failures:
        print("FAILED: " + message)
    if failures:
        sys.exit(1)
    print("UMAT interface checks passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
