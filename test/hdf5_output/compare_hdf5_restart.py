#!/usr/bin/env python3
"""Check that a restarted run continued the HDF5 post-processing output without repeats.

Usage: compare_hdf5_restart.py <reference post_processing.hdf5> <restarted post_processing.hdf5>
                               <scripts/postprocessing>

The reference is the file of an uninterrupted run that wrote checkpoints; the restarted file is
the same file after a restart from an earlier checkpoint continued in the same results folder.
The file held every step of the reference before the restart, so the steps after the checkpoint
must have been removed and written again once: the two files must have the same groups and
datasets with identical values (a same-rank restart is bitwise identical), and every group's
times must increase strictly.
Exits with status 1 and a message for every failed check.
"""

import sys

import h5py
import numpy as np


def main(reference_path, restarted_path, scripts_dir):
    sys.path.insert(0, scripts_dir)
    import read_post_processing as reader

    failures = []
    with h5py.File(reference_path, "r") as ref, h5py.File(restarted_path, "r") as new:
        ref_groups, new_groups = reader.list_groups(ref), reader.list_groups(new)
        if ref_groups != new_groups:
            failures.append("the files have different groups")
        for group in ref_groups:
            if group not in new_groups:
                continue
            a, b = reader.read_group(ref, group), reader.read_group(new, group)
            if list(a) != list(b):
                failures.append(f"{group}: different datasets")
                continue
            if "time" in b and not np.all(np.diff(b["time"]) > 0):
                failures.append(f"{group}: times do not increase (a step was written twice)")
            for name in a:
                if not np.array_equal(a[name], b[name]):
                    failures.append(f"{group}/{name}: {len(b[name])} values, reference "
                                    f"{len(a[name])}, or the values differ")
        for path in ("volume/stress/global/histogram/bin_edges", "surface/surface_1/description"):
            if path in ref and not np.array_equal(ref[path][()], new[path][()]):
                failures.append(f"{path} differs")

    for message in failures[:20]:
        print("FAILED:", message)
    if not failures:
        print(f"The restarted file equals the reference ({len(ref_groups)} groups)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
