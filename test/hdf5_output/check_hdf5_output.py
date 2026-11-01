#!/usr/bin/env python3
"""Check the HDF5 post-processing output against the text reference files.

Usage: check_hdf5_output.py <post_processing.hdf5> <text reference directory>
                            <results directory of the HDF5 run> <scripts/postprocessing>

A deck run with output_format = "hdf5" must hold, for every text output file of the same deck
run with text output (e.g. the multi-material regression deck, multi_material_hdf5.toml.in,
against test/data/test_results/multi_material_test), a group with the same columns and values:
- avg_<file>_<global|region_<region>>.txt   -> volume/<quantity>/<region> (user functions:
                                                avg_<name> -> volume/<name>)
- stats/<name>_<statistic>_<...>.txt         -> volume/<name>/<region>/statistics/<statistic>
- stats/<name>_hist_<...>.txt                -> volume/<name>/<region>/histogram (+ bin_edges,
                                                equal to the "# Bin edges:" line)
- surface_quantities_surface_<id>.txt        -> surface/surface_<id> (+ description, the text
                                                file's comment lines)
Groups are read with read_post_processing.py (the example reader), so the reader is checked too.
Also checks that the HDF5 run wrote no text output and no stats/ folder.
Exits with status 1 and a message for every failed check.
"""

import glob
import os
import sys

import h5py
import numpy as np

# Text file stems of the built-in averages -> quantity names in the HDF5 file
AVERAGE_STEMS = {"avg_stress": "stress", "avg_def_grad": "def_grad",
                 "avg_euler_strain": "euler_strain", "avg_pl_work": "plastic_work",
                 "avg_eq_pl_strain": "eq_pl_strain", "avg_elastic_strain": "elastic_strain"}


def read_text_file(path):
    """Comment lines, column names and values of a text output file."""
    with open(path) as f:
        lines = f.read().splitlines()
    comments = [line for line in lines if line.lstrip().startswith("#")]
    # The column header is the last comment line; its first word is "#"
    columns = comments[-1].split()[1:]
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    return comments[:-1], columns, data


def split_suffix(stem):
    """'<base>_global' or '<base>_region_<region>' -> (base, region)"""
    if stem.endswith("_global"):
        return stem[: -len("_global")], "global"
    base, region = stem.split("_region_", 1)
    return base, region


def group_of(relative_path):
    """HDF5 group of a text reference file (relative to the reference directory)."""
    folder, name = os.path.split(relative_path)
    stem = os.path.splitext(name)[0]
    if stem.startswith("surface_quantities_surface_"):
        return "surface/surface_" + stem.rsplit("_", 1)[1], "surface"
    base, region = split_suffix(stem)
    if folder == "stats":
        name, kind = base.rsplit("_", 1)
        if kind == "hist":
            return f"volume/{name}/{region}/histogram", "histogram"
        return f"volume/{name}/{region}/statistics/{kind}", "statistic"
    # Built-in averages have their own file stems; user functions are avg_<name>
    quantity = AVERAGE_STEMS.get(base, base[len("avg_"):])
    return f"volume/{quantity}/{region}", "average"


def main(hdf5_path, reference_dir, results_dir, scripts_dir):
    sys.path.insert(0, scripts_dir)
    import read_post_processing as reader

    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    references = sorted(os.path.relpath(p, reference_dir) for p in
                        glob.glob(os.path.join(reference_dir, "**", "*.txt"), recursive=True))
    check(len(references) > 0, f"no reference files in {reference_dir}")

    with h5py.File(hdf5_path, "r") as h5file:
        groups = set(reader.list_groups(h5file))
        matched = set()
        for relative in references:
            group, kind = group_of(relative)
            if group not in groups:
                failures.append(f"{relative}: no group {group}")
                continue
            matched.add(group)
            comments, columns, text = read_text_file(os.path.join(reference_dir, relative))
            values = reader.read_group(h5file, group)
            # The text files' first columns: Time, Volume (Time, Surface_ID, Area for surfaces)
            if kind == "surface":
                columns = ["time", "area"] + columns[3:]
                text = np.delete(text, 1, axis=1)
            else:
                columns = ["time", "volume"] + columns[2:]
            check(list(values.keys()) == columns,
                  f"{group}: datasets {list(values.keys())} are not the text columns {columns}")
            if list(values.keys()) != columns:
                continue
            hdf5 = np.column_stack([values[c] for c in columns])
            check(hdf5.shape == text.shape,
                  f"{group}: {hdf5.shape} values, the text file has {text.shape}")
            if hdf5.shape != text.shape:
                continue
            # The text files carry 8 significant digits; the reference run's solver round-off
            # is scaled by each column's size (skewness and kurtosis amplify it, see
            # test_mechanics.py)
            rtol = 1e-5 if ("skewness" in group or "kurtosis" in group) else 1e-6
            scale = np.maximum(np.abs(text).max(axis=0), 1e-30)
            scale = np.maximum(scale, 1e-6 * scale.max())
            bad = np.abs(hdf5 - text) > rtol * scale + 1e-12
            check(not bad.any(), f"{group}: {bad.sum()} values differ from {relative}")

            if kind == "histogram":
                edges_line = [c for c in comments if c.startswith("# Bin edges")]
                edges = [float(v) for v in edges_line[0].split(":")[1].split()]
                check(np.allclose(h5file[group]["bin_edges"][:], edges, rtol=1e-7, atol=1e-12),
                      f"{group}: bin_edges differ from the text header")
                histogram = reader.read_histogram(h5file, group.split("/")[1],
                                                  group.split("/")[2])
                for name, fractions in histogram.fractions.items():
                    check(np.allclose(fractions.sum(axis=1), 1.0, atol=1e-10),
                          f"{group}: {name} fractions do not sum to 1")
            if kind == "surface":
                description = reader.read_text(h5file, group + "/description")
                check(description.splitlines() == comments,
                      f"{group}: description is not the text file's comment lines")

        extra = sorted(g for g in groups if g not in matched)
        check(not extra, f"groups without a text reference file: {extra}")

    written_text = [p for p in glob.glob(os.path.join(results_dir, "**", "*.txt"), recursive=True)
                    if os.sep + "logs" + os.sep not in p]
    check(not written_text, f"the HDF5 run wrote text output: {written_text[:5]}")
    check(not os.path.exists(os.path.join(results_dir, "stats")),
          "the HDF5 run created a stats/ folder")

    for message in failures:
        print("FAILED:", message)
    if not failures:
        print(f"All {len(references)} text reference files match the HDF5 output")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
