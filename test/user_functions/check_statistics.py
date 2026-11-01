#!/usr/bin/env python3
"""Check the volume statistics test (statistics.toml.in).

Usage: check_statistics.py <results of the multi-rank run> <results of the 1-rank run>
                           <working directory of the multi-rank run> <its number of ranks>

Material 1 (material_A_1) is most of the 5x5x5 auto mesh, material 2 (material_B_2) a 2x2x2
corner block; the multi-rank run reads a decomposition in which rank 0 owns none of it.
Checks, on the stats/ files (one file per statistic, <name>_<statistic>_<region|global>.txt,
and one histogram file per quantity, <name>_hist_<region|global>.txt):
- the multi-rank run had ranks, rank 0 included, owning no element of material 2 (marker files of
  RecordActiveRanks), so the inactive-rank paths and the gather to rank 0 were exercised;
- the exact column headers of every file kind;
- two_value (1 on material 1, 3 on material 2): exact per-material and global statistics and
  histogram volume fractions from the material volumes, in both runs;
- built-in stress statistics: mean equals the stress volume average, min <= mean <= max, std >= 0;
- every histogram's fractions sum to 1 per component;
- the multi-rank and 1-rank statistics agree (two_value and stress).
Exits with status 1 and a message for every failed check.
"""

import glob
import os
import sys

import numpy as np

REGIONS = ["material_A_1", "material_B_2"]
STATS = ["mean", "variance", "std", "skewness", "kurtosis", "min", "max"]  # statistics = "all"
STRESS = ["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"]
EPS = ["Equiv_Plastic_Strain"]
NUM_STEPS = 3
FILE_RTOL = 1e-7  # the files carry 8 significant digits


def load(path):
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    if data.shape[0] != NUM_STEPS:
        raise SystemExit(f"FAILED: {path} has {data.shape[0]} rows, expected {NUM_STEPS}")
    return data


def header_lines(path):
    with open(path) as f:
        return [f.readline().rstrip("\n") for _ in range(2)]


def hist_columns(components, bins):
    columns = []
    for c in components:
        columns += [c + "_below"] + [f"{c}_bin_{b}" for b in range(1, bins + 1)] + [c + "_above"]
    return columns


def main(results_np, results_np1, workdir_np, num_ranks):
    num_ranks = int(num_ranks)
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    def close(a, b, rtol=FILE_RTOL, atol=1e-12):
        return np.allclose(a, b, rtol=rtol, atol=atol)

    # Ranks without material 2, rank 0 among them
    owners = sorted(glob.glob(os.path.join(workdir_np, "active_rank_*_region_2")))
    check(0 < len(owners) < num_ranks,
          f"{len(owners)} of {num_ranks} ranks own material 2; the test needs some rank "
          "without it")
    check(not os.path.exists(os.path.join(workdir_np, "active_rank_0_region_2")),
          "rank 0 owns material 2; the test needs rank 0 without it")

    for results in (results_np, results_np1):
        stats_dir = os.path.join(results, "stats")

        def path(name, kind, region=None):
            suffix = "global" if region is None else "region_" + region
            return os.path.join(stats_dir, f"{name}_{kind}_{suffix}.txt")

        # Headers: statistic files have the average file's columns; histogram files start with
        # the bin edges, then <component>_below, _bin_k, _above for each component
        expected = {
            path("two_value", "mean"): ["#", "Time", "Volume", "two_value"],
            path("stress", "kurtosis", REGIONS[0]): ["#", "Time", "Volume"] + STRESS,
            path("eq_pl_strain", "std"): ["#", "Time", "Volume"] + EPS,
        }
        for p, columns in expected.items():
            check(os.path.exists(p) and header_lines(p)[0].split() == columns,
                  f"{p}: header is not {columns}")
        for name, components, bins, edges in (("two_value", ["two_value"], 4, [0, 1, 2, 3, 4]),
                                              ("eq_pl_strain", EPS, 5,
                                               [0, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4])):
            p = path(name, "hist")
            check(os.path.exists(p), f"missing {p}")
            if os.path.exists(p):
                comment, columns = header_lines(p)
                check(comment.startswith("# Bin edges") and
                      close([float(v) for v in comment.split(":")[1].split()], edges),
                      f"{p}: bin edge line is not {edges}")
                check(columns.split() == ["#", "Time", "Volume"] + hist_columns(components, bins),
                      f"{p}: unexpected histogram header")
        # Every statistic has its own file
        for name in ("stress", "eq_pl_strain", "two_value"):
            for region in REGIONS + [None]:
                for statistic in STATS:
                    check(os.path.exists(path(name, statistic, region)),
                          f"missing {path(name, statistic, region)}")

        # two_value per material: constant 1 or 3
        volumes = {}
        for region, value in zip(REGIONS, (1.0, 3.0)):
            mean = load(path("two_value", "mean", region))
            volumes[region] = mean[:, 1]
            check(close(mean[:, 2], value), f"{region}: two_value mean is not {value}")
            for statistic in ("variance", "std", "skewness", "kurtosis"):
                check(close(load(path("two_value", statistic, region))[:, 2], 0.0),
                      f"{region}: two_value {statistic} is not 0")
            for statistic in ("min", "max"):
                check(close(load(path("two_value", statistic, region))[:, 2], value),
                      f"{region}: two_value {statistic} is not {value}")
            hist = load(path("two_value", "hist", region))
            expected_bin = 2 if value == 1.0 else 4  # [1, 2) is bin 2, [3, 4] is bin 4
            check(close(hist[:, 2 + expected_bin], 1.0),
                  f"{region}: two_value histogram is not all in bin {expected_bin}")

        # two_value over the whole sample: two-value distribution with p = V2 / V
        total = volumes[REGIONS[0]] + volumes[REGIONS[1]]
        p = volumes[REGIONS[1]] / total
        q = 1.0 - p
        exact = {"mean": 1.0 + 2.0 * p, "variance": 4.0 * p * q,
                 "std": 2.0 * np.sqrt(p * q), "skewness": (1.0 - 2.0 * p) / np.sqrt(p * q),
                 "kurtosis": (1.0 - 6.0 * p * q) / (p * q), "min": 1.0, "max": 3.0}
        for statistic, value in exact.items():
            data = load(path("two_value", statistic))
            check(close(data[:, 1], total), f"global two_value {statistic}: volume is wrong")
            check(close(data[:, 2], value), f"global two_value {statistic} is wrong")
        hist = load(path("two_value", "hist"))
        check(close(hist[:, 4], q) and close(hist[:, 6], p),
              "global two_value histogram fractions are not the material volume fractions")

        # Built-in stress: mean equals the volume average, min <= mean <= max
        for region in REGIONS + [None]:
            suffix = "global" if region is None else "region_" + region
            avg = load(os.path.join(results, f"avg_stress_{suffix}.txt"))
            mean = load(path("stress", "mean", region))
            low = load(path("stress", "min", region))
            high = load(path("stress", "max", region))
            std = load(path("stress", "std", region))
            scale = max(np.abs(avg[:, 2:]).max(), 1e-30)
            check(np.abs(mean[:, 2:] - avg[:, 2:]).max() <= FILE_RTOL * scale,
                  f"stress {suffix}: mean differs from the volume average")
            check(np.all(low[:, 2:] <= mean[:, 2:] + FILE_RTOL * scale) and
                  np.all(mean[:, 2:] <= high[:, 2:] + FILE_RTOL * scale),
                  f"stress {suffix}: mean outside [min, max]")
            check(np.all(std[:, 2:] >= 0.0), f"stress {suffix}: negative std")

        # Histogram fractions sum to 1 (both histograms are of one-component quantities)
        for name in ("two_value", "eq_pl_strain"):
            for region in REGIONS + [None]:
                hist = load(path(name, "hist", region))
                check(close(hist[:, 2:].sum(axis=1), 1.0),
                      f"{path(name, 'hist', region)}: fractions do not sum to 1")

    # The multi-rank and 1-rank runs agree (different partitions, same answer). The
    # equivalent plastic strain is left out: after 3 elastic steps it is round-off, so its
    # moments are too.
    compared = sorted(glob.glob(os.path.join(results_np1, "stats", "two_value_*.txt")) +
                      glob.glob(os.path.join(results_np1, "stats", "stress_*.txt")))
    for stats_path in compared:
        other = os.path.join(results_np, "stats", os.path.basename(stats_path))
        check(os.path.exists(other), f"missing {other}")
        if os.path.exists(other):
            a, b = load(stats_path), load(other)
            # Each column relative to its own size, but at least to 1e-6 of the file's largest
            # value: near-zero columns (e.g. transverse stresses) differ at solver tolerance
            scale = np.maximum(np.abs(a), np.abs(b)).max(axis=0)
            scale = np.maximum(scale, 1e-6 * scale.max())
            check(np.all(np.abs(a - b) <= 1e-6 * scale),
                  f"{os.path.basename(stats_path)}: 1-rank and {num_ranks}-rank results differ")

    for message in failures:
        print("FAILED:", message)
    if not failures:
        print("All volume statistics checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
