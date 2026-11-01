#!/usr/bin/env python3
"""
Read ExaConstit's HDF5 post-processing output (``post_processing.hdf5``).

With ``output_format = "hdf5"`` in ``[PostProcessing]``, ExaConstit writes
every volume average, volume statistic, histogram, user function average /
integral and surface quantity of a run into one file,
``<output_directory>/<basename>/post_processing.hdf5``, instead of the text
files. This module reads it back into numpy arrays (and, optionally, pandas
tables that look like the text files), and doubles as an example of how to
work with the file directly through h5py.

File layout
-----------
Each text file is a group and each text column a dataset of the same name,
holding one value per output step (row ``k`` of every dataset in a group
belongs to ``time[k]`` of that group)::

    volume/<quantity>/<region>/time, volume, <columns>
    volume/<quantity>/<region>/statistics/<statistic>/<columns>
    volume/<quantity>/<region>/histogram/<column>_below, <column>_bin_1, ...,
                                         <column>_above, bin_edges
    surface/surface_<id>/time, area, Disp_x, ..., description

``<quantity>`` is ``stress``, ``def_grad``, ``euler_strain``,
``plastic_work``, ``eq_pl_strain``, ``elastic_strain`` or a user function's
name; ``<region>`` is ``global`` or ``<material name>_<material number>``.
Statistics and histograms share the time and volume of their average.
``bin_edges`` and ``description`` are the only datasets that are not time
series.

With h5py alone::

    import h5py
    with h5py.File("results/tensile_test/post_processing.hdf5", "r") as f:
        t = f["volume/stress/global/time"][:]
        szz = f["volume/stress/global/Szz"][:]
        szz_std = f["volume/stress/material_A_1/statistics/std/Szz"][:]

What is here
------------
1. :func:`list_groups` — every group holding time series (the "files").
2. :func:`read_group` / :func:`read_table` — one group as a dict of arrays
   or a pandas DataFrame with the text file's columns (statistics groups get
   their average's time and volume).
3. :func:`read_histogram` — a histogram as a :class:`Histogram` record with
   the bin edges and a (steps x slots) array of volume fractions per column.
4. :func:`read_text` — text datasets such as a surface's ``description``.
5. Command line: print the file's groups, or plot one column of a quantity
   (its mean with a +/- one standard deviation band when the statistics are
   in the file). matplotlib is imported only for the plot.

Needs numpy and h5py; pandas only for :func:`read_table`.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)

#: Datasets that are not one value per output step
FIXED_DATASETS = ("bin_edges", "description")


# --- Reading groups ---------------------------------------------------------


def list_groups(h5file: h5py.File) -> List[str]:
    """Every group of the file that holds time series, in file order.

    Args:
        h5file: Open post-processing file.

    Returns:
        Group paths such as ``volume/stress/global``,
        ``volume/stress/global/statistics/std`` and ``surface/surface_1``.
    """
    groups: List[str] = []

    def visit(name: str, item: Union[h5py.Group, h5py.Dataset]) -> None:
        if isinstance(item, h5py.Group) and any(
                isinstance(child, h5py.Dataset) and key not in FIXED_DATASETS
                for key, child in item.items()):
            groups.append(name)

    h5file.visititems(visit)
    return groups


def read_group(h5file: h5py.File, group: str) -> Dict[str, np.ndarray]:
    """The time series of one group, by dataset name, in column order.

    Statistics and histogram groups hold only their columns; their time and
    volume are added from the average's group (two levels up), so every
    result has ``time`` (and ``volume`` for volume quantities) first.

    Args:
        h5file: Open post-processing file.
        group: Group path, e.g. ``volume/stress/material_A_1`` or
            ``volume/stress/global/statistics/mean``.

    Returns:
        Dataset name -> 1-D array with one value per output step.
    """
    node = h5file[group]
    columns: Dict[str, np.ndarray] = {}
    if "time" not in node:
        parts = group.split("/")
        owner = "/".join(parts[:-2] if parts[-2] == "statistics" else parts[:-1])
        for key in ("time", "volume"):
            if key in h5file[owner]:
                columns[key] = h5file[owner][key][:]
    for key, child in node.items():
        if isinstance(child, h5py.Dataset) and key not in FIXED_DATASETS:
            columns[key] = child[:]
    return columns


def read_table(h5file: h5py.File, group: str):
    """One group as a pandas DataFrame, with the columns of its text file.

    Args:
        h5file: Open post-processing file.
        group: Group path (see :func:`read_group`).

    Returns:
        ``pandas.DataFrame`` with one row per output step.
    """
    import pandas as pd  # soft dependency

    return pd.DataFrame(read_group(h5file, group))


def read_text(h5file: h5py.File, path: str) -> str:
    """A text dataset (stored as characters), e.g. a surface's description.

    Args:
        h5file: Open post-processing file.
        path: Dataset path, e.g. ``surface/surface_1/description``.
    """
    raw = h5file[path][()]
    if isinstance(raw, bytes):
        return raw.decode().rstrip("\x00")
    return b"".join(np.asarray(raw).ravel().tolist()).decode().rstrip("\x00")


# --- Histograms -------------------------------------------------------------


@dataclass
class Histogram:
    """Volume fraction histogram of one quantity in one region.

    Fields:
        time: Output times (steps).
        volume: Region volume at each step.
        bin_edges: ``bins + 1`` edges; slot 0 is below ``bin_edges[0]``,
            slots ``1..bins`` the bins, slot ``bins + 1`` above
            ``bin_edges[-1]``.
        fractions: Column name (e.g. ``Szz``) -> array of shape
            ``(steps, bins + 2)``; each row sums to 1.
    """

    time: np.ndarray
    volume: np.ndarray
    bin_edges: np.ndarray
    fractions: Dict[str, np.ndarray]


def read_histogram(h5file: h5py.File, quantity: str, region: str = "global") -> Histogram:
    """The histogram of a quantity in a region.

    Args:
        h5file: Open post-processing file.
        quantity: e.g. ``stress`` or a user function's name.
        region: ``global`` or ``<material name>_<material number>``.
    """
    group = f"volume/{quantity}/{region}/histogram"
    columns = read_group(h5file, group)
    edges = h5file[group]["bin_edges"][:]
    bins = len(edges) - 1
    fractions: Dict[str, np.ndarray] = {}
    for key in columns:
        if key.endswith("_below"):
            name = key[: -len("_below")]
            slots = [f"{name}_below"] + [f"{name}_bin_{b}" for b in range(1, bins + 1)] + \
                [f"{name}_above"]
            fractions[name] = np.column_stack([columns[s] for s in slots])
    return Histogram(time=columns["time"], volume=columns["volume"], bin_edges=edges,
                     fractions=fractions)


# --- Command line -----------------------------------------------------------


def plot_column(h5file: h5py.File, quantity: str, region: str, column: str,
                output: Optional[Path] = None) -> None:
    """Plot one column of a quantity over time, with a +/- 1 std band if available.

    Args:
        h5file: Open post-processing file.
        quantity: e.g. ``stress``.
        region: ``global`` or ``<material name>_<material number>``.
        column: e.g. ``Szz``.
        output: Image file to save; shown on screen when None.
    """
    import matplotlib.pyplot as plt  # soft dependency

    base = f"volume/{quantity}/{region}"
    data = read_group(h5file, base)
    fig, ax = plt.subplots()
    ax.plot(data["time"], data[column], label=f"{column} average")
    std_group = f"{base}/statistics/std"
    if std_group in h5file and column in h5file[std_group]:
        std = h5file[std_group][column][:]
        ax.fill_between(data["time"], data[column] - std, data[column] + std, alpha=0.3,
                        label="+/- 1 std")
    ax.set_xlabel("time")
    ax.set_ylabel(f"{quantity} {column}")
    ax.set_title(f"{quantity} in {region}")
    ax.legend()
    if output is None:
        plt.show()
    else:
        fig.savefig(output, dpi=150)
        logger.info("Saved %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List or plot ExaConstit's HDF5 post-processing output")
    parser.add_argument("file", type=Path, help="post_processing.hdf5")
    parser.add_argument("--plot", nargs=3, metavar=("QUANTITY", "REGION", "COLUMN"),
                        help="plot a column over time, e.g. --plot stress global Szz")
    parser.add_argument("--output", type=Path, default=None,
                        help="save the plot to this image file instead of showing it")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with h5py.File(args.file, "r") as h5file:
        if args.plot:
            plot_column(h5file, *args.plot, output=args.output)
            return
        for group in list_groups(h5file):
            columns = read_group(h5file, group)
            steps = len(next(iter(columns.values())))
            names = [k for k in columns if k not in ("time", "volume", "area")]
            shown = ", ".join(names[:6]) + (", ..." if len(names) > 6 else "")
            print(f"{group}  ({steps} steps): {shown}")


if __name__ == "__main__":
    main()
