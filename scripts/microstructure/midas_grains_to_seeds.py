#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a MIDAS ``Grains.csv`` file (far-field HEDM grain list) to an ExaConstit seed table.

Each grain becomes one seed of the microstructure generator: the grain centre of mass is the
seed position, the orientation matrix gives the seed orientation, and the equivalent grain
radius gives the seed size (used by the generator's size control, e.g. ``distribution =
"from_source"`` with Laguerre weight fitting).

Source format (verified against the MIDAS source and manual, October 2026)
--------------------------------------------------------------------------
``Grains.csv`` is written by ``FF_HEDM/src/ProcessGrains.c`` (https://github.com/marinerhemant/
MIDAS) and described in ``manuals/FF_Analysis.md`` ("Grains.csv Column Format"):

* metadata header lines start with ``%`` (``%NumGrains N``, ``%BeamCenter``, ``%PhaseInfo``,
  lattice parameters, ...);
* the last header line names the columns, tab separated, starting with ``%GrainID``;
* then one tab-separated line per grain (47 columns, each value followed by a tab), values
  printed with ``%lf`` (6 decimals).

Columns used here (1-based positions from FF_Analysis.md):

* 1 ``GrainID``: grain identifier;
* 2-10 ``O11 ... O33``: orientation matrix, row-major (O11 O12 O13 O21 ...);
* 11-13 ``X, Y, Z``: grain centre-of-mass position, micrometres;
* 23 ``GrainRadius``: equivalent grain radius, micrometres;
* 24 ``Confidence``: indexing confidence (optional filter);
* 44 ``PhaseNr``: phase number (optional phase column).

The MIDAS manual (manuals/ff-hedm/phase-4-read-report.md, "IPF maps") states that "MIDAS
matrices map crystal -> lab" and that crystal symmetry acts on the right (g S). This converter
does not rely on it: ``--conjugate`` is a required, explicit statement by the user. The columns
are found by name from the ``%GrainID`` header line; without that line the converter falls back
to the column positions listed above (``--no-header-columns`` documents this fallback).

What the converter writes
-------------------------
Columns ``id, x, y, z, qw, qx, qy, qz`` and, by default, ``phase`` and ``radius`` of the
"ExaConstit seed table v1" (see exaconstit_microstructure_io.py):
id = GrainID, (x, y, z) = units_scale * (X, Y, Z), radius = units_scale * GrainRadius,
q = quaternion of the orientation matrix (Shepperd's method, w >= 0), conjugated when
``--conjugate yes``. The matrix is used as read (with ``--matrix-layout row``, the default and
the documented MIDAS layout); converting it to a quaternion only changes the parametrization.
Matrices are checked to be rotations to ``--orthogonality-tolerance`` (default 1e-3, which
accepts the ~1e-6 rounding of ``%lf`` output).

Example::

    python midas_grains_to_seeds.py Grains.csv seeds.csv --conjugate no --units-scale 1e-3
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exaconstit_microstructure_io as msio  # noqa: E402

# Column names in the order ProcessGrains.c writes them (FF_HEDM/src/ProcessGrains.c, the
# "%GrainID\tO11..." header). Used only when the file has no %GrainID header line.
MIDAS_DEFAULT_COLUMNS = (
    ["GrainID"]
    + ["O%d%d" % (i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
       "DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
    + ["eFab%d%d" % (i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["eKen%d%d" % (i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2"])

MATRIX_COLUMNS = ["O%d%d" % (i, j) for i in (1, 2, 3) for j in (1, 2, 3)]


def read_midas_grains(path, use_header_columns=True):
    """
    Read a MIDAS Grains.csv file.

    Header lines start with '%'. The one starting with '%GrainID' gives the column names (split
    on whitespace). Every other non-blank line is one grain, split on whitespace (tabs; the
    trailing tab gives no extra field).

    Input:
      path - the Grains.csv file;
      use_header_columns - find columns by the names in the %GrainID line (default). When
        False, or when no such line exists, the ProcessGrains.c column order is assumed.
    Output: (names, data) where names is the list of column names and data is an
    (num_grains x num_columns) float64 array.
    Raises ConverterError naming the line for malformed input.
    """
    names = None
    rows = []
    with open(path, "r") as src:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("%"):
                # The column-name line: "%GrainID\tO11\t..."
                if stripped[1:].lstrip().startswith("GrainID") and use_header_columns:
                    names = stripped[1:].split()
                continue
            fields = stripped.split()
            try:
                rows.append([float(f) for f in fields])
            except ValueError:
                raise msio.ConverterError("%s:%d: non-numeric field" % (path, line_number))
            if len(rows[-1]) != len(rows[0]):
                raise msio.ConverterError("%s:%d: %d columns, first grain line has %d"
                                          % (path, line_number, len(rows[-1]), len(rows[0])))
    if not rows:
        raise msio.ConverterError("%s: no grain lines found" % path)
    if names is None:
        names = list(MIDAS_DEFAULT_COLUMNS)
    if len(rows[0]) < len(names):
        raise msio.ConverterError("%s: grain lines have %d columns but %d column names are "
                                  "expected" % (path, len(rows[0]), len(names)))
    return names, np.array(rows, dtype=np.float64)


def column(names, data, name, path):
    """Return the data column called ``name`` (exact match), or raise naming the file."""
    if name not in names:
        raise msio.ConverterError("%s: column '%s' not found (columns: %s)"
                                  % (path, name, " ".join(names)))
    return data[:, names.index(name)]


def build_parser():
    """Command-line options of this converter."""
    parser = argparse.ArgumentParser(
        description="Convert a MIDAS Grains.csv file to an ExaConstit seed table (CSV). "
                    "Position = X,Y,Z; orientation = quaternion of O11..O33; "
                    "radius = GrainRadius; phase = PhaseNr.")
    parser.add_argument("input", help="MIDAS Grains.csv file")
    parser.add_argument("output", help="seed table CSV to write")
    msio.add_conjugate_argument(parser)
    msio.add_units_scale_argument(parser, "positions and radii; MIDAS uses micrometres")
    parser.add_argument("--matrix-layout", choices=("row", "column"), default="row",
                        help="order of O11..O33 in the file: 'row' (default; O11 O12 O13 O21 "
                             "..., the documented MIDAS layout) or 'column' (read the 9 values "
                             "column by column, i.e. use the transpose)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="keep grains with Confidence >= this value (default 0: keep all)")
    parser.add_argument("--no-radius", action="store_true",
                        help="do not write the radius column (use when GrainRadius is missing "
                             "or zero)")
    parser.add_argument("--no-phase", action="store_true",
                        help="do not write the phase column")
    parser.add_argument("--no-header-columns", action="store_true",
                        help="ignore the %%GrainID header line and assume the ProcessGrains.c "
                             "column order (GrainID, O11..O33, X, Y, Z, ...)")
    parser.add_argument("--orthogonality-tolerance", type=float, default=1.0e-3,
                        help="allowed max |O O^T - I| and |det O - 1| (default 1e-3)")
    return parser


def convert(args):
    """
    Read, filter, convert and write.

    Steps: read the table; find the needed columns; drop grains below the confidence filter;
    check each orientation matrix and convert it to a quaternion (conjugating when asked);
    scale positions and radii; write the seed table.
    """
    names, data = read_midas_grains(args.input, not args.no_header_columns)

    # Step 1: pick the columns and apply the confidence filter
    keep = np.ones(data.shape[0], dtype=bool)
    if args.min_confidence > 0.0:
        keep &= column(names, data, "Confidence", args.input) >= args.min_confidence
    if not np.any(keep):
        raise msio.ConverterError("no grains left after the confidence filter")
    data = data[keep]
    grain_ids = column(names, data, "GrainID", args.input)
    matrices = np.stack([column(names, data, c, args.input) for c in MATRIX_COLUMNS], axis=1)
    positions = np.stack([column(names, data, c, args.input) for c in ("X", "Y", "Z")], axis=1)

    # Step 2: grain ids must be integers >= 0
    if np.any(grain_ids != np.floor(grain_ids)) or np.any(grain_ids < 0):
        raise msio.ConverterError("GrainID values must be integers >= 0")

    # Step 3: orientation matrix -> quaternion, grain by grain
    quats = np.zeros((data.shape[0], 4))
    for row in range(data.shape[0]):
        matrix = msio.matrix_from_nine_values(matrices[row], args.matrix_layout)
        problem = msio.rotation_matrix_problem(matrix, args.orthogonality_tolerance)
        if problem is not None:
            raise msio.ConverterError("GrainID %d: %s" % (int(grain_ids[row]), problem))
        q = msio.quat_from_matrix(matrix)
        if args.conjugate == "yes":
            q = msio.conjugate_quat(q)
        quats[row] = q

    columns = {"id": grain_ids.astype(np.int64),
               "x": args.units_scale * positions[:, 0],
               "y": args.units_scale * positions[:, 1],
               "z": args.units_scale * positions[:, 2],
               "qw": quats[:, 0], "qx": quats[:, 1], "qy": quats[:, 2], "qz": quats[:, 3]}

    # Step 4: optional phase (PhaseNr is printed as a real, e.g. "1.000000")
    if not args.no_phase and "PhaseNr" in names:
        phases = column(names, data, "PhaseNr", args.input)
        if np.any(phases != np.floor(phases)) or np.any(phases < 0):
            raise msio.ConverterError("PhaseNr values must be integers >= 0 (use --no-phase "
                                      "to skip them)")
        columns["phase"] = phases.astype(np.int64)

    # Step 5: optional radius
    if not args.no_radius:
        radii = column(names, data, "GrainRadius", args.input)
        bad = np.nonzero(~(radii > 0.0))[0]
        if bad.size > 0:
            raise msio.ConverterError("GrainID %d has GrainRadius %s <= 0 (use --no-radius to "
                                      "omit sizes)" % (int(grain_ids[bad[0]]), radii[bad[0]]))
        columns["radius"] = args.units_scale * radii

    comments = msio.provenance_lines(
        "midas_grains_to_seeds.py", args.input,
        ["conjugate: %s, matrix_layout: %s, units_scale: %s, min_confidence: %s"
         % (args.conjugate, args.matrix_layout, msio.format_real(args.units_scale),
            msio.format_real(args.min_confidence))])
    msio.write_seed_table(args.output, columns, comments)
    print("midas_grains_to_seeds: wrote %d seeds to %s (%d grains dropped by filters)"
          % (data.shape[0], args.output, int(np.count_nonzero(~keep))))


def main(argv=None):
    """Parse the command line and run the conversion."""
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.units_scale) or args.units_scale <= 0.0:
        raise msio.ConverterError("--units-scale must be > 0")
    convert(args)


if __name__ == "__main__":
    sys.exit(msio.run_main(main))
