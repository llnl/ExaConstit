#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a HEXRD ``grains.out`` file (far-field HEDM grain list) to an ExaConstit seed table.

Each grain of the HEDM reconstruction becomes one seed of the microstructure generator: the
grain centroid is the seed position and the fitted grain orientation is the seed orientation.
The generator then grows grains around these seeds (Voronoi or a weighted variant).

Source format (verified against the HEXRD source, master branch, October 2026)
-----------------------------------------------------------------------------
``grains.out`` is written by ``GrainDataWriter`` in hexrd/core/instrument/hedm_instrument.py
(https://github.com/HEXRD/hexrd):

* one header line starting with ``#``:
  ``# grain ID  completeness  chi^2  exp_map_c[0] exp_map_c[1] exp_map_c[2]  t_vec_c[0]
  t_vec_c[1] t_vec_c[2]  inv(V_s)[0,0] ... ln(V_s)[0,1]``;
* then one line per grain with 21 whitespace-separated columns (two-space delimiter):
  ``grain ID`` (integer), ``completeness`` (0..1), ``chi^2``, ``exp_map_c`` (3),
  ``t_vec_c`` (3), ``inv(V_s)`` in Mandel-Voigt order (6), ``ln(V_s)`` (6).
  Reals are printed with ``{:<23.16e}``, so they carry full double precision.

Meaning of the columns used here:

* ``exp_map_c``: exponential map (rotation vector, angle in radians times unit axis) of the
  grain orientation. HEXRD turns it into the matrix ``rmat_c`` with
  ``scipy.spatial.transform.Rotation.from_rotvec`` (``rotMatOfExpMap`` in
  hexrd/core/rotations.py; ``make_rmat_of_expmap`` in hexrd/core/transforms/xfcapi.py).
  HEXRD documents ``rmat_c`` as "the COB [change of basis] matrix taking components in the
  CRYSTAL FRAME to the SAMPLE FRAME" (xfcapi.py, ``gvec_to_xy`` docstring).
* ``t_vec_c``: "translation vector connecting SAMPLE FRAME to CRYSTAL FRAME", i.e. the grain
  centroid in sample coordinates. HEXRD lengths are in millimetres (its instrument units).
* ``completeness`` and ``chi^2``: fit quality, used by the optional filters.

What the converter writes
-------------------------
Columns ``id, x, y, z, qw, qx, qy, qz`` of the "ExaConstit seed table v1" (see
exaconstit_microstructure_io.py): id = HEXRD grain ID, (x, y, z) = units_scale * t_vec_c,
q = quaternion of exp_map_c (cos(theta/2), sin(theta/2) n), conjugated when ``--conjugate yes``.
The quaternion conversion only changes the parametrization; ``--conjugate`` is a required,
explicit statement by the user of whether HEXRD's crystal->sample change-of-basis rotation must
be inverted to match ExaConstit's crystal->sample passive convention.

Example::

    python hexrd_grains_to_seeds.py grains.out seeds.csv --conjugate no \
        --units-scale 1.0 --min-completeness 0.8 --max-chi2 0.01
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exaconstit_microstructure_io as msio  # noqa: E402

# Number of columns of a grains.out data line (GrainDataWriter asserts 21 columns)
HEXRD_NUM_COLUMNS = 21


def read_hexrd_grains(path):
    """
    Read a HEXRD grains.out file.

    Lines that are blank or start with '#' are skipped (the header). Every other line must have
    21 whitespace-separated numbers.

    Input: path - the grains.out file.
    Output: dict of numpy arrays, one entry per grain:
      "id" (int64), "completeness", "chi2", "expmap" (n x 3, radians), "tvec" (n x 3, source
      length units).
    Raises ConverterError naming the line for malformed input.
    """
    ids, completeness, chi2, expmaps, tvecs = [], [], [], [], []
    with open(path, "r") as src:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != HEXRD_NUM_COLUMNS:
                raise msio.ConverterError("%s:%d: expected %d columns (HEXRD grains.out), found "
                                          "%d" % (path, line_number, HEXRD_NUM_COLUMNS,
                                                  len(fields)))
            try:
                values = [float(f) for f in fields]
            except ValueError:
                raise msio.ConverterError("%s:%d: non-numeric field" % (path, line_number))
            # Column 0 is the integer grain ID, written with {:<12d}
            if values[0] != math.floor(values[0]):
                raise msio.ConverterError("%s:%d: grain ID '%s' is not an integer"
                                          % (path, line_number, fields[0]))
            ids.append(int(values[0]))
            completeness.append(values[1])
            chi2.append(values[2])
            expmaps.append(values[3:6])
            tvecs.append(values[6:9])
    if not ids:
        raise msio.ConverterError("%s: no grains found" % path)
    return {"id": np.array(ids, dtype=np.int64),
            "completeness": np.array(completeness),
            "chi2": np.array(chi2),
            "expmap": np.array(expmaps).reshape(-1, 3),
            "tvec": np.array(tvecs).reshape(-1, 3)}


def build_parser():
    """Command-line options of this converter."""
    parser = argparse.ArgumentParser(
        description="Convert a HEXRD grains.out file to an ExaConstit seed table (CSV). "
                    "Position = t_vec_c, orientation = quaternion of exp_map_c.")
    parser.add_argument("input", help="HEXRD grains.out file")
    parser.add_argument("output", help="seed table CSV to write")
    msio.add_conjugate_argument(parser)
    msio.add_units_scale_argument(parser, "grain positions; HEXRD uses millimetres")
    parser.add_argument("--min-completeness", type=float, default=0.0,
                        help="keep grains with completeness >= this value (default 0: keep all)")
    parser.add_argument("--max-chi2", type=float, default=math.inf,
                        help="keep grains with chi^2 <= this value (default: keep all)")
    parser.add_argument("--drop-nan", action="store_true",
                        help="drop grains whose completeness or chi^2 is NaN and report how "
                             "many (default: such grains are an error)")
    return parser


def convert(args):
    """
    Read, filter, convert and write.

    Steps: read the grain list; drop grains failing the completeness/chi^2 filters; turn each
    exponential map into a quaternion (and conjugate it when asked); scale positions; write the
    seed table. Prints a one-line summary.
    """
    grains = read_hexrd_grains(args.input)

    # Step 1: grains whose completeness or chi^2 is NaN (a failed fit). A comparison with NaN
    # is always False, so the filters below would drop them silently; refuse them instead,
    # or drop and count them with --drop-nan
    not_a_number = np.isnan(grains["completeness"]) | np.isnan(grains["chi2"])
    num_nan = int(np.count_nonzero(not_a_number))
    if num_nan > 0 and not args.drop_nan:
        raise msio.ConverterError("%d grain(s) have NaN completeness or chi^2 (first: grain ID "
                                  "%d); use --drop-nan to drop them"
                                  % (num_nan, int(grains["id"][not_a_number][0])))

    # Step 2: quality filters (NaN rows, if allowed, fail them and are counted separately)
    keep = (grains["completeness"] >= args.min_completeness) & (grains["chi2"] <= args.max_chi2)
    num_dropped = int(np.count_nonzero(~keep)) - num_nan
    if not np.any(keep):
        raise msio.ConverterError("no grains left after the completeness/chi^2 filters")

    # Step 3: exponential map -> quaternion, row by row (a parametrization change only)
    quats = []
    for expmap in grains["expmap"][keep]:
        q = msio.quat_from_expmap(expmap)
        if args.conjugate == "yes":
            q = msio.conjugate_quat(q)
        quats.append(q)
    quats = np.array(quats).reshape(-1, 4)

    # Step 4: positions in output units
    positions = args.units_scale * grains["tvec"][keep]

    columns = {"id": grains["id"][keep],
               "x": positions[:, 0], "y": positions[:, 1], "z": positions[:, 2],
               "qw": quats[:, 0], "qx": quats[:, 1], "qy": quats[:, 2], "qz": quats[:, 3]}
    comments = msio.provenance_lines(
        "hexrd_grains_to_seeds.py", args.input,
        ["conjugate: %s, units_scale: %s, min_completeness: %s, max_chi2: %s"
         % (args.conjugate, msio.format_real(args.units_scale),
            msio.format_real(args.min_completeness), msio.format_real(args.max_chi2))])
    msio.write_seed_table(args.output, columns, comments)
    print("hexrd_grains_to_seeds: wrote %d seeds to %s (%d grains dropped by filters, %d with "
          "NaN completeness/chi^2 dropped)" % (positions.shape[0], args.output, num_dropped,
                                               num_nan))


def main(argv=None):
    """Parse the command line and run the conversion."""
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.units_scale) or args.units_scale <= 0.0:
        raise msio.ConverterError("--units-scale must be > 0")
    convert(args)


if __name__ == "__main__":
    sys.exit(msio.run_main(main))
