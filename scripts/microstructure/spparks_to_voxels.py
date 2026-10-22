#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a SPPARKS site dump (text style) to an ExaConstit voxel map (HDF5).

SPPARKS (https://spparks.github.io) runs on-lattice kinetic Monte Carlo models such as the Potts
grain-growth model. Each lattice site carries an integer "spin"; neighbouring sites with the same
spin belong to the same grain. This converter puts every site of a simple cubic (or square)
lattice into one voxel and uses the spin as the voxel's grain id. SPPARKS spins carry no
crystal orientation, so no orientations are written; give the grains orientations in the
generator (e.g. random).

Note: in a Potts model with a finite number of spin values, two grains that do not touch can
have the same spin. The generator's fragment cleanup (``fragments`` option) separates such
disconnected pieces.

Source format (verified against the SPPARKS source and docs, github.com/spparks/spparks, Oct 2026)
---------------------------------------------------------------------------------------------
``dump ID text delta file [fields]`` writes (doc/dump.txt; src/dump_text.cpp, ``write_header``):

    ITEM: TIMESTEP
    <snapshot index> <time>
    ITEM: NUMBER OF ATOMS
    <number of sites>
    ITEM: BOX BOUNDS
    xlo xhi
    ylo yhi
    zlo zhi
    ITEM: ATOMS id type x y z
    <one line per site>

* The documentation shows the first line as ``ITEM: TIMESTEP TIME``; the source writes
  ``ITEM: TIMESTEP``. Both are accepted.
* The per-site column names follow the dump command's fields; the field ``site`` is renamed
  ``type`` in the header "to be LAMMPS compatible" (dump_text.cpp). ``site`` is the same as
  ``i1`` (doc/dump.txt). The default fields are ``id site x y z``.
* Sites can be in any order (unless ``dump_modify sort``), and a file can hold several
  snapshots. Reals (coordinates) are printed with ``%g`` (6 significant digits).
* Lattices ``sc/6n``, ``sc/26n`` (3d) and ``sq/4n``, ``sq/8n`` (2d) have one basis site per unit
  cell at (0,0,0) (doc/lattice.txt), so sites sit at ``xlo + i*a`` and the box spans N cells.
  Each site is therefore taken as the LOWER CORNER of its voxel by default
  (``--site-location corner``): the voxel grid then coincides with the simulation box.
  ``--site-location center`` makes each site the centre of its voxel instead.

What the converter writes
-------------------------
The "ExaConstit voxel map v1" (see exaconstit_microstructure_io.py): ``/grain_ids`` (nz, ny, nx)
with the spin of each site, ``/origin`` (lower corner of voxel (0,0,0)) and ``/spacing``, both
multiplied by ``--units-scale``. The voxel index of a site is round((x - x_min) / a) per axis,
where a = (max - min) / (number of planes - 1) along that axis (or given with ``--spacing``).
In ``corner`` mode the detected lattice is cross-checked against ``ITEM: BOX BOUNDS`` (the box
must start at the first plane of sites and span n * a); a mismatch beyond the %g print
precision is an error.
Spins <= 0 are written as given and mean "unassigned" to the generator.

Example::

    python spparks_to_voxels.py potts.dump grains.h5 --units-scale 1.0
"""

import argparse
import gzip
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exaconstit_microstructure_io as msio  # noqa: E402


def open_text(path):
    """Open a text file, transparently decompressing it when the name ends in .gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def read_spparks_dump(path, snapshot=-1):
    """
    Read one snapshot of a SPPARKS text dump.

    The file is scanned for "ITEM:" sections. Each "ITEM: TIMESTEP" starts a new snapshot;
    "ITEM: NUMBER OF ATOMS" gives the site count; "ITEM: BOX BOUNDS" the box; "ITEM: ATOMS
    <names>" the column names, followed by one line per site.

    Input:
      path - dump file (optionally .gz);
      snapshot - which snapshot to return, Python-style index (default -1 = the last one).
    Output: dict with "columns" (list of names), "data" (num_sites x num_columns float64),
      "box" (3 x 2 array, or None), "timestep" (text of the line after ITEM: TIMESTEP),
      "num_snapshots".
    Raises ConverterError for malformed files.
    """
    snapshots = []
    with open_text(path) as src:
        lines = iter(enumerate(src, start=1))
        current = None
        for line_number, line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("ITEM: TIMESTEP"):
                current = {"timestep": next(lines, (0, ""))[1].strip(), "box": None,
                           "num_sites": None, "columns": None, "rows": []}
                snapshots.append(current)
            elif current is None:
                raise msio.ConverterError("%s:%d: expected 'ITEM: TIMESTEP' (is this a SPPARKS "
                                          "text dump?)" % (path, line_number))
            elif stripped.startswith("ITEM: NUMBER OF ATOMS"):
                current["num_sites"] = int(next(lines, (0, "0"))[1])
            elif stripped.startswith("ITEM: BOX BOUNDS"):
                box = []
                for _ in range(3):
                    box.append([float(v) for v in next(lines, (0, "0 0"))[1].split()[:2]])
                current["box"] = np.array(box)
            elif stripped.startswith("ITEM: ATOMS"):
                current["columns"] = stripped.split()[2:]
                # Read exactly num_sites lines of per-site values
                if current["num_sites"] is None:
                    raise msio.ConverterError("%s:%d: 'ITEM: ATOMS' before 'ITEM: NUMBER OF "
                                              "ATOMS'" % (path, line_number))
                for _ in range(current["num_sites"]):
                    entry = next(lines, None)
                    if entry is None:
                        raise msio.ConverterError("%s: snapshot %s ends early"
                                                  % (path, current["timestep"]))
                    fields = entry[1].split()
                    if len(fields) != len(current["columns"]):
                        raise msio.ConverterError("%s:%d: %d values, expected %d"
                                                  % (path, entry[0], len(fields),
                                                     len(current["columns"])))
                    current["rows"].append([float(f) for f in fields])
            elif stripped.startswith("ITEM:"):
                raise msio.ConverterError("%s:%d: unexpected section '%s'"
                                          % (path, line_number, stripped))
    if not snapshots:
        raise msio.ConverterError("%s: no snapshots found" % path)
    try:
        chosen = snapshots[snapshot]
    except IndexError:
        raise msio.ConverterError("%s: snapshot %d requested, file has %d"
                                  % (path, snapshot, len(snapshots)))
    if chosen["columns"] is None:
        raise msio.ConverterError("%s: snapshot %s has no 'ITEM: ATOMS' section"
                                  % (path, chosen["timestep"]))
    return {"columns": chosen["columns"],
            "data": np.array(chosen["rows"], dtype=np.float64).reshape(-1,
                                                                        len(chosen["columns"])),
            "box": chosen["box"], "timestep": chosen["timestep"],
            "num_snapshots": len(snapshots)}


# Relative accuracy of a number printed with %g (6 significant digits): at most half a unit
# in the 6th digit, i.e. 5e-6 of its magnitude. Tolerances below are a few times this.
PRINT_RELATIVE_ERROR = 5.0e-6


def lattice_spacing(coords):
    """
    Spacing and number of lattice planes of a regular 1D set of site coordinates.

    Why not the smallest gap: coordinates are printed with %g (6 significant digits), so each
    gap carries a rounding error of up to 1e-5 of the coordinate magnitude, and the smallest of
    many gaps is biased low (200 sites at a = 0.123457 gave 0.1234). Instead we count the
    distinct planes n and use the whole extent: a = (max - min) / (n - 1). The two end values
    carry the only rounding error, which is then divided by n - 1.

    Steps: sort the distinct values; values closer than the print rounding are the same plane;
    n = 1 + number of real gaps; a = (max - min) / (n - 1).

    Input: coords - coordinates of all sites along one axis.
    Output: (spacing, num_planes), or (None, 1) when all sites share one plane (e.g. z in a 2D
    model).
    """
    values = np.unique(coords)
    if values.size < 2:
        return None, 1
    tolerance = 4.0 * PRINT_RELATIVE_ERROR * float(np.max(np.abs(values)))
    num_planes = 1 + int(np.count_nonzero(np.diff(values) > tolerance))
    if num_planes < 2:
        return None, 1
    return float(values[-1] - values[0]) / (num_planes - 1), num_planes


def check_box_bounds(box, lower, spacing, num_planes, path):
    """
    Cross-check the detected lattice against ``ITEM: BOX BOUNDS`` (corner sites only).

    For sc/sq lattices the basis site is at the cell corner, so the first plane of sites lies on
    the box's lower bound and the box spans exactly n cells: lo = x_min and hi - lo = n * a.
    A disagreement beyond the %g print precision means the detected spacing or origin is wrong
    (missing planes, a dump_modify subset, a non-cubic lattice, or sites at cell centres).

    Input: box - (3, 2) array of (lo, hi) per axis; lower - x_min per axis; spacing and
      num_planes per axis from ``lattice_spacing`` (axes with one plane are skipped: 2D models
      put a single layer of sites inside a box of any thickness).
    Raises ConverterError naming the axis and the numbers.
    """
    for axis in range(3):
        if num_planes[axis] < 2:
            continue
        lo, hi = float(box[axis][0]), float(box[axis][1])
        magnitude = max(abs(lo), abs(hi), abs(float(lower[axis])), spacing[axis])
        tolerance = 6.0 * PRINT_RELATIVE_ERROR * magnitude
        box_extent = num_planes[axis] * spacing[axis]
        if abs(lo - lower[axis]) > tolerance or abs((hi - lo) - box_extent) > tolerance:
            raise msio.ConverterError(
                "%s: axis %d: the sites (first plane %.6g, %d planes, spacing %.6g, so the box "
                "should be [%.6g, %.6g]) do not match ITEM: BOX BOUNDS [%.6g, %.6g]. Use "
                "--site-location center if sites are cell centres, or give --spacing to skip "
                "this check" % (path, axis, lower[axis], num_planes[axis], spacing[axis],
                                lower[axis], lower[axis] + box_extent, lo, hi))


def sites_to_grid(xyz, spins, spacing, site_location, path):
    """
    Place every site in its voxel.

    Steps: per axis, the voxel index is round((x - x_min) / spacing); the grid size is
    max index + 1. Each voxel must receive exactly one site (otherwise the lattice is not simple
    cubic/square, or the snapshot is a subset of the sites).

    Input:
      xyz - (n, 3) site coordinates; spins - (n,) integer spins;
      spacing - 3 spacings (source units); site_location - "corner" or "center";
      path - for error messages.
    Output: (grain_ids (nz, ny, nx) int64, origin (3,), spacing (3,)), source units.
    """
    lower = xyz.min(axis=0)
    index = np.zeros(xyz.shape, dtype=np.int64)
    for axis in range(3):
        scaled = (xyz[:, axis] - lower[axis]) / spacing[axis]
        index[:, axis] = np.rint(scaled).astype(np.int64)
        # Coordinates are printed with %g (6 significant digits); a site more than a quarter
        # voxel away from a lattice plane cannot be explained by that rounding
        off_lattice = np.abs(scaled - index[:, axis]) > 0.25
        if np.any(off_lattice):
            raise msio.ConverterError("%s: site coordinates along axis %d are not on a regular "
                                      "lattice with spacing %.6g" % (path, axis,
                                                                     spacing[axis]))
    dims = index.max(axis=0) + 1  # (nx, ny, nz)

    # Fill the grid; mark visited voxels to catch duplicates and holes
    grain_ids = np.zeros((dims[2], dims[1], dims[0]), dtype=np.int64)
    count = np.zeros_like(grain_ids)
    np.add.at(count, (index[:, 2], index[:, 1], index[:, 0]), 1)
    if np.any(count > 1):
        raise msio.ConverterError("%s: several sites fall in one voxel; the lattice is not "
                                  "simple cubic/square with the detected spacing" % path)
    if np.any(count == 0):
        raise msio.ConverterError("%s: %d of %d voxels have no site (a dump_modify subset, or "
                                  "a non-cubic lattice)" % (path, int(np.count_nonzero(count == 0)),
                                                            count.size))
    grain_ids[index[:, 2], index[:, 1], index[:, 0]] = spins

    # Lower corner of voxel (0,0,0)
    origin = lower.copy()
    if site_location == "center":
        origin -= 0.5 * np.asarray(spacing)
    return grain_ids, origin, np.asarray(spacing, dtype=np.float64)


def build_parser():
    """Command-line options of this converter."""
    parser = argparse.ArgumentParser(
        description="Convert a SPPARKS text site dump to an ExaConstit voxel map (HDF5). "
                    "Each lattice site becomes one voxel; its spin is the grain id.")
    parser.add_argument("input", help="SPPARKS text dump (dump ... text ...), optionally .gz")
    parser.add_argument("output", help="voxel map .h5 file to write")
    msio.add_units_scale_argument(parser, "origin and spacing; SPPARKS uses lattice units")
    parser.add_argument("--snapshot", type=int, default=-1,
                        help="snapshot to convert, Python-style index (default -1: the last)")
    parser.add_argument("--spin-column", default="auto",
                        help="per-site column holding the grain id (default 'auto': 'type', "
                             "which is how the 'site' field is labelled, else 'i1')")
    parser.add_argument("--site-location", choices=("corner", "center"), default="corner",
                        help="where a site sits in its voxel: 'corner' (default; lower corner, "
                             "so the voxel grid matches the SPPARKS box for sc/sq lattices) or "
                             "'center'")
    parser.add_argument("--spacing", type=float, nargs=3, metavar=("AX", "AY", "AZ"),
                        help="lattice spacing per axis in source units (default: detected "
                             "from the site coordinates as (max - min)/(planes - 1) and "
                             "checked against the box bounds; an axis with a single layer "
                             "of sites uses the spacing of the other axes). Giving it skips "
                             "the box check.")
    return parser


def convert(args):
    """
    Read the snapshot, place sites in voxels, scale, and write the voxel map.
    """
    dump = read_spparks_dump(args.input, args.snapshot)
    columns = dump["columns"]

    # Step 1: find the coordinate and spin columns
    for name in ("x", "y", "z"):
        if name not in columns:
            raise msio.ConverterError("%s: per-site column '%s' missing (dump the fields "
                                      "'id site x y z')" % (args.input, name))
    spin_name = args.spin_column
    if spin_name == "auto":
        spin_name = "type" if "type" in columns else "i1"
    if spin_name not in columns:
        raise msio.ConverterError("%s: spin column '%s' missing (columns: %s)"
                                  % (args.input, spin_name, " ".join(columns)))
    data = dump["data"]
    xyz = data[:, [columns.index("x"), columns.index("y"), columns.index("z")]]
    spins_real = data[:, columns.index(spin_name)]
    if np.any(spins_real != np.floor(spins_real)):
        raise msio.ConverterError("%s: column '%s' is not integer" % (args.input, spin_name))
    spins = spins_real.astype(np.int64)

    # Step 2: lattice spacing (detected unless given); single-layer axes borrow it
    if args.spacing is not None:
        spacing = [float(s) for s in args.spacing]
        if any(not (s > 0.0) for s in spacing):
            raise msio.ConverterError("--spacing values must be > 0")
    else:
        detected = [lattice_spacing(xyz[:, axis]) for axis in range(3)]
        known = [s for s, _ in detected if s is not None]
        fallback = min(known) if known else 1.0
        spacing = [s if s is not None else fallback for s, _ in detected]
        # Corner sites: the box must start at the first plane and span n * spacing
        if args.site_location == "corner" and dump["box"] is not None:
            check_box_bounds(dump["box"], xyz.min(axis=0), spacing,
                             [n for _, n in detected], args.input)

    # Step 3: sites -> voxels, then scale to output units
    grain_ids, origin, spacing = sites_to_grid(xyz, spins, spacing, args.site_location,
                                               args.input)
    origin = args.units_scale * origin
    spacing = args.units_scale * spacing

    attributes = {"source": "SPPARKS text dump %s, timestep %s"
                            % (os.path.basename(args.input), dump["timestep"]),
                  "converter": "spparks_to_voxels.py",
                  "site_location": args.site_location}
    msio.write_voxel_map(args.output, grain_ids, origin, spacing, attributes=attributes)
    nz, ny, nx = grain_ids.shape
    print("spparks_to_voxels: wrote %d x %d x %d voxels, %d distinct spins > 0, to %s"
          % (nx, ny, nz, np.unique(grain_ids[grain_ids > 0]).size, args.output))


def main(argv=None):
    """Parse the command line and run the conversion."""
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.units_scale) or args.units_scale <= 0.0:
        raise msio.ConverterError("--units-scale must be > 0")
    convert(args)


if __name__ == "__main__":
    sys.exit(msio.run_main(main))
