#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an ExaCA grain structure (VTK structured points with a ``GrainID`` field) to an
ExaConstit voxel map (HDF5), optionally with per-grain orientations.

ExaCA (https://github.com/LLNL/ExaCA) is a cellular automaton code for solidification
microstructure (e.g. additive manufacturing). Its output is a regular grid of cells, each with
an integer ``GrainID``. A grain's crystal orientation is not stored per cell; it is one of a
fixed list of orientations read from a separate file, selected from the GrainID.

Source format (verified against the ExaCA source, master branch, October 2026)
-----------------------------------------------------------------------------
VTK file, written by ``Print::writeHeader`` / ``Print::printViewData`` in src/CAprint.hpp:

    # vtk DataFile Version 3.0
    vtk output
    ASCII                      (or BINARY)
    DATASET STRUCTURED_POINTS
    DIMENSIONS nx ny nz
    ORIGIN x_min y_min z_min
    SPACING deltax deltax deltax
    POINT_DATA nx*ny*nz
    SCALARS GrainID int 1
    LOOKUP_TABLE default
    <values, x fastest, then y, then z>
    SCALARS LayerID short 1    (further fields may follow)
    ...

* Lengths are in metres (``deltax`` is in m; the ExaConstit RVE output of CAprint.hpp says
  "1 cell = <deltax> m. Data is cell-centered").
* Values are ``POINT_DATA``: one value per grid point, and each point is the centre of an ExaCA
  cell. The lower corner of voxel (0,0,0) is therefore ORIGIN - SPACING/2 (the default
  ``--point-location center``). ``--point-location corner`` uses ORIGIN as the corner instead.
  The header numbers are printed with C++ stream defaults (6 significant digits);
  ``--origin``/``--spacing`` override them when more precision is needed.
* Binary files are big-endian (``writeData(..., swap_endian_yn = true)``), as the legacy VTK
  format requires. Data types used by ExaCA: ``int`` (GrainID), ``short``, ``float``.
* Several ``SCALARS`` blocks can follow one another; this converter skips all but the one named
  by ``--field`` (default ``GrainID``). Generic legacy-VTK ``CELL_DATA``, ``VECTORS``,
  ``NORMALS`` and ``FIELD`` blocks are also handled (skipped or read), following the VTK file
  format documentation (https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf).

GrainID meaning (src/CAorientation.hpp, ``getGrainOrientation`` / ``getGrainNumber``):

* GrainID > 0: grains grown from the substrate (epitaxial); GrainID < 0: nucleated grains.
  The grain with id g and the grain with id -g are different grains.
* The orientation of grain g is entry (|g| - 1) mod N (0-based) of the orientation file, where
  N is the number of orientations in that file.
* GrainID 0 is written as unassigned (<= 0) and gets no orientation.

The canonical voxel map treats ids <= 0 as unassigned, so negative ExaCA ids are renumbered
(``--negative-ids offset``, the default): id -g becomes max_positive_id + g, where
max_positive_id is the largest positive GrainID in the file. ``--negative-ids error`` refuses
files with negative ids instead.

Orientation file (src/CAorientation.hpp, ``getOrientationsFromFile``; example
examples/Substrate/GrainOrientationVectors.csv): the first line is the number of orientations
N, followed by N lines of 9 comma-separated values. ``misorientationCalc`` reads value
``9*n + 3*l + d`` as component d (sample x, y, z) of crystal axis l, so with
``--matrix-layout row`` (default) the rows of the matrix are the crystal axes written in sample
coordinates. Files with 4 values per line are read as quaternions in the order given by
``--quat-layout`` (no default; ExaCA's own files are 9-value matrices, so this is for files
from other tools). Other ExaCA orientation files (Bunge Euler angles, IPF colours) are not
accepted: Euler angles are a convention conversion.

What the converter writes
-------------------------
The "ExaConstit voxel map v1" (see exaconstit_microstructure_io.py): ``/grain_ids`` (nz, ny,
nx), ``/origin`` (lower corner) and ``/spacing``, scaled by ``--units-scale``; with
``--orientations``, ``/features/id`` and ``/features/quat`` (w x y z) for every grain present.
Matrices become quaternions by Shepperd's method (w >= 0), conjugated when ``--conjugate yes``
(required with ``--orientations``). This only changes the parametrization; the user decides the
convention.

Example::

    python exaca_to_voxels.py TestProblem.vtk grains.h5 --units-scale 1e3 \
        --orientations GrainOrientationVectors.csv --conjugate yes
"""

import argparse
import math
import os
import re
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exaconstit_microstructure_io as msio  # noqa: E402

# Legacy VTK data type names -> numpy dtypes. Binary data is big-endian ('>').
# "long"/"unsigned_long" are taken as 64-bit (unverified: the legacy format does not fix it).
VTK_TYPES = {
    "bit": None,
    "unsigned_char": "u1", "char": "i1",
    "unsigned_short": "u2", "short": "i2",
    "unsigned_int": "u4", "int": "i4",
    "unsigned_long": "u8", "long": "i8",
    "vtktypeint8": "i1", "vtktypeuint8": "u1",
    "vtktypeint16": "i2", "vtktypeuint16": "u2",
    "vtktypeint32": "i4", "vtktypeuint32": "u4",
    "vtktypeint64": "i8", "vtktypeuint64": "u8",
    "float": "f4", "double": "f8",
}


# A line (after a newline) whose first word starts with a letter, other than nan/inf: the next
# keyword after a text data block
_TEXT_KEYWORD_LINE = re.compile(rb"[\r\n][ \t]*(?![nN][aA][nN]|[iI][nN][fF])[A-Za-z_]")


class _ByteCursor:
    """
    Read position inside a legacy VTK file held in memory as bytes.

    Legacy VTK files mix text lines (keywords) with data blocks that are either text or raw
    big-endian binary, so the file is read as bytes and walked with an explicit position.
    """

    def __init__(self, data, path):
        self.data = data
        self.pos = 0
        self.path = path

    def _skip_whitespace(self):
        """Move past spaces, tabs and newlines (binary blocks may end without a newline)."""
        n = len(self.data)
        while self.pos < n and self.data[self.pos:self.pos + 1] in (b" ", b"\t", b"\r", b"\n"):
            self.pos += 1

    def read_line(self, skip_blank=True):
        """Next text line (without the newline), or None at the end of the file."""
        if skip_blank:
            self._skip_whitespace()
        if self.pos >= len(self.data):
            return None
        end = self.data.find(b"\n", self.pos)
        if end < 0:
            end = len(self.data)
        line = self.data[self.pos:end].decode("ascii", errors="replace").strip()
        self.pos = end + 1
        return line

    def peek_line(self):
        """Next non-blank text line without consuming it."""
        saved = self.pos
        line = self.read_line()
        self.pos = saved
        return line

    def _text_block_end(self):
        """
        Byte position where the text data block starting at ``self.pos`` ends.

        A text block ends at the next line whose first word starts with a letter (the next
        keyword: SCALARS, VECTORS, FIELD, POINT_DATA, a FIELD array name, ...), or at the end
        of the file. ``nan``/``inf`` are numbers, not keywords. The search runs inside the
        regular-expression engine on the file bytes, so nothing is copied.
        """
        match = _TEXT_KEYWORD_LINE.search(self.data, self.pos)
        return match.start() + 1 if match else len(self.data)

    def _parse_text_numbers(self, start, end, count):
        """
        Parse the whitespace-separated numbers in data[start:end] into a float64 array.

        The block is parsed in slices of about 16 MB, each cut at a whitespace byte, with
        ``numpy.fromstring(..., sep=" ")`` (a C parser; ``sep=" "`` matches any run of spaces,
        tabs and newlines). Peak extra memory is one slice plus the result array, instead of one
        Python object per value. The block must hold exactly ``count`` numbers.
        """
        values = np.empty(count, dtype=np.float64)
        filled = 0
        chunk_bytes = 1 << 24
        pos = start
        while pos < end:
            # Cut the slice at the last whitespace byte so no number is split in two
            stop = min(pos + chunk_bytes, end)
            if stop < end:
                cut = max(self.data.rfind(c, pos, stop) for c in (b" ", b"\t", b"\n", b"\r"))
                if cut > pos:
                    stop = cut + 1
            chunk = self.data[pos:stop]
            pos = stop
            if not chunk.strip():
                continue
            # numpy only warns (DeprecationWarning) when it meets a non-number; make that an
            # error so a malformed block is reported instead of silently truncated
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                try:
                    parsed = np.fromstring(chunk, dtype=np.float64, sep=" ")
                except (DeprecationWarning, ValueError):
                    raise msio.ConverterError("%s: non-numeric value in a text data block"
                                              % self.path)
            if filled + parsed.size > count:
                raise msio.ConverterError("%s: text data block has more than the %d values "
                                          "expected" % (self.path, count))
            values[filled:filled + parsed.size] = parsed
            filled += parsed.size
        if filled != count:
            raise msio.ConverterError("%s: text data block has %d values, expected %d"
                                      % (self.path, filled, count))
        return values

    def read_values(self, count, vtk_type, binary, keep=True):
        """
        Read (or skip) ``count`` values of the given VTK type.

        Text files: whitespace-separated numbers up to the next keyword line. Binary files:
        count * itemsize raw big-endian bytes starting right after the previous line's newline.
        With keep=False the cursor just moves past the block: a binary block by its byte size,
        a text block to the next keyword line, without parsing or copying it.
        Output: numpy array (int64 for integer types, float64 for real types), or None when
        keep is False.
        """
        if vtk_type not in VTK_TYPES or VTK_TYPES[vtk_type] is None:
            raise msio.ConverterError("%s: unsupported VTK data type '%s'" % (self.path, vtk_type))
        code = VTK_TYPES[vtk_type]
        is_integer = code[0] in ("i", "u")
        if binary:
            dtype = np.dtype(">" + code)
            nbytes = count * dtype.itemsize
            if self.pos + nbytes > len(self.data):
                raise msio.ConverterError("%s: binary data block ends early" % self.path)
            values = None
            if keep:
                # frombuffer makes a view into the file bytes (no copy)
                values = np.frombuffer(self.data, dtype=dtype, count=count, offset=self.pos)
            self.pos += nbytes
        else:
            end = self._text_block_end()
            values = self._parse_text_numbers(self.pos, end, count) if keep else None
            self.pos = end
        if not keep:
            return None
        if is_integer:
            # Integers up to 2^53 are exact in float64, far above the 2^31 id limit
            return np.asarray(values, dtype=np.int64)
        return np.asarray(values, dtype=np.float64)


def _read_field_block(cursor, header_words, binary, want, num_values_hint):
    """
    Read or skip a legacy VTK ``FIELD <name> <numArrays>`` block.

    Each array is "<arrayName> <numComponents> <numTuples> <dataType>" followed by the data.
    Output: the array named ``want`` (as a flat numpy array) if present, else None.
    """
    if len(header_words) < 3:
        raise msio.ConverterError("%s: malformed FIELD line" % cursor.path)
    found = None
    for _ in range(int(header_words[2])):
        words = cursor.read_line().split()
        if len(words) < 4:
            raise msio.ConverterError("%s: malformed FIELD array line" % cursor.path)
        name, ncomp, ntuples, vtk_type = words[0], int(words[1]), int(words[2]), words[3]
        keep = name == want and ncomp == 1
        values = cursor.read_values(ncomp * ntuples, vtk_type.lower(), binary, keep)
        if keep:
            found = values
    return found


def read_vtk_structured_points(path, field_name):
    """
    Read one scalar field and the grid of a legacy VTK STRUCTURED_POINTS file.

    Input: path - .vtk file; field_name - the SCALARS (or FIELD array) name to return.
    Output: dict with "dims" (nx, ny, nz point counts), "origin", "spacing" (3 floats, file
      units), "association" ("POINT_DATA" or "CELL_DATA"), "values" (flat array, x fastest),
      "binary" (bool).
    Raises ConverterError for anything it cannot read.
    """
    with open(path, "rb") as src:
        cursor = _ByteCursor(src.read(), path)

    # Step 1: the three fixed header lines
    first = cursor.read_line(skip_blank=False)
    if first is None or not first.lower().startswith("# vtk datafile"):
        raise msio.ConverterError("%s: not a legacy VTK file (first line '%s')" % (path, first))
    cursor.read_line(skip_blank=False)  # title
    encoding = (cursor.read_line() or "").upper()
    if encoding not in ("ASCII", "BINARY"):
        raise msio.ConverterError("%s: third line must be ASCII or BINARY, got '%s'"
                                  % (path, encoding))
    binary = encoding == "BINARY"

    # Step 2: dataset keywords up to POINT_DATA / CELL_DATA
    dims = origin = spacing = None
    association = None
    num_values = None
    while association is None:
        line = cursor.read_line()
        if line is None:
            raise msio.ConverterError("%s: no POINT_DATA or CELL_DATA section" % path)
        words = line.split()
        key = words[0].upper()
        if key == "DATASET":
            if len(words) < 2 or words[1].upper() != "STRUCTURED_POINTS":
                raise msio.ConverterError("%s: dataset '%s' is not STRUCTURED_POINTS"
                                          % (path, " ".join(words[1:])))
        elif key == "DIMENSIONS":
            dims = [int(w) for w in words[1:4]]
        elif key == "ORIGIN":
            origin = [float(w) for w in words[1:4]]
        elif key in ("SPACING", "ASPECT_RATIO"):
            spacing = [float(w) for w in words[1:4]]
        elif key == "FIELD":
            _read_field_block(cursor, words, binary, None, None)  # dataset-level metadata
        elif key in ("POINT_DATA", "CELL_DATA"):
            association = key
            num_values = int(words[1])
        else:
            raise msio.ConverterError("%s: unexpected keyword '%s' in the dataset header"
                                      % (path, words[0]))
    if dims is None or origin is None or spacing is None:
        raise msio.ConverterError("%s: DIMENSIONS, ORIGIN and SPACING are all required" % path)

    # Step 3: the expected number of values per field
    if association == "POINT_DATA":
        expected = dims[0] * dims[1] * dims[2]
    else:
        expected = max(dims[0] - 1, 1) * max(dims[1] - 1, 1) * max(dims[2] - 1, 1)
    if num_values != expected:
        raise msio.ConverterError("%s: %s %d does not match DIMENSIONS %s"
                                  % (path, association, num_values, dims))

    # Step 4: walk the attribute blocks until the requested field is found
    seen = []
    while True:
        line = cursor.read_line()
        if line is None:
            break
        words = line.split()
        key = words[0].upper()
        if key in ("POINT_DATA", "CELL_DATA"):
            # A second association section (e.g. CELL_DATA after POINT_DATA): its fields
            # have a different size, so stop here rather than misread them
            break
        if key == "SCALARS":
            name, vtk_type = words[1], words[2].lower()
            ncomp = int(words[3]) if len(words) > 3 else 1
            if (cursor.peek_line() or "").upper().startswith("LOOKUP_TABLE"):
                cursor.read_line()
            # Only the requested field is parsed; the others are skipped without reading them
            values = cursor.read_values(num_values * ncomp, vtk_type, binary,
                                        keep=(name == field_name))
            seen.append(name)
            if name == field_name:
                if ncomp != 1:
                    raise msio.ConverterError("%s: field '%s' has %d components, expected 1"
                                              % (path, name, ncomp))
                return {"dims": dims, "origin": origin, "spacing": spacing,
                        "association": association, "values": values, "binary": binary}
        elif key in ("VECTORS", "NORMALS"):
            cursor.read_values(3 * num_values, words[2].lower(), binary, keep=False)
            seen.append(words[1])
        elif key == "FIELD":
            values = _read_field_block(cursor, words, binary, field_name, num_values)
            if values is not None:
                if values.size != num_values:
                    raise msio.ConverterError("%s: FIELD array '%s' has %d values, expected %d"
                                              % (path, field_name, values.size, num_values))
                return {"dims": dims, "origin": origin, "spacing": spacing,
                        "association": association, "values": values, "binary": binary}
        else:
            raise msio.ConverterError("%s: cannot read the VTK section '%s' that comes before "
                                      "field '%s'" % (path, words[0], field_name))
    raise msio.ConverterError("%s: field '%s' not found (fields seen: %s)"
                              % (path, field_name, ", ".join(seen) or "none"))


def read_exaca_orientations(path, values_per_line=None):
    """
    Read an ExaCA orientation file: first line N, then N lines of comma-separated values.

    Input: path - the file; values_per_line - 9 or 4, or None to detect from the first entry.
    Output: (N x values_per_line) float64 array.
    Commas and/or whitespace separate values; blank lines are skipped.
    """
    with open(path, "r") as src:
        lines = [ln.strip() for ln in src if ln.strip()]
    if not lines:
        raise msio.ConverterError("%s: empty orientation file" % path)
    try:
        count = int(lines[0].replace(",", " ").split()[0])
    except (ValueError, IndexError):
        raise msio.ConverterError("%s: first line must be the number of orientations" % path)
    if count < 1 or len(lines) - 1 < count:
        raise msio.ConverterError("%s: header says %d orientations, file has %d lines after it"
                                  % (path, count, len(lines) - 1))
    rows = []
    for k in range(count):
        try:
            values = [float(v) for v in lines[1 + k].replace(",", " ").split()]
        except ValueError:
            raise msio.ConverterError("%s: orientation %d has a non-numeric value" % (path, k))
        rows.append(values)
    if values_per_line is None:
        values_per_line = len(rows[0])
    if values_per_line not in (4, 9):
        raise msio.ConverterError("%s: %d values per orientation; expected 9 (rotation matrix) "
                                  "or 4 (quaternion)" % (path, values_per_line))
    for k, values in enumerate(rows):
        if len(values) != values_per_line:
            raise msio.ConverterError("%s: orientation %d has %d values, expected %d"
                                      % (path, k, len(values), values_per_line))
    return np.array(rows, dtype=np.float64)


def orientation_quaternions(table, args):
    """
    Turn each orientation of the file into a quaternion (w x y z), conjugated when asked.

    9-value rows: matrix in ``--matrix-layout`` order, checked to be a rotation, Shepperd's
    method. 4-value rows: reordered from ``--quat-layout`` to w x y z (no other change).
    Output: (N x 4) array.
    """
    quats = np.zeros((table.shape[0], 4))
    for k in range(table.shape[0]):
        if table.shape[1] == 9:
            matrix = msio.matrix_from_nine_values(table[k], args.matrix_layout)
            problem = msio.rotation_matrix_problem(matrix, args.orthogonality_tolerance)
            if problem is not None:
                raise msio.ConverterError("orientation %d: %s" % (k, problem))
            q = msio.quat_from_matrix(matrix)
        else:
            if args.quat_layout is None:
                raise msio.ConverterError("the orientation file has 4 values per line "
                                          "(quaternions): --quat-layout {wxyz,xyzw} is required")
            q = table[k] if args.quat_layout == "wxyz" else table[k][[3, 0, 1, 2]]
            norm = float(np.linalg.norm(q))
            if abs(norm - 1.0) > msio.QUAT_NORM_TOLERANCE:
                raise msio.ConverterError("orientation %d: quaternion norm %.6g is not 1"
                                          % (k, norm))
        if args.conjugate == "yes":
            q = msio.conjugate_quat(q)
        quats[k] = q
    return quats


def build_parser():
    """Command-line options of this converter."""
    parser = argparse.ArgumentParser(
        description="Convert an ExaCA VTK grain structure (GrainID field) to an ExaConstit "
                    "voxel map (HDF5), optionally with orientations from an ExaCA orientation "
                    "file.")
    parser.add_argument("input", help="ExaCA .vtk file (legacy VTK, STRUCTURED_POINTS)")
    parser.add_argument("output", help="voxel map .h5 file to write")
    msio.add_units_scale_argument(parser, "origin and spacing; ExaCA uses metres")
    parser.add_argument("--field", default="GrainID",
                        help="name of the grain id field (default GrainID)")
    parser.add_argument("--orientations",
                        help="ExaCA orientation file (e.g. GrainOrientationVectors.csv): first "
                             "line N, then N lines of 9 matrix values (or 4 quaternion values)")
    msio.add_conjugate_argument(parser, required=False)
    parser.add_argument("--matrix-layout", choices=("row", "column"), default="row",
                        help="order of the 9 values per orientation: 'row' (default; value "
                             "3*l+d is row l, column d) or 'column' (the transpose)")
    parser.add_argument("--quat-layout", choices=("wxyz", "xyzw"),
                        help="component order of 4-value orientation files (required for them)")
    parser.add_argument("--orthogonality-tolerance", type=float, default=1.0e-3,
                        help="allowed max |R R^T - I| and |det R - 1| for matrices (default "
                             "1e-3; ExaCA prints about 6 significant digits)")
    parser.add_argument("--point-location", choices=("center", "corner"), default="center",
                        help="for POINT_DATA files: whether each VTK point is the centre "
                             "(default, ExaCA cells are cell-centred) or the lower corner of "
                             "its voxel. Ignored for CELL_DATA files.")
    parser.add_argument("--negative-ids", choices=("offset", "error"), default="offset",
                        help="how to handle negative GrainIDs (nucleated grains): 'offset' "
                             "(default) renumbers -g to max_positive_id + g; 'error' refuses "
                             "them")
    parser.add_argument("--origin", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="override the VTK ORIGIN (file units, before --units-scale)")
    parser.add_argument("--spacing", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                        help="override the VTK SPACING (file units, before --units-scale)")
    return parser


def convert(args):
    """
    Read the grid and grain ids, renumber negative ids, attach orientations, write the map.
    """
    vtk = read_vtk_structured_points(args.input, args.field)

    # Step 1: grid geometry in file units (overrides first), then the voxel lower corner
    origin = np.array(args.origin if args.origin is not None else vtk["origin"], dtype=float)
    spacing = np.array(args.spacing if args.spacing is not None else vtk["spacing"],
                       dtype=float)
    if np.any(~(spacing > 0.0)):
        raise msio.ConverterError("spacing must be > 0, got %s" % spacing)
    nx, ny, nz = vtk["dims"]
    if vtk["association"] == "CELL_DATA":
        nx, ny, nz = max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1)
    elif args.point_location == "center":
        origin = origin - 0.5 * spacing

    # Step 2: grain ids, x fastest -> (nz, ny, nx)
    raw = vtk["values"]
    if raw.dtype.kind == "f":
        if np.any(raw != np.floor(raw)):
            raise msio.ConverterError("field '%s' has non-integer values" % args.field)
        raw = raw.astype(np.int64)
    exaca_ids = raw.reshape(nz, ny, nx)

    # Step 3: renumber negative (nucleated) grain ids so every grain id is >= 1
    max_positive = int(exaca_ids.max()) if np.any(exaca_ids > 0) else 0
    grain_ids = exaca_ids.copy()
    negative = exaca_ids < 0
    if np.any(negative):
        if args.negative_ids == "error":
            raise msio.ConverterError("field '%s' has negative ids (nucleated grains); use "
                                      "--negative-ids offset" % args.field)
        grain_ids[negative] = max_positive - exaca_ids[negative]
    if grain_ids.size and int(grain_ids.max()) >= 2 ** 31:
        raise msio.ConverterError("renumbered grain ids reach %d, above the generator's limit "
                                  "2^31 - 1" % int(grain_ids.max()))

    # Step 4: per-grain orientations for every grain present
    feature_ids = feature_quats = None
    if args.orientations:
        table = read_exaca_orientations(args.orientations)
        quats = orientation_quaternions(table, args)
        # Each canonical id comes from exactly one ExaCA id; find that id for each grain
        present = np.unique(exaca_ids[exaca_ids != 0])
        canonical = np.where(present > 0, present, max_positive - present)
        order = np.argsort(canonical)
        present, canonical = present[order], canonical[order]
        # ExaCA: orientation index of grain g is (|g| - 1) mod N, 0-based
        orientation_index = (np.abs(present) - 1) % table.shape[0]
        feature_ids = canonical
        feature_quats = quats[orientation_index]

    attributes = {"source": "ExaCA VTK %s, field %s" % (os.path.basename(args.input),
                                                         args.field),
                  "converter": "exaca_to_voxels.py",
                  "point_location": args.point_location,
                  "negative_id_offset": max_positive}
    if args.orientations:
        attributes["orientations"] = "%s, matrix_layout %s, conjugate %s" % (
            os.path.basename(args.orientations), args.matrix_layout, args.conjugate)
    msio.write_voxel_map(args.output, grain_ids, args.units_scale * origin,
                         args.units_scale * spacing, feature_ids=feature_ids,
                         feature_quats=feature_quats, attributes=attributes)
    print("exaca_to_voxels: wrote %d x %d x %d voxels, %d grains (%d nucleated), %s"
          "orientations, to %s"
          % (nx, ny, nz, np.unique(grain_ids[grain_ids > 0]).size,
             np.unique(exaca_ids[negative]).size, "" if args.orientations else "no ",
             args.output))


def main(argv=None):
    """Parse the command line, check the option combinations, and run the conversion."""
    parser = build_parser()
    args = parser.parse_args(argv)
    # --conjugate is required exactly when orientations are read
    if args.orientations and args.conjugate is None:
        parser.error("--conjugate {yes,no} is required with --orientations")
    if not args.orientations and args.conjugate is not None:
        parser.error("--conjugate only applies together with --orientations")
    if not math.isfinite(args.units_scale) or args.units_scale <= 0.0:
        raise msio.ConverterError("--units-scale must be > 0")
    convert(args)


if __name__ == "__main__":
    sys.exit(msio.run_main(main))
