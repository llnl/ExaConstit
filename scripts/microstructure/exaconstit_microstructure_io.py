#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared writers, readers and quaternion helpers for the ExaConstit microstructure converters.

The converters in this directory turn the output of other microstructure tools (HEXRD, MIDAS,
SPPARKS, ExaCA) into the input files the ExaConstit microstructure generator reads:

* the "ExaConstit microstructure file" (HDF5): a grain table (``grains/id``, optional
  orientation, phase, seed position, size, weight, shape) and, optionally, a grain id per cell of
  a regular grid stored as a Conduit Blueprint mesh under ``mesh/``. A file with a grid is used
  with ``[Source] type = "voxel_map"``, ``[VoxelMap] format = "exaconstit"``; a file with only a
  grain table (with positions) with ``[Seeds] source = "file"``. Written by
  ``write_microstructure_file`` (and its grid shortcut ``write_voxel_map``) with h5py only: the
  Conduit Python module is not needed.
* the optional CSV seed table, for grain lists written by hand or with a spreadsheet
  (``write_seed_table``). Used with ``[Seeds] source = "file"`` like a grain-table-only HDF5 file.

Both formats are specified in the C++ reader headers,
``src/microstructure/io/microstructure_file_io.hpp`` and ``src/microstructure/io/seed_table_io.hpp``
(user-level description: ``doc/microstructure.md``, sections 9 and 13); the functions here
write exactly what the readers expect.

Orientations
------------
ExaConstit orientations are unit quaternions stored as ``w x y z`` and interpreted as
crystal->sample, passive. The generator never converts conventions; it passes the user's
rotation through unchanged apart from normalizing every quaternion it writes to unit length.
The converters follow the same rule: they only change the
*parametrization* of a rotation (a 3x3 matrix or an exponential map / rotation vector becomes
the quaternion of the same rotation). Whether the source tool's rotation has to be inverted
(conjugated) to match ExaConstit's convention is a decision only the user can make, so every
converter that reads orientations has a required ``--conjugate {yes,no}`` option. Conjugating a
unit quaternion (w, x, y, z) -> (w, -x, -y, -z) is the same as transposing (inverting) its
rotation matrix.

Mapping between parametrizations (all standard; no convention is chosen here):

* rotation matrix R <-> quaternion q = (w, x, y, z):
  R = [[1-2(y^2+z^2), 2(xy-wz),     2(xz+wy)    ],
       [2(xy+wz),     1-2(x^2+z^2), 2(yz-wx)    ],
       [2(xz-wy),     2(yz+wx),     1-2(x^2+y^2)]]
  This is the same formula as HEXRD's ``rotMatOfQuat`` (hexrd/core/rotations.py) and the usual
  Rodrigues formula, so a rotation of angle theta about the unit axis n has
  q = (cos(theta/2), sin(theta/2) n).
* exponential map (rotation vector) e = theta * n -> q = (cos(theta/2), sin(theta/2) n).
  HEXRD builds its rotation matrix from the exponential map with
  ``scipy.spatial.transform.Rotation.from_rotvec`` (hexrd/core/rotations.py,
  ``rotMatOfExpMap``), which is this same mapping.

Both conversions return w >= 0. q and -q are the same rotation, so picking the sign is not a
convention change; it only makes the output unique.

Requirements: Python 3 and numpy. h5py is needed only to write or read microstructure files;
without it those functions raise ``ConverterError`` with a clear message.
"""

import math
import os
import sys

import numpy as np

try:
    import h5py  # noqa: F401  (optional; only the HDF5 microstructure file functions need it)
    HAVE_H5PY = True
except ImportError:  # pragma: no cover - depends on the local Python installation
    h5py = None
    HAVE_H5PY = False


# Names written into the files so a human can tell what produced them.
SEED_TABLE_FORMAT_NAME = "ExaConstit seed table v1"
# The microstructure file's `format` value and the version this module writes (the C++ reader
# refuses versions newer than it knows; adding optional datasets never changes the version)
MICROSTRUCTURE_FORMAT_NAME = "ExaConstit microstructure"
MICROSTRUCTURE_FORMAT_VERSION = 1

# The seed-table reader rejects a quaternion whose norm differs from 1 by more than this.
QUAT_NORM_TOLERANCE = 1.0e-3

# Largest value of a C int: phases and grain/feature ids become C ints in the generator
INT32_MAX = 2 ** 31 - 1

# Column order used when writing a seed table. Only the columns that are present are written.
SEED_TABLE_COLUMN_ORDER = ("id", "x", "y", "z", "qw", "qx", "qy", "qz", "phase", "weight",
                           "volume", "radius", "diameter",
                           "m11", "m22", "m33", "m23", "m13", "m12")
SEED_TABLE_INTEGER_COLUMNS = ("id", "phase")
SEED_TABLE_SIZE_COLUMNS = ("volume", "radius", "diameter")
SEED_TABLE_QUAT_COLUMNS = ("qw", "qx", "qy", "qz")
SEED_TABLE_SHAPE_COLUMNS = ("m11", "m22", "m33", "m23", "m13", "m12")


class ConverterError(Exception):
    """An error in the input data or options that the user has to fix (bad file, bad flag)."""


# -------------------------------------------------------------------------------------------------
# Number formatting
# -------------------------------------------------------------------------------------------------

def format_real(value):
    """
    Format a floating-point number with 17 significant digits.

    17 significant digits are enough to write any IEEE double so that reading the text back
    gives exactly the same double (a bitwise round trip), which the generator's seed-table tests
    rely on.

    Input: value - a Python or numpy float.
    Output: the text, e.g. "0.10000000000000001" or "-1.2345678901234567e-05".
    """
    return "%.17g" % float(value)


def format_integer(value):
    """Format an integer value (Python or numpy integer, or an integral float) as plain text."""
    return "%d" % int(value)


# -------------------------------------------------------------------------------------------------
# Quaternion and rotation helpers (parametrization changes only)
# -------------------------------------------------------------------------------------------------

def quat_from_matrix(matrix):
    """
    Quaternion (w, x, y, z) of a 3x3 rotation matrix, by Shepperd's method, with w >= 0.

    Shepperd's method (S. W. Shepperd, "Quaternion from rotation matrix", J. Guidance and
    Control 1(3), 1978) computes four candidate values 4w^2, 4x^2, 4y^2, 4z^2 from the trace and
    the diagonal, takes the square root of the largest one (so we never divide by a small
    number), and gets the other three components from the off-diagonal sums and differences.
    The result is the quaternion q with R = R(q) for the formula in the module docstring.

    The matrix is used as given: it is not transposed and no convention is assumed. The result
    is normalized to unit length (a matrix read from text with 6 decimals is orthonormal only to
    about 1e-6, so the raw result is unit only to that accuracy); callers should check that the
    matrix is a rotation first with ``rotation_matrix_problem``.

    Input: matrix - array-like 3x3, matrix[i][j] = row i, column j.
    Output: numpy array of 4 floats (w, x, y, z), unit length, w >= 0.
    """
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    # Step 1: the four candidates 4w^2, 4x^2, 4y^2, 4z^2 (each up to a common factor)
    candidates = (1.0 + trace,
                  1.0 + 2.0 * m[0, 0] - trace,
                  1.0 + 2.0 * m[1, 1] - trace,
                  1.0 + 2.0 * m[2, 2] - trace)

    # Step 2: pick the largest; ties go to the first (w), so the identity gives (1, 0, 0, 0)
    largest = 0
    for k in range(1, 4):
        if candidates[k] > candidates[largest]:
            largest = k

    # Step 3: that component is sqrt(candidate)/2; the other three come from the off-diagonal
    # terms divided by 4 * (that component)
    if largest == 0:
        w = 0.5 * math.sqrt(candidates[0])
        s = 0.25 / w
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif largest == 1:
        x = 0.5 * math.sqrt(candidates[1])
        s = 0.25 / x
        w = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 1] + m[1, 0]) * s
        z = (m[0, 2] + m[2, 0]) * s
    elif largest == 2:
        y = 0.5 * math.sqrt(candidates[2])
        s = 0.25 / y
        w = (m[0, 2] - m[2, 0]) * s
        x = (m[0, 1] + m[1, 0]) * s
        z = (m[1, 2] + m[2, 1]) * s
    else:
        z = 0.5 * math.sqrt(candidates[3])
        s = 0.25 / z
        w = (m[1, 0] - m[0, 1]) * s
        x = (m[0, 2] + m[2, 0]) * s
        y = (m[1, 2] + m[2, 1]) * s

    q = np.array([w, x, y, z], dtype=np.float64)

    # Step 4: q and -q are the same rotation; choose w >= 0 so the answer is unique
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


def matrix_from_quat(quat):
    """
    3x3 rotation matrix of a quaternion (w, x, y, z), using the formula in the module docstring.

    The quaternion is normalized first. This is the inverse of ``quat_from_matrix``.

    Input: quat - array-like of 4 floats (w, x, y, z), non-zero.
    Output: numpy 3x3 array.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                     [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                     [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]])


def quat_from_expmap(expmap):
    """
    Quaternion (w, x, y, z) of an exponential map (rotation vector) e = theta * n, with w >= 0.

    The rotation angle is theta = |e| (radians) and the axis is n = e / theta, so
    q = (cos(theta/2), sin(theta/2) * n) = (cos(theta/2), [sin(theta/2)/theta] * e).
    For small theta the factor sin(theta/2)/theta is evaluated from its Taylor series
    1/2 - theta^2/48 + theta^4/3840 to avoid dividing by (almost) zero.

    Input: expmap - array-like of 3 floats (radians).
    Output: numpy array of 4 floats (w, x, y, z), unit length, w >= 0.
    """
    e = np.asarray(expmap, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(e))

    # Factor that multiplies e to give the vector part sin(theta/2) * n
    if theta < 1.0e-4:
        factor = 0.5 - theta * theta / 48.0 + theta ** 4 / 3840.0
    else:
        factor = math.sin(0.5 * theta) / theta

    q = np.array([math.cos(0.5 * theta), factor * e[0], factor * e[1], factor * e[2]])

    # Angles above pi give w < 0; flip to the equivalent quaternion with w >= 0
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


def conjugate_quat(quat):
    """
    Conjugate (inverse) of a unit quaternion: (w, x, y, z) -> (w, -x, -y, -z).

    This inverts the rotation; for the matrix form it is the transpose. The converters apply it
    only when the user passes ``--conjugate yes``.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotation_matrix_problem(matrix, tolerance=1.0e-3):
    """
    Check that a 3x3 matrix is a proper rotation (orthonormal, determinant +1).

    A matrix read from a text file is a rotation only to the precision it was printed with, so
    the check uses a tolerance on max |R R^T - I| and on |det R - 1|. A determinant near -1
    means a reflection (often a left-handed frame or a transposed/garbled row) and is reported
    separately.

    Input: matrix - array-like 3x3; tolerance - allowed deviation (dimensionless).
    Output: None when the matrix is a rotation, otherwise a short description of the problem.
    """
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(m)):
        return "matrix has non-finite entries"
    orthogonality_error = float(np.max(np.abs(m @ m.T - np.eye(3))))
    determinant = float(np.linalg.det(m))
    if determinant < 0.0:
        return "matrix has determinant %.6g (a reflection, not a rotation)" % determinant
    if orthogonality_error > tolerance or abs(determinant - 1.0) > tolerance:
        return ("matrix is not orthonormal: max |R R^T - I| = %.3g, det = %.6g (tolerance %.3g)"
                % (orthogonality_error, determinant, tolerance))
    return None


def matrix_from_nine_values(values, layout):
    """
    Build a 3x3 matrix from 9 numbers in file order.

    Input:
      values - 9 numbers as they appear in the file;
      layout - "row": values are M11 M12 M13 M21 ... (row by row);
               "column": values are M11 M21 M31 M12 ... (column by column).
    Output: numpy 3x3 array M with M[i][j] = row i, column j.

    Reading the same 9 numbers with the other layout gives the transpose, i.e. the inverse
    rotation. That is a convention question the user answers with ``--matrix-layout`` and
    ``--conjugate``.
    """
    v = np.asarray(values, dtype=np.float64).reshape(9)
    if layout == "row":
        return v.reshape(3, 3)
    if layout == "column":
        return v.reshape(3, 3).T.copy()
    raise ConverterError("unknown matrix layout '%s' (expected row or column)" % layout)


# -------------------------------------------------------------------------------------------------
# Command-line helpers shared by the converter scripts
# -------------------------------------------------------------------------------------------------

def add_conjugate_argument(parser, required=True):
    """
    Add the ``--conjugate {yes,no}`` option to an argparse parser.

    There is deliberately no default: the user must state whether the source tool's rotation is
    the inverse of ExaConstit's crystal->sample passive rotation.
    """
    parser.add_argument(
        "--conjugate", choices=("yes", "no"), required=required,
        help="REQUIRED for orientation data. 'no' writes the quaternion of the source rotation "
             "as is; 'yes' writes its conjugate (the inverse rotation, i.e. the transposed "
             "matrix). ExaConstit expects crystal->sample passive quaternions (w x y z); choose "
             "the value that makes the source's rotation match that convention.")


def add_units_scale_argument(parser, what):
    """Add ``--units-scale`` (multiplies every length written to the output)."""
    parser.add_argument(
        "--units-scale", type=float, default=1.0,
        help="factor multiplying every length written to the output (%s). Default 1.0 keeps "
             "the source units. Note that the generator also has a units_scale option; use "
             "one or the other, not both." % what)


def run_main(main_function, argv=None):
    """
    Run a converter's main function and turn ``ConverterError`` into a clean exit.

    Input errors are printed as one line "error: ..." on stderr with exit status 1, without a
    Python traceback (which would only confuse users).
    """
    try:
        main_function(argv)
    except ConverterError as err:
        sys.stderr.write("error: %s\n" % err)
        return 1
    return 0


# -------------------------------------------------------------------------------------------------
# Seed table (CSV)
# -------------------------------------------------------------------------------------------------

def write_seed_table(path, columns, comments=()):
    """
    Write an "ExaConstit seed table v1" CSV file.

    Format (see WORK_PACKAGES.md, "Shared formats"): comment lines start with '#', then a header
    line of comma-separated column names, then one line per seed. Real numbers are written with
    17 significant digits so they read back bit for bit.

    Input:
      path     - output file path;
      columns  - dict mapping column name -> sequence of values, one per seed. Required:
                 "x", "y", "z". Optional: "id" (int >= 1, unique), "qw","qx","qy","qz" (all
                 four or none), "phase" (1 <= int <= 2^31-1), "weight", one of "volume"/"radius"/
                 "diameter" (> 0), and the six shape entries "m11".."m12" (all or none).
                 Unknown names are refused.
      comments - lines written first, each prefixed with "# ".
    Output: none. Raises ConverterError on invalid content (the same rules the C++ reader
    enforces, so a file this function writes is always readable), including a table with no
    seeds.
    """
    # Step 1: check the column names and the column groups
    for name in columns:
        if name not in SEED_TABLE_COLUMN_ORDER:
            raise ConverterError("seed table: unknown column '%s'" % name)
    for name in ("x", "y", "z"):
        if name not in columns:
            raise ConverterError("seed table: required column '%s' missing" % name)
    present_quat = [c for c in SEED_TABLE_QUAT_COLUMNS if c in columns]
    if present_quat and len(present_quat) != 4:
        raise ConverterError("seed table: give all four of qw,qx,qy,qz or none")
    present_shape = [c for c in SEED_TABLE_SHAPE_COLUMNS if c in columns]
    if present_shape and len(present_shape) != 6:
        raise ConverterError("seed table: give all six shape entries m11..m12 or none")
    if len([c for c in SEED_TABLE_SIZE_COLUMNS if c in columns]) > 1:
        raise ConverterError("seed table: at most one of volume, radius, diameter")

    # Step 2: convert every column to a numpy array and check that all have one value per seed
    arrays = {}
    num_seeds = None
    for name in SEED_TABLE_COLUMN_ORDER:
        if name not in columns:
            continue
        if name in SEED_TABLE_INTEGER_COLUMNS:
            values = np.asarray(columns[name], dtype=np.int64).reshape(-1)
        else:
            values = np.asarray(columns[name], dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(values)):
                raise ConverterError("seed table: column '%s' has non-finite values" % name)
        if num_seeds is None:
            num_seeds = values.size
        elif values.size != num_seeds:
            raise ConverterError("seed table: column '%s' has %d values, expected %d"
                                 % (name, values.size, num_seeds))
        arrays[name] = values

    # Step 3: value rules (unique non-negative ids, unit quaternions, positive sizes, ...)
    if "id" in arrays:
        ids = arrays["id"]
        if np.any(ids < 1):
            raise ConverterError("seed table: ids must be >= 1")
        unique_ids, counts = np.unique(ids, return_counts=True)
        if np.any(counts > 1):
            raise ConverterError("seed table: duplicate id %d" % unique_ids[counts > 1][0])
    if not num_seeds:
        raise ConverterError("seed table: no seeds to write (the generator refuses an empty "
                             "table)")
    # Phase ids start at 1, like ExaConstit's region ids (phase k is region k by default)
    if "phase" in arrays and (np.any(arrays["phase"] < 1)
                              or np.any(arrays["phase"] > INT32_MAX)):
        raise ConverterError("seed table: phases must be in [1, 2^31 - 1]")
    for name in SEED_TABLE_SIZE_COLUMNS:
        if name in arrays and np.any(arrays[name] <= 0.0):
            raise ConverterError("seed table: column '%s' must be > 0" % name)
    if present_quat:
        norms = np.sqrt(arrays["qw"] ** 2 + arrays["qx"] ** 2 + arrays["qy"] ** 2
                        + arrays["qz"] ** 2)
        bad = np.nonzero(np.abs(norms - 1.0) > QUAT_NORM_TOLERANCE)[0]
        if bad.size > 0:
            raise ConverterError("seed table: row %d has quaternion norm %.6g (not unit)"
                                 % (bad[0], norms[bad[0]]))

    # Step 4: write comments, header and rows
    names = [name for name in SEED_TABLE_COLUMN_ORDER if name in arrays]
    with open(path, "w") as out:
        out.write("# %s\n" % SEED_TABLE_FORMAT_NAME)
        for line in comments:
            out.write("# %s\n" % line)
        out.write(",".join(names) + "\n")
        for row in range(num_seeds or 0):
            fields = []
            for name in names:
                if name in SEED_TABLE_INTEGER_COLUMNS:
                    fields.append(format_integer(arrays[name][row]))
                else:
                    fields.append(format_real(arrays[name][row]))
            out.write(",".join(fields) + "\n")


def read_seed_table(path):
    """
    Read a seed table CSV file (the syntax rules of "ExaConstit seed table v1").

    This is a plain reader for tests and quick checks: it parses the syntax (comments, header,
    equal field counts) but does not apply the generator's units_scale/translate or its value
    checks.

    Input: path - the CSV file.
    Output: dict mapping lower-case column name -> numpy array (int64 for "id" and "phase",
    float64 otherwise).
    """
    header = None
    rows = []
    # utf-8-sig drops a leading byte-order mark (added by some spreadsheet programs), as the
    # C++ reader does
    with open(path, "r", encoding="utf-8-sig") as src:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            # Blank lines and comment lines are skipped
            if not stripped or stripped.startswith("#"):
                continue
            fields = [f.strip() for f in stripped.split(",")]
            if header is None:
                header = [f.lower() for f in fields]
                continue
            if len(fields) != len(header):
                raise ConverterError("%s:%d: %d fields, header has %d"
                                     % (path, line_number, len(fields), len(header)))
            rows.append(fields)
    if header is None:
        raise ConverterError("%s: no header line" % path)

    table = {}
    for k, name in enumerate(header):
        texts = [row[k] for row in rows]
        if name in SEED_TABLE_INTEGER_COLUMNS:
            table[name] = np.array([int(t) for t in texts], dtype=np.int64)
        else:
            table[name] = np.array([float(t) for t in texts], dtype=np.float64)
    return table


# -------------------------------------------------------------------------------------------------
# Voxel map (HDF5)
# -------------------------------------------------------------------------------------------------

def require_h5py():
    """Raise ConverterError with an explanation when h5py is not installed."""
    if not HAVE_H5PY:
        raise ConverterError("writing or reading an HDF5 microstructure file needs the Python "
                             "package h5py, which is not installed (try: pip install h5py, or "
                             "conda install h5py)")


# Per-grain datasets of the grain table: name -> (values per grain, numpy type)
GRAIN_DATASETS = (("quat", 4, np.float64), ("phase", 1, np.int32), ("position", 3, np.float64),
                  ("volume", 1, np.float64), ("weight", 1, np.float64),
                  ("shape", 6, np.float64))


def _check_grains(grains):
    """
    Check a grain table dict and return a new dict sorted by id with numpy arrays: "id" int64
    (G,), and the optional entries of GRAIN_DATASETS shaped (G,) or (G, c). Raises
    ConverterError for any rule of the format (ids in [1, 2^31 - 1] and unique, counts, unit
    quaternions, phases >= 1, finite values, volumes > 0).
    """
    if "id" not in grains:
        raise ConverterError("microstructure file: the grain table needs 'id'")
    unknown = set(grains) - {"id"} - {name for name, _, _ in GRAIN_DATASETS}
    if unknown:
        raise ConverterError("microstructure file: unknown grain table entries %s"
                             % sorted(unknown))
    ids = np.asarray(grains["id"])
    if ids.size == 0 or not np.issubdtype(ids.dtype, np.integer):
        raise ConverterError("microstructure file: 'id' must be a non-empty integer array")
    ids = ids.astype(np.int64).reshape(-1)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    if ids[0] < 1 or ids[-1] > INT32_MAX:
        raise ConverterError("microstructure file: grain ids must be in [1, 2^31 - 1]")
    if np.any(ids[1:] == ids[:-1]):
        raise ConverterError("microstructure file: duplicate grain id %d"
                             % ids[1:][ids[1:] == ids[:-1]][0])
    result = {"id": ids}
    for name, per_grain, dtype in GRAIN_DATASETS:
        if grains.get(name) is None:
            continue
        values = np.asarray(grains[name], dtype=np.float64)
        if values.size != per_grain * ids.size:
            raise ConverterError("microstructure file: '%s' holds %d values; expected %d per "
                                 "grain for %d grains" % (name, values.size, per_grain, ids.size))
        values = values.reshape((ids.size, per_grain) if per_grain > 1 else (ids.size,))[order]
        if not np.all(np.isfinite(values)):
            raise ConverterError("microstructure file: '%s' holds values that are not finite"
                                 % name)
        if name == "quat":
            norms = np.linalg.norm(values, axis=1)
            bad = np.nonzero(np.abs(norms - 1.0) > QUAT_NORM_TOLERANCE)[0]
            if bad.size > 0:
                raise ConverterError("microstructure file: grain %d has quaternion norm %.6g"
                                     % (ids[bad[0]], norms[bad[0]]))
        if name == "phase":
            if np.any(values != np.floor(values)) or np.any(values < 1) or \
                    np.any(values > INT32_MAX):
                raise ConverterError("microstructure file: phases must be whole numbers in "
                                     "[1, 2^31 - 1]")
        if name == "volume" and np.any(values <= 0.0):
            raise ConverterError("microstructure file: volumes must be > 0")
        result[name] = values.astype(dtype)
    return result


def _check_grid(grid):
    """
    Check a grid dict ("grain_ids" integer (nz, ny, nx), "origin", "spacing" 3 floats each) and
    return (ids int64 (nz, ny, nx), origin float64[3], spacing float64[3]).
    """
    for key in ("grain_ids", "origin", "spacing"):
        if key not in grid:
            raise ConverterError("microstructure file: the grid needs '%s'" % key)
    ids = np.asarray(grid["grain_ids"])
    if ids.ndim != 3:
        raise ConverterError("microstructure file: grain_ids must have shape (nz, ny, nx) (x "
                             "varying fastest), got %s" % (ids.shape,))
    if not np.issubdtype(ids.dtype, np.integer):
        raise ConverterError("microstructure file: grain_ids must be integers")
    ids = ids.astype(np.int64)
    # Grain ids become C ints in the generator
    if ids.size and int(ids.max()) > INT32_MAX:
        raise ConverterError("microstructure file: grain id %d is above the limit 2^31 - 1"
                             % int(ids.max()))
    origin = np.asarray(grid["origin"], dtype=np.float64).reshape(3)
    spacing = np.asarray(grid["spacing"], dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(spacing)):
        raise ConverterError("microstructure file: origin and spacing must be finite")
    if np.any(spacing <= 0.0):
        raise ConverterError("microstructure file: spacing must be > 0, got %s" % spacing)
    return ids, origin, spacing


def write_microstructure_file(path, grains, grid=None, provenance=None, compression=None):
    """
    Write an "ExaConstit microstructure file" (HDF5, format_version 1) with h5py.

    Layout (every value is a dataset, never an HDF5 attribute, so Conduit can load the file;
    exact rules in src/microstructure/io/microstructure_file_io.hpp):
      format, format_version          "ExaConstit microstructure", 1
      grains/id                       int64 (G,), in [1, 2^31 - 1], unique (sorted on write)
      grains/quat                     float64 (G, 4), w x y z, crystal-to-sample   (optional)
      grains/phase                    int32 (G,), >= 1                             (optional)
      grains/position                 float64 (G, 3), seed point                   (optional)
      grains/volume                   float64 (G,), > 0, target size               (optional)
      grains/weight                   float64 (G,), laguerre / additive weight     (optional)
      grains/shape                    float64 (G, 6), m11 m22 m33 m23 m13 m12      (optional)
      mesh/...                        only with `grid`: a Conduit Blueprint uniform mesh,
                                      coordset "coords" (dims = POINTS per axis = cells + 1,
                                      origin = lower corner of cell (0,0,0), spacing), topology
                                      "topo", element field "grain_id" with values (nz, ny, nx),
                                      x varying fastest; values <= 0 = unassigned
      provenance/<key>                strings, informational (readers ignore them)

    Input:
      path       - output .h5 file (overwritten);
      grains     - dict: "id" plus any of the optional entries above, one row per grain in the
                   same order (rows are sorted by id before writing);
      grid       - optional dict: "grain_ids" (integer array (nz, ny, nx)), "origin", "spacing"
                   (3 floats each, the units the user will give the generator);
      provenance - optional dict of strings (converter name, source file, options);
      compression - None (default) or "gzip": compress the grid values. Off by default because
                   the generator can only read gzip data when its HDF5 library was built with
                   zlib (ExaConstit's own HDF5 build is not).
    Output: none. Raises ConverterError on invalid content or missing h5py.
    """
    require_h5py()
    if compression not in (None, "gzip"):
        raise ConverterError("microstructure file: compression must be None or \"gzip\"")
    table = _check_grains(grains)
    if grid is not None:
        ids, origin, spacing = _check_grid(grid)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("format", data=MICROSTRUCTURE_FORMAT_NAME)
        h5.create_dataset("format_version", data=np.int64(MICROSTRUCTURE_FORMAT_VERSION))
        for name, values in table.items():
            h5.create_dataset("grains/" + name, data=values)
        if grid is not None:
            nz, ny, nx = ids.shape
            cs = h5.create_group("mesh/coordsets/coords")
            cs.create_dataset("type", data="uniform")
            for axis, cells in zip(("i", "j", "k"), (nx, ny, nz)):
                cs.create_dataset("dims/" + axis, data=np.int64(cells + 1))
            for d, axis in enumerate(("x", "y", "z")):
                cs.create_dataset("origin/" + axis, data=origin[d])
                cs.create_dataset("spacing/d" + axis, data=spacing[d])
            topo = h5.create_group("mesh/topologies/topo")
            topo.create_dataset("type", data="uniform")
            topo.create_dataset("coordset", data="coords")
            field = h5.create_group("mesh/fields/grain_id")
            field.create_dataset("association", data="element")
            field.create_dataset("topology", data="topo")
            # Chunks of up to 32^3 cells match the generator's default read tiles
            chunks = tuple(min(32, n) for n in ids.shape)
            if compression is None:
                field.create_dataset("values", data=ids, chunks=chunks)
            else:
                field.create_dataset("values", data=ids, chunks=chunks, compression=compression,
                                     shuffle=True)
        for key, value in (provenance or {}).items():
            h5.create_dataset("provenance/" + key, data=str(value))


def _as_text(value):
    """
    A string dataset's value as str: h5py variable-length strings (str or bytes), fixed-length
    strings (bytes), and Conduit's form, an array of 1-byte characters ending in a NUL.
    """
    if isinstance(value, np.ndarray):
        value = b"".join(value.reshape(-1).tolist())
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).rstrip("\0")


def _scalar(value):
    """A one-number dataset's value: an HDF5 scalar (h5py) or a 1-element array (Conduit)."""
    return np.asarray(value).reshape(-1)[0]


def read_microstructure_file(path):
    """
    Read an ExaConstit microstructure file written by write_microstructure_file or the generator
    (for tests and quick checks; the generator has its own reader).

    Output: dict with "format_version", "grains" (dict of numpy arrays, per-grain rows reshaped
    to (G, c)), "provenance" (dict of strings) and, for a file with a grid, "grain_ids" (int64,
    (nz, ny, nx)), "origin" and "spacing" (float64[3]).
    """
    require_h5py()
    result = {"grains": {}, "provenance": {}}
    with h5py.File(path, "r") as h5:
        name = _as_text(h5["format"][()])
        if name != MICROSTRUCTURE_FORMAT_NAME:
            raise ConverterError("%s is not an ExaConstit microstructure file" % path)
        result["format_version"] = int(_scalar(h5["format_version"][()]))
        grains = h5["grains"]
        result["grains"]["id"] = np.asarray(grains["id"][...], dtype=np.int64).reshape(-1)
        count = result["grains"]["id"].size
        for key, per_grain, _ in GRAIN_DATASETS:
            if key in grains:
                values = grains[key][...]
                result["grains"][key] = values.reshape((count, per_grain) if per_grain > 1
                                                       else (count,))
        if "mesh" in h5:
            cs = h5["mesh/coordsets/coords"]
            dims = [int(_scalar(cs["dims/" + axis][()])) - 1 for axis in ("i", "j", "k")]
            result["origin"] = np.array([_scalar(cs["origin/" + a][()]) for a in ("x", "y", "z")],
                                        dtype=np.float64)
            result["spacing"] = np.array([_scalar(cs["spacing/d" + a][()])
                                          for a in ("x", "y", "z")], dtype=np.float64)
            values = np.asarray(h5["mesh/fields/grain_id/values"][...], dtype=np.int64)
            result["grain_ids"] = values.reshape(dims[2], dims[1], dims[0])
        if "provenance" in h5:
            for key in h5["provenance"]:
                result["provenance"][key] = _as_text(h5["provenance"][key][()])
    return result


def write_voxel_map(path, grain_ids, origin, spacing, feature_ids=None, feature_quats=None,
                    feature_phases=None, attributes=None):
    """
    Write a microstructure file with a grid (a "voxel map"): the shortcut the grid converters
    use for write_microstructure_file.

    Input:
      path        - output .h5 file (overwritten);
      grain_ids   - integer array of shape (nz, ny, nx), x varying fastest; values <= 0 =
                    unassigned, values < 2^31;
      origin, spacing - 3 floats each: lower corner of voxel (0, 0, 0) and voxel edge lengths;
      feature_ids, feature_quats, feature_phases - optional per-grain table, rows in the same
                    order. Without feature_ids the grain table lists the positive ids of the
                    grid (the format always has grains/id);
      attributes  - optional dict of provenance strings.
    Output: none. Raises ConverterError on invalid content or missing h5py.
    """
    if feature_ids is None and (feature_quats is not None or feature_phases is not None):
        raise ConverterError("voxel map: feature quaternions/phases need feature ids")
    grid = {"grain_ids": grain_ids, "origin": origin, "spacing": spacing}
    ids, _, _ = _check_grid(grid)
    if feature_ids is None:
        present = np.unique(ids[ids > 0])
        if present.size == 0:
            raise ConverterError("voxel map: no voxel holds a grain id > 0")
        grains = {"id": present}
    else:
        grains = {"id": feature_ids, "quat": feature_quats, "phase": feature_phases}
    write_microstructure_file(path, grains, grid=grid, provenance=attributes)


def read_voxel_map(path):
    """
    Read a microstructure file with a grid (for tests and quick checks).

    Output: dict with "grain_ids" (int64, (nz, ny, nx)), "origin", "spacing" (float64[3]),
    "feature_id" and, when present, "feature_quat", "feature_phase", plus "provenance".
    """
    data = read_microstructure_file(path)
    if "grain_ids" not in data:
        raise ConverterError("%s has no grid (it holds only a grain table)" % path)
    result = {"grain_ids": data["grain_ids"], "origin": data["origin"],
              "spacing": data["spacing"], "provenance": data["provenance"]}
    for name in ("id", "quat", "phase"):
        if name in data["grains"]:
            result["feature_" + name] = data["grains"][name]
    return result


def provenance_lines(tool_name, source_path, extra=()):
    """
    Short provenance lines (converter name, source file, options) for file headers.

    Input: tool_name - script name; source_path - input file; extra - more "key: value" lines.
    Output: list of strings.
    """
    lines = ["written by %s from %s" % (tool_name, os.path.basename(str(source_path)))]
    lines.extend(extra)
    return lines
