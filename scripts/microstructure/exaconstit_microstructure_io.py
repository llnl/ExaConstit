#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared writers, readers and quaternion helpers for the ExaConstit microstructure converters.

The converters in this directory turn the output of other microstructure tools (HEXRD, MIDAS,
SPPARKS, ExaCA) into the two input formats the ExaConstit microstructure generator reads:

* "ExaConstit seed table v1" (CSV): one seed (grain centre) per row, with optional
  orientation, phase and grain size. Used with ``[Seeds] source = "file"``.
* "ExaConstit voxel map v1" (HDF5): a grain id per voxel of a regular grid, plus an optional
  per-grain table of orientations and phases. Used with ``[Source] type = "voxel_map"``,
  ``format = "exaconstit"``.

Both formats are specified in the C++ reader headers, ``src/microstructure/io/seed_table_io.hpp``
and ``src/microstructure/io/voxel_map_io.hpp``; the functions here write exactly what the readers
(``seed_table_io.cpp`` and ``voxel_map_io.cpp``) expect.

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

Requirements: Python 3 and numpy. h5py is needed only to write or read voxel maps; without it
those functions raise ``ConverterError`` with a clear message.
"""

import math
import os
import sys

import numpy as np

try:
    import h5py  # noqa: F401  (optional; only the voxel map functions need it)
    HAVE_H5PY = True
except ImportError:  # pragma: no cover - depends on the local Python installation
    h5py = None
    HAVE_H5PY = False


# Names written into the files so a human can tell what produced them.
SEED_TABLE_FORMAT_NAME = "ExaConstit seed table v1"
VOXEL_MAP_FORMAT_NAME = "ExaConstit voxel map v1"

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
                 "x", "y", "z". Optional: "id" (int >= 0, unique), "qw","qx","qy","qz" (all
                 four or none), "phase" (0 <= int <= 2^31-1), "weight", one of "volume"/"radius"/
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
        if np.any(ids < 0):
            raise ConverterError("seed table: ids must be >= 0")
        unique_ids, counts = np.unique(ids, return_counts=True)
        if np.any(counts > 1):
            raise ConverterError("seed table: duplicate id %d" % unique_ids[counts > 1][0])
    if not num_seeds:
        raise ConverterError("seed table: no seeds to write (the generator refuses an empty "
                             "table)")
    if "phase" in arrays and (np.any(arrays["phase"] < 0)
                              or np.any(arrays["phase"] > INT32_MAX)):
        raise ConverterError("seed table: phases must be in [0, 2^31 - 1]")
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
        raise ConverterError("writing or reading an HDF5 voxel map needs the Python package "
                             "h5py, which is not installed (try: pip install h5py, or conda "
                             "install h5py)")


def write_voxel_map(path, grain_ids, origin, spacing, feature_ids=None, feature_quats=None,
                    feature_phases=None, attributes=None):
    """
    Write an "ExaConstit voxel map v1" HDF5 file.

    Layout (see WORK_PACKAGES.md, "Shared formats"):
      /grain_ids      int64, shape (nz, ny, nx): x varies fastest. Values <= 0 = unassigned;
                      values must be < 2^31 (the generator stores grains as C ints).
      /origin         float64[3]: x, y, z of the LOWER CORNER of voxel (0,0,0). Voxel (i,j,k)
                      covers origin + [i, i+1]*spacing_x, etc.
      /spacing        float64[3], all > 0.
      /features/id    int64[nf], unique, in [1, 2^31 - 1]   (optional)
      /features/quat  float64[nf, 4], w x y z               (optional, needs /features/id)
      /features/phase int32[nf]                             (optional, needs /features/id)
    String attributes on the root group (format name, provenance) are informational only.

    Input:
      path        - output .h5 file (overwritten);
      grain_ids   - integer array of shape (nz, ny, nx);
      origin, spacing - 3 floats each, in the units the user will give the generator;
      feature_ids, feature_quats, feature_phases - optional per-grain table, rows in the same
                    order (they are sorted by id before writing);
      attributes  - optional dict of extra root attributes (strings or numbers).
    Output: none. Raises ConverterError on invalid content or missing h5py.
    """
    require_h5py()

    # Step 1: check the grid
    ids = np.asarray(grain_ids)
    if ids.ndim != 3:
        raise ConverterError("voxel map: grain_ids must have shape (nz, ny, nx), got %s"
                             % (ids.shape,))
    if not np.issubdtype(ids.dtype, np.integer):
        raise ConverterError("voxel map: grain_ids must be integers")
    ids = ids.astype(np.int64)
    # Grain ids become C ints in the generator
    if ids.size and int(ids.max()) > INT32_MAX:
        raise ConverterError("voxel map: grain id %d is above the limit 2^31 - 1"
                             % int(ids.max()))
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(spacing)):
        raise ConverterError("voxel map: origin and spacing must be finite")
    if np.any(spacing <= 0.0):
        raise ConverterError("voxel map: spacing must be > 0, got %s" % spacing)

    # Step 2: check the optional per-grain table and sort it by id
    if feature_ids is None and (feature_quats is not None or feature_phases is not None):
        raise ConverterError("voxel map: feature quaternions/phases need feature ids")
    if feature_ids is not None:
        fids = np.asarray(feature_ids, dtype=np.int64).reshape(-1)
        order = np.argsort(fids, kind="stable")
        fids = fids[order]
        if fids.size > 0 and (fids[0] < 1 or fids[-1] > INT32_MAX):
            raise ConverterError("voxel map: feature ids must be in [1, 2^31 - 1]")
        if np.any(fids[1:] == fids[:-1]):
            raise ConverterError("voxel map: duplicate feature id")
        if feature_quats is not None:
            fquats = np.asarray(feature_quats, dtype=np.float64).reshape(-1, 4)
            if fquats.shape[0] != fids.size:
                raise ConverterError("voxel map: %d quaternions for %d feature ids"
                                     % (fquats.shape[0], fids.size))
            fquats = fquats[order]
            norms = np.linalg.norm(fquats, axis=1)
            bad = np.nonzero(np.abs(norms - 1.0) > QUAT_NORM_TOLERANCE)[0]
            if bad.size > 0:
                raise ConverterError("voxel map: feature id %d has quaternion norm %.6g"
                                     % (fids[bad[0]], norms[bad[0]]))
        if feature_phases is not None:
            fphases = np.asarray(feature_phases, dtype=np.int64).reshape(-1)
            if fphases.size != fids.size:
                raise ConverterError("voxel map: %d phases for %d feature ids"
                                     % (fphases.size, fids.size))
            fphases = fphases[order]
            if np.any(fphases < 0) or np.any(fphases > INT32_MAX):
                raise ConverterError("voxel map: phases must be in [0, 2^31)")

    # Step 3: write the datasets
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = VOXEL_MAP_FORMAT_NAME
        for key, value in (attributes or {}).items():
            h5.attrs[key] = value
        h5.create_dataset("grain_ids", data=ids, dtype=np.int64)
        h5.create_dataset("origin", data=origin, dtype=np.float64)
        h5.create_dataset("spacing", data=spacing, dtype=np.float64)
        if feature_ids is not None:
            group = h5.create_group("features")
            group.create_dataset("id", data=fids, dtype=np.int64)
            if feature_quats is not None:
                group.create_dataset("quat", data=fquats, dtype=np.float64)
            if feature_phases is not None:
                group.create_dataset("phase", data=fphases.astype(np.int32), dtype=np.int32)


def read_voxel_map(path):
    """
    Read an "ExaConstit voxel map v1" HDF5 file (for tests and quick checks).

    Output: dict with "grain_ids" (int64, (nz, ny, nx)), "origin", "spacing" (float64[3]) and,
    when present, "feature_id", "feature_quat", "feature_phase", plus "attrs" (root attributes).
    """
    require_h5py()
    result = {}
    with h5py.File(path, "r") as h5:
        result["grain_ids"] = np.asarray(h5["grain_ids"][...], dtype=np.int64)
        result["origin"] = np.asarray(h5["origin"][...], dtype=np.float64)
        result["spacing"] = np.asarray(h5["spacing"][...], dtype=np.float64)
        if "features" in h5:
            group = h5["features"]
            for name in ("id", "quat", "phase"):
                if name in group:
                    result["feature_" + name] = group[name][...]
        result["attrs"] = {key: h5.attrs[key] for key in h5.attrs}
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
