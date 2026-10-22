#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of the microstructure converters in scripts/microstructure/.

Each test writes a tiny synthetic input in the format of the source tool (HEXRD grains.out,
MIDAS Grains.csv, SPPARKS text dump, ExaCA VTK + orientation file) into its own directory under
the current working directory, runs the converter script as a user would (a subprocess with the
same Python), reads the output back, and compares it with values computed independently in the
test. The quaternion helpers are tested directly against rotations whose quaternion is known in
closed form.

Where the scripts are: the environment variable EXACONSTIT_CONVERTER_DIR (set by CTest), else
../../../scripts/microstructure relative to this file. Outputs go to
./converters_test_output/ (CTest runs this in the build tree; never /tmp).

Run directly:  python test_converters.py
"""

import math
import os
import shutil
import subprocess
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.environ.get(
    "EXACONSTIT_CONVERTER_DIR",
    os.path.normpath(os.path.join(HERE, "..", "..", "..", "scripts", "microstructure")))
sys.path.insert(0, SCRIPT_DIR)
import exaconstit_microstructure_io as msio  # noqa: E402

OUTPUT_ROOT = os.path.join(os.getcwd(), "converters_test_output")


def setUpModule():
    """Start from an empty output directory so stale files cannot make a test pass."""
    if os.path.isdir(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT)


def case_dir(name):
    """Directory for one test case's inputs and outputs."""
    path = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


def run_script(script, args):
    """Run a converter script with this Python; return the CompletedProcess (text output)."""
    return subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, script)] + list(args),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


def rodrigues_matrix(axis, angle):
    """
    Rotation matrix of a rotation by ``angle`` (radians) about ``axis``, by Rodrigues' formula
    R = I + sin(angle) K + (1 - cos(angle)) K^2 with K the cross-product matrix of the unit
    axis. Written out here independently of the module under test.
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    k = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def axis_angle_quat(axis, angle):
    """Closed-form quaternion (cos(angle/2), sin(angle/2) n), sign chosen so that w >= 0."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    q = np.concatenate([[math.cos(0.5 * angle)], math.sin(0.5 * angle) * n])
    return -q if q[0] < 0.0 else q


def same_rotation(q1, q2, tol):
    """True when q1 and q2 are the same rotation (q or -q) to within tol."""
    return min(np.max(np.abs(q1 - q2)), np.max(np.abs(q1 + q2))) <= tol


# -------------------------------------------------------------------------------------------------
# Quaternion math
# -------------------------------------------------------------------------------------------------

class QuaternionMathTest(unittest.TestCase):
    """The matrix and exponential-map conversions give the standard quaternion of a rotation."""

    def test_quarter_turn_about_z(self):
        # 90 degrees about z maps x -> y: R = [[0,-1,0],[1,0,0],[0,0,1]]; its quaternion is
        # (cos 45, 0, 0, sin 45). The exp map of the same rotation is (0, 0, pi/2).
        expected = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
        from_matrix = msio.quat_from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        from_expmap = msio.quat_from_expmap([0.0, 0.0, 0.5 * math.pi])
        np.testing.assert_allclose(from_matrix, expected, atol=1e-15)
        np.testing.assert_allclose(from_expmap, expected, atol=1e-15)

    def test_general_rotation_matrix_and_expmap_agree(self):
        # The same rotation (2 rad about (1,2,3)) given as a Rodrigues matrix and as an exp map
        # must give the same quaternion, equal to the closed form.
        axis, angle = np.array([1.0, 2.0, 3.0]), 2.0
        expected = axis_angle_quat(axis, angle)
        q_matrix = msio.quat_from_matrix(rodrigues_matrix(axis, angle))
        q_expmap = msio.quat_from_expmap(angle * axis / np.linalg.norm(axis))
        np.testing.assert_allclose(q_matrix, expected, atol=1e-14)
        np.testing.assert_allclose(q_expmap, expected, atol=1e-14)

    def test_matrix_quat_matrix_round_trip(self):
        # Random rotations plus half turns about x, y, z and a diagonal: half turns make
        # trace = -1, so Shepperd's method must use each of its x, y, z branches.
        rng = np.random.default_rng(12345)
        cases = [(rng.normal(size=3), rng.uniform(0.0, math.pi)) for _ in range(50)]
        cases += [([1, 0, 0], math.pi), ([0, 1, 0], math.pi), ([0, 0, 1], math.pi),
                  ([1, 1, 1], math.pi), ([1, 0, 0], 0.0)]
        for axis, angle in cases:
            matrix = rodrigues_matrix(axis, angle)
            q = msio.quat_from_matrix(matrix)
            self.assertGreaterEqual(q[0], 0.0)
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=14)
            np.testing.assert_allclose(msio.matrix_from_quat(q), matrix, atol=1e-14)
            self.assertTrue(same_rotation(q, axis_angle_quat(axis, angle), 1e-14))

    def test_expmap_large_and_small_angles(self):
        # An angle above pi gives w < 0 in the closed form; the helper returns -q (same
        # rotation) with w >= 0. A tiny angle uses the series and still matches.
        axis = np.array([0.3, -0.4, 0.5])
        unit = axis / np.linalg.norm(axis)
        q = msio.quat_from_expmap(4.0 * unit)
        self.assertGreaterEqual(q[0], 0.0)
        np.testing.assert_allclose(msio.matrix_from_quat(q), rodrigues_matrix(axis, 4.0),
                                   atol=1e-14)
        q_small = msio.quat_from_expmap(1e-7 * unit)
        np.testing.assert_allclose(q_small, axis_angle_quat(axis, 1e-7), atol=1e-16)

    def test_conjugate_is_transpose(self):
        # Conjugating the quaternion inverts the rotation, i.e. transposes the matrix.
        matrix = rodrigues_matrix([2.0, -1.0, 0.5], 1.1)
        q = msio.quat_from_matrix(matrix)
        np.testing.assert_allclose(msio.matrix_from_quat(msio.conjugate_quat(q)), matrix.T,
                                   atol=1e-14)

    def test_rotation_matrix_problem(self):
        # A rotation passes; a reflection and a sheared matrix are reported.
        self.assertIsNone(msio.rotation_matrix_problem(rodrigues_matrix([1, 1, 0], 0.7)))
        self.assertIn("reflection", msio.rotation_matrix_problem(np.diag([1.0, 1.0, -1.0])))
        sheared = np.eye(3)
        sheared[0, 1] = 0.01
        self.assertIn("not orthonormal", msio.rotation_matrix_problem(sheared))


# -------------------------------------------------------------------------------------------------
# Shared writers
# -------------------------------------------------------------------------------------------------

class SeedTableWriterTest(unittest.TestCase):
    """The seed table writer round-trips exactly and refuses what the C++ reader refuses."""

    def test_round_trip_is_bitwise(self):
        # 17 significant digits make text -> double exact: values like 0.1 and 1/3 come back
        # with the identical bit pattern.
        path = os.path.join(case_dir("seed_writer"), "round_trip.csv")
        q = axis_angle_quat([1, 2, 3], 0.9)
        columns = {"id": [7, 3], "x": [0.1, 1.0 / 3.0], "y": [-2.5e-7, 1e10], "z": [0.0, -1.0],
                   "qw": [q[0], 1.0], "qx": [q[1], 0.0], "qy": [q[2], 0.0], "qz": [q[3], 0.0],
                   "phase": [0, 2], "radius": [0.25, 1.0 / 7.0]}
        msio.write_seed_table(path, columns, ["test"])
        table = msio.read_seed_table(path)
        for name, values in columns.items():
            self.assertTrue(np.array_equal(table[name], np.asarray(values)), name)
        with open(path) as f:
            self.assertEqual(f.readline().strip(), "# ExaConstit seed table v1")

    def test_byte_order_mark_is_ignored(self):
        # Spreadsheet programs may start a UTF-8 file with a byte-order mark; the reader drops
        # it (like the C++ reader), so the first header name is still "x"
        path = os.path.join(case_dir("seed_writer"), "bom.csv")
        with open(path, "w", encoding="utf-8-sig") as out:
            out.write("x,y,z\n1.5,2,3\n")
        table = msio.read_seed_table(path)
        self.assertEqual(sorted(table), ["x", "y", "z"])
        self.assertEqual(table["x"][0], 1.5)

    def test_invalid_tables_are_refused(self):
        path = os.path.join(case_dir("seed_writer"), "bad.csv")
        base = {"x": [0.0], "y": [0.0], "z": [0.0]}
        bad_cases = [
            dict(base, qw=[1.0], qx=[0.0]),                              # partial quaternion
            dict(base, qw=[0.9], qx=[0.0], qy=[0.0], qz=[0.0]),          # norm 0.9
            {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0], "id": [4, 4]},  # duplicate id
            dict(base, radius=[1.0], volume=[1.0]),                      # two size columns
            {"x": [0.0], "y": [0.0]},                                    # z missing
            dict(base, phase=[2 ** 31]),                                 # phase above C int
            {"x": [], "y": [], "z": []},                                 # no seeds
        ]
        for columns in bad_cases:
            with self.assertRaises(msio.ConverterError):
                msio.write_seed_table(path, columns)


@unittest.skipUnless(msio.HAVE_H5PY, "h5py is not installed")
class VoxelMapWriterTest(unittest.TestCase):
    """The voxel map writer stores the canonical layout and sorts the per-grain table by id."""

    def test_round_trip_and_feature_sorting(self):
        path = os.path.join(case_dir("voxel_writer"), "map.h5")
        ids = np.arange(24, dtype=np.int32).reshape(2, 3, 4) % 3   # values 0, 1, 2
        quats = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        msio.write_voxel_map(path, ids, [1.0, 2.0, 3.0], [0.5, 0.5, 0.25],
                             feature_ids=[2, 1], feature_quats=quats, feature_phases=[5, 4])
        result = msio.read_voxel_map(path)
        self.assertEqual(result["grain_ids"].dtype, np.int64)
        self.assertTrue(np.array_equal(result["grain_ids"], ids))
        np.testing.assert_array_equal(result["origin"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result["spacing"], [0.5, 0.5, 0.25])
        np.testing.assert_array_equal(result["feature_id"], [1, 2])
        np.testing.assert_array_equal(result["feature_quat"], quats[::-1])
        np.testing.assert_array_equal(result["feature_phase"], [4, 5])
        self.assertEqual(result["feature_phase"].dtype, np.int32)

    def test_invalid_maps_are_refused(self):
        path = os.path.join(case_dir("voxel_writer"), "bad.h5")
        ids = np.ones((2, 2, 2), dtype=np.int64)
        with self.assertRaises(msio.ConverterError):
            msio.write_voxel_map(path, ids, [0, 0, 0], [1.0, 0.0, 1.0])     # zero spacing
        with self.assertRaises(msio.ConverterError):
            msio.write_voxel_map(path, ids[0], [0, 0, 0], [1, 1, 1])        # rank 2
        with self.assertRaises(msio.ConverterError):
            msio.write_voxel_map(path, ids, [0, 0, 0], [1, 1, 1], feature_ids=[0])  # id 0
        with self.assertRaises(msio.ConverterError):
            big = ids.copy()
            big[0, 0, 0] = 2 ** 31                                           # above C int
            msio.write_voxel_map(path, big, [0, 0, 0], [1, 1, 1])


# -------------------------------------------------------------------------------------------------
# Converters
# -------------------------------------------------------------------------------------------------

class HelpTest(unittest.TestCase):
    """Every converter answers --help (argparse) and exits cleanly."""

    def test_help(self):
        for script in ("hexrd_grains_to_seeds.py", "midas_grains_to_seeds.py",
                       "spparks_to_voxels.py", "exaca_to_voxels.py"):
            result = run_script(script, ["--help"])
            self.assertEqual(result.returncode, 0, script + result.stderr)
            self.assertIn("usage", result.stdout)


def write_hexrd_grains(path, grains):
    """
    Write a grains.out file exactly as HEXRD's GrainDataWriter does: header line, then
    '{:<12d}  {:<12f}  {:<12e}' for id/completeness/chi^2 and '{:<23.16e}' for the 18 reals,
    joined by two spaces. ``grains`` is a list of (id, completeness, chi2, expmap, tvec).
    """
    header_items = ['# grain ID', 'completeness', 'chi^2',
                    'exp_map_c[0]', 'exp_map_c[1]', 'exp_map_c[2]',
                    't_vec_c[0]', 't_vec_c[1]', 't_vec_c[2]',
                    'inv(V_s)[0,0]', 'inv(V_s)[1,1]', 'inv(V_s)[2,2]',
                    'inv(V_s)[1,2]*sqrt(2)', 'inv(V_s)[0,2]*sqrt(2)', 'inv(V_s)[0,1]*sqrt(2)',
                    'ln(V_s)[0,0]', 'ln(V_s)[1,1]', 'ln(V_s)[2,2]',
                    'ln(V_s)[1,2]', 'ln(V_s)[0,2]', 'ln(V_s)[0,1]']
    delim = '  '
    header = delim.join([delim.join(['{:<12}'] * 3).format(*header_items[:3]),
                         delim.join(['{:<23}'] * 18).format(*header_items[3:])])
    with open(path, "w") as out:
        out.write(header + "\n")
        for gid, completeness, chi2, expmap, tvec in grains:
            reals = list(expmap) + list(tvec) + [1.0, 1.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6
            out.write(delim.join([delim.join(['{:<12d}', '{:<12f}', '{:<12e}'])
                                  .format(gid, completeness, chi2),
                                  delim.join(['{:<23.16e}'] * 18).format(*reals)]) + "\n")


class HexrdConverterTest(unittest.TestCase):
    """grains.out -> seed table: position t_vec_c, quaternion of exp_map_c, filters."""

    def setUp(self):
        self.dir = case_dir("hexrd")
        self.input = os.path.join(self.dir, "grains.out")
        # Three grains; grain 2 has low completeness and grain 5 a large chi^2
        self.grains = [(0, 0.95, 1e-3, [0.1, -0.2, 0.3], [0.05, -0.1, 0.2]),
                       (2, 0.40, 1e-3, [1.0, 0.5, -2.5], [0.3, 0.3, 0.3]),
                       (5, 0.99, 5e-2, [0.0, 0.0, 0.0], [-0.2, 0.1, 0.0])]
        write_hexrd_grains(self.input, self.grains)

    def test_conversion_and_filters(self):
        output = os.path.join(self.dir, "seeds.csv")
        result = run_script("hexrd_grains_to_seeds.py",
                            [self.input, output, "--conjugate", "no", "--units-scale", "10",
                             "--min-completeness", "0.5", "--max-chi2", "0.01"])
        self.assertEqual(result.returncode, 0, result.stderr)
        table = msio.read_seed_table(output)
        # Only grain 0 passes both filters; position is 10 * t_vec_c
        np.testing.assert_array_equal(table["id"], [0])
        np.testing.assert_allclose([table["x"][0], table["y"][0], table["z"][0]],
                                   [0.5, -1.0, 2.0], rtol=1e-15)
        expmap = np.array(self.grains[0][3])
        expected = axis_angle_quat(expmap, np.linalg.norm(expmap))
        q = np.array([table[c][0] for c in ("qw", "qx", "qy", "qz")])
        np.testing.assert_allclose(q, expected, atol=1e-15)

    def test_conjugate_yes_and_large_angle(self):
        # Grain 2 has |exp map| = 2.78 rad; --conjugate yes flips the vector part
        output = os.path.join(self.dir, "seeds_conj.csv")
        result = run_script("hexrd_grains_to_seeds.py", [self.input, output, "--conjugate", "yes"])
        self.assertEqual(result.returncode, 0, result.stderr)
        table = msio.read_seed_table(output)
        np.testing.assert_array_equal(table["id"], [0, 2, 5])
        for row, grain in enumerate(self.grains):
            expmap = np.array(grain[3])
            angle = np.linalg.norm(expmap)
            expected = axis_angle_quat(expmap, angle) if angle > 0 else np.array([1., 0, 0, 0])
            expected = msio.conjugate_quat(expected)
            q = np.array([table[c][row] for c in ("qw", "qx", "qy", "qz")])
            np.testing.assert_allclose(q, expected, atol=1e-15)

    def test_nan_fit_quality(self):
        # A NaN chi^2 would fail every filter comparison and vanish silently; it is an error
        # unless --drop-nan, which drops it and reports the count
        nan_input = os.path.join(self.dir, "grains_nan.out")
        write_hexrd_grains(nan_input, self.grains + [(8, 0.9, float("nan"), [0.1, 0.1, 0.1],
                                                      [0.0, 0.0, 0.0])])
        output = os.path.join(self.dir, "seeds_nan.csv")
        result = run_script("hexrd_grains_to_seeds.py", [nan_input, output, "--conjugate", "no"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("grain ID 8", result.stderr)
        result = run_script("hexrd_grains_to_seeds.py",
                            [nan_input, output, "--conjugate", "no", "--drop-nan"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 with NaN", result.stdout)
        np.testing.assert_array_equal(msio.read_seed_table(output)["id"], [0, 2, 5])

    def test_required_conjugate_flag(self):
        result = run_script("hexrd_grains_to_seeds.py",
                            [self.input, os.path.join(self.dir, "x.csv")])
        self.assertEqual(result.returncode, 2)            # argparse usage error
        self.assertIn("--conjugate", result.stderr)

    def test_bad_column_count_names_line(self):
        bad = os.path.join(self.dir, "bad.out")
        with open(bad, "w") as out:
            out.write("# header\n1 0.9 0.001 0.1 0.2\n")
        result = run_script("hexrd_grains_to_seeds.py",
                            [bad, os.path.join(self.dir, "x.csv"), "--conjugate", "no"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("bad.out:2", result.stderr)


def write_midas_grains(path, grains, header=True):
    """
    Write a Grains.csv as ProcessGrains.c does: '%' metadata lines, the '%GrainID\\tO11...'
    column line, then 47 tab-separated values per grain, each '%lf' (6 decimals) followed by a
    tab. ``grains`` is a list of dicts with id, matrix (3x3), pos (3), radius, confidence, phase.
    """
    names = (["GrainID"] + ["O%d%d" % (i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
             + ["X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma", "DiffPos", "DiffOme",
                "DiffAngle", "GrainRadius", "Confidence"]
             + ["eFab%d%d" % (i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
             + ["eKen%d%d" % (i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
             + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2"])
    with open(path, "w") as out:
        out.write("%%NumGrains %d\n%%BeamCenter 0.000000\n%%BeamThickness 200.000000\n"
                  "%%GlobalPosition 0.000000\n%%NumPhases 1\n%%PhaseInfo\n%%\tSpaceGroup:225\n"
                  "%%\tLattice Parameter: 3.600000 3.600000 3.600000 90.000000 90.000000 "
                  "90.000000\n" % len(grains))
        if header:
            out.write("%" + "\t".join(names) + "\t\n")
        for g in grains:
            values = ([g["id"]] + list(np.asarray(g["matrix"]).reshape(9)) + list(g["pos"])
                      + [3.6, 3.6, 3.6, 90.0, 90.0, 90.0, 10.0, 0.01, 0.02, g["radius"],
                         g["confidence"]] + [0.0] * 19 + [g["phase"], 1.0, 2.0, 3.0])
            out.write("%d\t" % values[0] + "".join("%lf\t" % v for v in values[1:]) + "\n")


class MidasConverterTest(unittest.TestCase):
    """Grains.csv -> seed table: quaternion of O (6-decimal input), radius, phase, filters."""

    def setUp(self):
        self.dir = case_dir("midas")
        self.input = os.path.join(self.dir, "Grains.csv")
        self.rotations = [([1, 2, 3], 0.7), ([0, 0, 1], math.pi), ([-1, 0.5, 0.2], 2.9)]
        self.grains = []
        for k, (axis, angle) in enumerate(self.rotations):
            self.grains.append({"id": 10 + k, "matrix": rodrigues_matrix(axis, angle),
                                "pos": [100.0 * k, -50.0, 25.5], "radius": 40.0 + k,
                                "confidence": [0.9, 0.3, 0.8][k], "phase": [1, 1, 2][k]})
        write_midas_grains(self.input, self.grains)

    def check_table(self, table, kept, conjugate):
        """Compare the seed table rows with the grains listed in ``kept``."""
        np.testing.assert_array_equal(table["id"], [self.grains[k]["id"] for k in kept])
        for row, k in enumerate(kept):
            expected = axis_angle_quat(*self.rotations[k])
            if conjugate:
                expected = msio.conjugate_quat(expected)
            q = np.array([table[c][row] for c in ("qw", "qx", "qy", "qz")])
            # The matrix was printed with 6 decimals, so agreement is to about 1e-6
            self.assertTrue(same_rotation(q, expected, 5e-6), (q, expected))
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=14)
            # Positions and radius scaled by 1e-3 (micrometres -> millimetres)
            np.testing.assert_allclose([table["x"][row], table["y"][row], table["z"][row]],
                                       1e-3 * np.array(self.grains[k]["pos"]), rtol=1e-15)
            self.assertAlmostEqual(table["radius"][row], 1e-3 * self.grains[k]["radius"],
                                   places=15)
            self.assertEqual(table["phase"][row], self.grains[k]["phase"])

    def test_conversion(self):
        output = os.path.join(self.dir, "seeds.csv")
        result = run_script("midas_grains_to_seeds.py",
                            [self.input, output, "--conjugate", "no", "--units-scale", "1e-3",
                             "--min-confidence", "0.5"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.check_table(msio.read_seed_table(output), [0, 2], conjugate=False)

    def test_conjugate_and_no_header_fallback(self):
        # Without the %GrainID line the ProcessGrains.c column order is assumed
        no_header = os.path.join(self.dir, "Grains_noheader.csv")
        write_midas_grains(no_header, self.grains, header=False)
        output = os.path.join(self.dir, "seeds_conj.csv")
        result = run_script("midas_grains_to_seeds.py",
                            [no_header, output, "--conjugate", "yes", "--units-scale", "1e-3"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.check_table(msio.read_seed_table(output), [0, 1, 2], conjugate=True)

    def test_errors(self):
        out = os.path.join(self.dir, "x.csv")
        # Missing --conjugate is a usage error
        result = run_script("midas_grains_to_seeds.py", [self.input, out])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--conjugate", result.stderr)
        # A zero radius is refused unless --no-radius is given
        grains = [dict(g) for g in self.grains]
        grains[1]["radius"] = 0.0
        zero_radius = os.path.join(self.dir, "Grains_zero_radius.csv")
        write_midas_grains(zero_radius, grains)
        result = run_script("midas_grains_to_seeds.py", [zero_radius, out, "--conjugate", "no"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("GrainRadius", result.stderr)
        result = run_script("midas_grains_to_seeds.py",
                            [zero_radius, out, "--conjugate", "no", "--no-radius"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("radius", msio.read_seed_table(out))
        # A matrix that is not a rotation (a reflection) is refused, naming the grain
        grains[0]["matrix"] = np.diag([1.0, 1.0, -1.0])
        reflection = os.path.join(self.dir, "Grains_reflection.csv")
        write_midas_grains(reflection, grains)
        result = run_script("midas_grains_to_seeds.py",
                            [reflection, out, "--conjugate", "no", "--no-radius"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("GrainID 10", result.stderr)


def spparks_spin(i, j, k):
    """Spin of lattice site (i, j, k) in the synthetic SPPARKS model (values 1..5)."""
    return 1 + (i + 2 * j + 3 * k) % 5


def write_spparks_dump(path, dims, spacing, snapshots, drop_site=False, box_cells=None):
    """
    Write a SPPARKS text dump (dump_text.cpp layout) with sites in a shuffled order, as a
    parallel run without dump_modify sort would. ``snapshots`` is a list of spin functions.
    Sites sit at xlo + i * spacing (sc lattice, basis site at the cell corner). ``box_cells``
    overrides the number of cells per axis written in ITEM: BOX BOUNDS (to fake a bad box).
    """
    nx, ny, nz = dims
    sites = [(i, j, k) for k in range(nz) for j in range(ny) for i in range(nx)]
    order = np.random.default_rng(7).permutation(len(sites))
    if drop_site:
        order = order[1:]
    with open(path, "w") as out:
        for step, spin in enumerate(snapshots):
            out.write("ITEM: TIMESTEP\n%d %10g\n" % (step, 1.5 * step))
            out.write("ITEM: NUMBER OF ATOMS\n%d\n" % len(order))
            out.write("ITEM: BOX BOUNDS\n")
            for n in (box_cells or dims):
                out.write("%g %g\n" % (0.0, n * spacing))
            out.write("ITEM: ATOMS id type x y z\n")
            for s in order:
                i, j, k = sites[s]
                out.write("%d %d %g %g %g \n" % (s + 1, spin(i, j, k), i * spacing,
                                                 j * spacing, k * spacing))


@unittest.skipUnless(msio.HAVE_H5PY, "h5py is not installed")
class SpparksConverterTest(unittest.TestCase):
    """SPPARKS dump -> voxel map: each site is one voxel holding its spin."""

    def setUp(self):
        self.dir = case_dir("spparks")
        self.dims = (4, 3, 2)
        self.input = os.path.join(self.dir, "potts.dump")
        # Two snapshots: the first all spin 9, the second the spparks_spin pattern
        write_spparks_dump(self.input, self.dims, 0.5,
                           [lambda i, j, k: 9, spparks_spin])

    def expected_ids(self):
        nx, ny, nz = self.dims
        ids = np.zeros((nz, ny, nx), dtype=np.int64)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    ids[k, j, i] = spparks_spin(i, j, k)
        return ids

    def test_last_snapshot_corner_sites(self):
        output = os.path.join(self.dir, "map.h5")
        result = run_script("spparks_to_voxels.py", [self.input, output, "--units-scale", "2"])
        self.assertEqual(result.returncode, 0, result.stderr)
        voxels = msio.read_voxel_map(output)
        # Voxel (i,j,k) holds the spin of site (i,j,k); x varies fastest in (nz, ny, nx)
        self.assertTrue(np.array_equal(voxels["grain_ids"], self.expected_ids()))
        # Corner sites: the first site (0,0,0) is the lower corner; spacing 0.5 * 2 = 1
        np.testing.assert_array_equal(voxels["origin"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(voxels["spacing"], [1.0, 1.0, 1.0])
        self.assertNotIn("feature_id", voxels)

    def test_first_snapshot_center_sites(self):
        output = os.path.join(self.dir, "map_first.h5")
        result = run_script("spparks_to_voxels.py",
                            [self.input, output, "--snapshot", "0", "--site-location", "center"])
        self.assertEqual(result.returncode, 0, result.stderr)
        voxels = msio.read_voxel_map(output)
        self.assertTrue(np.all(voxels["grain_ids"] == 9))
        # Centre sites: the lower corner is half a spacing below the first site
        np.testing.assert_array_equal(voxels["origin"], [-0.25, -0.25, -0.25])

    def test_spacing_from_extent_not_smallest_gap(self):
        # 200 sites at a = 0.123457 printed with %g (6 digits): the smallest printed gap is
        # 0.1234 (4.6e-4 relative error), while (max - min)/(n - 1) = 24.5679/199 recovers a
        # to about 2e-6 relative (the %g rounding of the last coordinate divided by 199)
        fine = os.path.join(self.dir, "fine.dump")
        write_spparks_dump(fine, (200, 2, 1), 0.123457, [spparks_spin])
        output = os.path.join(self.dir, "fine.h5")
        result = run_script("spparks_to_voxels.py", [fine, output])
        self.assertEqual(result.returncode, 0, result.stderr)
        voxels = msio.read_voxel_map(output)
        self.assertEqual(voxels["grain_ids"].shape, (1, 2, 200))
        np.testing.assert_allclose(voxels["spacing"][:2], [0.123457, 0.123457], rtol=1e-5)

    def test_box_bounds_mismatch_is_refused(self):
        # A box one cell longer in y than the sites (e.g. a missing plane) disagrees with the
        # detected lattice far beyond the %g precision
        bad_box = os.path.join(self.dir, "bad_box.dump")
        write_spparks_dump(bad_box, self.dims, 0.5, [spparks_spin], box_cells=(4, 4, 2))
        result = run_script("spparks_to_voxels.py", [bad_box, os.path.join(self.dir, "x.h5")])
        self.assertEqual(result.returncode, 1)
        self.assertIn("BOX BOUNDS", result.stderr)
        self.assertIn("axis 1", result.stderr)

    def test_missing_site_is_refused(self):
        holes = os.path.join(self.dir, "holes.dump")
        write_spparks_dump(holes, self.dims, 1.0, [spparks_spin], drop_site=True)
        result = run_script("spparks_to_voxels.py", [holes, os.path.join(self.dir, "x.h5")])
        self.assertEqual(result.returncode, 1)
        self.assertIn("have no site", result.stderr)


def write_exaca_vtk(path, dims, origin, spacing, fields, binary):
    """
    Write a legacy VTK file like ExaCA's Print::writeHeader/printViewData: STRUCTURED_POINTS,
    POINT_DATA, one 'SCALARS name type 1' + 'LOOKUP_TABLE default' block per field, values x
    fastest. Binary values are big-endian with no newline after a block (as ExaCA writes them);
    text values are space separated with a newline per z layer.
    ``fields`` is a list of (name, vtk_type, values (nz, ny, nx)).
    """
    nx, ny, nz = dims
    numpy_type = {"int": ">i4", "short": ">i2", "float": ">f4"}
    with open(path, "wb") as out:
        header = ("# vtk DataFile Version 3.0\nvtk output\n%s\nDATASET STRUCTURED_POINTS\n"
                  "DIMENSIONS %d %d %d\nORIGIN %g %g %g\nSPACING %g %g %g\nPOINT_DATA %d\n"
                  % ("BINARY" if binary else "ASCII", nx, ny, nz, origin[0], origin[1],
                     origin[2], spacing, spacing, spacing, nx * ny * nz))
        out.write(header.encode("ascii"))
        for name, vtk_type, values in fields:
            out.write(("SCALARS %s %s 1\nLOOKUP_TABLE default\n" % (name, vtk_type))
                      .encode("ascii"))
            if binary:
                out.write(np.asarray(values).astype(numpy_type[vtk_type]).tobytes())
            else:
                for k in range(nz):
                    layer = np.asarray(values)[k].reshape(-1)
                    out.write((" ".join("%g" % v for v in layer) + " \n").encode("ascii"))


@unittest.skipUnless(msio.HAVE_H5PY, "h5py is not installed")
class ExacaConverterTest(unittest.TestCase):
    """ExaCA VTK + orientation file -> voxel map with renumbered ids and per-grain quaternions."""

    def setUp(self):
        self.dir = case_dir("exaca")
        self.dims = (4, 3, 2)
        nx, ny, nz = self.dims
        # GrainIDs: positive (epitaxial) 1..4, negative (nucleated) -1 and -5, and one 0 cell
        pattern = np.array([1, 2, 3, 4, -1, -5], dtype=np.int64)
        self.ids = pattern[np.arange(nx * ny * nz) % 6].reshape(nz, ny, nx)
        self.ids[1, 2, 3] = 0
        layer = np.arange(nx * ny * nz, dtype=np.int64).reshape(nz, ny, nx) % 3
        self.fields = [("LayerID", "short", layer), ("GrainID", "int", self.ids),
                       ("Temperature", "float", 0.5 * layer)]
        self.origin = (1e-5, 2e-5, 0.0)
        self.spacing = 2.5e-6
        # Orientation file: N = 3 rotation matrices, 9 comma-separated values per line
        self.rotations = [([1, 0, 0], 0.3), ([0, 1, 1], 1.2), ([1, -2, 0.5], 2.0)]
        self.orient = os.path.join(self.dir, "GrainOrientationVectors.csv")
        with open(self.orient, "w") as out:
            out.write("%d\n" % len(self.rotations))
            for axis, angle in self.rotations:
                out.write(",".join("%.17g" % v for v in rodrigues_matrix(axis, angle).reshape(9))
                          + "\n")

    def convert(self, binary, extra):
        vtk = os.path.join(self.dir, "grains_%s.vtk" % ("bin" if binary else "ascii"))
        write_exaca_vtk(vtk, self.dims, self.origin, self.spacing, self.fields, binary)
        output = os.path.join(self.dir, "map_%s.h5" % ("bin" if binary else "ascii"))
        result = run_script("exaca_to_voxels.py", [vtk, output] + extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return msio.read_voxel_map(output)

    def check_map(self, voxels, conjugate):
        # Negative ids: -g -> max positive id (4) + g, so -1 -> 5 and -5 -> 9; 0 stays 0
        expected = np.where(self.ids < 0, 4 - self.ids, self.ids)
        self.assertTrue(np.array_equal(voxels["grain_ids"], expected))
        # Lengths times 1e6 (m -> um); POINT_DATA points are cell centres, so the lower corner
        # is ORIGIN - spacing/2
        np.testing.assert_allclose(voxels["origin"],
                                   1e6 * (np.array(self.origin) - 0.5 * self.spacing), rtol=1e-14)
        np.testing.assert_allclose(voxels["spacing"], [2.5, 2.5, 2.5], rtol=1e-14)
        # One feature row per grain present; ExaCA grain g uses orientation (|g| - 1) mod 3
        np.testing.assert_array_equal(voxels["feature_id"], [1, 2, 3, 4, 5, 9])
        exaca_of_feature = [1, 2, 3, 4, -1, -5]
        for row, g in enumerate(exaca_of_feature):
            q_expected = axis_angle_quat(*self.rotations[(abs(g) - 1) % 3])
            if conjugate:
                q_expected = msio.conjugate_quat(q_expected)
            np.testing.assert_allclose(voxels["feature_quat"][row], q_expected, atol=1e-14)

    def test_ascii_and_binary(self):
        # The same grain structure in text and binary VTK gives the same voxel map
        args = ["--units-scale", "1e6", "--orientations", self.orient, "--conjugate", "no"]
        ascii_map = self.convert(False, args)
        binary_map = self.convert(True, args)
        self.check_map(ascii_map, conjugate=False)
        self.check_map(binary_map, conjugate=False)
        self.assertTrue(np.array_equal(ascii_map["grain_ids"], binary_map["grain_ids"]))

    def test_conjugate_yes(self):
        voxels = self.convert(True, ["--units-scale", "1e6", "--orientations", self.orient,
                                     "--conjugate", "yes"])
        self.check_map(voxels, conjugate=True)

    def test_quaternion_file_layout(self):
        # A 4-value file in x y z w order is reordered to w x y z (and nothing else)
        quat_file = os.path.join(self.dir, "orient_xyzw.csv")
        quats = [axis_angle_quat(*r) for r in self.rotations]
        with open(quat_file, "w") as out:
            out.write("3\n")
            for q in quats:
                out.write("%.17g,%.17g,%.17g,%.17g\n" % (q[1], q[2], q[3], q[0]))
        voxels = self.convert(False, ["--orientations", quat_file, "--conjugate", "no",
                                      "--quat-layout", "xyzw"])
        np.testing.assert_array_equal(voxels["feature_quat"][0], quats[0])
        # Without --quat-layout the 4-value file is refused
        vtk = os.path.join(self.dir, "grains_ascii.vtk")
        result = run_script("exaca_to_voxels.py", [vtk, os.path.join(self.dir, "x.h5"),
                                                   "--orientations", quat_file,
                                                   "--conjugate", "no"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("--quat-layout", result.stderr)

    def test_large_ascii_grid_is_fast(self):
        # 64^3 = 262144 points with three text fields: the reader parses only GrainID, with
        # numpy in bounded slices, and skips the other two blocks without parsing them. The
        # time limit is generous; the old per-value reader also used about 100 bytes per value
        n = 64
        rng = np.random.default_rng(3)
        ids = rng.integers(-50, 200, size=(n, n, n))
        ids[ids == 0] = 1
        layer = np.zeros((n, n, n), dtype=np.int64)
        vtk = os.path.join(self.dir, "large_ascii.vtk")
        write_exaca_vtk(vtk, (n, n, n), (0.0, 0.0, 0.0), 1e-6,
                        [("LayerID", "short", layer), ("GrainID", "int", ids),
                         ("Temperature", "float", layer)], False)
        output = os.path.join(self.dir, "large_ascii.h5")
        start = time.time()
        result = run_script("exaca_to_voxels.py", [vtk, output, "--orientations", self.orient,
                                                   "--conjugate", "no"])
        elapsed = time.time() - start
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(elapsed, 30.0)
        voxels = msio.read_voxel_map(output)
        self.assertTrue(np.array_equal(voxels["grain_ids"], np.where(ids < 0, 199 - ids, ids)))

    def test_required_flags_and_negative_id_error(self):
        vtk = os.path.join(self.dir, "grains_flags.vtk")
        write_exaca_vtk(vtk, self.dims, self.origin, self.spacing, self.fields, False)
        out = os.path.join(self.dir, "x.h5")
        # --orientations without --conjugate is a usage error
        result = run_script("exaca_to_voxels.py", [vtk, out, "--orientations", self.orient])
        self.assertEqual(result.returncode, 2)
        self.assertIn("--conjugate", result.stderr)
        # --negative-ids error refuses nucleated grains
        result = run_script("exaca_to_voxels.py", [vtk, out, "--negative-ids", "error"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("negative ids", result.stderr)
        # A missing field lists the fields that were seen
        result = run_script("exaca_to_voxels.py", [vtk, out, "--field", "Nope"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("LayerID, GrainID, Temperature", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
