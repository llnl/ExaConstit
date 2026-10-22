"""Comparisons for the [Mesh.Microstructure] end-to-end tests (test/CMakeLists.txt).

Two checks, used where the hook and the run it is compared with label meshes that are
equivalent but not bit-identical (the hook works in place on ExaConstit's mesh with Nodes; the
tool reads the mesh without them, or ExaConstit re-projects curved Nodes; or a different element
order or refinement), so byte-identical record files are not expected:

  seeds TABLE_A TABLE_B
      The two grain tables (<basename>_grains.csv, written by the hook and the tool) must come
      from the same seed set. Every grain carries the source id of its seed and the orientation
      keyed by that seed, so for every source id present in both tables the quaternion (qw, qx,
      qy, qz) must be identical, digit for digit. At least one source id must be shared. Grains
      present in only one table are allowed (a finer mesh can resolve a grain, or a fragment kept
      by the cleanup, that a coarser one misses). Prints the number of shared grains.

  labels LABELS_A LABELS_B MIN_FRACTION [COMMAND ...]
      Two labelings of the same mesh with the same element numbering must give the same grain id
      to at least MIN_FRACTION of the elements. Prints the measured fraction. Each LABELS is
      either an MFEM mesh file whose element attributes are the grain ids (the hook's or the
      tool's Output.mesh_format = "parallel" part, written as plain text by a 1-rank run, so the
      elements are in serial order), several such files as a comma-separated list, or a grain
      file with one grain id per line (the tool's <basename>.grains for an auto mesh, in serial
      element order). Grain ids are compared as numbers: both writers number grains in the same
      seed order, so equal ids mean the same grain when the seed sets are equal (check "seeds").
      If COMMAND is given, it is run only when ALL labels agree (e.g. a --compare-mechanics run,
      whose results can only be expected to match for identical microstructures) and its exit
      status decides; otherwise the comparison is reported as skipped.

Exit status 0 on success, 1 otherwise. Plain Python 3, no third-party modules.
"""

import csv
import subprocess
import sys


def read_grain_table(path):
    """Map source id -> (qw, qx, qy, qz) as the exact strings written in the table."""
    rows = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["source_id"]] = (row["qw"], row["qx"], row["qy"], row["qz"])
    return rows


def check_seeds(table_a, table_b):
    a = read_grain_table(table_a)
    b = read_grain_table(table_b)
    shared = sorted(set(a) & set(b), key=int)
    if not shared:
        print("no source id is shared by the two grain tables")
        return 1
    different = [s for s in shared if a[s] != b[s]]
    print("%d grains share a seed (%d and %d grains in the tables)" % (len(shared), len(a), len(b)))
    if different:
        print("orientations differ for source ids: " + ", ".join(different))
        return 1
    return 0


def read_labels(paths):
    """Grain ids in element order: element attributes of MFEM mesh files ('elements' section),
    or the numbers of a grain file (one per line)."""
    labels = []
    for path in paths:
        with open(path) as handle:
            lines = handle.read().splitlines()
        if "elements" in lines:
            start = lines.index("elements")
            count = int(lines[start + 1])
            for line in lines[start + 2:start + 2 + count]:
                labels.append(int(line.split()[0]))
        else:
            labels.extend(int(float(line)) for line in lines if line.strip())
    return labels


def check_labels(labels_a, labels_b, min_fraction, command):
    a = read_labels(labels_a.split(","))
    b = read_labels(labels_b.split(","))
    if len(a) != len(b) or not a:
        print("the labelings have %d and %d elements" % (len(a), len(b)))
        return 1
    same = sum(1 for x, y in zip(a, b) if x == y)
    fraction = same / len(a)
    print("%d of %d elements (%.4f) have the same grain id" % (same, len(a), fraction))
    if fraction < min_fraction:
        return 1
    if command:
        if same != len(a):
            print("labels are not identical: skipped " + " ".join(command))
            return 0
        print("labels identical: running " + " ".join(command))
        sys.stdout.flush()
        return subprocess.call(command)
    return 0


def main(argv):
    if len(argv) == 4 and argv[1] == "seeds":
        return check_seeds(argv[2], argv[3])
    if len(argv) >= 5 and argv[1] == "labels":
        return check_labels(argv[2], argv[3], float(argv[4]), argv[5:])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
