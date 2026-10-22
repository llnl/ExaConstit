#include "microstructure/io/seed_table_io.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/seeding/roi_filler.hpp"
#include "microstructure/seeding/seed_generators.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <fstream>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * Tests for seed tables (ReadSeedTable / WriteSeedTable / KeepSeedsInBox), the exclusion box and
 * id offset of the seed generator, and the ROI filler (FillOutsideRoi).
 *
 * Every test runs on all ranks. Collective results are compared bitwise with a reference that each
 * rank computes alone (MPI_COMM_SELF) on the whole serial mesh, which is the np = 1 answer, so
 * running the executable at np = 1..4 checks rank-count independence.
 *
 * Expected numbers are built from the same decimal strings as the files with std::strtod (the
 * function the reader uses), so "bitwise" really means the reader adds no rounding of its own.
 */

namespace ms = exaconstit::microstructure;
namespace fs = std::filesystem;

namespace {

const fs::path data_dir = fs::path(MS_SEED_TABLE_DATA_DIR);
constexpr double PI = 3.141592653589793238462643383279502884;

/** The double that strtod reads from `text` (the reader's own conversion). */
double D(const char* text) {
    return std::strtod(text, nullptr);
}

int WorldRank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int WorldSize() {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

/**
 * Writes `content` to a file in the working directory whose name includes the rank count (so the
 * np variants of this test never touch the same file), on rank 0 only, then waits for it on all
 * ranks. Collective.
 */
fs::path WriteText(const std::string& name, const std::string& content) {
    const fs::path path = "seed_table_np" + std::to_string(WorldSize()) + "_" + name + ".csv";
    if (WorldRank() == 0) {
        std::ofstream out(path);
        out << content;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return path;
}

ms::SeedTable Read(const fs::path& path, double scale = 1.0, std::array<double, 3> shift = {}) {
    ms::SeedTableParameters params;
    params.path = path;
    params.units_scale = scale;
    params.translate = shift;
    return ms::ReadSeedTable(params, MPI_COMM_WORLD);
}

/**
 * Reads `path` and expects every rank to throw std::runtime_error with the same message, which
 * must contain every string of `parts` (e.g. ":3:" for line 3). Collective.
 */
void ExpectCollectiveError(const fs::path& path, const std::vector<std::string>& parts) {
    std::string message;
    bool threw = false;
    try {
        Read(path);
    } catch (const std::runtime_error& error) {
        threw = true;
        message = error.what();
    }
    // Every rank threw ...
    int all_threw = threw ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &all_threw, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    EXPECT_EQ(all_threw, 1) << path;
    // ... the same message
    const std::string root_message = ms::BroadcastString(message, 0, MPI_COMM_WORLD);
    EXPECT_EQ(message, root_message);
    for (const std::string& part : parts) {
        EXPECT_NE(message.find(part), std::string::npos)
            << "message '" << message << "' lacks '" << part << "'";
    }
}

/** Bitwise equality of every array of two tables. */
void ExpectSameTable(const ms::SeedTable& a, const ms::SeedTable& b) {
    EXPECT_EQ(a.seeds.x, b.seeds.x);
    EXPECT_EQ(a.seeds.y, b.seeds.y);
    EXPECT_EQ(a.seeds.z, b.seeds.z);
    EXPECT_EQ(a.seeds.source_id, b.seeds.source_id);
    EXPECT_EQ(a.seeds.quat, b.seeds.quat);
    EXPECT_EQ(a.seeds.phase, b.seeds.phase);
    EXPECT_EQ(a.seeds.weight, b.seeds.weight);
    EXPECT_EQ(a.seeds.shape, b.seeds.shape);
    EXPECT_EQ(a.volume, b.volume);
    EXPECT_EQ(a.shape_tensor, b.shape_tensor);
    EXPECT_EQ(a.has_quat, b.has_quat);
    EXPECT_EQ(a.has_phase, b.has_phase);
    EXPECT_EQ(a.has_weight, b.has_weight);
    EXPECT_EQ(a.has_volume, b.has_volume);
    EXPECT_EQ(a.has_shape, b.has_shape);
}

/** Distributes `serial` over MPI_COMM_WORLD (collective). */
std::unique_ptr<mfem::ParMesh> Distribute(mfem::Mesh& serial) {
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    return ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
}

/** Bitwise equality of positions and ids. */
void ExpectSameSeeds(const ms::SeedSet& a, const ms::SeedSet& b) {
    EXPECT_EQ(a.x, b.x);
    EXPECT_EQ(a.y, b.y);
    EXPECT_EQ(a.z, b.z);
    EXPECT_EQ(a.source_id, b.source_id);
}

bool StrictlyInside(double x, double y, double z, const std::array<double, 3>& lo,
                    const std::array<double, 3>& hi) {
    return x > lo[0] && x < hi[0] && y > lo[1] && y < hi[1] && z > lo[2] && z < hi[2];
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Parsing the checked-in tables
// ------------------------------------------------------------------------------------------------

// Only x, y, z: positions copied, ids default to the 0-based row index, every optional array empty.
TEST(SeedTable, MinimalXyz) {
    const ms::SeedTable t = Read(data_dir / "minimal_xyz.csv");
    ASSERT_EQ(t.seeds.Size(), 3u);
    EXPECT_EQ(t.seeds.x, (std::vector<double>{0.25, 0.75, 0.5}));
    EXPECT_EQ(t.seeds.y, (std::vector<double>{0.25, 0.25, 0.75}));
    EXPECT_EQ(t.seeds.z, (std::vector<double>{0.25, 0.25, 0.75}));
    EXPECT_EQ(t.seeds.source_id, (std::vector<std::int64_t>{0, 1, 2}));
    EXPECT_FALSE(t.has_quat || t.has_phase || t.has_weight || t.has_volume || t.has_shape);
    EXPECT_TRUE(t.seeds.quat.empty() && t.seeds.phase.empty() && t.seeds.weight.empty() &&
                t.seeds.shape.empty() && t.volume.empty() && t.shape_tensor.empty());
    EXPECT_TRUE(t.ignored_columns.empty());
}

// Every column, shuffled. Quaternions must come out in w x y z order, bitwise equal to the file
// text, including a non-unit one (|q| = 1.0004, inside the 1e-3 tolerance, NOT normalized) and a
// negative qw (NOT sign-flipped). Radii become volumes 4/3 pi r^3; the shape tensor is stored as M
// and as A = M^-1 (checked through A M = I).
TEST(SeedTable, AllColumns) {
    const ms::SeedTable t = Read(data_dir / "all_columns.csv");
    ASSERT_EQ(t.seeds.Size(), 3u);
    EXPECT_TRUE(t.has_quat && t.has_phase && t.has_weight && t.has_volume && t.has_shape);
    EXPECT_EQ(t.seeds.source_id, (std::vector<std::int64_t>{10, 3, 7}));
    EXPECT_EQ(t.seeds.x, (std::vector<double>{D("0.1"), D("0.6"), D("0.4")}));
    EXPECT_EQ(t.seeds.z, (std::vector<double>{D("0.3"), D("0.8"), D("0.9")}));
    const std::vector<double> quat = {D("-0.5"), D("0.5"), D("0.5"), D("0.5"),
                                      D("1.0004"), 0.0, 0.0, 0.0,
                                      D("0.70710678118654757"), 0.0, 0.0,
                                      D("0.70710678118654757")};
    EXPECT_EQ(t.seeds.quat, quat);
    EXPECT_EQ(t.seeds.phase, (std::vector<int>{1, 0, 2}));
    EXPECT_EQ(t.seeds.weight, (std::vector<double>{D("0.25"), D("-0.125"), 0.0}));
    const double radii[3] = {D("0.05"), D("0.1"), D("0.2")};
    for (int i = 0; i < 3; i++) {
        const double r = radii[i];
        EXPECT_EQ(t.volume[i], 4.0 / 3.0 * PI * r * r * r);
    }
    // A M = I for every seed (M symmetric, stored m11 m22 m33 m23 m13 m12)
    for (std::size_t i = 0; i < 3; i++) {
        const double* m = &t.shape_tensor[6 * i];
        const double* a = &t.seeds.shape[6 * i];
        const double M[3][3] = {{m[0], m[5], m[4]}, {m[5], m[1], m[3]}, {m[4], m[3], m[2]}};
        const double A[3][3] = {{a[0], a[5], a[4]}, {a[5], a[1], a[3]}, {a[4], a[3], a[2]}};
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                double sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += A[r][k] * M[k][c];
                }
                EXPECT_NEAR(sum, r == c ? 1.0 : 0.0, 1e-14) << "seed " << i;
            }
        }
    }
    // Diagonal M = diag(4, 9, 1) inverts to diag(1/4, 1/9, 1) (to round-off: M is first divided
    // by its largest diagonal entry 9)
    EXPECT_DOUBLE_EQ(t.seeds.shape[0], 1.0 / 4.0);
    EXPECT_DOUBLE_EQ(t.seeds.shape[1], 1.0 / 9.0);
    EXPECT_DOUBLE_EQ(t.seeds.shape[2], 1.0);
    EXPECT_EQ(t.seeds.shape[3], 0.0);
}

// Comment lines (also indented and between rows), blank lines, spaces and tabs around fields and
// names are all ignored.
TEST(SeedTable, CommentsAndWhitespace) {
    const ms::SeedTable t = Read(data_dir / "comments_whitespace.csv");
    ASSERT_EQ(t.seeds.Size(), 3u);
    EXPECT_EQ(t.seeds.x, (std::vector<double>{D("0.1"), D("0.4"), D("0.7")}));
    EXPECT_EQ(t.seeds.y, (std::vector<double>{D("0.2"), D("0.5"), D("0.8")}));
    EXPECT_EQ(t.seeds.z, (std::vector<double>{D("0.3"), D("0.6"), D("0.9")}));
    EXPECT_EQ(t.seeds.source_id, (std::vector<std::int64_t>{5, 6, 7}));
}

// Column names are case-insensitive: "X", " Y ", "QW", "VOLUME", ... are recognized.
TEST(SeedTable, MixedCaseHeader) {
    const ms::SeedTable t = Read(data_dir / "mixed_case_header.csv");
    ASSERT_EQ(t.seeds.Size(), 2u);
    EXPECT_TRUE(t.has_quat && t.has_phase && t.has_volume);
    EXPECT_FALSE(t.has_weight || t.has_shape);
    EXPECT_EQ(t.seeds.quat, (std::vector<double>{1, 0, 0, 0, 0, 1, 0, 0}));
    EXPECT_EQ(t.seeds.phase, (std::vector<int>{0, 3}));
    EXPECT_EQ(t.volume, (std::vector<double>{D("0.001"), D("0.002")}));
}

// units_scale s and translate t: positions s x + t, volumes s^3 (here from radii, (s r)^3),
// shape tensors s^2 M (so A is divided by s^2), weights and quaternions untouched. s = 2 makes
// every scaling exact, so the comparisons are bitwise.
TEST(SeedTable, UnitsScaleAndTranslate) {
    const ms::SeedTable base = Read(data_dir / "all_columns.csv");
    const std::array<double, 3> shift = {1.0, -2.0, 0.5};
    const ms::SeedTable t = Read(data_dir / "all_columns.csv", 2.0, shift);
    ASSERT_EQ(t.seeds.Size(), base.seeds.Size());
    for (std::size_t i = 0; i < t.seeds.Size(); i++) {
        EXPECT_EQ(t.seeds.x[i], 2.0 * base.seeds.x[i] + shift[0]);
        EXPECT_EQ(t.seeds.y[i], 2.0 * base.seeds.y[i] + shift[1]);
        EXPECT_EQ(t.seeds.z[i], 2.0 * base.seeds.z[i] + shift[2]);
        EXPECT_EQ(t.volume[i], 8.0 * base.volume[i]);
        for (int k = 0; k < 6; k++) {
            EXPECT_EQ(t.shape_tensor[6 * i + k], 4.0 * base.shape_tensor[6 * i + k]);
            EXPECT_EQ(t.seeds.shape[6 * i + k], base.seeds.shape[6 * i + k] / 4.0);
        }
    }
    EXPECT_EQ(t.seeds.weight, base.seeds.weight);
    EXPECT_EQ(t.seeds.quat, base.seeds.quat);

    // A volume column scales with s^3, a diameter d gives 4/3 pi (s d / 2)^3 (s = 0.5: exact)
    const fs::path vol = WriteText("volume", "x,y,z,volume\n0,0,0,0.25\n");
    EXPECT_EQ(Read(vol, 0.5).volume[0], 0.125 * D("0.25"));
    const fs::path dia = WriteText("diameter", "x,y,z,diameter\n0,0,0,0.3\n");
    const double r = 0.5 * (0.5 * D("0.3"));
    EXPECT_EQ(Read(dia, 0.5).volume[0], 4.0 / 3.0 * PI * r * r * r);

    // Invalid parameters are refused on every rank
    EXPECT_THROW(Read(data_dir / "minimal_xyz.csv", 0.0), std::invalid_argument);
}

// Unknown columns are ignored (and listed); duplicate positions are allowed.
TEST(SeedTable, IgnoredColumnsAndDuplicatePositions) {
    const fs::path p =
        WriteText("ignored", "x,y,z,confidence,Grain_Name\n0,0,0,0.9,a\n0,0,0,0.8,b\n");
    const ms::SeedTable t = Read(p);
    EXPECT_EQ(t.seeds.Size(), 2u);
    EXPECT_EQ(t.ignored_columns, (std::vector<std::string>{"confidence", "grain_name"}));
}

// ------------------------------------------------------------------------------------------------
// One error per rule: thrown on every rank with the same message naming the line
// ------------------------------------------------------------------------------------------------

// Header errors are reported on the header line (line 2, after a comment line); row errors on
// the row's own line, counting comments and blank lines.
TEST(SeedTable, Errors) {
    ExpectCollectiveError(WriteText("missing", "# c\nx,z\n0,0\n"), {":2:", "missing", "'y'"});
    ExpectCollectiveError(WriteText("partial_quat", "# c\nx,y,z,qw,qx,qy\n0,0,0,1,0,0\n"),
                          {":2:", "incomplete quaternion", "qz"});
    ExpectCollectiveError(WriteText("quat_norm", "# c\nx,y,z,qw,qx,qy,qz\n0,0,0,1,0,0,0\n"
                                                 "1,1,1,0.7,0.7,0.1,0\n"),
                          {":4:", "norm"});
    ExpectCollectiveError(WriteText("euler", "# c\nx,y,z,phi1,Phi,phi2\n0,0,0,1,2,3\n"),
                          {":2:", "'phi1'", "qw,qx,qy,qz"});
    ExpectCollectiveError(WriteText("euler2", "x,y,z,euler_1\n0,0,0,1\n"), {":1:", "euler_1"});
    ExpectCollectiveError(WriteText("rotmat", "x,y,z,r12\n0,0,0,1\n"), {":1:", "'r12'"});
    ExpectCollectiveError(WriteText("non_spd", "# c\nx,y,z,m11,m22,m33,m23,m13,m12\n"
                                               "0,0,0,1,1,1,0,0,0\n\n0,0,0,1,1,1,0,0,2\n"),
                          {":5:", "positive definite"});
    ExpectCollectiveError(WriteText("dup_id", "# c\nid,x,y,z\n4,0,0,0\n2,1,0,0\n4,0,1,0\n"),
                          {":5:", "duplicate id 4", "line 3"});
    ExpectCollectiveError(WriteText("two_sizes", "# c\nx,y,z,radius,volume\n0,0,0,1,1\n"),
                          {":2:", "size column"});
    ExpectCollectiveError(WriteText("non_numeric", "# c\nx,y,z\n0,0,0\n0,abc,0\n"),
                          {":4:", "'y'", "'abc'"});
    ExpectCollectiveError(WriteText("field_count", "x,y,z\n0,0,0\n0,0\n"), {":3:", "fields"});
    ExpectCollectiveError(WriteText("neg_radius", "x,y,z,radius\n0,0,0,-1\n"),
                          {":2:", "'radius'", "> 0"});
    ExpectCollectiveError(WriteText("neg_id", "id,x,y,z\n-1,0,0,0\n"), {":2:", ">= 0"});
    ExpectCollectiveError(WriteText("empty", "# only a comment\n"), {"no header"});
    ExpectCollectiveError(WriteText("no_rows", "x,y,z\n"), {"no seed rows"});
    ExpectCollectiveError("no_such_seed_table.csv", {"cannot open"});
}

// Input-robustness cases from review R8: Windows line endings and a UTF-8 byte-order mark are
// accepted; non-finite values (inf, nan, 1e400 overflows to inf) and hexadecimal floats are
// refused with the field text in the message; ids up to INT64_MAX are accepted, INT64_MAX + 1
// is not an int64; ambiguous quaternion columns q0..q3 and quat* are refused.
TEST(SeedTable, InputRobustness) {
    // CRLF: the trailing carriage return is whitespace and is trimmed with the field
    const ms::SeedTable crlf = Read(WriteText("crlf", "# c\r\nx,y,z,id\r\n0.5,0.25,1,3\r\n"));
    ASSERT_EQ(crlf.seeds.Size(), 1u);
    EXPECT_EQ(crlf.seeds.x[0], 0.5);
    EXPECT_EQ(crlf.seeds.source_id[0], 3);

    // UTF-8 BOM before the header (Excel "CSV UTF-8"), and before a leading comment
    const ms::SeedTable bom = Read(WriteText("bom", "\xEF\xBB\xBFx,y,z\n1,2,3\n"));
    ASSERT_EQ(bom.seeds.Size(), 1u);
    EXPECT_EQ(bom.seeds.z[0], 3.0);
    EXPECT_EQ(Read(WriteText("bom_comment", "\xEF\xBB\xBF# c\nX,y,z\n1,2,3\n")).seeds.Size(),
              1u);

    // Non-finite values and hexadecimal floats
    ExpectCollectiveError(WriteText("inf", "x,y,z\n0,inf,0\n"), {":2:", "'y'", "'inf'", "finite"});
    ExpectCollectiveError(WriteText("nan", "x,y,z\n0,0,nan\n"), {":2:", "'z'", "'nan'"});
    ExpectCollectiveError(WriteText("huge", "x,y,z\n1e400,0,0\n"), {":2:", "'x'", "'1e400'"});
    ExpectCollectiveError(WriteText("hex", "x,y,z\n0x1p3,0,0\n"),
                          {":2:", "'0x1p3'", "not a number"});

    // Id range: INT64_MAX = 9223372036854775807 fits, one more does not
    const ms::SeedTable big = Read(WriteText("id_max", "id,x,y,z\n9223372036854775807,0,0,0\n"));
    EXPECT_EQ(big.seeds.source_id[0], std::numeric_limits<std::int64_t>::max());
    ExpectCollectiveError(WriteText("id_over", "id,x,y,z\n9223372036854775808,0,0,0\n"),
                          {":2:", "'id'", "not an integer"});

    // Ambiguous quaternion columns
    ExpectCollectiveError(WriteText("q0", "x,y,z,q0,q1,q2,q3\n0,0,0,1,0,0,0\n"),
                          {":1:", "'q0'", "qw,qx,qy,qz"});
    ExpectCollectiveError(WriteText("quat", "x,y,z,quat_w\n0,0,0,1\n"), {":1:", "'quat_w'"});
}

// A shape tensor in very small units (m11 = 1e-200: det M = 1e-600 would underflow to 0) is
// still recognized as SPD, because M is divided by its largest diagonal entry before the test;
// its inverse is 1e200 times that of the unit tensor.
TEST(SeedTable, TinyShapeTensor) {
    const ms::SeedTable t = Read(WriteText(
        "tiny_shape", "x,y,z,m11,m22,m33,m23,m13,m12\n0,0,0,1e-200,2e-200,4e-200,0,0,0\n"));
    ASSERT_EQ(t.seeds.shape.size(), 6u);
    EXPECT_DOUBLE_EQ(t.seeds.shape[0], 1e200);
    EXPECT_DOUBLE_EQ(t.seeds.shape[1], 0.5e200);
    EXPECT_DOUBLE_EQ(t.seeds.shape[2], 0.25e200);
}

// ------------------------------------------------------------------------------------------------
// Write / read round trip and KeepSeedsInBox
// ------------------------------------------------------------------------------------------------

// WriteSeedTable uses 17 significant digits and writes M (not A), so reading the file back with
// units_scale = 1 reproduces every array bit for bit, also after a non-trivial unit conversion.
TEST(SeedTable, WriteReadRoundTrip) {
    for (const char* name : {"all_columns.csv", "minimal_xyz.csv", "mixed_case_header.csv"}) {
        const ms::SeedTable original = Read(data_dir / name, 1.0e-3, {0.1, 0.2, 0.3});
        const fs::path path = "seed_table_np" + std::to_string(WorldSize()) + "_roundtrip.csv";
        if (WorldRank() == 0) {
            ms::WriteSeedTable(path, original);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const ms::SeedTable again = Read(path);
        ExpectSameTable(original, again);
        MPI_Barrier(MPI_COMM_WORLD); // before rank 0 overwrites the file in the next pass
    }
}

// The box [0, 0.5] x [0, 1] x [0, 1] keeps the seeds at x = 0.1 and 0.4 (ids 10 and 7) and drops
// the one at x = 0.6; every array is compacted in the same way.
TEST(SeedTable, KeepSeedsInBox) {
    ms::SeedTable t = Read(data_dir / "all_columns.csv");
    const ms::SeedTable full = t;
    EXPECT_EQ(ms::KeepSeedsInBox(t, {0.0, 0.0, 0.0}, {0.5, 1.0, 1.0}), 1);
    ASSERT_EQ(t.seeds.Size(), 2u);
    EXPECT_EQ(t.seeds.source_id, (std::vector<std::int64_t>{10, 7}));
    EXPECT_EQ(t.seeds.phase, (std::vector<int>{1, 2}));
    EXPECT_EQ(t.volume, (std::vector<double>{full.volume[0], full.volume[2]}));
    for (int k = 0; k < 4; k++) {
        EXPECT_EQ(t.seeds.quat[4 + k], full.seeds.quat[8 + k]);
    }
    for (int k = 0; k < 6; k++) {
        EXPECT_EQ(t.seeds.shape[6 + k], full.seeds.shape[12 + k]);
        EXPECT_EQ(t.shape_tensor[6 + k], full.shape_tensor[12 + k]);
    }
    // A box containing every seed drops nothing; an empty-overlap box drops all
    ms::SeedTable u = full;
    EXPECT_EQ(ms::KeepSeedsInBox(u, {0, 0, 0}, {1, 1, 1}), 0);
    EXPECT_EQ(ms::KeepSeedsInBox(u, {5, 5, 5}, {6, 6, 6}), 3);
    EXPECT_EQ(u.seeds.Size(), 0u);
    EXPECT_TRUE(u.seeds.quat.empty() && u.volume.empty() && u.shape_tensor.empty());
}

// ------------------------------------------------------------------------------------------------
// Exclusion box and id offset of the seed generator
// ------------------------------------------------------------------------------------------------

// The new fields change nothing by default: an exclusion box that misses the mesh and a zero
// offset give bitwise the seeds of the default run; a nonzero offset shifts every id and nothing
// else.
TEST(SeedGeneratorExtensions, DefaultsUnchangedAndOffset) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::TETRAHEDRON);
    auto pmesh = Distribute(serial);
    ms::SeedingParameters params;
    params.num_seeds = 50;
    params.rng_seed = 17;
    params.min_distance_factor = 0.3;
    const ms::SeedSet reference = ms::GenerateSeeds(*pmesh, params);

    ms::SeedingParameters missing_box = params;
    missing_box.use_exclude_box = true;
    missing_box.exclude_box_lo = {5.0, 5.0, 5.0};
    missing_box.exclude_box_hi = {6.0, 6.0, 6.0};
    ExpectSameSeeds(ms::GenerateSeeds(*pmesh, missing_box), reference);

    ms::SeedingParameters offset = params;
    offset.source_id_offset = 1000;
    const ms::SeedSet shifted = ms::GenerateSeeds(*pmesh, offset);
    EXPECT_EQ(shifted.x, reference.x);
    ASSERT_EQ(shifted.source_id.size(), reference.source_id.size());
    for (std::size_t i = 0; i < shifted.Size(); i++) {
        EXPECT_EQ(shifted.source_id[i], reference.source_id[i] + 1000);
    }

    ms::SeedingParameters bad = params;
    bad.source_id_offset = -1;
    EXPECT_THROW(ms::ValidateSeedingParameters(bad), std::invalid_argument);
    bad = missing_box;
    bad.exclude_box_hi[1] = bad.exclude_box_lo[1];
    EXPECT_THROW(ms::ValidateSeedingParameters(bad), std::invalid_argument);
}

// Every placement honors the box: no seed strictly inside it, for rejection, element-weighted
// sampling and lattices, and the distributed run equals the single-rank run bitwise.
TEST(SeedGeneratorExtensions, ExclusionBoxAllPlacements) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial);
    const std::array<double, 3> lo = {0.2, 0.3, -1.0};
    const std::array<double, 3> hi = {0.7, 0.8, 2.0}; // a column through the cube
    for (const ms::SeedPlacement placement :
         {ms::SeedPlacement::REJECTION, ms::SeedPlacement::ELEMENT_WEIGHTED,
          ms::SeedPlacement::LATTICE}) {
        ms::SeedingParameters params;
        params.placement = placement;
        params.num_seeds = 64;
        params.rng_seed = 3;
        params.lattice_jitter = 0.2;
        params.use_exclude_box = true;
        params.exclude_box_lo = lo;
        params.exclude_box_hi = hi;
        const ms::SeedSet reference = ms::GenerateSeeds(serial, MPI_COMM_SELF, params);
        ms::SeedingReport report;
        const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params, &report);
        ExpectSameSeeds(seeds, reference);
        EXPECT_GT(seeds.Size(), 0u);
        EXPECT_GT(report.rejected_by_exclude_box, 0);
        for (std::size_t i = 0; i < seeds.Size(); i++) {
            EXPECT_FALSE(StrictlyInside(seeds.x[i], seeds.y[i], seeds.z[i], lo, hi));
        }
        if (placement != ms::SeedPlacement::LATTICE) {
            EXPECT_EQ(static_cast<std::int64_t>(seeds.Size()), params.num_seeds);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// ROI filler
// ------------------------------------------------------------------------------------------------

// Unit cube of 6^3 hexes with the ROI [0, 0.5]^3 in one corner. The ROI faces lie on element
// faces, so every quadrature point is strictly on one side and V_roi = 1/8, V_outside = 7/8 (up
// to the fixed-point resolution). 20 ROI seeds then ask for round(20 * 7) = 140 filler seeds,
// none strictly inside the ROI, ids from the offset on, identical to the single-rank run.
TEST(RoiFiller, CornerRoiAutoCount) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial);
    ms::RoiFillerParameters params;
    params.lo = {0.0, 0.0, 0.0};
    params.hi = {0.5, 0.5, 0.5};
    params.rng_seed = 99;
    params.source_id_offset = 1000;
    params.min_distance_factor = 0.3;

    const ms::RoiFillerResult reference =
        ms::FillOutsideRoi(serial, MPI_COMM_SELF, params, 20);
    const ms::RoiFillerResult result = ms::FillOutsideRoi(*pmesh, MPI_COMM_WORLD, params, 20);

    EXPECT_NEAR(result.roi_material_volume, 0.125, 1e-12);
    EXPECT_NEAR(result.outside_material_volume, 0.875, 1e-12);
    EXPECT_EQ(result.roi_material_volume, reference.roi_material_volume);
    EXPECT_EQ(result.outside_material_volume, reference.outside_material_volume);
    EXPECT_EQ(result.num_requested, 140);
    ASSERT_EQ(result.filler.Size(), 140u);
    ExpectSameSeeds(result.filler, reference.filler);
    for (std::size_t i = 0; i < result.filler.Size(); i++) {
        const ms::SeedSet& f = result.filler;
        EXPECT_FALSE(StrictlyInside(f.x[i], f.y[i], f.z[i], params.lo, params.hi));
        EXPECT_GE(f.source_id[i], params.source_id_offset);
        if (i > 0) {
            EXPECT_LT(f.source_id[i - 1], f.source_id[i]);
        }
    }

    // The filler ids are exactly the generator's candidate indices shifted by the offset
    ms::SeedingParameters plain;
    plain.num_seeds = 140;
    plain.rng_seed = 99;
    plain.use_exclude_box = true;
    plain.exclude_box_lo = params.lo;
    plain.exclude_box_hi = params.hi;
    plain.min_distance = 0.3 * std::cbrt(result.outside_material_volume / 140.0);
    const ms::SeedSet unshifted = ms::GenerateSeeds(*pmesh, plain);
    ASSERT_EQ(unshifted.Size(), 140u);
    for (std::size_t i = 0; i < unshifted.Size(); i++) {
        EXPECT_EQ(result.filler.source_id[i], unshifted.source_id[i] + 1000);
        EXPECT_EQ(result.filler.x[i], unshifted.x[i]);
    }
}

// A tet mesh with an ROI that cuts elements: volumes and seeds still identical to the single-rank
// run, V_roi + V_outside equals the cube volume, an explicit count is honored, and an ROI covering
// the whole mesh needs no filler.
TEST(RoiFiller, CutRoiTetsExplicitCountAndFullCover) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(5, 5, 5, mfem::Element::TETRAHEDRON);
    auto pmesh = Distribute(serial);
    ms::RoiFillerParameters params;
    params.lo = {0.33, 0.21, 0.1};
    params.hi = {0.87, 0.74, 0.66};
    params.num_filler = 37;
    params.rng_seed = 5;
    params.source_id_offset = 12;

    const ms::RoiFillerResult reference = ms::FillOutsideRoi(serial, MPI_COMM_SELF, params, 0);
    const ms::RoiFillerResult result = ms::FillOutsideRoi(*pmesh, MPI_COMM_WORLD, params, 0);
    EXPECT_EQ(result.roi_material_volume, reference.roi_material_volume);
    EXPECT_EQ(result.outside_material_volume, reference.outside_material_volume);
    EXPECT_NEAR(result.roi_material_volume + result.outside_material_volume, 1.0, 1e-12);
    EXPECT_GT(result.roi_material_volume, 0.0);
    EXPECT_EQ(result.num_requested, 37);
    ASSERT_EQ(result.filler.Size(), 37u);
    ExpectSameSeeds(result.filler, reference.filler);
    for (std::size_t i = 0; i < result.filler.Size(); i++) {
        const ms::SeedSet& f = result.filler;
        EXPECT_FALSE(StrictlyInside(f.x[i], f.y[i], f.z[i], params.lo, params.hi));
    }

    // ROI box enclosing the whole cube: V_outside = 0, so the automatic count is 0
    ms::RoiFillerParameters cover;
    cover.lo = {-1.0, -1.0, -1.0};
    cover.hi = {2.0, 2.0, 2.0};
    const ms::RoiFillerResult none = ms::FillOutsideRoi(*pmesh, MPI_COMM_WORLD, cover, 10);
    EXPECT_EQ(none.outside_material_volume, 0.0);
    EXPECT_EQ(none.num_requested, 0);
    EXPECT_EQ(none.filler.Size(), 0u);

    // An id offset that leaves no room for the generator's candidate indices (< 2^40) is refused
    ms::RoiFillerParameters no_room = params;
    no_room.source_id_offset = std::numeric_limits<std::int64_t>::max() - 5;
    EXPECT_THROW(ms::FillOutsideRoi(*pmesh, MPI_COMM_WORLD, no_room, 0), std::invalid_argument);

    // An ROI outside the mesh cannot set the automatic count
    ms::RoiFillerParameters outside;
    outside.lo = {5.0, 5.0, 5.0};
    outside.hi = {6.0, 6.0, 6.0};
    EXPECT_THROW(ms::FillOutsideRoi(*pmesh, MPI_COMM_WORLD, outside, 10), std::invalid_argument);
}

// Thin outside shell: the exclusion box [0.05, 0.95]^3 leaves 1 - 0.9^3 = 0.271 of the unit cube
// outside it (the fraction the ROI filler would pass). With a small stall cap the run must stop,
// and the message must name the exclusion box and the outside volume instead of only suggesting a
// smaller min_distance; every rank throws the same message.
TEST(SeedGeneratorExtensions, StallMessageNamesExclusionBox) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(10, 10, 10, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial);
    ms::SeedingParameters params;
    params.num_seeds = 5000;
    params.rng_seed = 1;
    params.use_exclude_box = true;
    params.exclude_box_lo = {0.05, 0.05, 0.05};
    params.exclude_box_hi = {0.95, 0.95, 0.95};
    params.exclude_box_outside_fraction = 0.271;
    params.max_candidates = 10000; // ~2710 of them land outside the box: fewer than 5000
    std::string message;
    try {
        ms::GenerateSeeds(*pmesh, params);
    } catch (const std::runtime_error& error) {
        message = error.what();
    }
    ASSERT_FALSE(message.empty()) << "expected a stall";
    EXPECT_EQ(message, ms::BroadcastString(message, 0, MPI_COMM_WORLD));
    EXPECT_NE(message.find("by the exclusion box"), std::string::npos) << message;
    EXPECT_NE(message.find("fraction 0.271"), std::string::npos) << message;
    EXPECT_NE(message.find("outside volume"), std::string::npos) << message;
    EXPECT_NE(message.find("only the material outside the exclusion box"), std::string::npos)
        << message;

    // The fraction is validated
    params.exclude_box_outside_fraction = 0.0;
    EXPECT_THROW(ms::ValidateSeedingParameters(params), std::invalid_argument);
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
