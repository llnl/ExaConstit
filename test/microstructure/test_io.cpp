/**
 * @file test_io.cpp
 * @brief Tests of the microstructure output writer (src/microstructure/io/output_writer.*).
 *
 * The central check throughout is that every file reads back with the same code ExaConstit uses
 * to read it (mfem::Mesh / ParMesh for meshes, `while (f >> v)` for orientations, getline +
 * `>>` for the region map, mfem::Vector::Load for the grain file), and
 * that it reads back exactly: attributes and coordinates compared with ==, not a tolerance.
 *
 * The executable runs at np = 1..4 (test_io.cmake). Tests that write per-rank files use a
 * directory private to each rank; tests of collective writers use one directory shared by all
 * ranks of the run. Output goes to test_io_output/np<N>/ under the ctest working directory, so the
 * different np variants never overwrite each other when ctest runs them in parallel.
 */

#include "microstructure/io/output_writer.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

namespace ms = exaconstit::microstructure;
namespace fs = std::filesystem;

namespace {

/** This process's rank in MPI_COMM_WORLD. */
int Rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

/** Number of ranks in MPI_COMM_WORLD. */
int NumRanks() {
    int n = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}

/**
 * Output directory shared by all ranks of one run, emptied and created by rank 0. Collective:
 * the barrier makes sure no rank writes into it before rank 0 has (re)created it.
 */
fs::path SharedDir(const std::string& test) {
    const fs::path dir = fs::path("test_io_output") / ("np" + std::to_string(NumRanks())) / test;
    if (Rank() == 0) {
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return dir;
}

/** Output directory private to this rank (rank<r> below the test directory). */
fs::path RankDir(const std::string& test) {
    const fs::path dir = fs::path("test_io_output") / ("np" + std::to_string(NumRanks())) /
                         test / ("rank" + std::to_string(Rank()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

/**
 * A 3x2x2 box of hexahedra (or the tetrahedra MFEM splits them into) with grain ids
 * 1 + (7e mod 5) as element attributes. The ids are deliberately not monotone in the element
 * index, so a writer that reordered elements or rewrote attributes would be caught.
 */
mfem::Mesh MakeLabeledMesh(mfem::Element::Type type) {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(3, 2, 2, type, 1.5, 1.0, 1.0 / 3.0, false);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        mesh.SetAttribute(e, 1 + (e * 7) % 5);
    }
    mesh.SetAttributes();
    return mesh;
}

/** Element attributes in element order. */
std::vector<int> Attributes(const mfem::Mesh& mesh) {
    std::vector<int> a(static_cast<std::size_t>(mesh.GetNE()));
    for (int e = 0; e < mesh.GetNE(); ++e) {
        a[static_cast<std::size_t>(e)] = mesh.GetAttribute(e);
    }
    return a;
}

/**
 * Asserts two serial meshes are identical: same counts, same element attributes in the same
 * order, bit-identical vertex coordinates, same boundary attributes.
 */
void ExpectSameMesh(const mfem::Mesh& a, const mfem::Mesh& b) {
    ASSERT_EQ(a.GetNE(), b.GetNE());
    ASSERT_EQ(a.GetNV(), b.GetNV());
    ASSERT_EQ(a.GetNBE(), b.GetNBE());
    EXPECT_EQ(Attributes(a), Attributes(b));
    for (int v = 0; v < a.GetNV(); ++v) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_EQ(a.GetVertex(v)[d], b.GetVertex(v)[d]) << "vertex " << v;
        }
    }
    for (int i = 0; i < a.GetNBE(); ++i) {
        EXPECT_EQ(a.GetBdrAttribute(i), b.GetBdrAttribute(i));
    }
}

/**
 * True if the file starts with the gzip magic bytes 0x1f 0x8b (the fixed first two bytes every
 * gzip file begins with, which readers use to recognize the format).
 */
bool IsGzip(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    unsigned char b[2] = {0, 0};
    f.read(reinterpret_cast<char*>(b), 2);
    return f && b[0] == 0x1f && b[1] == 0x8b;
}

/** The first @p n lines of a plain or gzip file (mfem::ifgzstream decompresses if needed). */
std::string FirstLines(const fs::path& path, int n) {
    mfem::ifgzstream in(path.string());
    std::string out, line;
    for (int i = 0; i < n && std::getline(in, line); ++i) {
        out += line + "\n";
    }
    return out;
}

/** Concatenates every rank's vector, in rank order, on every rank (MPI_Allgatherv). */
std::vector<int> GatherAll(const std::vector<int>& local) {
    int n = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(NumRanks()));
    MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    // displs[r] = where rank r's values start = sum of the counts of ranks before it
    std::vector<int> displs(counts.size(), 0);
    for (std::size_t r = 1; r < counts.size(); ++r) {
        displs[r] = displs[r - 1] + counts[r - 1];
    }
    std::vector<int> all(static_cast<std::size_t>(displs.back() + counts.back()));
    MPI_Allgatherv(local.data(), n, MPI_INT, all.data(), counts.data(), displs.data(), MPI_INT,
                   MPI_COMM_WORLD);
    return all;
}

/** Sum of the element volumes of a (possibly curved) mesh. */
double TotalVolume(mfem::Mesh& mesh) {
    double v = 0.0;
    for (int e = 0; e < mesh.GetNE(); ++e) {
        v += mesh.GetElementVolume(e);
    }
    return v;
}

/**
 * A stream buffer standing in for a file on a disk that fills up: it accepts the first
 * @p capacity bytes and then refuses the rest (reports fewer bytes written), as std::filebuf does
 * when the operating system's write fails.
 */
class FullDiskSink : public std::streambuf {
public:
    explicit FullDiskSink(std::size_t capacity) : m_capacity(capacity) {}
    std::size_t BytesAccepted() const { return m_stored.size(); }

protected:
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        const std::size_t room = m_capacity - m_stored.size();
        const std::size_t take = std::min(room, static_cast<std::size_t>(count));
        m_stored.append(data, take);
        return static_cast<std::streamsize>(take);
    }
    int_type overflow(int_type c) override {
        if (traits_type::eq_int_type(c, traits_type::eof())) {
            return traits_type::not_eof(c);
        }
        const char ch = traits_type::to_char_type(c);
        return xsputn(&ch, 1) == 1 ? c : traits_type::eof();
    }

private:
    std::size_t m_capacity;
    std::string m_stored;
};

/**
 * Writes about 3.5 MB of poorly compressible numbers (one per line). That is more than zstr's
 * 1 MiB buffer, so compressed data reaches the sink in the middle of the write, not only at the
 * final flush.
 */
void WriteManyNumbers(std::ostream& os) {
    for (std::uint32_t i = 0; i < 500000; ++i) {
        os << (i * 2654435761u) % 1000003u << '\n';
    }
}

} // namespace

// Serial mesh, hex and tet, gzip and plain: the file reads back with mfem::Mesh (what ExaConstit
// uses for type = "file") to exactly the same mesh, the ".gz" suffix follows the compression,
// and the provenance comments appear right after the first line with a "# " prefix added.
TEST(MicrostructureIo, SerialMeshRoundTrip) {
    const fs::path dir = RankDir("serial");
    for (const auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        const mfem::Mesh mesh = MakeLabeledMesh(type);
        const std::string tag = type == mfem::Element::HEXAHEDRON ? "hex" : "tet";

        // gzip: ".gz" is appended
        const fs::path gz = ms::WriteSerialMesh(mesh, dir / (tag + ".mesh"), 6,
                                                "generated by test_io\n# second line");
#ifdef MFEM_USE_ZLIB
        EXPECT_EQ(gz, dir / (tag + ".mesh.gz"));
        EXPECT_TRUE(IsGzip(gz));
#else
        // Without zlib the writer falls back to a plain file and drops the ".gz"
        EXPECT_EQ(gz, dir / (tag + ".mesh"));
        EXPECT_FALSE(IsGzip(gz));
#endif
        EXPECT_NE(FirstLines(gz, 5).find("# generated by test_io\n# second line"),
                  std::string::npos);
        mfem::Mesh gz_back(gz.string(), 1, 1, false);
        ExpectSameMesh(mesh, gz_back);

        // plain: a ".gz" in the requested name is dropped
        const fs::path plain = ms::WriteSerialMesh(mesh, dir / (tag + "_plain.mesh.gz"), 0);
        EXPECT_EQ(plain, dir / (tag + "_plain.mesh"));
        EXPECT_FALSE(IsGzip(plain));
        mfem::Mesh plain_back(plain.string(), 1, 1, false);
        ExpectSameMesh(mesh, plain_back);
    }
}

// Coordinates that need all 17 significant digits (thirds, offsets of 1e-13, the next double
// after x + 0.1) must come back bit-identical, which only holds if the writer really uses
// precision 17. Denormals (doubles smaller in magnitude than about 2.2e-308, stored with reduced
// precision) are avoided: whether operator>> reads them back depends on the C++ library (libc++
// and libstdc++, through strtod's range error, both can set failbit), so no reader can rely on
// them.
TEST(MicrostructureIo, SerialMeshKeepsFullPrecision) {
    const fs::path dir = RankDir("precision");
    mfem::Mesh mesh = MakeLabeledMesh(mfem::Element::HEXAHEDRON);
    for (int v = 0; v < mesh.GetNV(); ++v) {
        double* x = mesh.GetVertex(v);
        x[0] = x[0] / 3.0 + 1e-13 * v;
        x[1] = std::nextafter(x[1] + 0.1, 2.0);
    }
    const fs::path out = ms::WriteSerialMesh(mesh, dir / "p.mesh", 1);
    mfem::Mesh back(out.string(), 1, 1, false);
    ExpectSameMesh(mesh, back);
}

// A ParMesh would print only its local piece, so it is rejected; so is a gzip level above 9.
TEST(MicrostructureIo, SerialMeshRejectsParMesh) {
    mfem::Mesh mesh = MakeLabeledMesh(mfem::Element::HEXAHEDRON);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    EXPECT_THROW(ms::WriteSerialMesh(pmesh, RankDir("reject") / "x.mesh", 0),
                 std::invalid_argument);
    EXPECT_THROW(ms::WriteSerialMesh(mesh, RankDir("reject") / "x.mesh", 10),
                 std::invalid_argument);
}

// Three parts written from any number of ranks (round-robin): each part is a valid
// "MFEM mesh v1.2" file named MakeParFilename(prefix, p), every part is non-empty, no fourth part
// exists, and together the parts hold every element exactly once with its grain id (the sorted
// list of all part attributes equals the serial mesh's sorted attribute list).
TEST(MicrostructureIo, PartitionedMeshThreeParts) {
    const fs::path dir = SharedDir("partitioned");
    mfem::Mesh mesh = MakeLabeledMesh(mfem::Element::HEXAHEDRON);
    const std::string prefix = (dir / "part.mesh.").string();
    const auto names = ms::WritePartitionedMesh(mesh, 3, prefix, 6, MPI_COMM_WORLD);
    ASSERT_EQ(names.size(), 3u);
    MPI_Barrier(MPI_COMM_WORLD);

    // Every rank checks every part; a serial mfem::Mesh can read a v1.2 part on its own
    std::vector<int> attributes;
    int total = 0;
    for (int p = 0; p < 3; ++p) {
        EXPECT_EQ(names[static_cast<std::size_t>(p)].string(), mfem::MakeParFilename(prefix, p));
        EXPECT_EQ(FirstLines(names[static_cast<std::size_t>(p)], 1), "MFEM mesh v1.2\n");
        mfem::Mesh part(names[static_cast<std::size_t>(p)].string(), 1, 1, false);
        EXPECT_GT(part.GetNE(), 0);
        total += part.GetNE();
        const auto a = Attributes(part);
        attributes.insert(attributes.end(), a.begin(), a.end());
    }
    EXPECT_EQ(total, mesh.GetNE());
    auto expected = Attributes(mesh);
    std::sort(expected.begin(), expected.end());
    std::sort(attributes.begin(), attributes.end());
    EXPECT_EQ(attributes, expected);
    EXPECT_FALSE(fs::exists(mfem::MakeParFilename(prefix, 3)));
}

// One part per rank, read back exactly the way ExaConstit's readParallelMesh does
// (mfem::ifgzstream + ParMesh(comm, istream)). The global element count and the sorted list of
// all attributes match the serial mesh, and the global boundary-element count equals the serial
// one: if rank-interface faces had been written as boundary (the ParMesh::Save problem), it would
// be larger.
TEST(MicrostructureIo, PartitionedMeshReadsAsParMesh) {
    const fs::path dir = SharedDir("partitioned_parmesh");
    mfem::Mesh mesh = MakeLabeledMesh(mfem::Element::TETRAHEDRON);
    const std::string prefix = (dir / "pmesh.").string();
    ms::WritePartitionedMesh(mesh, NumRanks(), prefix, 6, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    mfem::ifgzstream in(mfem::MakeParFilename(prefix, Rank()));
    mfem::ParMesh pmesh(MPI_COMM_WORLD, in, true, 1, true);
    EXPECT_EQ(pmesh.GetGlobalNE(), static_cast<HYPRE_BigInt>(mesh.GetNE()));
    auto all = GatherAll(Attributes(pmesh));
    auto expected = Attributes(mesh);
    std::sort(all.begin(), all.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(all, expected);

    long long nbe = pmesh.GetNBE();
    long long total_nbe = 0;
    MPI_Allreduce(&nbe, &total_nbe, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(total_nbe, mesh.GetNBE());
}

// A curved (order-2 Nodes) mesh split in two: each part keeps order-2 Nodes, and the part
// volumes add up to the serial volume (to round-off), so the curved geometry was carried over.
TEST(MicrostructureIo, PartitionedCurvedMesh) {
    const fs::path dir = SharedDir("partitioned_curved");
    mfem::Mesh mesh = MakeLabeledMesh(mfem::Element::HEXAHEDRON);
    mesh.SetCurvature(2);
    const std::string prefix = (dir / "curved.").string();
    const auto names = ms::WritePartitionedMesh(mesh, 2, prefix, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    int total = 0;
    double volume = 0.0;
    for (const auto& name : names) {
        mfem::Mesh part(name.string(), 1, 1, false);
        ASSERT_NE(part.GetNodes(), nullptr);
        EXPECT_EQ(part.GetNodes()->FESpace()->GetMaxElementOrder(), 2);
        total += part.GetNE();
        volume += TotalVolume(part);
    }
    EXPECT_EQ(total, mesh.GetNE());
    EXPECT_NEAR(volume, TotalVolume(mesh), 1e-13);
}

// num_parts must be in 1..NE: 0 parts, or more parts than elements (some part would be empty),
// throw std::invalid_argument on every rank before any communication.
TEST(MicrostructureIo, PartitionedMeshRejectsBadPartCount) {
    mfem::Mesh mesh = MakeLabeledMesh(mfem::Element::HEXAHEDRON);
    const std::string prefix = (SharedDir("partitioned_bad") / "x.").string();
    EXPECT_THROW(ms::WritePartitionedMesh(mesh, 0, prefix, 0, MPI_COMM_WORLD),
                 std::invalid_argument);
    EXPECT_THROW(ms::WritePartitionedMesh(mesh, mesh.GetNE() + 1, prefix, 0, MPI_COMM_WORLD),
                 std::invalid_argument);
}

// ParPrint path: each rank's part holds exactly its local elements with their grain ids, carries
// the comment, and reloading all parts as a ParMesh gives back the same distributed mesh.
TEST(MicrostructureIo, ParallelMeshParPrint) {
    const fs::path dir = SharedDir("parprint");
    mfem::Mesh mesh = MakeLabeledMesh(mfem::Element::HEXAHEDRON);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    const std::string prefix = (dir / "pp.").string();
    const auto names = ms::WriteParallelMesh(pmesh, prefix, 6, "parallel provenance");
    ASSERT_EQ(names.size(), static_cast<std::size_t>(NumRanks()));
    MPI_Barrier(MPI_COMM_WORLD);

    const fs::path& mine = names[static_cast<std::size_t>(Rank())];
    mfem::Mesh part(mine.string(), 1, 1, false);
    EXPECT_EQ(Attributes(part), Attributes(pmesh));
    EXPECT_NE(FirstLines(mine, 5).find("# parallel provenance"), std::string::npos);

    mfem::ifgzstream in(mine.string());
    mfem::ParMesh back(MPI_COMM_WORLD, in, true, 1, true);
    EXPECT_EQ(back.GetGlobalNE(), pmesh.GetGlobalNE());
    EXPECT_EQ(Attributes(back), Attributes(pmesh));
}

// Orientation file read the way ExaConstit's LoadSharedOrientationData reads it
// (`while (f >> v)`): every value comes back bit-identical, including 1e-17, -0.0 and a value one
// ulp (unit in the last place: the gap to the next representable double) from 1/sqrt(2); there
// is one line per grain; a size that is not a multiple of 4 is rejected.
TEST(MicrostructureIo, OrientationFileReadsLikeExaConstit) {
    const fs::path dir = RankDir("ori");
    const std::vector<double> quat = {1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, 0.0,
                                      1e-17,     -0.0,       0.7071067811865476,
                                      std::nextafter(0.7071067811865476, 1.0)};
    ms::WriteOrientationFile(dir / "grains.ori", quat);

    std::ifstream f(dir / "grains.ori");
    std::vector<double> back;
    double v;
    while (f >> v) {
        back.push_back(v);
    }
    ASSERT_EQ(back.size(), quat.size());
    for (std::size_t i = 0; i < quat.size(); ++i) {
        EXPECT_EQ(back[i], quat[i]) << i;
    }
    std::ifstream lines(dir / "grains.ori");
    std::string line;
    int n = 0;
    while (std::getline(lines, line)) {
        ++n;
    }
    EXPECT_EQ(n, 2);

    EXPECT_THROW(ms::WriteOrientationFile(dir / "bad.ori", {1.0, 0.0, 0.0}),
                 std::invalid_argument);
}

// Region map read with ExaConstit's create_grains_to_map parse (getline, then `>> g >> r`) gives
// the pairs (g, region of g) for g = 1..4. Grain file read with mfem::Vector::Load, as makeMesh
// does for auto meshes, gives the ids in element order and nothing after them.
TEST(MicrostructureIo, RegionMapAndGrainFile) {
    const fs::path dir = RankDir("text");
    ms::WriteRegionMap(dir / "region_map.txt", {1, 2, 2, 1});
    std::ifstream rf(dir / "region_map.txt");
    std::string line;
    std::vector<std::pair<int, int>> pairs;
    while (std::getline(rf, line)) {
        std::istringstream iss(line);
        int g, r;
        ASSERT_TRUE(static_cast<bool>(iss >> g >> r)) << line;
        pairs.emplace_back(g, r);
    }
    const std::vector<std::pair<int, int>> expected = {{1, 1}, {2, 2}, {3, 2}, {4, 1}};
    EXPECT_EQ(pairs, expected);

    mfem::Array<int> ids({3, 1, 2, 2, 5});
    ms::WriteGrainFile(dir / "grains.txt", ids);
    std::ifstream gf(dir / "grains.txt");
    mfem::Vector gmap(ids.Size());
    gmap.Load(gf, ids.Size());
    ASSERT_FALSE(gf.fail());
    for (int i = 0; i < ids.Size(); ++i) {
        EXPECT_EQ(gmap(i), ids[i]);
    }
    double extra;
    EXPECT_FALSE(static_cast<bool>(gf >> extra));
}

// Grain table CSV: the header lists exactly the non-empty columns, a row holds
// grain id, quaternion and the optional values in header order, and a centroid column of the
// wrong length (2 values instead of 3 per grain) is rejected.
TEST(MicrostructureIo, GrainTableCsv) {
    const fs::path dir = RankDir("csv");
    ms::MicrostructureResult result;
    result.num_grains = 2;
    result.grain_quat = {1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5};
    result.grain_region = {1, 2};
    result.grain_source_id = {7, 3};
    ms::GrainTable table = ms::MakeGrainTable(result);
    table.volume = {0.25, 0.75};
    ms::WriteGrainTable(dir / "grains.csv", table);

    std::ifstream f(dir / "grains.csv");
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        lines.push_back(line);
    }
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(lines[0], "grain_id,qw,qx,qy,qz,region,source_id,volume");
    EXPECT_EQ(lines[2], "2,0.5,0.5,0.5,0.5,2,3,0.75");

    table.centroid = {0.0, 1.0};
    EXPECT_THROW(ms::WriteGrainTable(dir / "bad.csv", table), std::invalid_argument);
}

// ParaView output: the .pvd index exists after a collective write. Then only the last rank passes
// an element_grain array one entry short: every rank must still throw (collective agreement), the
// short rank with its own message and the others with "failed on another rank", instead of the
// healthy ranks going on into the collective space construction and hanging there. (At np = 1
// the only rank is the short one.)
TEST(MicrostructureIo, ParaViewCollection) {
    const fs::path dir = SharedDir("paraview");
    mfem::Mesh mesh = MakeLabeledMesh(mfem::Element::HEXAHEDRON);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mfem::Array<int> grains(pmesh.GetNE());
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        grains[e] = pmesh.GetAttribute(e);
    }
    ms::WriteParaView(pmesh, grains, {1, 1, 2, 2, 1}, dir / "sample_paraview");
    MPI_Barrier(MPI_COMM_WORLD);
    EXPECT_TRUE(fs::exists(dir / "sample_paraview" / "sample_paraview.pvd"));

    const bool is_short_rank = Rank() == NumRanks() - 1;
    mfem::Array<int> bad_grains(grains);
    if (is_short_rank) {
        bad_grains.SetSize(pmesh.GetNE() + 1, 1); // one entry too many is as wrong as too few
    }
    std::string message;
    try {
        ms::WriteParaView(pmesh, bad_grains, {}, dir / "bad");
    } catch (const std::runtime_error& e) {
        message = e.what();
    }
    ASSERT_FALSE(message.empty()) << "WriteParaView did not throw on rank " << Rank();
    if (is_short_rank) {
        EXPECT_NE(message.find("element_grain has"), std::string::npos) << message;
    } else {
        EXPECT_NE(message.find("failed on another rank"), std::string::npos) << message;
    }
}

// WriteParallelMesh accepts only conforming, non-NURBS meshes, because ParPrint writes the other
// kinds in formats readParallelMesh cannot use as a decomposition. A nonconforming ParMesh (the
// NC mesh MFEM uses for local refinement with hanging nodes) is rejected on every rank before any
// file is written.
TEST(MicrostructureIo, ParallelMeshRejectsNonconforming) {
    const fs::path dir = SharedDir("parprint_nc");
    mfem::Mesh mesh = MakeLabeledMesh(mfem::Element::HEXAHEDRON);
    mesh.EnsureNCMesh();
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    ASSERT_TRUE(pmesh.Nonconforming());
    const std::string prefix = (dir / "nc.").string();
    EXPECT_THROW(ms::WriteParallelMesh(pmesh, prefix, 0), std::invalid_argument);
    MPI_Barrier(MPI_COMM_WORLD);
    EXPECT_FALSE(fs::exists(mfem::MakeParFilename(prefix, Rank())));
}

#ifdef MFEM_USE_ZLIB

// A successful gzip write produces data that starts with the gzip magic bytes and decompresses
// (with zstr::istreambuf, the decompressor mfem::ifgzstream uses) to exactly the text written.
// The trailing empty gzip member left by zstr's destructor is read through transparently.
TEST(MicrostructureIo, GzipWriterRoundTrip) {
    std::stringbuf compressed;
    ms::detail::WriteGzip(compressed, 6, WriteManyNumbers, "memory");
    const std::string bytes = compressed.str();
    ASSERT_GT(bytes.size(), 2u);
    EXPECT_EQ(static_cast<unsigned char>(bytes[0]), 0x1f);
    EXPECT_EQ(static_cast<unsigned char>(bytes[1]), 0x8b);

    std::ostringstream expected;
    WriteManyNumbers(expected);

    std::stringbuf source(bytes);
    zstr::istreambuf unzip(&source);
    std::istream in(&unzip);
    std::ostringstream back;
    back << in.rdbuf();
    EXPECT_EQ(back.str(), expected.str());
}

// A disk that fills up during a gzip write must give a catchable std::runtime_error, not
// std::terminate from zstr::ostreambuf's destructor (which flushes again and must not throw).
// Covered: a sink that refuses everything (failure seen at the final flush), one that fills
// after 100 bytes while more than zstr's 1 MiB buffer is being written (failure in mid-write),
// and one that fills during an intermediate flush.
// If the destructor threw, the test executable would abort here instead of reporting a pass.
TEST(MicrostructureIo, GzipWriterReportsFullDisk) {
    FullDiskSink refuses_all(0);
    EXPECT_THROW(ms::detail::WriteGzip(
                     refuses_all, 6, [](std::ostream& os) { os << "one line\n"; }, "full disk"),
                 std::runtime_error);
    EXPECT_EQ(refuses_all.BytesAccepted(), 0u);

    FullDiskSink fills_mid_write(100);
    EXPECT_THROW(ms::detail::WriteGzip(fills_mid_write, 6, WriteManyNumbers, "full disk"),
                 std::runtime_error);
    EXPECT_EQ(fills_mid_write.BytesAccepted(), 100u);

    // MFEM's printers flush the stream several times, and every flush makes zstr finish a gzip
    // member. A disk that fills during one of those intermediate flushes must be reported too.
    FullDiskSink fills_at_flush(30);
    const auto flush_often = [](std::ostream& os) {
        for (int i = 0; i < 2000; ++i) {
            os << "line " << i << std::endl;
        }
    };
    EXPECT_THROW(ms::detail::WriteGzip(fills_at_flush, 6, flush_often, "full disk"),
                 std::runtime_error);
}

// An exception thrown by the content producer itself, in the middle of a compressed write,
// propagates unchanged (here std::logic_error), and unwinding through the compressor is safe.
TEST(MicrostructureIo, GzipWriterPropagatesWriterException) {
    std::stringbuf compressed;
    const auto throw_midway = [](std::ostream& os) {
        WriteManyNumbers(os);
        throw std::logic_error("producer failed");
    };
    EXPECT_THROW(ms::detail::WriteGzip(compressed, 6, throw_midway, "memory"), std::logic_error);
    EXPECT_THROW(ms::detail::WriteGzip(compressed, 0, WriteManyNumbers, "memory"),
                 std::invalid_argument);
}

#endif // MFEM_USE_ZLIB

int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
