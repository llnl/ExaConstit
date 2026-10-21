/**
 * @file test_parallel_mesh_input.cpp
 * @brief Tests for reading a parallel mesh decomposition ([Mesh] type = "parallel")
 *
 * Covers the options, the PARALLEL branch of makeMesh in src/sim_state/simulation_state.cpp,
 * and mechanics runs (serial file vs decomposition, and the wrong-rank-count abort).
 *
 * A parallel mesh decomposition is a mesh that was split into parts ahead of time and saved as
 * one file per part, in MFEM's parallel mesh format (each part holds its own elements plus the
 * list of vertices, edges and faces it shares with neighboring parts). Rank r of the run reads
 * part r, the file mfem::MakeParFilename(prefix, r) = prefix + r zero-padded to six digits, so
 * the number of ranks must equal the number of parts. The parts may be gzip-compressed.
 *
 * The fixture mesh is test/data/cube_tet.mesh (750 tets whose element attributes are the 125
 * grain IDs of test/data/grains.txt), mapped to two regions by test/data/region_mapping.txt.
 * The decompositions are written here with mfem::MeshPartitioner and mfem::MeshPart::Print,
 * partitioned exactly as ParMesh(comm, serial_mesh) partitions a serial mesh, so a parallel
 * read and a serial read of the same mesh give each rank the same elements in the same order.
 *
 * The executable has three modes:
 * - no arguments (run under MPI with any number of ranks): the gtest cases below. Each case
 *   writes a decomposition with one part per rank of the current run, so every rank count
 *   tests a different partition. Scratch files go to parallel_mesh_input_np<ranks>/ in the
 *   working directory.
 * - `--write-mechanics-case <directory> <np>...` (one rank): writes, for each rank count np, a
 *   decomposition of cube_tet.mesh into np gzip parts plus two input decks,
 *   serial_np<np>.toml (the serial mesh file) and parallel_np<np>.toml (the decomposition),
 *   both derived from test/data/voce_full_multi_tet.toml and shortened to 5 steps. CTest then
 *   runs `mechanics` on both decks in <directory>.
 * - `--compare-mechanics <directory_a> <directory_b>`: compares every .txt result file of
 *   directory_a with the file of the same name in directory_b, with the default tolerances of
 *   test_mechanics.py (a value matches if it is within relative 1e-8, or within the absolute
 *   tolerance max(1e-10, 1e-8 * largest magnitude in the file)).
 *
 * The tests are registered only with EXACONSTIT_ENABLE_EXTENSIVE_TESTS=ON (test/CMakeLists.txt).
 *
 * MPI notes for readers new to it: MPI_Barrier makes every rank wait until all ranks reach it;
 * the tests use it so that no rank reads a file before rank 0 has finished writing it (or
 * deleting it). MPI_Allreduce combines one value from each rank (sum, min, max) and gives the
 * result to every rank; the tests use it to compare quantities summed over the whole mesh.
 */

#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace fs = std::filesystem;

namespace {

/** @brief Directory with the test input files (test/data in the source tree) */
const fs::path kDataDir = EXACONSTIT_TEST_DATA_DIR;

/**
 * @brief Write a parallel decomposition of a serial mesh, one file per part
 *
 * @param mesh Serial mesh whose element attributes are grain IDs
 * @param prefix File prefix; part p goes to mfem::MakeParFilename(prefix, p)
 * @param num_parts Number of parts
 * @param compress Whether to gzip the parts
 *
 * @details Runs on one rank (the caller decides which). MeshPartitioner with no explicit
 * partitioning uses METIS through Mesh::GeneratePartitioning(num_parts, 1), as
 * ParMesh(comm, mesh) does by default (mfem mesh/pmesh.hpp:340, mesh/mesh.hpp:3003-3037), so the
 * parts match the partition of a serial read on num_parts ranks, and each part lists its
 * elements in ascending serial order. MeshPart::Print writes MFEM's parallel mesh format.
 * Coordinates are printed with 17 significant digits so they round-trip exactly.
 */
void WriteDecomposition(mfem::Mesh& mesh,
                        const std::string& prefix,
                        const int num_parts,
                        const bool compress) {
    mfem::MeshPartitioner partitioner(mesh, num_parts);
    mfem::MeshPart mesh_part;
    for (int part = 0; part < num_parts; ++part) {
        // Copy the elements of this part (and its shared-entity lists) into mesh_part, then
        // write it to the part's file, gzip-compressed or plain
        partitioner.ExtractPart(part, mesh_part);
        mfem::ofgzstream part_stream(mfem::MakeParFilename(prefix, part), compress);
        part_stream.precision(17);
        mesh_part.Print(part_stream);
    }
}

/**
 * @brief Options for a SimulationState built from cube_tet.mesh or one of its decompositions
 *
 * @param mesh_type MeshType::FILE (mesh_location is the mesh file) or MeshType::PARALLEL
 *        (mesh_location is the decomposition prefix)
 * @param mesh_location Mesh file or prefix
 * @param ref_ser Serial refinement levels
 * @param ref_par Parallel refinement levels
 * @return Options with two UMAT regions assigned through test/data/region_mapping.txt
 *
 * @details Only the options SimulationState needs to build the mesh, the grain and region
 * maps and the finite element spaces are set. The two materials are placeholder UMATs (one
 * property, one state variable); they are never evaluated, because no time step is taken.
 */
ExaOptions MakeOptions(const MeshType mesh_type,
                       const fs::path& mesh_location,
                       const int ref_ser,
                       const int ref_par) {
    ExaOptions options;
    options.basename = "parallel_mesh_input";
    options.mesh.mesh_type = mesh_type;
    options.mesh.mesh_file = mesh_location;
    options.mesh.order = 1;
    options.mesh.ref_ser = ref_ser;
    options.mesh.ref_par = ref_par;
    options.time.time_type = TimeStepType::FIXED;
    options.time.fixed_time = TimeOptions::FixedTimeOptions{};
    options.time.fixed_time->dt = 0.1;
    options.time.fixed_time->t_final = 1.0;
    options.solvers.rtmodel = RTModel::CPU;
    options.region_mapping_file = kDataDir / "region_mapping.txt";
    for (int region = 0; region < 2; ++region) {
        MaterialOptions material;
        if (region == 0) {
            material.material_name = "material_A";
        } else {
            material.material_name = "material_B";
        }
        material.region_id = region + 1;
        material.mech_type = MechType::UMAT;
        material.properties.num_props = 1;
        material.properties.properties = {1.0};
        material.state_vars.num_vars = 1;
        material.state_vars.initial_values = {0.0};
        options.materials.push_back(material);
    }
    return options;
}

/**
 * @brief Per-element data of a rank's part of the mesh, for comparing two reads
 */
struct LocalMeshData {
    /** @brief Grain ID of each local element */
    std::vector<int> grains;
    /** @brief Region ID (element attribute after region mapping) of each local element */
    std::vector<int> regions;
    /** @brief Element centers, three per element */
    std::vector<double> centers;
    /** @brief Global number of elements */
    long long global_elements = 0;
    /** @brief Volume of each grain summed over all ranks, indexed by grain ID */
    std::vector<double> grain_volumes;
};

/** @brief Largest grain ID in cube_tet.mesh */
constexpr int kNumGrains = 125;

/**
 * @brief Build a SimulationState and collect its mesh data
 *
 * @param options Options from MakeOptions()
 * @return The rank's element data and the global element count and grain volumes
 *
 * @details Collective: every rank must call it (SimulationState construction and the
 * MPI_Allreduce of the grain volumes communicate). The grain IDs come from
 * SimulationState::GetGrains() (the mesh's element attributes before region mapping) and the
 * region IDs from the element attributes after the constructor has replaced grain IDs by
 * region IDs.
 */
LocalMeshData CollectMeshData(ExaOptions options) {
    SimulationState sim_state(options);
    auto mesh = sim_state.GetMesh();
    auto grains = sim_state.GetGrains();
    LocalMeshData data;
    data.grain_volumes.assign(kNumGrains + 1, 0.0);
    std::vector<double> local_volumes(kNumGrains + 1, 0.0);
    mfem::Vector center(3);
    const double* grain_data = grains->HostRead();
    // Record, for each local element in local order: grain ID, region ID, center (x, y, z),
    // and add its volume to its grain's local total
    for (int i = 0; i < mesh->GetNE(); ++i) {
        const int grain = static_cast<int>(grain_data[i]);
        data.grains.push_back(grain);
        data.regions.push_back(mesh->GetAttribute(i));
        mesh->GetElementCenter(i, center);
        data.centers.insert(data.centers.end(), {center[0], center[1], center[2]});
        if (grain >= 1 && grain <= kNumGrains) {
            local_volumes[grain] += mesh->GetElementVolume(i);
        }
    }
    // Global element count and per-grain volumes summed over all ranks, so they can be compared
    // between two reads whose partitions differ
    data.global_elements = mesh->GetGlobalNE();
    MPI_Allreduce(local_volumes.data(),
                  data.grain_volumes.data(),
                  kNumGrains + 1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    return data;
}

/**
 * @brief Scratch directory for this rank count, created by rank 0
 *
 * @return parallel_mesh_input_np<ranks>/ in the working directory
 *
 * @details Collective. One directory per rank count, so ctest can run the np1-np4 variants at
 * the same time without them overwriting each other's files. The barrier makes the other ranks
 * wait until rank 0 has created the directory.
 */
fs::path ScratchDirectory() {
    int rank = 0;
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    const fs::path directory = "parallel_mesh_input_np" + std::to_string(num_ranks);
    if (rank == 0) {
        fs::create_directories(directory);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return directory;
}

/**
 * @brief Write a decomposition of cube_tet.mesh into one part per rank (rank 0 writes)
 *
 * @param prefix File prefix of the parts
 * @param compress Whether to gzip the parts
 *
 * @details Collective. Only rank 0 reads cube_tet.mesh and writes the parts; the barrier makes
 * the other ranks wait until every part is on disk before anyone tries to read it.
 */
void WriteCubeDecomposition(const std::string& prefix, const bool compress) {
    int rank = 0;
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (rank == 0) {
        mfem::Mesh mesh((kDataDir / "cube_tet.mesh").string().c_str(), 1, 1, true);
        WriteDecomposition(mesh, prefix, num_ranks, compress);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace

// The [Mesh] table accepts type = "parallel" with floc as the prefix, and validation looks for
// part 0 unless the input files are not needed (restart). The expected results follow from the
// rules in MeshOptions::validate: part 0 is deleted first, so validation with file checks must
// fail; it must pass with file checks off (a restart) and, once a file named like part 0
// exists, with file checks on (validation only checks that the file exists, not its contents).
// Without a prefix validation must fail even with file checks off.
TEST(ParallelMeshInput, Options) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const fs::path directory = ScratchDirectory();
    const std::string prefix = (directory / "options_check.mesh.").string();
    // Start from a missing part 0 (a previous run may have left one behind)
    if (rank == 0) {
        fs::remove(mfem::MakeParFilename(prefix, 0));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Parse a [Mesh] table from a string, as the options parser would from options.toml
    std::istringstream input("type = \"parallel\"\nfloc = \"" + prefix + "\"\nref_ser = 1\n");
    const toml::value table = toml::parse(input, "mesh_table");
    const MeshOptions mesh = MeshOptions::from_toml(table);
    EXPECT_EQ(mesh.mesh_type, MeshType::PARALLEL);
    EXPECT_EQ(mesh.mesh_file.string(), prefix);
    EXPECT_EQ(mesh.ref_ser, 1);

    EXPECT_FALSE(mesh.validate(true)) << "part 0 is missing";
    EXPECT_TRUE(mesh.validate(false)) << "a restart does not need the mesh files";

    // Every rank must finish the checks above before part 0 appears; without this barrier a
    // slow rank could see the file rank 0 creates next and wrongly pass the "missing" check
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::ofstream(mfem::MakeParFilename(prefix, 0)) << "placeholder\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    EXPECT_TRUE(mesh.validate(true)) << "part 0 exists";

    MeshOptions no_prefix = mesh;
    no_prefix.mesh_file.clear();
    EXPECT_FALSE(no_prefix.validate(false)) << "a parallel mesh needs a prefix";

    EXPECT_EQ(string_to_mesh_type("parallel"), MeshType::PARALLEL);
}

// A gzip decomposition read through makeMesh gives every rank the same elements, grain IDs,
// region IDs and element centers as the serial mesh file read through the FILE path. This
// holds exactly (not just globally) because WriteDecomposition partitions the mesh the same
// way ParMesh(comm, serial_mesh) does, and both keep each part's elements in ascending serial
// order. cube_tet.mesh has 750 elements; the centers are computed from coordinates that were
// written with 17 digits, so they agree to round-off (1e-14 on a unit cube).
TEST(ParallelMeshInput, MatchesSerialRead) {
    const fs::path directory = ScratchDirectory();
    const std::string prefix = (directory / "cube_tet_gz.mesh.").string();
    WriteCubeDecomposition(prefix, true);

    // Read the same mesh both ways, without refinement
    const LocalMeshData serial = CollectMeshData(
        MakeOptions(MeshType::FILE, kDataDir / "cube_tet.mesh", 0, 0));
    const LocalMeshData parallel = CollectMeshData(MakeOptions(MeshType::PARALLEL, prefix, 0, 0));

    // Same global size, and element by element the same data on this rank
    EXPECT_EQ(serial.global_elements, 750);
    EXPECT_EQ(parallel.global_elements, serial.global_elements);
    ASSERT_EQ(parallel.grains.size(), serial.grains.size()) << "local element count";
    EXPECT_EQ(parallel.grains, serial.grains) << "grain IDs";
    EXPECT_EQ(parallel.regions, serial.regions) << "region IDs";
    double max_center_difference = 0.0;
    for (size_t k = 0; k < serial.centers.size(); ++k) {
        max_center_difference = std::max(max_center_difference,
                                         std::abs(serial.centers[k] - parallel.centers[k]));
    }
    EXPECT_LE(max_center_difference, 1e-14) << "element centers";

    // Both region IDs occur somewhere in the mesh (region_mapping.txt uses regions 1 and 2), so
    // the region map was applied to the decomposition. Each rank flags the regions it has, and
    // MPI_MAX over the ranks gives 1 for every region found on any rank.
    int local_regions[2] = {0, 0};
    for (const int region : parallel.regions) {
        ASSERT_TRUE(region == 1 || region == 2) << "region " << region;
        local_regions[region - 1] = 1;
    }
    int global_regions[2] = {0, 0};
    MPI_Allreduce(local_regions, global_regions, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(global_regions[0], 1);
    EXPECT_EQ(global_regions[1], 1);
}

// ref_ser and ref_par are both applied to a decomposition as parallel refinements: the refined
// mesh has the serially refined mesh's element count and per-grain volumes (the partition
// differs, so only global quantities are compared). The parts are plain text here, which also
// covers the uncompressed read.
//
// Expected values: ref_ser = 1 and ref_par = 1 are two uniform refinements in total, and each
// splits every tet into 8, so 750 * 8 * 8 = 750 * 64 elements. Refinement does not change the
// volume of a straight-sided element's children combined, so every grain keeps its volume and
// the grain volumes add up to the unit cube that cube_tet.mesh fills.
TEST(ParallelMeshInput, RefinementMatchesSerialRead) {
    const fs::path directory = ScratchDirectory();
    const std::string prefix = (directory / "cube_tet_plain.mesh.").string();
    WriteCubeDecomposition(prefix, false);

    const LocalMeshData serial = CollectMeshData(
        MakeOptions(MeshType::FILE, kDataDir / "cube_tet.mesh", 1, 1));
    const LocalMeshData parallel = CollectMeshData(MakeOptions(MeshType::PARALLEL, prefix, 1, 1));

    EXPECT_EQ(serial.global_elements, 750 * 64);
    EXPECT_EQ(parallel.global_elements, serial.global_elements);
    // Compare grain by grain; every grain must still exist (positive volume) after refinement
    double total = 0.0;
    double max_difference = 0.0;
    for (int grain = 1; grain <= kNumGrains; ++grain) {
        total += serial.grain_volumes[grain];
        max_difference = std::max(
            max_difference, std::abs(serial.grain_volumes[grain] - parallel.grain_volumes[grain]));
        EXPECT_GT(parallel.grain_volumes[grain], 0.0) << "grain " << grain;
    }
    EXPECT_NEAR(total, 1.0, 1e-12) << "cube_tet.mesh fills the unit cube";
    EXPECT_LE(max_difference, 1e-14) << "per-grain volumes";
}

namespace {

/**
 * @brief Write the decompositions and input decks of the mechanics comparison
 *
 * @param directory Output directory (created if needed)
 * @param rank_counts Rank counts to prepare a decomposition and a pair of decks for
 * @return Process exit code (0 on success; toml11 and file errors throw)
 *
 * @details Runs on one rank (the setup step of the CTest chain). Both decks start from
 * test/data/voce_full_multi_tet.toml and differ only in basename and the [Mesh] table, so any
 * difference in the results comes from how the mesh was read. Steps:
 *  1. load the base deck and make every data path absolute, so the decks run from the scratch
 *     directory;
 *  2. shorten the run to 5 fixed steps and turn off refinement;
 *  3. for each rank count np: clear old results, write an np-part gzip decomposition, and
 *     write serial_np<np>.toml (type = "file") and parallel_np<np>.toml (type = "parallel").
 */
int WriteMechanicsCase(const fs::path& directory, const std::vector<int>& rank_counts) {
    fs::create_directories(directory);
    // Absolute paths, so the decks work from any working directory
    const fs::path data_dir = fs::absolute(kDataDir);
    const fs::path out_dir = fs::absolute(directory);

    toml::value base = toml::parse((data_dir / "voce_full_multi_tet.toml").string());
    // grain_file is not used by file or parallel meshes (and would only print a warning). The
    // base deck no longer sets it; the erase (a no-op then) guards against it coming back.
    base.as_table().erase("grain_file");
    base["region_mapping_file"] = (data_dir / "region_mapping.txt").string();
    for (auto& material : base["Materials"].as_array()) {
        material["Properties"]["floc"] = (data_dir / "props_cp_voce.txt").string();
        material["State_Vars"]["floc"] = (data_dir / "state_cp_voce.txt").string();
        material["Grain"]["orientation_file"] = (data_dir / "voce_quats.ori").string();
    }
    // Five fixed steps keep the runs short
    auto& time = base["Time"].as_table();
    time.erase("Auto");
    time.erase("Custom");
    base["Time"]["Fixed"]["dt"] = 0.1;
    base["Time"]["Fixed"]["t_final"] = 0.5;
    base["Mesh"]["ref_ser"] = 0;
    base["Mesh"]["ref_par"] = 0;

    // Wide enough to keep each absolute path on one line (toml11 wraps longer strings)
    const std::size_t no_wrap = 250;
    mfem::Mesh mesh((data_dir / "cube_tet.mesh").string().c_str(), 1, 1, true);
    for (const int np : rank_counts) {
        const std::string tag = "_np" + std::to_string(np);
        // mechanics appends to existing result files, so clear the previous runs' output
        fs::remove_all(out_dir / "results" / ("serial" + tag));
        fs::remove_all(out_dir / "results" / ("parallel" + tag));
        const std::string prefix = (out_dir / ("cube_tet" + tag + ".mesh.")).string();
        WriteDecomposition(mesh, prefix, np, true);

        toml::value serial = base;
        serial["basename"] = "serial" + tag;
        serial["Mesh"]["type"] = "file";
        serial["Mesh"]["floc"] = (data_dir / "cube_tet.mesh").string();
        std::ofstream(out_dir / ("serial" + tag + ".toml")) << toml::format(serial, no_wrap);

        toml::value parallel = base;
        parallel["basename"] = "parallel" + tag;
        parallel["Mesh"]["type"] = "parallel";
        parallel["Mesh"]["floc"] = prefix;
        std::ofstream(out_dir / ("parallel" + tag + ".toml")) << toml::format(parallel, no_wrap);
        std::cout << "Wrote " << np << "-part decomposition " << prefix << "* and decks "
                  << "serial" << tag << ".toml, parallel" << tag << ".toml in " << out_dir
                  << std::endl;
    }
    return 0;
}

/**
 * @brief Read the numeric rows of a whitespace-separated result file
 *
 * @param file Result file (for example avg_stress_<basename>.txt)
 * @return One vector per line that parses entirely as numbers (header lines are skipped);
 *         empty if the file cannot be opened
 *
 * @details A line counts as numeric only if every whitespace-separated field is consumed
 * completely by std::strtod, so a header such as "# time sxx syy ..." is skipped.
 */
std::vector<std::vector<double>> ReadNumericRows(const fs::path& file) {
    std::vector<std::vector<double>> rows;
    std::ifstream input(file);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::vector<double> row;
        std::string field;
        bool numeric = true;
        while (fields >> field) {
            char* end = nullptr;
            const double value = std::strtod(field.c_str(), &end);
            if (end == field.c_str() || *end != '\0') {
                numeric = false;
                break;
            }
            row.push_back(value);
        }
        if (numeric && !row.empty()) {
            rows.push_back(row);
        }
    }
    return rows;
}

/**
 * @brief Compare the result files of two mechanics runs
 *
 * @param directory_a Results of the reference run
 * @param directory_b Results of the run to check
 * @return 0 if every file matches, 1 otherwise (also 1 if directory_a has no .txt files)
 *
 * @details Runs on one rank. For every .txt file in directory_a (sorted by name), the file of
 * the same name in directory_b must have the same number of numeric rows and columns, and every
 * value must agree within the test_mechanics.py tolerances: relative 1e-8, or absolute
 * max(1e-10, 1e-8 * largest magnitude in the reference file). The absolute tolerance keeps
 * values that should be zero (and so are only round-off) from failing the relative test.
 * Prints one line per file and the first mismatching value.
 */
int CompareMechanics(const fs::path& directory_a, const fs::path& directory_b) {
    int files = 0;
    int failures = 0;
    // Collect the reference run's result files; sorting gives a stable output order
    std::vector<fs::path> names;
    if (fs::is_directory(directory_a)) {
        for (const auto& entry : fs::directory_iterator(directory_a)) {
            if (entry.path().extension() == ".txt") {
                names.push_back(entry.path().filename());
            }
        }
    }
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
        ++files;
        const auto rows_a = ReadNumericRows(directory_a / name);
        const auto rows_b = ReadNumericRows(directory_b / name);
        if (rows_a.empty() || rows_a.size() != rows_b.size()) {
            std::cout << "MISMATCH " << name << ": " << rows_a.size() << " vs " << rows_b.size()
                      << " numeric rows" << std::endl;
            ++failures;
            continue;
        }
        // Tolerances scale with the largest value in the reference file
        double max_magnitude = 0.0;
        for (const auto& row : rows_a) {
            for (const double value : row) {
                max_magnitude = std::max(max_magnitude, std::abs(value));
            }
        }
        const double rel_tolerance = 1e-8;
        const double abs_tolerance = std::max(1e-10, rel_tolerance * max_magnitude);
        // Compare value by value; stop at the first value outside both tolerances
        double max_difference = 0.0;
        bool match = true;
        for (size_t r = 0; r < rows_a.size() && match; ++r) {
            if (rows_a[r].size() != rows_b[r].size()) {
                match = false;
                break;
            }
            for (size_t c = 0; c < rows_a[r].size(); ++c) {
                const double difference = std::abs(rows_a[r][c] - rows_b[r][c]);
                max_difference = std::max(max_difference, difference);
                if (difference > abs_tolerance &&
                    difference > rel_tolerance * std::abs(rows_a[r][c])) {
                    std::cout << "MISMATCH " << name << " row " << r << " column " << c << ": "
                              << std::setprecision(17) << rows_a[r][c] << " vs " << rows_b[r][c]
                              << std::endl;
                    match = false;
                    break;
                }
            }
        }
        if (match) {
            std::cout << "match    ";
        } else {
            std::cout << "MISMATCH ";
            ++failures;
        }
        std::cout << name << " (" << rows_a.size()
                  << " rows, max difference " << max_difference << ")" << std::endl;
    }
    if (files == 0) {
        std::cout << "No result files found in " << directory_a << std::endl;
        return 1;
    }
    std::cout << files - failures << " of " << files << " result files match" << std::endl;
    if (failures != 0) {
        return 1;
    }
    return 0;
}

} // namespace

/**
 * @brief Entry point: run the gtest cases, or one of the two helper modes of the mechanics
 * comparison (see the comment at the top of this file)
 *
 * @details MPI is initialized in every mode, because SimulationState and the mesh writers use
 * it. The gtest mode also initializes HYPRE, which the finite element spaces built by
 * SimulationState need.
 */
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int result = 0;
    std::string mode;
    if (argc > 1) {
        mode = argv[1];
    }
    if (mode == "--write-mechanics-case" && argc > 3) {
        // Parse the rank counts strictly: std::atoi would turn a typo into 0, which would reach
        // MeshPartitioner(mesh, 0). Every argument must be a whole positive number.
        std::vector<int> rank_counts;
        bool valid_counts = true;
        for (int k = 3; k < argc; ++k) {
            char* end = nullptr;
            const long count = std::strtol(argv[k], &end, 10);
            if (end == argv[k] || *end != '\0' || count <= 0 || count > 1000000) {
                std::cerr << "--write-mechanics-case: rank count '" << argv[k]
                          << "' is not a positive integer" << std::endl;
                valid_counts = false;
                break;
            }
            rank_counts.push_back(static_cast<int>(count));
        }
        if (valid_counts) {
            result = WriteMechanicsCase(argv[2], rank_counts);
        } else {
            result = 1;
        }
    } else if (mode == "--compare-mechanics" && argc == 4) {
        result = CompareMechanics(argv[2], argv[3]);
    } else {
        mfem::Hypre::Init();
        ::testing::InitGoogleTest(&argc, argv);
        result = RUN_ALL_TESTS();
    }
    MPI_Finalize();
    return result;
}
