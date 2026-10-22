/*
 * microstructure_generator: standalone MPI tool that paints grains onto a finite element mesh and
 * writes the files an ExaConstit run needs (mesh with grain ids as element attributes, or an
 * auto-mesh grain file; orientation file; region map when there is more than one region), plus a
 * grain table, an optional grain adjacency table and a ParaView view.
 *
 * Usage:
 *     microstructure_generator -i options.toml [-m mesh] [-n N] [-s seed] [-o base] [--parts P]
 *     microstructure_generator -m mesh.mesh[.gz] -n N [-s seed] [-o base] [--parts P]
 *
 * Flow of main():
 *  1. MPI start-up and command line.
 *  2. Options: the TOML file (or the defaults of the "-n N" quick path), the command-line
 *     overrides, then validate(). Every rank checks its own copy; the verdicts are combined with
 *     MPI_Allreduce(MIN) so all ranks stop together (a "collective" decision: one that every rank
 *     takes part in and every rank learns).
 *  3. mfem::Device from `exec` (ConfigureMfemDevice), exactly like ExaConstit's mechanics driver.
 *  4. The serial mesh on every rank: a mesh file (plain or gzip) or ExaConstit's auto box. The
 *     auto box is labeled unrefined, which is what ExaConstit's grain_file refers to.
 *  5. Partition (BuildSerialOrder + MakeParMesh, which keeps a map from local to serial element
 *     ids) and GenerateMicrostructure.
 *  6. Grain statistics on the distributed mesh (collective; fixed-point sums, so the numbers do
 *     not depend on the rank count): element count, volume, centroid and neighbor count of every
 *     grain, and the shared face area of every pair of touching grains
 *     (labeling/grain_statistics.hpp over the face adjacency the driver built for the fragment
 *     cleanup, with periodic face pairing for periodic microstructures). The pair table is
 *     gathered on rank 0.
 *  7. Gather the grain ids to serial element order, set them as element attributes of the serial
 *     mesh, and write the outputs selected by [Output]: mesh file(s) or grain file, orientation
 *     file, region map, grain table <base>_grains.csv, optional grain adjacency table
 *     <base>_adjacency.csv ([Output] grain_adjacency), ParaView view.
 *  8. Rank-0 summary: provenance, what each optional stage did (Johnson-Mehl nucleation, size
 *     targets, Lloyd relaxation, volume fit with the element-level volume errors against the
 *     targets, periodic images, quadrature vote, fragment cleanup), grain statistics, and how to
 *     point an ExaConstit input at the outputs. No ExaConstit input file is generated.
 *
 * Errors inside the pipeline are normally thrown on all ranks together by the library (its
 * checks are either made on identical data or globally reduced). The tool catches them, prints
 * the message on every rank that caught it (with the rank number) and calls MPI_Abort, which also
 * stops the other ranks when only some ranks threw (e.g. a file write failing on one rank).
 */
#include "microstructure/exec_dispatch.hpp"
#include "microstructure/io/output_writer.hpp"
#include "microstructure/labeling/grain_statistics.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/microstructure_options.hpp"
#include "microstructure/util/collective_errors.hpp"
#include "utilities/auto_mesh.hpp"

#include "ExaConstit_Version.h"
#include "mfem.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

/** Elements per grain below which a coarse-resolution warning is printed. */
constexpr double MIN_ELEMENTS_PER_GRAIN = 50.0;

/** Parsed command line. Identical on every rank (every rank receives the same argv). */
struct CommandLine {
    bool help = false;                ///< -h / --help
    std::optional<fs::path> input;    ///< -i FILE
    ms::CliOverrides overrides;       ///< -m -n -s -o --parts
    std::string error;                ///< non-empty when the command line is invalid
    std::string text;                 ///< the full command line, for provenance
};

/** Prints the usage text. Host only. */
void PrintUsage(std::ostream& os) {
    os << "Usage:\n"
       << "  microstructure_generator -i options.toml [overrides]\n"
       << "  microstructure_generator -m mesh.mesh[.gz] -n N [overrides]\n"
       << "Options:\n"
       << "  -i, --input FILE      generator options (TOML)\n"
       << "  -m, --mesh FILE       serial MFEM mesh file (plain or gzip); sets [Mesh] type=file\n"
       << "  -n, --num-grains N    number of random seeds (grains)\n"
       << "  -s, --seed S          random number key Seeds.rng_seed (integer >= 0)\n"
       << "  -o, --output NAME     output basename\n"
       << "      --parts P         also write a P-part parallel decomposition\n"
       << "  -h, --help            this text\n";
}

/*
 * ParseInteger: strict decimal parse with std::strtoll; the whole word must be a number in
 * [min_value, max_value]. Returns false otherwise (no exceptions: the command-line parser
 * collects a message instead).
 */
bool ParseInteger(const std::string& word, long long min_value, long long max_value,
                  long long& value) {
    if (word.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(word.c_str(), &end, 10);
    if (errno != 0 || end == word.c_str() || *end != '\0' || parsed < min_value ||
        parsed > max_value) {
        return false;
    }
    value = parsed;
    return true;
}

/*
 * ParseCommandLine
 *
 * Walks argv once. Every flag except -h takes one value. Either -i is given, or both -m and -n
 * (the quick path); anything else fills CommandLine::error. The parse is pure string handling,
 * so every rank reaches the same result.
 */
CommandLine ParseCommandLine(int argc, char** argv) {
    CommandLine cl;
    for (int k = 0; k < argc; ++k) {
        cl.text += (k > 0 ? " " : "");
        cl.text += argv[k];
    }

    for (int k = 1; k < argc; ++k) {
        const std::string flag = argv[k];
        if (flag == "-h" || flag == "--help") {
            cl.help = true;
            continue;
        }
        // Every other flag needs a value
        if (k + 1 >= argc) {
            cl.error = "missing value after " + flag;
            return cl;
        }
        const std::string value = argv[++k];
        long long number = 0;
        if (flag == "-i" || flag == "--input") {
            cl.input = fs::path(value);
        } else if (flag == "-m" || flag == "--mesh") {
            cl.overrides.mesh = fs::path(value);
        } else if (flag == "-n" || flag == "--num-grains") {
            if (!ParseInteger(value, 1, (1LL << 31) - 1, number)) {
                cl.error = "-n needs an integer in [1, 2^31 - 1], got '" + value + "'";
                return cl;
            }
            cl.overrides.num_seeds = static_cast<std::int64_t>(number);
        } else if (flag == "-s" || flag == "--seed") {
            if (!ParseInteger(value, 0, std::numeric_limits<long long>::max(), number)) {
                cl.error = "-s needs an integer >= 0, got '" + value + "'";
                return cl;
            }
            cl.overrides.rng_seed = static_cast<std::uint64_t>(number);
        } else if (flag == "-o" || flag == "--output") {
            cl.overrides.basename = value;
        } else if (flag == "--parts") {
            if (!ParseInteger(value, 1, (1LL << 31) - 1, number)) {
                cl.error = "--parts needs an integer >= 1, got '" + value + "'";
                return cl;
            }
            cl.overrides.num_parts = static_cast<int>(number);
        } else {
            cl.error = "unknown option '" + flag + "'";
            return cl;
        }
    }

    // Either an options file, or the quick path with both a mesh and a grain count
    if (!cl.help && !cl.input && (!cl.overrides.mesh || !cl.overrides.num_seeds)) {
        cl.error = "give an options file (-i) or both a mesh (-m) and a grain count (-n)";
    }
    return cl;
}

/** The tool's collective verdict helper: AgreeOrReport (util/collective_errors.hpp) with the
 * program name as the prefix of the error line. Collective. */
bool AgreeOrReport(bool local_ok, const std::string& message, MPI_Comm comm) {
    return ms::AgreeOrReport(local_ok, message, comm, "microstructure_generator");
}

/** "major.minor.patch" of ExaConstit (MAJOR/MINOR are integers, PATCH a string literal). */
std::string VersionString() {
    std::ostringstream os;
    os << EXACONSTIT_VERSION_MAJOR << "." << EXACONSTIT_VERSION_MINOR << "."
       << EXACONSTIT_VERSION_PATCH;
    return os.str();
}

/** Lower-case option spelling of a metric, for logs. */
std::string MetricName(ms::MetricType metric) {
    switch (metric) {
    case ms::MetricType::VORONOI:
        return "voronoi";
    case ms::MetricType::LAGUERRE:
        return "laguerre";
    case ms::MetricType::QUADRIC:
        return "quadric";
    case ms::MetricType::ADDITIVE:
        return "additive";
    }
    return "unknown";
}

/** Human-readable description of the input mesh, for provenance. */
std::string DescribeMeshInput(const ms::MicrostructureOptions& options) {
    std::ostringstream os;
    if (options.mesh.type == "auto") {
        os << "auto box, ncuts = [" << options.mesh.nxyz[0] << ", " << options.mesh.nxyz[1]
           << ", " << options.mesh.nxyz[2] << "], length = [" << options.mesh.mxyz[0] << ", "
           << options.mesh.mxyz[1] << ", " << options.mesh.mxyz[2] << "]";
    } else {
        os << "file " << options.mesh.floc.string();
    }
    return os.str();
}

/*
 * ReadSerialMesh: the whole serial mesh, built identically on every rank.
 * - "file": mfem::Mesh(path, generate_edges = 1, refine = 1, fix_orientation = true), the same
 *   call ExaConstit's makeMesh uses for [Mesh] type = "file" (simulation_state.cpp), so the
 *   element order is the one ExaConstit will see. MFEM detects gzip from the file contents.
 * - "auto": exaconstit::MakeAutoHexMesh (utilities/auto_mesh.hpp), the box ExaConstit builds for
 *   [Mesh] type = "auto", unrefined, in the same lexicographic element order. Its boundary
 *   attributes are remapped with ApplyAutoMeshBoundaryAttributes exactly once, here, right after
 *   the box is built, so a mesh file written from it carries the same boundary ids ExaConstit
 *   gives its auto meshes (ExaConstit remaps auto meshes but not file meshes). The remap is a
 *   permutation of the ids 1..6 (MFEM's face numbering -> ExaConstit's), not an idempotent
 *   assignment: applying it twice would scramble the ids (e.g. the right face 3 -> 5 -> 2) and put
 *   boundary conditions on the wrong faces. No later step may apply it again.
 */
mfem::Mesh ReadSerialMesh(const ms::MicrostructureOptions& options) {
    if (options.mesh.type == "auto") {
        mfem::Mesh mesh = exaconstit::MakeAutoHexMesh(options.mesh.nxyz, options.mesh.mxyz);
        exaconstit::ApplyAutoMeshBoundaryAttributes(mesh);
        mesh.SetAttributes(); // refresh the list of boundary attributes after the remap
        return mesh;
    }
    if (options.mesh.type == "file") {
        return mfem::Mesh(options.mesh.floc.string().c_str(), 1, 1, true);
    }
    throw std::invalid_argument(
        "the generator tool reads [Mesh] type = \"file\" or \"auto\"; type = \"" +
        options.mesh.type + "\" input (an existing parallel decomposition) is not supported yet");
}

/*
 * ComputeStatistics (collective): ComputeGrainStatistics over the face adjacency the driver's
 * fragment cleanup already built (GenerationReport::adjacency; with periodic face pairing when
 * any axis is periodic, so a grain that wraps around the box counts its wrapped faces), so the
 * face-neighbor exchange and the periodic gather are not repeated. Every number is a fixed-point
 * sum, so the grain table and adjacency table are byte-identical for every rank count. The pair
 * table is gathered on rank 0, which writes the files.
 */
ms::GrainStatistics ComputeStatistics(const ms::GenerationReport& report,
                                      const ms::MicrostructureResult& result) {
    if (!report.adjacency) {
        // The driver always fills it; identical on every rank, so all ranks throw together
        throw std::logic_error("the driver report carries no mesh adjacency");
    }
    return ms::ComputeGrainStatistics(*report.adjacency, result.element_grain, result.num_grains,
                                      0);
}

/*
 * BuildGrainTable: the grain table columns. Orientation, region and source id come from the
 * result; element count, volume, centroid and neighbor count from the statistics; the target
 * volume column (the target of the seed each grain grew from) only when size targets were used.
 */
ms::GrainTable BuildGrainTable(const ms::MicrostructureResult& result,
                               const ms::GenerationReport& report,
                               const ms::GrainStatistics& stats) {
    ms::GrainTable table = ms::MakeGrainTable(result);
    table.num_elements = stats.num_elements;
    table.volume = stats.volume;
    table.centroid = stats.centroid;
    table.num_neighbors = stats.num_neighbors;
    if (!report.seed_target_volume.empty()) {
        table.target_volume.resize(report.grain_seed.size());
        for (std::size_t g = 0; g < report.grain_seed.size(); ++g) {
            table.target_volume[g] =
                report.seed_target_volume[static_cast<std::size_t>(report.grain_seed[g])];
        }
    }
    return table;
}

/** The pair columns of the statistics as the writer's adjacency table. */
ms::GrainAdjacencyTable BuildAdjacencyTable(const ms::GrainStatistics& stats) {
    ms::GrainAdjacencyTable table;
    table.grain_a = stats.pair_a;
    table.grain_b = stats.pair_b;
    table.shared_area = stats.pair_area;
    table.num_faces = stats.pair_faces;
    return table;
}

/** "first -> last" of a non-empty history, or "n/a". */
template <typename T>
std::string FirstToLast(const std::vector<T>& history) {
    if (history.empty()) {
        return "n/a";
    }
    std::ostringstream os;
    os << history.front() << " -> " << history.back();
    return os.str();
}

/** One line summarizing a volume-fit history (FitWeights; also each centroidal power refit). */
std::string DescribeFit(const ms::WeightFitReport& fit) {
    std::ostringstream os;
    os << fit.stop_reason << " after " << fit.iterations << " step(s) (" << fit.evaluations
       << " evaluations); max relative volume error on the fit samples "
       << FirstToLast(fit.max_relative_error) << "; dual objective Phi " << FirstToLast(fit.phi);
    if (!fit.num_empty.empty()) {
        os << "; empty cells at the end " << fit.num_empty.back();
    }
    os << "; sample resolution floor " << fit.resolution_floor;
    return os.str();
}

/*
 * RequestedGrains: the requested grain count for the log and the mesh header. With Johnson-Mehl
 * nucleation the count is not requested but follows from the nucleation (num_nuclei is only an
 * optional cap; 0 means no cap), so say that instead of printing "0 requested".
 */
std::string RequestedGrains(const ms::GenerationReport& report) {
    if (!report.used_johnson_mehl) {
        return std::to_string(report.seeding.requested) + " requested";
    }
    if (report.seeding.requested > 0) {
        return "set by Johnson-Mehl nucleation (at most " +
               std::to_string(report.seeding.requested) + " nuclei)";
    }
    return "set by Johnson-Mehl nucleation (no nucleus limit)";
}

/**
 * SeedingText: the seeding report for the log and the mesh header: the Johnson-Mehl report for
 * nucleation runs (the generic seeding summary would show "N of 0 requested"), the seed
 * generator's summary otherwise.
 */
std::string SeedingText(const ms::GenerationReport& report) {
    if (report.used_johnson_mehl) {
        return report.johnson_mehl.Summary();
    }
    return report.seeding.Summary();
}

/*
 * DescribeStages: rank-0 text for every optional stage that ran after seeding (the seeding or
 * Johnson-Mehl report itself is printed by SeedingText), from the driver's report, plus
 * the element-level comparison of seed volumes with their targets and the grain statistics.
 * Stages that did not run print nothing (the fragment cleanup always runs).
 */
std::string DescribeStages(const ms::MicrostructureOptions& options,
                           const ms::GenerationReport& report,
                           const ms::GrainStatistics& stats) {
    std::ostringstream os;
    const ms::TessellationOptions& tess = options.tessellation;

    // Size targets: one target volume per seed
    if (!report.seed_target_volume.empty()) {
        os << "size targets: " << report.seed_target_volume.size() << " seed target volumes ("
           << options.seeds.size.distribution << " distribution), summing to the material volume "
           << report.material_volume;
        if (report.used_polydisperse_placement) {
            os << "; polydisperse placement, largest first";
        }
        os << "\n";
    }

    // Lloyd relaxation (and the refits of centroidal power mode)
    if (report.ran_lloyd) {
        const ms::LloydReport& lloyd = report.lloyd;
        int clamped = 0;
        for (const int c : lloyd.num_clamped) {
            clamped += c;
        }
        os << "Lloyd relaxation: " << tess.lloyd_iterations << " step(s); CVT energy "
           << FirstToLast(lloyd.energy);
        if (!lloyd.max_displacement.empty()) {
            os << "; largest seed move in the last step " << lloyd.max_displacement.back();
        }
        os << "; " << clamped << " seed move(s) clamped back into the material";
        if (!lloyd.num_empty.empty()) {
            os << "; empty cells in the last step " << lloyd.num_empty.back();
        }
        os << "\n";
        if (!lloyd.fits.empty()) {
            os << "  centroidal power: " << lloyd.fits.size() << " refit(s); last refit: "
               << DescribeFit(lloyd.fits.back()) << "\n";
        }
    }

    // Volume fit (sample level)
    if (report.ran_weight_fit) {
        os << "volume fit (" << tess.fit_volumes.samples
           << " samples): " << DescribeFit(report.weight_fit) << "\n";
    }

    // Element level: the final grains are whole elements, so on coarse meshes their volumes
    // deviate from the targets by more than the fit tolerance reached on the fit samples
    if (!report.seed_target_volume.empty()) {
        const ms::TargetVolumeComparison cmp = ms::CompareWithTargetVolumes(
            stats.volume, report.grain_seed, report.seed_target_volume);
        os << "element-level volume error vs targets (final grains, summed per seed): max "
           << cmp.max_relative_error << " (seed " << cmp.worst_seed << "), mean "
           << cmp.mean_relative_error << "; " << cmp.num_empty_seeds
           << " seed(s) without elements\n";
    }

    // Periodic images and face pairing
    if (report.periodic_layers > 0) {
        os << "periodic: axes [" << (tess.periodic[0] ? "x" : "-")
           << (tess.periodic[1] ? "y" : "-") << (tess.periodic[2] ? "z" : "-") << "], "
           << report.periodic_layers << " image layer(s), " << report.num_periodic_images
           << " seed images; face pairing: " << report.periodic_faces.plane_faces
           << " faces on periodic planes, " << report.periodic_faces.matched_pairs
           << " matched pairs, " << report.periodic_faces.unmatched_faces << " unmatched\n";
    }

    // Quadrature vote
    if (tess.sampling == ms::SamplingMode::QUADRATURE_VOTE) {
        os << "quadrature vote: " << report.num_split_elements
           << " element(s) cut by a grain boundary (decided by majority volume)\n";
    }

    // Fragment cleanup (always runs; with fragments = "none" it only reports what it found)
    os << report.cleanup.Summary() << "\n";

    // Grain statistics (the same numbers as the grain table)
    if (stats.num_grains > 0) {
        std::int64_t min_elements = stats.num_elements[0];
        std::int64_t max_elements = stats.num_elements[0];
        int min_neighbors = stats.num_neighbors[0];
        int max_neighbors = stats.num_neighbors[0];
        double neighbor_sum = 0.0;
        for (std::size_t g = 0; g < static_cast<std::size_t>(stats.num_grains); ++g) {
            min_elements = std::min(min_elements, stats.num_elements[g]);
            max_elements = std::max(max_elements, stats.num_elements[g]);
            min_neighbors = std::min(min_neighbors, stats.num_neighbors[g]);
            max_neighbors = std::max(max_neighbors, stats.num_neighbors[g]);
            neighbor_sum += stats.num_neighbors[g];
        }
        os << "grain statistics: elements per grain " << min_elements << " - " << max_elements
           << ", neighbors per grain " << min_neighbors << " - " << max_neighbors << " (mean "
           << neighbor_sum / stats.num_grains << "), " << stats.pair_a.size()
           << " touching grain pairs, material volume " << stats.total_volume << "\n";
    }
    return os.str();
}

/** Paths of everything written, for the summary. */
struct WrittenFiles {
    fs::path serial_mesh;
    std::string parallel_prefix;
    int num_parts = 0;
    fs::path orientation_file;
    fs::path grain_file;
    fs::path region_map;
    fs::path grain_table;
    fs::path grain_adjacency;
    fs::path paraview;
};

/*
 * PrintSummary: rank-0 end-of-run log. Provenance first, then the stage reports (DescribeStages),
 * then the files, then how to reference
 * them from an ExaConstit options file (key names from src/options: [Mesh] type/floc,
 * [Mesh.Auto] length/ncuts, top-level grain_file and region_mapping_file, and
 * [Materials.Grain] ori_type/num_grains/orientation_file/ori_stride). No TOML is written.
 */
void PrintSummary(const ms::MicrostructureOptions& options,
                  const ms::GenerationReport& report,
                  const ms::GrainStatistics& stats,
                  const WrittenFiles& files,
                  int num_regions) {
    std::ostringstream os;
    os << "\n=== microstructure_generator summary ===\n"
       << "ExaConstit version " << VersionString() << "\n"
       << "input mesh: " << DescribeMeshInput(options) << " (" << report.num_elements
       << " elements)\n"
       << "rng_seed = " << options.seeds.rng_seed << ", metric = "
       << MetricName(options.tessellation.metric) << "\n"
       << "grains: " << RequestedGrains(report) << ", " << report.num_seeds
       << " seeds placed, " << report.num_grains << " final grains";
    if (report.num_empty_seeds > 0) {
        os << " (" << report.num_empty_seeds << " seeds won no element)";
    }
    os << "\n" << SeedingText(report);
    os << DescribeStages(options, report, stats);

    os << "Files written:\n";
    if (!files.serial_mesh.empty()) {
        os << "  serial mesh:       " << files.serial_mesh.string() << "\n";
    }
    if (files.num_parts > 0) {
        os << "  parallel mesh:     " << files.num_parts << " parts, prefix "
           << files.parallel_prefix << " (part p = prefix + p zero-padded to 6 digits)\n";
    }
    os << "  orientations:      " << files.orientation_file.string() << "\n";
    if (!files.grain_file.empty()) {
        os << "  grain file:        " << files.grain_file.string() << "\n";
    }
    if (!files.region_map.empty()) {
        os << "  region map:        " << files.region_map.string() << "\n";
    }
    os << "  grain table:       " << files.grain_table.string() << "\n";
    if (!files.grain_adjacency.empty()) {
        os << "  grain adjacency:   " << files.grain_adjacency.string() << "\n";
    }
    if (!files.paraview.empty()) {
        os << "  ParaView:          " << files.paraview.string() << "\n";
    }

    os << "\nTo use them in your own ExaConstit options file (the paths below are relative to "
          "the directory this tool ran in; adjust them to where your ExaConstit run starts):\n";
    if (!files.serial_mesh.empty()) {
        os << "  [Mesh] type = \"file\", floc = \"" << files.serial_mesh.string() << "\"\n";
    }
    if (files.num_parts > 0) {
        os << "  or [Mesh] type = \"parallel\", floc = \"" << files.parallel_prefix
           << "\" (run with exactly " << files.num_parts << " MPI ranks)\n";
    }
    if (options.mesh.type == "auto" && !files.grain_file.empty()) {
        os << "  or [Mesh] type = \"auto\" with [Mesh.Auto] length = [" << options.mesh.mxyz[0]
           << ", " << options.mesh.mxyz[1] << ", " << options.mesh.mxyz[2] << "], ncuts = ["
           << options.mesh.nxyz[0] << ", " << options.mesh.nxyz[1] << ", "
           << options.mesh.nxyz[2] << "] and top-level grain_file = \""
           << files.grain_file.string() << "\"\n";
    }
    os << "  [Materials.Grain] ori_type = \"quat\", ori_stride = 4, num_grains = "
       << report.num_grains << ", orientation_file = \"" << files.orientation_file.string()
       << "\"\n";
    if (num_regions > 1) {
        os << "  top-level region_mapping_file = \"" << files.region_map.string() << "\" and "
           << num_regions << " [[Materials]] blocks, in region order\n";
    } else {
        os << "  one [[Materials]] block (single region; no region mapping file needed)\n";
    }
    os << "  Mesh refinement (ref_ser/ref_par) in ExaConstit keeps the grain ids of each "
          "parent element.\n";
    std::cout << os.str() << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    // ---- 1. MPI and command line -------------------------------------------------------------
    mfem::Mpi::Init(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mfem::Mpi::WorldRank();
    const int nranks = mfem::Mpi::WorldSize();
    const bool root = (rank == 0);

    const CommandLine cl = ParseCommandLine(argc, argv);
    if (cl.help || !cl.error.empty()) {
        // Every rank parsed the same argv, so every rank takes this branch together
        if (root) {
            if (!cl.error.empty()) {
                std::cerr << "microstructure_generator: " << cl.error << "\n";
            }
            PrintUsage(cl.help ? std::cout : std::cerr);
        }
        return cl.help ? 0 : 1;
    }

    // ---- 2. Options: file (or defaults), overrides, validation (collective verdicts) ---------
    ms::MicrostructureOptions options;
    {
        bool ok = true;
        std::string message;
        if (cl.input) {
            try {
                options = ms::MicrostructureOptions::parse_options_file(*cl.input);
            } catch (const std::exception& e) {
                ok = false;
                message = "cannot read options file '" + cl.input->string() + "': " + e.what();
            }
        }
        if (!AgreeOrReport(ok, message, comm)) {
            return 1;
        }
    }
    options.apply_cli_overrides(cl.overrides);
    // validate() prints each problem on rank 0; the MIN-reduction stops all ranks together even
    // if the file system looked different on some rank. AgreeOrReport lets the lowest failing
    // rank speak: if that is rank 0, its problems are already printed above; if rank 0 passed
    // (so printed nothing) and another rank failed, that rank prints its own collected problems.
    {
        std::vector<std::string> problems;
        const bool ok = options.validate(&problems);
        std::string message = "invalid options (see the messages above)";
        if (!root) {
            message = "invalid options on this rank (rank 0 accepted them; check that every rank "
                      "sees the same files):";
            for (const std::string& problem : problems) {
                message += "\n  " + problem;
            }
        }
        if (!AgreeOrReport(ok, message, comm)) {
            return 1;
        }
    }

    // ---- 3. Backend (mfem::Device), as ExaConstit's mechanics driver configures it ----------
    mfem::Device device;
    {
        bool ok = true;
        std::string message;
        try {
            ms::ConfigureMfemDevice(device, options.exec);
        } catch (const std::exception& e) {
            ok = false;
            message = e.what();
        }
        if (!AgreeOrReport(ok, message, comm)) {
            return 1;
        }
    }

    // ---- 4. Serial mesh on every rank --------------------------------------------------------
    std::unique_ptr<mfem::Mesh> serial;
    {
        bool ok = true;
        std::string message;
        try {
            serial = std::make_unique<mfem::Mesh>(ReadSerialMesh(options));
        } catch (const std::exception& e) {
            ok = false;
            message = std::string("cannot build the mesh: ") + e.what();
        }
        if (!AgreeOrReport(ok, message, comm)) {
            return 1;
        }
    }

    try {
        // ---- 5. Partition and generate -------------------------------------------------------
        // BuildSerialOrder runs METIS once on rank 0 and broadcasts the partitioning; the
        // ParMesh keeps serial order within each rank, so local element i is serial element
        // order.local_to_serial[i] (mesh/serial_order.hpp).
        const ms::SerialOrder order = ms::BuildSerialOrder(*serial, comm);
        std::unique_ptr<mfem::ParMesh> pmesh = ms::MakeParMesh(comm, *serial, order);

        ms::GenerationReport report;
        ms::MicrostructureResult result = ms::GenerateMicrostructure(options, *pmesh, comm,
                                                                     &report);

        // Coarse-resolution warning: a grain needs many elements to deform realistically
        const double elements_per_grain =
            double(report.num_elements) / double(std::max(1, report.num_grains));
        if (root && elements_per_grain < MIN_ELEMENTS_PER_GRAIN) {
            std::cout << "Warning: only " << elements_per_grain
                      << " elements per grain on average (fewer than about "
                      << MIN_ELEMENTS_PER_GRAIN
                      << " resolves grains poorly); use a finer mesh or fewer grains.\n";
        }

        // ---- 6. Grain statistics (collective; fixed-point sums, rank-count independent) -------
        const ms::GrainStatistics stats = ComputeStatistics(report, result);

        // ---- 7. Gather labels and write outputs ----------------------------------------------
        // The grain ids of the local elements, gathered on rank 0 in serial element order
        // (GatherToSerial reads the host copy, so make it current first)
        result.element_grain.HostRead();
        std::vector<int> serial_grain = ms::GatherToSerial(comm, order, result.element_grain, 0);

        const ms::OutputOptions& out = options.output;
        const bool write_serial = (out.mesh_format == ms::MeshFormat::SERIAL ||
                                   out.mesh_format == ms::MeshFormat::BOTH);
        const bool write_parallel = (out.mesh_format == ms::MeshFormat::PARALLEL ||
                                     out.mesh_format == ms::MeshFormat::BOTH);

        // The partitioned writer is collective and needs the labeled serial mesh on every rank,
        // so broadcast the serial labels in that case
        const int serial_ne = serial->GetNE();
        if (write_parallel) {
            serial_grain.resize(static_cast<std::size_t>(serial_ne));
            MPI_Bcast(serial_grain.data(), serial_ne, MPI_INT, 0, comm);
        }
        const bool have_labels = root || write_parallel;
        if (have_labels) {
            for (int e = 0; e < serial_ne; ++e) {
                serial->SetAttribute(e, serial_grain[static_cast<std::size_t>(e)]);
            }
            serial->SetAttributes(); // refresh the attribute list (1..num_grains)
        }

        // Output names: <directory>/<basename>.<extension>
        const fs::path base = out.directory / out.basename;
        if (root) {
            fs::create_directories(out.directory);
        }
        MPI_Barrier(comm);

        // Provenance in the mesh header comments (each line becomes a "# ..." comment)
        std::ostringstream comments;
        comments << "Generated by ExaConstit microstructure_generator, ExaConstit version "
                 << VersionString() << "\n"
                 << "command: " << cl.text << "\n"
                 << "input mesh: " << DescribeMeshInput(options) << "\n"
                 << "rng_seed: " << options.seeds.rng_seed << "\n"
                 << "metric: " << MetricName(options.tessellation.metric) << "\n"
                 << "grains: " << RequestedGrains(report) << "\n"
                 << "final grains: " << report.num_grains << "\n"
                 << SeedingText(report);

        WrittenFiles files;
        if (write_serial && root) {
            files.serial_mesh = ms::WriteSerialMesh(*serial, fs::path(base.string() + ".mesh"),
                                                    out.compression, comments.str());
        }
        if (write_parallel) {
            files.num_parts = (out.num_parts > 0) ? out.num_parts : nranks;
            files.parallel_prefix = base.string() + ".mesh.";
            ms::WritePartitionedMesh(*serial, files.num_parts, files.parallel_prefix,
                                     out.compression, comm);
        }

        // Per-grain text files (rank 0): orientations always; region map only with >1 region;
        // grain file for auto input ("auto") or when forced ("true")
        files.orientation_file = base.string() + ".ori";
        const bool want_grain_file =
            (out.grain_file == "true") || (out.grain_file == "auto" && options.mesh.type == "auto");
        if (want_grain_file) {
            files.grain_file = base.string() + ".grains";
        }
        if (result.num_regions > 1) {
            files.region_map = base.string() + ".regions";
        }
        files.grain_table = base.string() + "_grains.csv";
        if (out.grain_adjacency) {
            files.grain_adjacency = base.string() + "_adjacency.csv";
        }
        if (root) {
            ms::WriteOrientationFile(files.orientation_file, result.grain_quat);
            if (want_grain_file) {
                ms::WriteGrainFile(files.grain_file, serial_grain);
            }
            if (result.num_regions > 1) {
                ms::WriteRegionMap(files.region_map, result.grain_region);
            }
            ms::WriteGrainTable(files.grain_table, BuildGrainTable(result, report, stats));
            if (out.grain_adjacency) {
                ms::WriteGrainAdjacency(files.grain_adjacency, BuildAdjacencyTable(stats));
            }
        }

        // ParaView view of the distributed mesh (collective); the region field only when there
        // is more than one region
        if (out.paraview) {
            files.paraview = base.string() + "_paraview";
            std::vector<int> regions_for_view;
            if (result.num_regions > 1) {
                regions_for_view = result.grain_region;
            }
            ms::WriteParaView(*pmesh, result.element_grain, regions_for_view, files.paraview);
        }
        MPI_Barrier(comm);

        // ---- 8. Summary ----------------------------------------------------------------------
        if (root) {
            PrintSummary(options, report, stats, files, result.num_regions);
        }
    } catch (const std::exception& e) {
        // Every rank that caught an error prints it with its rank number, then stops all ranks.
        // Library errors are normally thrown on all ranks together (then each rank prints the
        // same line), but some failures are local to one rank, e.g. writing its share of the
        // partitioned mesh or ParaView files; printing only on rank 0 would hide those.
        std::cerr << "microstructure_generator: error (rank " << rank << "): " << e.what()
                  << std::endl;
        MPI_Abort(comm, 1);
    }
    return 0;
}
