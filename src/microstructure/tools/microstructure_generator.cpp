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
 *     file, region map (whenever there is more than one region), grain table <base>_grains.csv
 *     (with phase, filler and voxel feature id columns when they carry information), optional
 *     grain adjacency table
 *     <base>_adjacency.csv ([Output] grain_adjacency), ParaView view.
 *  8. Rank-0 summary: provenance (including the grain source: generated seeds, a seed table or
 *     a voxel map, with its file and format), how the grains were made (seed generator,
 *     Johnson-Mehl nucleation, seed table with its region of interest and filler seeds, or voxel
 *     map grid, fill and read counts), what each optional stage did (size targets, Lloyd
 *     relaxation, volume fit with the element-level volume errors against the targets, periodic
 *     images, quadrature vote, fragment cleanup), grain statistics, phase fractions and regions,
 *     passthrough parts, and how to point an ExaConstit input at the outputs: the [[Materials]]
 *     blocks in region order with the phases, crystallinity and passthrough parts of each. No
 *     ExaConstit input file is generated.
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
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mpi.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
 * result, and so do the phase and filler columns when they carry information (MakeGrainTable:
 * phases only when some grain has one, the filler flag only when some grain is a filler grain);
 * element count, volume, centroid and neighbor count come from the statistics; the target volume
 * column (the target of the seed each grain grew from) only when size targets were used, with 0
 * for grains that did not grow from a seed (passthrough grains, grain_seed = -1); the feature id
 * column (the grain id in the voxel file, -1 for passthrough grains) only for voxel maps.
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
            const int seed = report.grain_seed[g];
            table.target_volume[g] =
                (seed >= 0) ? report.seed_target_volume[static_cast<std::size_t>(seed)] : 0.0;
        }
    }
    if (report.used_voxel_map) {
        table.feature_id = report.grain_feature;
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

/** "(x, y, z)" of a 3-vector, for logs. */
std::string Triple(const std::array<double, 3>& v) {
    std::ostringstream os;
    os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    return os.str();
}

/** Comma-separated list of integers ("none" when empty), for logs. */
template <typename T>
std::string JoinList(const std::vector<T>& values) {
    if (values.empty()) {
        return "none";
    }
    std::ostringstream os;
    for (std::size_t k = 0; k < values.size(); ++k) {
        os << (k > 0 ? ", " : "") << values[k];
    }
    return os.str();
}

/**
 * DescribeGrainSource: one line naming where the grains come from, for the log and the mesh
 * header provenance: the voxel map file and its format, the seed table file, Johnson-Mehl
 * nucleation, or generated seeds.
 */
std::string DescribeGrainSource(const ms::MicrostructureOptions& options,
                                const ms::GenerationReport& report) {
    if (report.used_voxel_map) {
        return "voxel map " + options.voxel_map.file.string() + " (format \"" +
               options.voxel_map.format + "\")";
    }
    if (report.used_seed_table) {
        return "seed table " + options.seeds.file.string() +
               " (format \"ExaConstit seed table v1\")";
    }
    if (report.used_johnson_mehl) {
        return "Johnson-Mehl nucleation";
    }
    return "generated seeds";
}

/*
 * RequestedGrains: the requested grain count for the log and the mesh header.
 * - Johnson-Mehl: the count is not requested but follows from the nucleation (num_nuclei is only
 *   an optional cap; 0 means no cap), so say that instead of printing "0 requested".
 * - Voxel map: the grains are the features of the map that the mesh touches; nothing is
 *   requested.
 * - Seed table: the table seeds kept inside the ROI, plus the filler seeds asked for (given, or
 *   the automatic count).
 */
std::string RequestedGrains(const ms::GenerationReport& report) {
    if (report.used_voxel_map) {
        return "read from the voxel map";
    }
    if (report.used_seed_table) {
        std::string text = std::to_string(report.num_table_seeds) + " from the seed table";
        if (report.used_filler) {
            text += " + " + std::to_string(report.num_filler_requested) + " filler requested";
        }
        return text;
    }
    if (!report.used_johnson_mehl) {
        return std::to_string(report.seeding.requested) + " requested";
    }
    if (report.seeding.requested > 0) {
        return "set by Johnson-Mehl nucleation (at most " +
               std::to_string(report.seeding.requested) + " nuclei)";
    }
    return "set by Johnson-Mehl nucleation (no nucleus limit)";
}

/** Prefixes every line of `text` with `indent` (for nesting a sub-report in the log). */
std::string Indent(const std::string& text, const std::string& indent) {
    std::ostringstream os;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        os << indent << line << "\n";
    }
    return os.str();
}

/*
 * SeedTableText: the seed-table section of the log (Seeds.source = "file"): rows read, rows
 * dropped because they lie outside the region of interest (ROI), the optional columns found and
 * the ignored ones; then the ROI box and, with random filler seeds, the material volumes inside
 * and outside the ROI, the filler seeds requested and placed, and the filler seed generator's own
 * report (indented).
 */
std::string SeedTableText(const ms::MicrostructureOptions& options,
                          const ms::GenerationReport& report) {
    std::ostringstream os;
    const ms::SeedOptions& seeds = options.seeds;
    os << "seed table: " << seeds.file.string() << " (units_scale " << seeds.units_scale
       << ", translate " << Triple(seeds.translate) << "): " << report.num_table_rows
       << " rows read, " << report.num_table_dropped << " dropped outside the ROI, "
       << report.num_table_seeds << " used\n";

    // Optional columns present in the table (x, y, z and id are always there or implied)
    std::vector<std::string> columns;
    if (report.table_has_quat) {
        columns.push_back("orientation (qw, qx, qy, qz)");
    }
    if (report.table_has_phase) {
        columns.push_back("phase");
    }
    if (report.table_has_weight) {
        columns.push_back("weight");
    }
    if (report.table_has_volume) {
        columns.push_back("size");
    }
    if (report.table_has_shape) {
        columns.push_back("shape tensor");
    }
    os << "  optional columns: " << JoinList(columns) << "; ignored columns: "
       << JoinList(report.table_ignored_columns) << "\n";

    // Region of interest and filler seeds
    if (report.used_roi) {
        os << "region of interest: " << Triple(report.roi_lo) << " - " << Triple(report.roi_hi)
           << " (mesh units)\n";
    }
    if (report.used_filler) {
        os << "filler seeds outside the ROI: material volume inside the ROI "
           << report.roi_material_volume << ", outside " << report.outside_material_volume
           << "; " << report.num_filler_requested << " requested ("
           << (seeds.filler_num_seeds > 0 ? "given"
                                          : "automatic: same mean grain volume as the ROI")
           << "), " << report.num_filler_seeds << " placed, source ids from "
           << report.filler_source_id_offset << "; " << report.num_filler_grains
           << " final filler grains\n";
        os << Indent(report.filler_seeding.Summary(), "  ");
    }
    return os.str();
}

/*
 * VoxelMapText: the voxel-map section of the log ([Source] type = "voxel_map"): the file, the
 * id dataset used, the grid (dimensions, and origin and spacing in mesh units after
 * units_scale/translate), the sampling, the elements filled from their neighbors (no voxel vote:
 * holes of unassigned voxels, or parts of the mesh outside the grid), the sample points outside
 * the grid, and (with_read_counts) the voxel tiles and voxels read by all ranks together. A tile
 * is a cube of tile_size^3 voxels ([VoxelMap] tile_size) that a rank reads from the file in one
 * piece; only the tiles its elements touch are read. The
 * read counts depend on the rank count (every rank reads the tiles its own elements touch, and
 * neighboring ranks read shared tiles twice), so they go to the log only, never into the mesh
 * header, which must be identical for every rank count.
 */
std::string VoxelMapText(const ms::MicrostructureOptions& options,
                         const ms::GenerationReport& report,
                         bool with_read_counts) {
    std::ostringstream os;
    const ms::VoxelMapOptions& voxel = options.voxel_map;
    const ms::VoxelGridInfo& grid = report.voxel_grid;
    os << "voxel map: " << voxel.file.string() << " (format \"" << voxel.format
       << "\", ids from " << grid.feature_ids_path << ")\n";
    os << "  grid " << grid.dims[0] << " x " << grid.dims[1] << " x " << grid.dims[2]
       << " voxels, origin " << Triple(grid.origin) << ", spacing " << Triple(grid.spacing)
       << " (mesh units)\n";
    os << "  sampling: "
       << (voxel.sampling == ms::SamplingMode::QUADRATURE_VOTE
               ? "quadrature_vote (vote_levels " + std::to_string(voxel.vote_levels) + ")"
               : std::string("centroid"))
       << "; " << report.voxel_num_filled << " element(s) filled from their neighbors ("
       << report.voxel_fill_rounds << " fill round(s)); " << report.voxel_num_outside_points
       << " sample point(s) outside the grid\n";
    if (with_read_counts) {
        os << "  read by all ranks: " << report.voxel_total_tiles_read << " tile(s), "
           << report.voxel_total_voxels_read
           << " voxel(s) (a tile is a cube of tile_size^3 voxels read from the file in one piece; "
              "these counts depend on the rank count)\n";
    }
    return os.str();
}

/**
 * SourceText: the report of how the grains were made, for the log and the mesh header: the
 * voxel-map section for voxel maps (the generic seeding summary would be all zeros), the
 * seed-table / ROI / filler section for seed tables, the Johnson-Mehl report for nucleation runs
 * (the generic seeding summary would show "N of 0 requested"), and the seed generator's summary
 * otherwise. `for_log` false (the mesh header) leaves out the rank-count-dependent voxel read
 * counts.
 */
std::string SourceText(const ms::MicrostructureOptions& options,
                       const ms::GenerationReport& report,
                       bool for_log) {
    if (report.used_voxel_map) {
        return VoxelMapText(options, report, for_log);
    }
    if (report.used_seed_table) {
        return SeedTableText(options, report);
    }
    if (report.used_johnson_mehl) {
        return report.johnson_mehl.Summary();
    }
    return report.seeding.Summary();
}

/**
 * Grain id of passthrough attribute number k (0-based, ascending attribute order): the
 * passthrough grains are always the last grains, num_grains - P + 1 .. num_grains.
 */
int PassthroughGrainId(const ms::GenerationReport& report,
                       const ms::MicrostructureResult& result,
                       std::size_t k) {
    return result.num_grains - report.num_passthrough_grains + 1 + static_cast<int>(k);
}

/*
 * RegionContents: what region r (1-based) holds, in words, for the region section and the
 * [[Materials]] guidance: its phases (MicrostructureResult::region_phases), whether it is
 * crystalline (region_crystalline), the passthrough attributes mapped to it (with their grain
 * ids), and whether filler grains are in it.
 */
std::string RegionContents(const ms::GenerationReport& report,
                           const ms::MicrostructureResult& result,
                           int r) {
    const auto ri = static_cast<std::size_t>(r - 1);
    std::ostringstream os;
    // Phases
    if (ri < result.region_phases.size() && !result.region_phases[ri].empty()) {
        const std::vector<int>& phases = result.region_phases[ri];
        os << (phases.size() > 1 ? "phases " : "phase ") << JoinList(phases);
    } else {
        os << "no phase data";
    }
    // Crystallinity
    const bool crystalline =
        ri < result.region_crystalline.size() && result.region_crystalline[ri] != 0;
    os << "; " << (crystalline ? "crystalline" : "not crystalline (identity orientations)");
    // Passthrough parts of the mesh in this region
    for (std::size_t k = 0; k < report.passthrough_attributes.size(); ++k) {
        if (k < report.passthrough_regions.size() && report.passthrough_regions[k] == r) {
            os << "; passthrough attribute " << report.passthrough_attributes[k] << " (grain "
               << PassthroughGrainId(report, result, k) << ")";
        }
    }
    // Filler grains in this region
    int filler = 0;
    for (std::size_t g = 0; g < result.grain_filler.size(); ++g) {
        if (result.grain_filler[g] != 0 && result.grain_region[g] == r) {
            ++filler;
        }
    }
    if (filler > 0) {
        os << "; " << filler << " filler grain(s)";
    }
    return os.str();
}

/*
 * DescribePhasesAndRegions: rank-0 text for phases and regions, printed when the grains carry
 * phases or there is more than one region.
 *  - Phase volume fractions over the tessellated / voxel-labeled material (passthrough grains
 *    have no phase and are left out of the denominator), from the grain volumes of the
 *    statistics (fixed-point sums, so identical for every rank count). For
 *    Regions.mode = "phase_fractions" the requested fractions and the ones the assignment reached
 *    (GenerationReport::phase_fraction_achieved) are shown too.
 *  - Per region: grain count, volume, volume fraction of the whole mesh, and RegionContents.
 */
std::string DescribePhasesAndRegions(const ms::MicrostructureOptions& options,
                                     const ms::GenerationReport& report,
                                     const ms::MicrostructureResult& result,
                                     const ms::GrainStatistics& stats) {
    std::ostringstream os;
    bool any_phase = false;
    for (const int p : result.grain_phase) {
        any_phase = any_phase || (p >= 0);
    }
    if (!any_phase && result.num_regions <= 1) {
        return "";
    }
    const auto ng = static_cast<std::size_t>(result.num_grains);
    const bool have_volumes = stats.volume.size() == ng;

    // Phase volume fractions (active material only)
    if (any_phase && have_volumes) {
        std::map<int, std::pair<int, double>> phase_sum; // phase -> (grains, volume)
        double active_volume = 0.0;
        for (std::size_t g = 0; g < ng; ++g) {
            const bool passthrough =
                static_cast<int>(g) >= result.num_grains - report.num_passthrough_grains;
            if (passthrough) {
                continue;
            }
            active_volume += stats.volume[g];
            if (g < result.grain_phase.size() && result.grain_phase[g] >= 0) {
                phase_sum[result.grain_phase[g]].first += 1;
                phase_sum[result.grain_phase[g]].second += stats.volume[g];
            }
        }
        os << "phases (Regions.mode = \"" << options.regions.mode << "\"):\n";
        for (const auto& [phase, sum] : phase_sum) {
            os << "  phase " << phase << ": " << sum.first << " grain(s), volume fraction "
               << (active_volume > 0.0 ? sum.second / active_volume : 0.0) << "\n";
        }
        if (!report.phase_fraction_ids.empty()) {
            os << "  phase_fractions targets -> achieved:";
            for (std::size_t k = 0; k < report.phase_fraction_ids.size(); ++k) {
                const double target = (k < options.regions.phase_fractions.size())
                                          ? options.regions.phase_fractions[k]
                                          : 0.0;
                const double achieved = (k < report.phase_fraction_achieved.size())
                                            ? report.phase_fraction_achieved[k]
                                            : 0.0;
                os << (k > 0 ? "," : "") << " phase " << report.phase_fraction_ids[k] << " "
                   << target << " -> " << achieved;
            }
            os << "\n";
        }
    }

    // Regions
    os << "regions: " << result.num_regions << "\n";
    for (int r = 1; r <= result.num_regions; ++r) {
        int grains = 0;
        double volume = 0.0;
        for (std::size_t g = 0; g < ng; ++g) {
            if (result.grain_region[g] == r) {
                ++grains;
                volume += have_volumes ? stats.volume[g] : 0.0;
            }
        }
        os << "  region " << r << ": " << grains << " grain(s), volume " << volume
           << " (fraction " << (stats.total_volume > 0.0 ? volume / stats.total_volume : 0.0)
           << "); " << RegionContents(report, result, r) << "\n";
    }
    return os.str();
}

/*
 * DescribePassthrough: rank-0 text for the passthrough parts of the mesh ([Mesh]
 * passthrough_attributes): for each attribute its grain id, its region and its element count,
 * plus the number of active (tessellated or voxel-labeled) elements.
 */
std::string DescribePassthrough(const ms::GenerationReport& report,
                                const ms::MicrostructureResult& result) {
    if (report.passthrough_attributes.empty()) {
        return "";
    }
    std::ostringstream os;
    os << "passthrough parts (one grain each, identity orientation; active attributes "
       << JoinList(report.active_attributes) << " with " << report.num_active_elements
       << " elements):\n";
    for (std::size_t k = 0; k < report.passthrough_attributes.size(); ++k) {
        os << "  attribute " << report.passthrough_attributes[k] << " -> grain "
           << PassthroughGrainId(report, result, k) << " -> region "
           << (k < report.passthrough_regions.size() ? report.passthrough_regions[k] : 0) << " ("
           << (k < report.passthrough_element_counts.size() ? report.passthrough_element_counts[k]
                                                            : 0)
           << " elements)\n";
    }
    return os.str();
}

/*
 * DescribeStages: rank-0 text for every optional stage that ran after seeding (the seeding or
 * Johnson-Mehl report itself is printed by SourceText), from the driver's report, plus
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
        // Only grains grown from a seed have a target (passthrough grains have grain_seed -1)
        std::vector<double> seeded_volume;
        std::vector<int> seeded_seed;
        for (std::size_t g = 0; g < report.grain_seed.size(); ++g) {
            if (report.grain_seed[g] >= 0) {
                seeded_volume.push_back(stats.volume[g]);
                seeded_seed.push_back(report.grain_seed[g]);
            }
        }
        const ms::TargetVolumeComparison cmp =
            ms::CompareWithTargetVolumes(seeded_volume, seeded_seed, report.seed_target_volume);
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

    // Quadrature vote of the tessellation (voxel maps have their own sampling, reported by
    // VoxelMapText; the Tessellation options are not used there)
    if (!report.used_voxel_map && tess.sampling == ms::SamplingMode::QUADRATURE_VOTE) {
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
 * MaterialsGuidance: the ExaConstit setup guidance for the materials (no TOML is generated; the
 * user writes the deck). ExaConstit numbers its [[Materials]] blocks 1, 2, ... in file order,
 * and block k serves region k of the region map (option_parser_v2.cpp sets region_id from the
 * block order; simulation_state.cpp create_grains_to_map reads the "grain region" lines). So the
 * guidance lists the R blocks in region order, each with what it holds (RegionContents).
 *
 * Every crystal-plasticity block reads the same orientation file and must use the same
 * num_grains = N, the total grain count including passthrough grains: the orientation file has
 * one row per grain, and ExaConstit checks every block's num_grains for equality with the grain
 * count of the shared orientation data: SimulationState::LoadSharedOrientationData
 * (simulation_state.cpp:1439, "Grain count mismatch") when the data is already loaded, and
 * PrepareOrientationForRegion (simulation_state.cpp:1575, per material). The parser's
 * finalize_material_options (option_parser_v2.cpp:70) only checks that later blocks do not
 * exceed the first one, which is weaker; the equality checks are the binding ones.
 * Non-crystalline and passthrough grains carry the identity orientation (1, 0, 0, 0) in that
 * file, so a crystal-plasticity model may be used for them too, or any other material model that
 * ignores orientations.
 */
std::string MaterialsGuidance(const ms::GenerationReport& report,
                              const ms::MicrostructureResult& result,
                              const WrittenFiles& files) {
    std::ostringstream os;
    const std::string grain_line = "[Materials.Grain] ori_type = \"quat\", ori_stride = 4, "
                                   "num_grains = " +
                                   std::to_string(result.num_grains) + ", orientation_file = \"" +
                                   files.orientation_file.string() + "\"";
    if (result.num_regions <= 1) {
        os << "  one [[Materials]] block (single region; no region mapping file needed) with\n"
           << "    " << grain_line << "\n";
    } else {
        os << "  top-level region_mapping_file = \"" << files.region_map.string() << "\" and "
           << result.num_regions
           << " [[Materials]] blocks, in region order (block k is region k):\n";
        for (int r = 1; r <= result.num_regions; ++r) {
            os << "    block " << r << " (region " << r << "): "
               << RegionContents(report, result, r) << "\n";
        }
        os << "  every crystal-plasticity block uses\n"
           << "    " << grain_line << "\n"
           << "  (num_grains counts all grains";
        if (report.num_passthrough_grains > 0) {
            os << ", including the " << report.num_passthrough_grains << " passthrough grain(s)";
        }
        os << "; the orientation file is shared by all blocks)\n";
    }
    bool identity = report.num_passthrough_grains > 0;
    for (const char c : result.region_crystalline) {
        identity = identity || (c == 0);
    }
    if (identity) {
        os << "  non-crystalline and passthrough grains carry the identity orientation "
              "(1, 0, 0, 0) in the orientation file\n";
    }
    return os.str();
}

/*
 * PrintSummary: rank-0 end-of-run log. Provenance first (version, mesh, grain source, random
 * key, metric for tessellations, grain counts), then how the grains were made (SourceText), the
 * stage reports (DescribeStages), phases and regions (DescribePhasesAndRegions), passthrough
 * parts (DescribePassthrough), the data warnings the driver raised, the files, and finally how
 * to reference them from an ExaConstit options file (key names from src/options: [Mesh]
 * type/floc, [Mesh.Auto] length/ncuts, top-level grain_file and region_mapping_file, and
 * [Materials.Grain] ori_type/num_grains/orientation_file/ori_stride; MaterialsGuidance). No TOML
 * is written.
 */
void PrintSummary(const ms::MicrostructureOptions& options,
                  const ms::GenerationReport& report,
                  const ms::MicrostructureResult& result,
                  const ms::GrainStatistics& stats,
                  const WrittenFiles& files) {
    std::ostringstream os;
    os << "\n=== microstructure_generator summary ===\n"
       << "ExaConstit version " << VersionString() << "\n"
       << "input mesh: " << DescribeMeshInput(options) << " (" << report.num_elements
       << " elements)\n"
       << "grain source: " << DescribeGrainSource(options, report) << "\n"
       << "rng_seed = " << options.seeds.rng_seed;
    if (!report.used_voxel_map) {
        os << ", metric = " << MetricName(options.tessellation.metric);
    }
    os << "\ngrains: " << RequestedGrains(report);
    if (!report.used_voxel_map) {
        os << ", " << report.num_seeds << " seeds placed";
    }
    os << ", " << report.num_grains << " final grains";
    if (report.num_passthrough_grains > 0) {
        os << " (including " << report.num_passthrough_grains << " passthrough grain(s))";
    }
    if (report.num_empty_seeds > 0) {
        os << " (" << report.num_empty_seeds << " seeds won no element)";
    }
    os << "\n" << SourceText(options, report, true);
    os << DescribeStages(options, report, stats);
    os << DescribePhasesAndRegions(options, report, result, stats);
    os << DescribePassthrough(report, result);
    for (const std::string& warning : report.warnings) {
        os << "data warning: " << warning << "\n";
    }

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
           << files.grain_file.string() << "\"";
        if (result.num_regions > 1) {
            os << " (required with several materials)";
        }
        os << "\n";
    }
    os << MaterialsGuidance(report, result, files);
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
                 << "grain source: " << DescribeGrainSource(options, report) << "\n"
                 << "rng_seed: " << options.seeds.rng_seed << "\n";
        if (!report.used_voxel_map) {
            comments << "metric: " << MetricName(options.tessellation.metric) << "\n";
        }
        comments << "grains: " << RequestedGrains(report) << "\n"
                 << "final grains: " << report.num_grains << "\n"
                 << "regions: " << result.num_regions << "\n"
                 << SourceText(options, report, false);

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
            PrintSummary(options, report, result, stats, files);
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
