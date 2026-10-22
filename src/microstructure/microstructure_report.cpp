#include "microstructure/microstructure_report.hpp"

#include <cstddef>
#include <sstream>

/*
 * The record tables and log phrases of a generated microstructure (see the header). These used to
 * live in tools/microstructure_generator.cpp; the code is unchanged, so the tool's files and log
 * text stay byte for byte the same.
 */

namespace exaconstit::microstructure {

namespace {

/** Comma-separated list of integers ("none" when empty), for logs. */
std::string JoinIntegers(const std::vector<int>& values) {
    if (values.empty()) {
        return "none";
    }
    std::ostringstream os;
    for (std::size_t k = 0; k < values.size(); ++k) {
        os << (k > 0 ? ", " : "") << values[k];
    }
    return os.str();
}

} // namespace

/*
 * BuildGrainTable: MakeGrainTable for the result columns, then the statistics columns, then the
 * optional target-volume (per parent seed) and feature-id columns.
 */
GrainTable BuildGrainTable(const MicrostructureResult& result,
                           const GenerationReport& report,
                           const GrainStatistics& stats) {
    GrainTable table = MakeGrainTable(result);
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

/* BuildAdjacencyTable: the pair columns of the statistics, renamed. */
GrainAdjacencyTable BuildAdjacencyTable(const GrainStatistics& stats) {
    GrainAdjacencyTable table;
    table.grain_a = stats.pair_a;
    table.grain_b = stats.pair_b;
    table.shared_area = stats.pair_area;
    table.num_faces = stats.pair_faces;
    return table;
}

/*
 * RequestedGrains:
 * - voxel map: the grains are the features of the map that the mesh touches; nothing is
 *   requested;
 * - seed table: the table seeds kept inside the ROI, plus the filler seeds asked for (given, or
 *   the automatic count);
 * - Johnson-Mehl: the count is not requested but follows from the nucleation (num_nuclei is only
 *   an optional cap; 0 means no cap), so say that instead of printing "0 requested".
 */
std::string RequestedGrains(const GenerationReport& report) {
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

int PassthroughGrainId(const GenerationReport& report, const MicrostructureResult& result,
                       std::size_t k) {
    return result.num_grains - report.num_passthrough_grains + 1 + static_cast<int>(k);
}

/*
 * RegionContents: phases, crystallinity, passthrough attributes (with their grain ids) and
 * filler grains of region r, separated by "; ".
 */
std::string RegionContents(const GenerationReport& report, const MicrostructureResult& result,
                           int r) {
    const auto ri = static_cast<std::size_t>(r - 1);
    std::ostringstream os;
    // Phases
    if (ri < result.region_phases.size() && !result.region_phases[ri].empty()) {
        const std::vector<int>& phases = result.region_phases[ri];
        os << (phases.size() > 1 ? "phases " : "phase ") << JoinIntegers(phases);
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

} // namespace exaconstit::microstructure
