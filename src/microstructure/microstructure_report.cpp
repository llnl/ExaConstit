/*
 * Implementation of microstructure_report.hpp: the record tables (grain table, microstructure
 * file grain table, grain adjacency) and the log phrases of a generated microstructure, shared by
 * the standalone tool and the mechanics hook so both write the same files and text.
 */
#include "microstructure/microstructure_report.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <utility>

#include "microstructure/io/seed_table_io.hpp"

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
            table.target_volume[g] = (seed >= 0)
                                         ? report.seed_target_volume[static_cast<std::size_t>(seed)]
                                         : 0.0;
        }
    }
    if (report.used_voxel_map) {
        table.feature_id = report.grain_feature;
    }
    return table;
}

/*
 * BuildMicrostructureGrainTable: the active grains (passthrough grains are the last
 * num_passthrough_grains ids), then each optional column only when every grain has a value.
 */
MicrostructureGrainTable BuildMicrostructureGrainTable(const MicrostructureResult& result,
                                                       const GenerationReport& report) {
    const auto n = static_cast<std::size_t>(result.num_grains - report.num_passthrough_grains);
    MicrostructureGrainTable table;
    table.id.resize(n);
    for (std::size_t g = 0; g < n; ++g) {
        table.id[g] = static_cast<std::int64_t>(g + 1);
    }
    table.quat.assign(result.grain_quat.begin(),
                      result.grain_quat.begin() + static_cast<std::ptrdiff_t>(4 * n));

    bool all_phases = n > 0 && result.grain_phase.size() >= n;
    for (std::size_t g = 0; all_phases && g < n; ++g) {
        all_phases = result.grain_phase[g] >= 1;
    }
    if (all_phases) {
        table.phase.assign(result.grain_phase.begin(),
                           result.grain_phase.begin() + static_cast<std::ptrdiff_t>(n));
    }

    const SeedSet& seeds = report.seeds;
    bool all_seeds = n > 0 && report.grain_seed.size() >= n && seeds.Size() > 0;
    for (std::size_t g = 0; all_seeds && g < n; ++g) {
        all_seeds = report.grain_seed[g] >= 0;
    }
    if (!all_seeds) {
        return table;
    }
    std::vector<double> shape;
    bool shapes_ok = !seeds.shape.empty();
    for (std::size_t g = 0; g < n; ++g) {
        const auto i = static_cast<std::size_t>(report.grain_seed[g]);
        table.position.push_back(seeds.x[i]);
        table.position.push_back(seeds.y[i]);
        table.position.push_back(seeds.z[i]);
        if (!seeds.weight.empty()) {
            table.weight.push_back(seeds.weight[i]);
        }
        if (!report.seed_target_volume.empty()) {
            table.volume.push_back(report.seed_target_volume[i]);
        }
        if (shapes_ok) {
            double m[6];
            shapes_ok = InvertShapeTensorPublic(&seeds.shape[6 * i], m);
            shape.insert(shape.end(), m, m + 6);
        }
    }
    if (shapes_ok) {
        table.shape = std::move(shape);
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
 * ConformalText: see the header. Repair, held nodes, grain adjacency, fit distance, quality.
 */
std::string ConformalText(const GenerationReport& report) {
    if (!report.ran_conformal) {
        return "";
    }
    const LabelRepairReport& repair = report.conformal_repair;
    const ConformalFitReport& fit = report.conformal_fit;
    std::ostringstream os;
    os << "conformal grain boundaries: label repair changed the grain of " << repair.flips
       << " element(s) (" << repair.passes << " pass(es)); unfit grain boundary nodes "
       << repair.bad_nodes_before << " -> " << repair.bad_nodes_after << "\n";
    if (report.conformal_pinched_nodes > 0) {
        os << "  " << report.conformal_pinched_nodes
           << " node(s) held where a grain touches itself only at a node or an edge\n";
    }
    if (report.conformal_under_resolved_grains > 0) {
        os << "  " << report.conformal_under_resolved_grains
           << " grain(s) too thin to fit (no mesh node inside the grain, e.g. one element thick): "
              "their boundaries keep the stair steps; refine the mesh to fit them\n";
    }
    const AdjacencyComparison& adj = report.conformal_adjacency;
    if (adj.discrete > 0 || adj.analytic > 0) {
        os << "  grain adjacency: " << adj.discrete << " seed pair(s) touch in the mesh, "
           << adj.analytic << " have an analytic boundary near the mesh; " << adj.fake
           << " touch only in the mesh (stair-step contacts), " << adj.missing
           << " only analytically (faces smaller than the mesh resolves)\n";
    }
    os << "  fit: " << fit.fit_nodes << " grain boundary nodes (" << report.conformal_smoothed_nodes
       << " of them on the smoothed staircase, the others on the analytic boundaries), distance "
       << "from their targets (element sizes h) mean " << fit.initial_deviation << " -> "
       << fit.mean_deviation << ", largest after " << fit.max_deviation << "; " << fit.rounds
       << " round(s), " << fit.solves << " elastic solve(s)\n";
    os << "  mesh quality: smallest scaled Jacobian " << fit.quality_before.min_scaled_jacobian
       << " -> " << fit.quality_after.min_scaled_jacobian << ", largest aspect ratio "
       << fit.quality_before.max_aspect_ratio << " -> " << fit.quality_after.max_aspect_ratio
       << ", smallest mean ratio " << fit.quality_before.min_mean_ratio << " -> "
       << fit.quality_after.min_mean_ratio << ", inverted elements " << fit.quality_after.inverted
       << "\n";
    return os.str();
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

/* PassthroughGrainId: the passthrough grains are the last P ids, in ascending attribute order. */
int PassthroughGrainId(const GenerationReport& report,
                       const MicrostructureResult& result,
                       std::size_t k) {
    return result.num_grains - report.num_passthrough_grains + 1 + static_cast<int>(k);
}

/*
 * RegionContents: phases, crystallinity, passthrough attributes (with their grain ids) and
 * filler grains of region r, separated by "; ".
 */
std::string
RegionContents(const GenerationReport& report, const MicrostructureResult& result, int r) {
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
    const bool crystalline = ri < result.region_crystalline.size() &&
                             result.region_crystalline[ri] != 0;
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
