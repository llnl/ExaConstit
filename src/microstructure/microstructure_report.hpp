#pragma once

#include "microstructure/io/output_writer.hpp"
#include "microstructure/labeling/grain_statistics.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/microstructure_types.hpp"

#include <string>
#include <vector>

/**
 * @file microstructure_report.hpp
 * @brief Turning a generated microstructure into its record tables and log text.
 *
 * Shared by the two consumers of GenerateMicrostructure, the standalone microstructure_generator
 * tool and ExaConstit's [Mesh.Microstructure] hook, so both write the same grain table and
 * describe grains and regions in the same words. Everything here is host only, not collective,
 * and only reads its inputs.
 */

namespace exaconstit::microstructure {

/**
 * @brief The grain table columns (io/output_writer.hpp GrainTable) of a finished run.
 *
 * Orientation, region and source id come from the result, and so do the phase and filler
 * columns when they carry information (MakeGrainTable: phases only when some grain has one, the
 * filler flag only when some grain is a filler grain); element count, volume (mesh length
 * units cubed), centroid and neighbor count from the grain statistics (ComputeGrainStatistics,
 * computed on the report's adjacency); the target volume column (the size target of the seed each
 * grain grew from) only when size targets were used, with 0 for grains that did not grow from a
 * seed (passthrough grains, grain_seed = -1); the voxel feature id column (the grain id in the
 * voxel file, -1 for passthrough grains) only for voxel maps.
 * @param stats the statistics of the root rank (where the table is written); the per-grain
 *        columns are replicated on every rank, so any rank can build the table.
 */
GrainTable BuildGrainTable(const MicrostructureResult& result,
                           const GenerationReport& report,
                           const GrainStatistics& stats);

/**
 * @brief The grain adjacency table (pairs of face-sharing grains with their shared area and face
 * count) from the grain statistics. The pair columns are filled on the statistics' root rank
 * only, so call this there.
 */
GrainAdjacencyTable BuildAdjacencyTable(const GrainStatistics& stats);

/**
 * @brief The requested grain count in words, for logs and mesh headers: "N requested" for
 * generated seeds; "set by Johnson-Mehl nucleation (at most N nuclei)" or "(no nucleus limit)"
 * for Johnson-Mehl (the count follows from the nucleation); "N from the seed table" plus
 * "+ M filler requested" for seed tables; "read from the voxel map" for voxel maps.
 */
std::string RequestedGrains(const GenerationReport& report);

/**
 * @brief Grain id of passthrough attribute number k (0-based, ascending attribute order): the
 * passthrough grains are always the last grains, num_grains - P + 1 .. num_grains.
 */
int PassthroughGrainId(const GenerationReport& report, const MicrostructureResult& result,
                       std::size_t k);

/**
 * @brief What material region r (1-based) holds, in words: its phases
 * (MicrostructureResult::region_phases, or "no phase data"), whether it is crystalline
 * (region_crystalline; non-crystalline regions get identity orientations), the passthrough
 * attributes mapped to it with their grain ids, and the number of filler grains in it, e.g.
 * "phase 1; crystalline; passthrough attribute 2 (grain 9)".
 */
std::string RegionContents(const GenerationReport& report, const MicrostructureResult& result,
                           int r);

} // namespace exaconstit::microstructure
