#pragma once

#include "microstructure/labeling/vertex_field.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <memory>

/**
 * @file geodesic_common.hpp
 * @brief Parameter, statistics and result types shared by the two geodesic labelers: the graph
 * geodesic (labeling/geodesic_graph.hpp, Dijkstra on element edges and diagonals) and the Fast
 * Iterative Method eikonal solver (labeling/geodesic_fim.hpp).
 *
 * Both labelers measure the distance from each seed along paths that stay inside the material,
 * store it on the mesh vertices (VertexField: T(v) and label(v)), and turn it into one seed label
 * per element. They take the same options and return the same result, so the driver can switch
 * between them with a single option (`geodesic = { solver = "fim" | "graph" }`).
 */

namespace exaconstit::microstructure {

/** @brief Options of the geodesic labelers (plain struct, filled by the driver). */
struct GeodesicParameters {
    /** @brief CENTROID: label at the element centroid; QUADRATURE_VOTE: subcell point vote. */
    SamplingMode sampling = SamplingMode::CENTROID;
    /** @brief Vote subdivision levels 0..3 (8^levels points per element), as for Euclidean. */
    int vote_levels = 2;
    /** @brief Upper bound on voting points processed at once (>= 1; memory only). */
    int max_points_per_chunk = 1 << 21;
    /**
     * @brief Safety cap on the exchange rounds of each phase (>= 1). Exceeding it throws: the
     * rounds always end on their own (see "Termination" in each solver's header), so hitting the
     * cap means a bug or an extreme mesh/partition.
     */
    int max_rounds = 100000;
};

/**
 * @brief Statistics of one run. Counts are global (summed over ranks, identical on every rank);
 * the round counts of the graph solver depend on the partition, everything else does not (the
 * FIM round counts are partition independent too, see geodesic_fim.hpp).
 */
struct GeodesicReport {
    /**
     * @brief Distance rounds (phase 1), each followed by a distance exchange between ranks.
     * Graph solver: local Dijkstra rounds, 1 on a single rank. FIM: synchronous update sweeps,
     * roughly the number of elements along the longest shortest path.
     */
    int distance_rounds = 0;
    /** @brief Label rounds (phase 2), each followed by a label exchange between ranks. */
    int label_rounds = 0;
    /** @brief Seeds outside the material, moved to the nearest element (SourceReport). */
    int seeds_outside_material = 0;
    /** @brief Vertices with T = +infinity (bodies without a seed), each counted once. */
    std::int64_t unreached_vertices = 0;
    /**
     * @brief Elements with element_seed = -1 because no seed reaches them: none of their vertices
     * has a finite T (a body of the mesh without a seed).
     */
    std::int64_t unreached_elements = 0;
    /**
     * @brief Elements with element_seed = -1 although some of their vertices are reached: their
     * winning vertex (or every voting point) carried an unlabeled vertex (see unlabeled_vertices).
     * Zero unless the mesh is degenerate. unreached_elements + unlabeled_elements = all elements
     * with element_seed = -1.
     */
    std::int64_t unlabeled_elements = 0;
    /** @brief Vote sampling only: elements whose voting points carried more than one label. */
    std::int64_t split_elements = 0;
    /**
     * @brief Reached vertices for which no upwind vertex (a vertex with smaller T on a path that
     * produced T(v)) and no source could be found, so their label is left -1. Zero unless the mesh
     * is degenerate (element edges shorter than the rounding error of T).
     */
    std::int64_t unlabeled_vertices = 0;
};

/** @brief Result of LabelGeodesicGraph and LabelGeodesicFim. */
struct GeodesicLabels {
    /** @brief Per local element: 0-based seed index, -1 for an element no seed reaches. */
    mfem::Array<int> element_seed;
    /**
     * @brief The converged vertex field (T, label, sources, numbering), for output or as the
     * starting point of another solver. Owns an MFEM space on the mesh, so it must not outlive the
     * mesh.
     */
    std::unique_ptr<VertexField> field;
    /** @brief Source element of every seed. */
    SourceReport sources;
    /** @brief Rounds and counts. */
    GeodesicReport report;
};

} // namespace exaconstit::microstructure
