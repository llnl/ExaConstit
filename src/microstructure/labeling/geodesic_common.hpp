#pragma once

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "microstructure/labeling/vertex_field.hpp"
#include "microstructure/microstructure_types.hpp"

/**
 * @file geodesic_common.hpp
 * @brief Parameter, statistics and result types shared by the geodesic labelers: the graph
 * geodesic (labeling/geodesic_graph.hpp, Dijkstra on element edges and diagonals), the Fast
 * Iterative Method eikonal solver (labeling/geodesic_fim.hpp) and its weighted multi-seed
 * variant for Laguerre and weighted Quadric metrics (labeling/geodesic_weighted.hpp).
 *
 * Both labelers measure the distance from each seed along paths that stay inside the material,
 * store it on the field nodes (VertexField: T(v) and label(v); the mesh vertices, or every nodal
 * point of a curved mesh), and turn it into one seed label
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
    /**
     * @brief FIM only: stopping tolerance of the distance rounds, relative to the diagonal D of
     * the bounding box of the field nodes (in the metric coordinates). A vertex takes a new
     * distance only when it is its first arrival or lowers T by more than tolerance * D, so the
     * long tail of tiny corrections (Jacobi sweeps converge geometrically, slowly on stretched
     * cells) is cut. The converged T then lies within a few tolerance * D of the fully converged
     * one, far below the discretization error (about 1e-2 h); the labels and the round counts stay
     * partition independent (geodesic_fim.hpp). In [0, 1); 0 iterates until no value changes.
     */
    double tolerance = 1.0e-6;
    /**
     * @brief Weighted solver only (labeling/geodesic_weighted.hpp): largest number of seeds a
     * node keeps (1 .. GEODESIC_MAX_SEEDS_PER_NODE_LIMIT = 64). Seeds that can no longer win are
     * dropped exactly; if more remain, the worst are cut and the node is counted in
     * GeodesicReport::overflow_nodes.
     */
    int max_seeds_per_node = 16;
    /**
     * @brief Weighted solver only: one stretch per seed (9 entries per seed, row-major S_i, the
     * Quadric tensor A_i = S_i^T S_i of anisotropy kind "per_seed"), or empty. When given, seed
     * i's path lengths are measured with S_i (stretch must then be the identity), and seed j may
     * drop seed i only where A_j <= A_i (geodesic_weighted.hpp, "Per-seed tensors").
     */
    std::vector<double> seed_stretch;
    /**
     * @brief Which points carry the distance field (labeling/vertex_field.hpp): AUTO (default)
     * uses the element corners on straight-sided meshes, exactly as before curved support, and
     * every nodal point of the mesh's own Nodes space with p^3 virtual sub-cells per element on
     * curved meshes; CORNERS / MESH_NODES force one of the two.
     */
    FieldNodes field_nodes = FieldNodes::AUTO;
    /**
     * @brief Optional active-element mask and shared locator (MaterialContext,
     * seeding/material_point_locator.hpp; not owned, must outlive the call). Paths run through
     * active elements only (a face to an inactive element is a free surface), seeds are located
     * in active elements, and inactive elements get element_seed = -1 without counting as
     * unreached. nullptr = every element.
     */
    const MaterialContext* context = nullptr;
    /**
     * @brief Metric of the path lengths (VertexField::SetMetricStretch): a vector r has length
     * |S r|, S row-major 3x3, i.e. paths are measured with the Quadric tensor A = S^T S
     * (labeling/geodesic_metric.hpp builds S from the tessellation options). The identity
     * (default) gives plain lengths, bitwise as before metrics existed.
     */
    std::array<double, 9> stretch = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
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
    /**
     * @brief Weighted solver only: nodes where more than max_seeds_per_node seeds could still win
     * and the worst were cut, at least once (0 = the result is exact and partition independent).
     */
    std::int64_t overflow_nodes = 0;
    /** @brief Weighted solver only: the longest seed list kept by any node (global maximum). */
    int max_list_length = 0;
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
