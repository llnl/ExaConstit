#pragma once

#include "microstructure/labeling/geodesic_limits.hpp"
#include "microstructure/mesh/element_mask.hpp"
#include "microstructure/microstructure_options.hpp"

#include "mfem.hpp"

#include <array>
#include <mpi.h>

/**
 * @file geodesic_metric.hpp
 * @brief The metric the geodesic solvers measure paths with, built from the [Tessellation]
 * options: plain lengths for Voronoi / Additive / Laguerre, the quadric tensor A for Quadric.
 *
 * The solvers take the metric as a "stretch" S (GeodesicParameters::stretch, row-major 3x3): a
 * vector r has length |S r|, the A-length for A = S^T S. Mapping the mesh with y = S x turns
 * A-lengths into Euclidean lengths, and a linear map keeps cells as cells and linear
 * interpolation linear, so the eikonal and graph solvers run unchanged on the mapped coordinates
 * (VertexField::SetMetricStretch). Geodesic Quadric labels are then the argmin of the A-length of
 * the shortest path inside the material, which on a convex part equals the straight-line Quadric
 * labels.
 *
 * **Which tensor.**
 *  - anisotropy kind "none" / "tensor": the stored global_tensor, factored A = L L^T, S = L^T
 *    (MetricStretchFromTensor; A must be positive definite, which validate() checks).
 *  - "columnar" / "lamellar": their tensors I - a a^T and n n^T are only semidefinite (no length
 *    along the column axis or within the layer), which no path-length solver can use: the mapped
 *    mesh would collapse onto a plane or a line. The geodesic mode therefore uses a
 *    *regularized* tensor: columns k times longer than wide (S = I - (1 - 1/k) a a^T,
 *    ColumnarStretch), layers k times wider than thick (S = n n^T + (I - n n^T) / k,
 *    LamellarStretch). k is Tessellation.geodesic_elongation, or chosen from the part when that is
 *    "auto" (AutoGeodesicElongation).
 *  - Voronoi, Additive, Laguerre: the identity (no mapping; bitwise the behavior without metrics).
 *  - "per_seed" has one tensor per seed and is not a single metric: the identity here, and the
 *    weighted solver (labeling/geodesic_weighted.hpp) maps each seed's lengths with its own
 *    stretch (GeodesicParameters::seed_stretch).
 *
 * **What finite k changes.** With the exact semidefinite tensor a column never ends; with finite
 * k, the boundary between two neighboring columns i and j at a point whose height differs from
 * seed i's by dz moves by about (dz / k)^2 / (2 d_ij) toward i (d_ij the distance between the
 * columns), and a column can end where a neighbor's seed is much closer in height. With
 * dz <= L (the part length along the axis) and d_ij ~ w (the column width), the shift is at most
 * about w (L / (k w))^2 / 2; the automatic k = 10 L / w keeps it below w / 200.
 */

namespace exaconstit::microstructure {

/** @brief The metric chosen for one run (ResolveGeodesicMetric). */
struct GeodesicMetric {
    /** @brief S, row-major (GeodesicParameters::stretch); identity for unstretched metrics. */
    std::array<double, 9> stretch = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    /** @brief Columnar / lamellar only: the elongation k used; 0 for every other kind. */
    double elongation = 0.0;
    /** @brief True when elongation was chosen automatically from the part. */
    bool automatic = false;
    /** @brief True when the automatic value hit GEODESIC_MAX_ELONGATION. */
    bool clamped = false;
};

/**
 * @brief Automatic elongation of the regularized columnar / lamellar metric, from the extents of
 * the active material (corner vertices of active elements, exact MIN/MAX reductions, so the value
 * is bitwise the same on every partition):
 *  - columnar (axis a): L = extent along a; w = sqrt(Lu Lv / N), the column width if the N seeds
 *    shared the bounding-box cross-section (Lu, Lv the extents along two directions
 *    perpendicular to a); k = 10 L / w;
 *  - lamellar (normal n): t = Ln / N, the layer thickness; L = sqrt(Lu^2 + Lv^2), the in-plane
 *    diameter of the bounding box; k = 10 L / t.
 * The result is rounded up to two significant digits (a readable value for the logs), then
 * limited to [GEODESIC_MIN_AUTO_ELONGATION, GEODESIC_MAX_ELONGATION].
 *
 * **Collective** over comm. Host.
 * @param columnar  true: columnar along `direction`; false: lamellar with normal `direction`.
 * @param direction axis or normal (any nonzero length).
 * @param num_seeds number of seeds N >= 1.
 * @param mask      optional active elements (nullptr = all).
 * @param clamped   optional out: true when the value was limited to GEODESIC_MAX_ELONGATION.
 * @throws std::invalid_argument for a zero direction or num_seeds < 1 (replicated inputs).
 */
double AutoGeodesicElongation(bool columnar, const std::array<double, 3>& direction,
                              mfem::ParMesh& pmesh, MPI_Comm comm, int num_seeds,
                              const ElementMask* mask, bool* clamped = nullptr);

/**
 * @brief The metric of the geodesic solvers for these options (see the file comment).
 *
 * **Collective** over comm when the elongation is automatic (columnar / lamellar with
 * geodesic_elongation = 0); every rank must call it in any case. Host.
 * @throws std::invalid_argument for an unknown kind, a non positive definite tensor, a zero
 *         direction, or an elongation outside [1, GEODESIC_MAX_ELONGATION].
 */
GeodesicMetric ResolveGeodesicMetric(const TessellationOptions& tessellation,
                                     mfem::ParMesh& pmesh, MPI_Comm comm, int num_seeds,
                                     const ElementMask* mask);

} // namespace exaconstit::microstructure
