#pragma once

/**
 * @file geodesic_limits.hpp
 * @brief Limits of the geodesic metric (labeling/geodesic_metric.hpp) and of the weighted solver
 * (labeling/geodesic_weighted.hpp), in a header of their own so the options code can check them
 * without including the labelers.
 */

namespace exaconstit::microstructure {

/** @brief Smallest automatic elongation of the regularized columnar / lamellar metric. */
inline constexpr double GEODESIC_MIN_AUTO_ELONGATION = 10.0;
/**
 * @brief Largest elongation (Tessellation.geodesic.elongation, or automatic) of the regularized
 * columnar / lamellar metric. Beyond it the mapped cells become so flat that the triangle update
 * of the eikonal solver discards them as degenerate (its test is relative, about 1e-12 on sin^2
 * of the triangle angles, i.e. k ~ 1e6), and long before that the cells are extreme slivers; 1e4
 * keeps a wide margin.
 */
inline constexpr double GEODESIC_MAX_ELONGATION = 1.0e4;

/**
 * @brief Largest number of seeds a node of the weighted geodesic solver keeps
 * (GeodesicParameters::max_seeds_per_node, Tessellation.geodesic.max_seeds_per_node;
 * labeling/geodesic_weighted.hpp). The kernels hold a working list of this size per node.
 */
inline constexpr int GEODESIC_MAX_SEEDS_PER_NODE_LIMIT = 64;

} // namespace exaconstit::microstructure
