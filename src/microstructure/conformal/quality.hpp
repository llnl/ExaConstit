#pragma once

#include "mfem.hpp"

#include <cstdint>

/**
 * @file quality.hpp
 * @brief Element quality of a (fitted) mesh: scaled Jacobian and aspect ratio at the element
 * corners (Stage 2d of conformal fitting).
 *
 * **Scaled Jacobian.** At a corner of an element, take the three edge vectors e1, e2, e3 that
 * leave the corner (for a hex: the columns of the Jacobian of the element map at that corner,
 * which are exactly those edges for a straight trilinear hex; for a tet: the three edges to the
 * other corners). The scaled Jacobian is det(e1, e2, e3) / (|e1| |e2| |e3|): 1 for a right-angled
 * corner, near 0 for a flat corner, negative for an inverted one. For a tet it is multiplied by
 * sqrt(2), so a regular tet scores 1. The element's value is the minimum over its corners. Common
 * rules of thumb (not measured here): above about 0.3 is usable for hex meshes, and a negative
 * value means the element is inverted.
 *
 * **Aspect ratio.** At every corner, the ratio of the largest to the smallest singular value of the
 * edge matrix (e1, e2, e3) relative to the ideal element's: E for a hex corner, E W^-1 for a tet
 * corner, W the corner edge matrix of the regular tetrahedron. 1 for a cube or a regular
 * tetrahedron; a corner squashed to a fifth along one direction scores 5. The element's value is
 * the maximum over its corners.
 *
 * Curved elements (Nodes of order > 1) are evaluated at every nodal point of the element (corners
 * included): at each point the corner edges are replaced by the Jacobian of the element map
 * applied to the reference corner edges, so a straight element gives the same values as from its
 * corners, and a curved element whose map folds inside is caught (min_det <= 0 at a point).
 *
 * Host; collective (global minimum / maximum / count).
 */

namespace exaconstit::microstructure {

/** @brief Global quality summary (identical on every rank). */
struct MeshQuality {
    double min_scaled_jacobian = 1.0; ///< minimum over all elements
    double max_aspect_ratio = 1.0;    ///< maximum over all elements
    double min_det = 0.0;             ///< smallest corner Jacobian determinant
    double min_mean_ratio = 1.0;      ///< minimum over all elements (ElementMeanRatio)
    std::int64_t inverted = 0;        ///< elements with a corner of non-positive determinant
    std::int64_t num_elements = 0;
};

/**
 * @brief Scaled Jacobian and aspect ratio of one local element (see the file comment).
 * Hex and tet elements only. Host.
 */
void ElementQuality(mfem::Mesh& mesh, int e, double& scaled_jacobian, double& aspect_ratio,
                    double& min_det);

/**
 * @brief Mean ratio of one local element: at every corner (every nodal point for curved
 * elements), with S the corner edge matrix relative to the ideal element (as for the aspect
 * ratio), eta = 3 |det S|^(2/3) / |S|_F^2 (1 for a cube or a regular tetrahedron, toward 0 for a
 * flat corner, negative when inverted); the minimum over the points. Host.
 */
double ElementMeanRatio(mfem::Mesh& mesh, int e);

/**
 * @brief Quality of all local elements, reduced over the communicator. **Collective.**
 * @throws std::invalid_argument on every rank if an element is neither a hex nor a tet.
 */
MeshQuality ComputeMeshQuality(mfem::ParMesh& pmesh);

} // namespace exaconstit::microstructure
