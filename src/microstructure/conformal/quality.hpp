#pragma once

#include "mfem.hpp"

#include <cstdint>

/**
 * @file quality.hpp
 * @brief Element quality of a mesh, as the conformal fit checks it: scaled Jacobian, aspect ratio
 * and mean ratio, measured at the element corners.
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
 * **Mean ratio.** At every corner, with S the edge matrix relative to the ideal element (as for
 * the aspect ratio), eta = 3 |det S|^(2/3) / |S|_F^2, with the sign of det S. It combines shape and
 * stretch in one smooth number: 1 for a cube or a regular tetrahedron, 0.76 for the six tets a cube
 * is cut into, toward 0 for a flat corner, negative when inverted. The element's value is the
 * minimum over its corners. The conformal fit judges tets by it (conformal_fitter.hpp).
 *
 * Curved elements (Nodes of order > 1) are evaluated at every nodal point of the element (corners
 * included): at each point the corner edges are replaced by the Jacobian of the element map
 * applied to the reference corner edges, so a straight element gives the same values as from its
 * corners, and a curved element whose map folds inside is caught (min_det <= 0 at a point).
 *
 * Host only. ElementQuality and ElementMeanRatio are local; ComputeMeshQuality is collective.
 */

namespace exaconstit::microstructure {

/** @brief Quality of a whole mesh, reduced over all ranks (identical on every rank). */
struct MeshQuality {
    double min_scaled_jacobian = 1.0; ///< smallest element scaled Jacobian
    double max_aspect_ratio = 1.0;    ///< largest element aspect ratio
    double min_det = 0.0;             ///< smallest corner Jacobian determinant (<= 0: inverted)
    double min_mean_ratio = 1.0;      ///< smallest element mean ratio (ElementMeanRatio)
    std::int64_t inverted = 0;        ///< elements with a corner of non-positive determinant
    std::int64_t num_elements = 0;    ///< elements in the mesh
};

/**
 * @brief Scaled Jacobian and aspect ratio of one local element (see the file comment). Host,
 * local.
 * @param mesh            the mesh (its Nodes, if any, define the element shape).
 * @param e               local element index.
 * @param scaled_jacobian out: smallest corner scaled Jacobian of the element.
 * @param aspect_ratio    out: largest corner aspect ratio of the element.
 * @param min_det         out: smallest corner Jacobian determinant (<= 0 = inverted).
 * @throws std::invalid_argument if the element is neither a hex nor a tet (on this rank only).
 */
void ElementQuality(
    mfem::Mesh& mesh, int e, double& scaled_jacobian, double& aspect_ratio, double& min_det);

/**
 * @brief Mean ratio of one local element: at every corner (every nodal point for curved
 * elements), with S the corner edge matrix relative to the ideal element (as for the aspect
 * ratio), eta = 3 |det S|^(2/3) / |S|_F^2 (1 for a cube or a regular tetrahedron, toward 0 for a
 * flat corner, negative when inverted); the minimum over the points. Host, local.
 * @throws std::invalid_argument if the element is neither a hex nor a tet (on this rank only).
 */
double ElementMeanRatio(mfem::Mesh& mesh, int e);

/**
 * @brief Quality of all elements, reduced over the mesh communicator. Host, collective.
 * @throws std::invalid_argument on every rank if an element is neither a hex nor a tet.
 */
MeshQuality ComputeMeshQuality(mfem::ParMesh& pmesh);

} // namespace exaconstit::microstructure
