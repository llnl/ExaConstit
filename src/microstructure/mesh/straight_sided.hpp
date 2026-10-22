#pragma once

#include "mfem.hpp"

#include <mpi.h>

/**
 * @file straight_sided.hpp
 * @brief Decides whether a mesh's elements are straight-sided, and gives their corner points.
 *
 * "Straight-sided" means every element is exactly the image of its reference element under the
 * multilinear map of its corner points (trilinear for a hex, linear for a tet), i.e. the shape a
 * mesh without high-order Nodes has. Such elements lie inside the convex hull of their corners,
 * so the generator can use exact corner boxes, the closed-form tet test and flat boundary faces.
 * A mesh with Nodes (a high-order coordinate field) is not automatically curved: ExaConstit
 * gives every mesh order-p Nodes (`Mesh::SetCurvature`), and Nodes projected from a straight
 * mesh describe the same straight elements. Treating those as curved changed the seed sampling
 * box (padding for bulges) and thus the random seeds, so the same mesh gave a different
 * microstructure with and without Nodes.
 *
 * The rules (StraightSidedLocal, per rank; IsStraightSided combines the ranks):
 *  - no Nodes: straight-sided;
 *  - Nodes in a continuous (H1) space of order 1 everywhere: straight-sided (the order-1 H1 map
 *    is the multilinear corner map);
 *  - continuous Nodes of higher order: straight-sided when every nodal point of every element
 *    coincides with the multilinear map of that element's corner nodes evaluated at the node's
 *    reference position, within STRAIGHT_SIDED_REL_TOL times the element size (largest edge of
 *    the corner bounding box) plus a round-off allowance of 16 machine epsilons times the largest
 *    coordinate magnitude of the element. For a Bernstein (positive) basis the dofs are control
 *    points, which for a multilinear map also equal the map at the uniformly spaced reference
 *    nodes (linear precision), so the same test applies;
 *  - discontinuous (L2) Nodes, used by MFEM for periodic meshes: never straight-sided (they
 *    keep the general, curved treatment, as before; preflight rejects them anyway).
 *
 * Corner points: for a straight-sided mesh with Nodes, the corners are read from the Nodes (the
 * vertex dofs, which every H1 element numbers first, in the element's vertex order), not from
 * the mesh vertex array, which MFEM does not keep in sync with moved Nodes. Without Nodes they
 * are the mesh vertices, exactly as before, so results for meshes without Nodes are unchanged
 * bit for bit.
 *
 * What the decision changes (in place, the mesh is never copied): the seeding element boxes and
 * point-in-element test (MaterialPointLocator), the unpadded seed sampling box, the surface
 * clearance triangles and the canonical grain-statistics face areas. A straight-sided mesh with
 * Nodes therefore draws the same seeds as the same mesh without Nodes. Element volumes,
 * centroids and the adjacency's face areas are still evaluated through the Nodes and agree with
 * the no-Nodes mesh only up to round-off.
 *
 * Everything runs on the host.
 */

namespace exaconstit::microstructure {

/** @brief Relative tolerance of the higher-order straightness test (times the element size). */
constexpr double STRAIGHT_SIDED_REL_TOL = 1.0e-12;

/**
 * @brief Whether this rank's local elements are straight-sided (rules in the file comment).
 * Not collective; host only. A rank with no elements answers true.
 * @param mesh this rank's mesh partition (any dimension; only 3D meshes are used by the
 *        generator).
 * @param rel_tol relative tolerance of the higher-order test (default STRAIGHT_SIDED_REL_TOL).
 */
bool StraightSidedLocal(const mfem::Mesh& mesh, double rel_tol = STRAIGHT_SIDED_REL_TOL);

/**
 * @brief Whether the whole distributed mesh is straight-sided: StraightSidedLocal on every rank,
 * combined with MPI_Allreduce(MIN) so every rank gets the same answer (one curved element on
 * one rank makes the whole mesh curved). Use this, not StraightSidedLocal, for decisions that
 * must not depend on the partition (element boxes, the seed sampling box).
 * Collective over `comm`: every rank must call it. Host only.
 */
bool IsStraightSided(const mfem::Mesh& mesh, MPI_Comm comm,
                     double rel_tol = STRAIGHT_SIDED_REL_TOL);

/**
 * @brief Corner points of local element e of a straight-sided mesh, as a 3 x n_corners matrix
 * (column k = corner k in the element's vertex order, mfem::Element::GetVertices): from the Nodes
 * vertex dofs when the mesh has continuous Nodes, else from the mesh vertices (this is then
 * exactly mfem::Mesh::GetPointMatrix). Host only, not collective.
 * @param e 0-based local element index.
 * @param corners resized to SpaceDimension() x n_corners.
 */
void ElementCornerPoints(const mfem::Mesh& mesh, int e, mfem::DenseMatrix& corners);

/**
 * @brief The multilinear (tet: linear) element transformation of local element e built from its
 * corner points (ElementCornerPoints): the transformation MFEM itself uses for a mesh without
 * Nodes (Mesh::GetElementTransformation, mesh.cpp:361-373: the corner point matrix with
 * Mesh::GetTransformationFEforElementType). For a straight-sided mesh this is the element's true
 * geometry, and it is bitwise the transformation of the same mesh without Nodes. Host only.
 */
void StraightElementTransformation(const mfem::Mesh& mesh, int e,
                                   mfem::IsoparametricTransformation& trans);

/**
 * @brief Coordinates of mesh vertex v (0-based local vertex index): from the Nodes value at the
 * vertex's dof when the mesh has continuous Nodes, else the mesh vertex array. Host only.
 * @param x receives SpaceDimension() values.
 */
void VertexPoint(const mfem::Mesh& mesh, int v, double* x);

/**
 * @brief The multilinear (triangle: linear) transformation of local face f built from its corner
 * vertices (VertexPoint): what Mesh::GetFaceTransformation does for a mesh without Nodes
 * (mesh.cpp:539-561), bitwise the same for such a mesh. Host only.
 */
void StraightFaceTransformation(const mfem::Mesh& mesh, int f,
                                mfem::IsoparametricTransformation& trans);

} // namespace exaconstit::microstructure
