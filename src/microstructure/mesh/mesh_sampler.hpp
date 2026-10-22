#pragma once

#include "mfem.hpp"

#include <memory>

/**
 * @file mesh_sampler.hpp
 * @brief Element volumes, volume centroids and quadrature-point coordinates of a mesh.
 *
 * The labeler needs, for each element, a representative point (its volume centroid) and, for
 * finer labeling, the physical coordinates and weights of the element's quadrature points.
 * Both come from the same integrals over each element e with quadrature points q:
 *
 *   volume_e   = sum_q w_q detJ_q
 *   centroid_e = sum_q (w_q detJ_q x_q) / volume_e
 *
 * where w_q is the reference-element quadrature weight, detJ_q the Jacobian determinant of the
 * reference-to-physical map and x_q the physical position of the point. This is the
 * *quadrature-weighted* (true volume) centroid. For a curved or distorted element it differs
 * from the plain average of the vertices, which is only the centroid of affine hexes and tets.
 *
 * **Data source.**
 * - Element volumes and centroids (ComputeElementSamples) come from `mfem::GeometricFactors`
 *   with COORDINATES | DETERMINANTS, which evaluates the mesh nodes at the quadrature points of
 *   every element with MFEM's QuadratureInterpolator. Its arrays use the "byNODES" layout
 *   (`mesh.cpp:15237-15253`), where NQ = quadrature points per element, SDIM = 3 and e, q, d are
 *   0-based: `X[q + NQ*(d + SDIM*e)]` is coordinate d (0=x, 1=y, 2=z) of point q of element e,
 *   and `detJ[q + NQ*e]` its signed Jacobian determinant.
 * - Quadrature points (ComputeQuadraturePointSamples, QuadraturePointSampler) are evaluated
 *   *per element range* instead: the element-wise node coordinates and the rule's shape-function
 *   tables (mfem::DofToQuad) are kept, and x_q and detJ_q are computed for one range at a time
 *   with the same arithmetic as MFEM's non-tensor interpolation kernel (bitwise equal to
 *   GeometricFactors for tets; equal to round-off for hexes, where GeometricFactors uses a
 *   sum-factorized kernel that adds the same terms in another order).
 *
 * **Memory.**
 * - ComputeElementSamples: GeometricFactors covers all local elements at once (it has no element
 *   range), so peak memory is about `(SDIM + 1) * NQ * NE` values plus the element-restricted
 *   nodes, freed on return.
 * - Quadrature points: the element-wise node coordinates, `3 * ND` doubles per element (ND =
 *   nodes per element: 24 doubles = 192 bytes for a linear hex, 96 bytes for a linear tet; plus a
 *   temporary copy of the vertex coordinates for a mesh without Nodes), kept for the whole local
 *   mesh, plus 4 doubles (32 bytes) per point of the *requested range only*.
 *
 * **The mesh is not modified.** A mesh without high-order Nodes is sampled through a temporary
 * order-1 node GridFunction; calling `Mesh::EnsureNodes()` instead would permanently give the
 * mesh Nodes and change how it is printed.
 * Caveat: for a mesh that has Nodes, MFEM caches a QuadratureInterpolator (its output layout set
 * to byNODES and its tensor-product switch set) and an element restriction inside the Nodes FE
 * space (`mesh.cpp:15253-15262`). These caches persist after sampling. The risk is low because
 * MFEM's own callers always set the layout explicitly before using the interpolator.
 *
 * **Where the work runs (host or GPU).** Everything follows the global `mfem::Device`
 * configuration, exactly like the rest of ExaConstit:
 * - the loops are `mfem::forall` kernels, which run as a GPU kernel when the Device has a CUDA
 *   or HIP backend and as a plain host loop otherwise;
 * - the outputs are `mfem::Vector`s. An mfem::Vector can hold a host copy and a device copy of
 *   its data, and MFEM moves the data between them on request: `Read()` / `Write()` give a
 *   pointer valid wherever the Device runs kernels, and `HostRead()` gives a host pointer
 *   (copying back from the GPU if needed). Read results on the host with `HostRead()`.
 * In GPU builds this source file must be compiled as device code (nvcc/hipcc), which the
 * microstructure CMake guarantees by linking blt::cuda / blt::hip directly; otherwise
 * `mfem::forall` would fall back to a host loop over device pointers.
 */

namespace exaconstit::microstructure {

/**
 * @brief Volume and volume centroid of every local element.
 *
 * Each Vector has one entry per local element, indexed by the 0-based local element id. Units
 * are those of the mesh coordinates (length for cx/cy/cz, length^3 for volume). The data may
 * be valid only on the device after a GPU run; use `HostRead()` to read it on the host.
 */
struct ElementSamples {
    /** @brief Centroid coordinates: sum_q(w_q detJ_q x_q) / sum_q(w_q detJ_q). */
    mfem::Vector cx, cy, cz;
    /** @brief Element volume: sum_q(w_q detJ_q). Negative for an inverted element. */
    mfem::Vector volume;
};

/**
 * @brief Quadrature points of the local element range [elem_begin, elem_end).
 *
 * The Vectors have `num_qp * (elem_end - elem_begin)` entries. Entry
 * `(e - elem_begin) * num_qp + q` holds 0-based quadrature point q of 0-based local element e.
 * As for ElementSamples, use `HostRead()` to read the data on the host.
 */
struct QuadraturePointSamples {
    /** @brief Quadrature points per element (0 when the rank has no elements). */
    int num_qp = 0;
    /** @brief First local element of the range (0-based). */
    int elem_begin = 0;
    /** @brief One past the last local element of the range. */
    int elem_end = 0;
    /** @brief Physical coordinates of the quadrature points (mesh length units). */
    mfem::Vector x, y, z;
    /**
     * @brief Integration weight times detJ (length^3), so the weights of an element sum to its
     *        volume.
     */
    mfem::Vector weight;
};

/**
 * @brief Quadrature order that integrates element volumes and centroids exactly.
 *
 * For a mesh whose nodes are polynomials of order p (p = 1 for a mesh without Nodes), detJ * x
 * has degree 4p - 1 per direction on tensor-product elements (hexes) and total degree 4p - 3 on
 * simplices (tets), so order 4p - 1 is exact for both. A linear mesh gives 3.
 *
 * For a variable-order node space, p is the largest order among this rank's local elements;
 * that is still exact element by element.
 *
 * Not collective (does not call `ParFiniteElementSpace::GetMaxElementOrder`, which is an
 * MPI_Allreduce for variable-order spaces, `pfespace.cpp:5081-5093`), so ranks with no elements
 * may skip it. Host only.
 *
 * @param mesh serial or parallel mesh (a ParMesh binds to this reference).
 * @return the quadrature order to pass to `mfem::IntRules.Get(geometry, order)`.
 */
int DefaultSampleQuadOrder(const mfem::Mesh& mesh);

/**
 * @brief Computes the volume and volume centroid of every local element.
 *
 * Not collective: each rank samples its own elements independently, and a rank with no
 * elements returns empty Vectors. The loop over elements runs where the `mfem::Device` runs
 * kernels (host or GPU); the setup runs on the host.
 *
 * @param mesh 3D volume mesh (dimension and space dimension 3) with one element geometry
 *        locally; run CheckMesh first. A ParMesh is accepted (its local part is sampled). Non-
 *        const because MFEM's finite element space constructor takes a non-const mesh; the mesh
 *        is not modified (see the file comment for the cache caveat).
 * @param quad_order order of `mfem::IntRules.Get(geom, quad_order)`; negative selects
 *        DefaultSampleQuadOrder(mesh).
 * @param out resized to the local element count and overwritten.
 * @throws std::runtime_error for a mixed-geometry or non-3D mesh.
 */
void ComputeElementSamples(mfem::Mesh& mesh, int quad_order, ElementSamples& out);

/**
 * @brief Physical quadrature-point coordinates and weights of a local element range.
 *
 * Not collective. Same requirements and execution model as ComputeElementSamples. Each call
 * gathers the element-wise node coordinates of the whole local mesh again (cheap, but use a
 * QuadraturePointSampler when walking the mesh in many ranges) and evaluates the requested range
 * only.
 *
 * @param mesh as for ComputeElementSamples.
 * @param quad_order as for ComputeElementSamples.
 * @param elem_begin first 0-based local element of the range.
 * @param elem_end one past the last local element; `0 <= elem_begin <= elem_end <= NE_local`.
 * @param out overwritten; see QuadraturePointSamples for the layout.
 * @throws std::runtime_error for an invalid range, a mixed-geometry or non-3D mesh.
 */
void ComputeQuadraturePointSamples(mfem::Mesh& mesh,
                                   int quad_order,
                                   int elem_begin,
                                   int elem_end,
                                   QuadraturePointSamples& out);

/**
 * @brief Physical coordinates and weights w_q detJ_q of the points of a user-supplied
 * integration rule, for a local element range.
 *
 * Same output layout and requirements as the quadrature-order overload, which calls this one with
 * `mfem::IntRules.Get(geometry, order)`. Use it for point sets that are not one of MFEM's standard
 * rules, e.g. the subcell-center voting points of labeling/quadrature_vote.hpp.
 *
 * Not collective. Host function launching device kernels.
 *
 * @param mesh       as for ComputeElementSamples.
 * @param rule       points and weights on the reference element of the mesh's (single) element
 *                   geometry; the weights of the rule are the reference weights w_q. MFEM caches
 *                   interpolation data per rule *address* (FiniteElementSpace::
 *                   GetQuadratureInterpolator, FiniteElement::GetDofToQuad), so a rule that is
 *                   not from mfem::IntRules must stay alive (at a fixed address) for as long as
 *                   the mesh and its FE spaces are used, e.g. in static storage.
 * @param elem_begin first 0-based local element of the range.
 * @param elem_end   one past the last local element; `0 <= elem_begin <= elem_end <= NE_local`.
 * @param out        overwritten; see QuadraturePointSamples for the layout (num_qp =
 *                   rule.GetNPoints(), or 0 on a rank without elements).
 * @throws std::runtime_error for an invalid range, a mixed-geometry or non-3D mesh, or more
 *         points in the range than fit in an int.
 */
void ComputeQuadraturePointSamples(mfem::Mesh& mesh,
                                   const mfem::IntegrationRule& rule,
                                   int elem_begin,
                                   int elem_end,
                                   QuadraturePointSamples& out);

/**
 * @brief Evaluates the points of one integration rule for any element range, with memory bounded
 * by the range.
 *
 * The constructor gathers, once, the element-wise node coordinates of all local elements (3 ND
 * doubles per element) and the rule's shape-function tables; Sample() computes the coordinates
 * and weights of the requested range only (4 doubles per point of the range). A caller that walks
 * the elements in chunks (as the quadrature vote does) builds one sampler and calls Sample() per
 * chunk, so its memory grows with the chunk, not with NE x points per element.
 *
 * Host object; Sample() launches device kernels on the mfem::Device backend. Not collective.
 * The mesh and the rule must outlive the sampler (see the rule-lifetime note above).
 */
class QuadraturePointSampler {
public:
    /**
     * @brief Checks the mesh and evaluates the rule's points on every local element (nothing is
     * evaluated on a rank without elements).
     * @param mesh as for ComputeElementSamples (not modified; see the cache caveat in the file
     *             comment).
     * @param rule integration rule on the reference element of the mesh's element geometry.
     * @throws std::runtime_error for a mixed-geometry or non-3D mesh.
     */
    QuadraturePointSampler(mfem::Mesh& mesh, const mfem::IntegrationRule& rule);
    ~QuadraturePointSampler();
    QuadraturePointSampler(const QuadraturePointSampler&) = delete;
    QuadraturePointSampler& operator=(const QuadraturePointSampler&) = delete;

    /** @brief Points per element (rule.GetNPoints()), or 0 on a rank without elements. */
    int NumPointsPerElement() const;

    /**
     * @brief Coordinates and weights w_q detJ_q of the points of local elements
     * [elem_begin, elem_end) (QuadraturePointSamples layout).
     * @throws std::runtime_error for an invalid range or more points than fit in an int.
     */
    void Sample(int elem_begin, int elem_end, QuadraturePointSamples& out) const;

private:
    struct Factors; ///< element node coordinates, shape tables and temporary nodes
    mfem::Mesh& m_mesh;
    const mfem::IntegrationRule& m_rule;
    std::unique_ptr<Factors> m_factors; ///< null on a rank without elements
};

} // namespace exaconstit::microstructure
