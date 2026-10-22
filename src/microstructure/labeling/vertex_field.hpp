#pragma once

#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <memory>
#include <mpi.h>
#include <vector>

/**
 * @file vertex_field.hpp
 * @brief A distance value and a seed label on every mesh vertex, with the bookkeeping that the
 * geodesic (path-inside-the-material) labelers share.
 *
 * **What it is for.** In `geodesic` distance mode a grain grows from its seed along paths that
 * stay inside the material, so grains wrap around corners and voids instead of reaching across
 * them. The distance from the seeds is computed on the mesh vertices: every vertex v carries
 *   - T(v): the shortest path length from the nearest seed (mesh length units; +infinity when no
 *     seed reaches the vertex, e.g. on a body of the mesh without a seed), and
 *   - label(v): the 0-based index of that seed in the SeedSet (-1 when T(v) = +infinity).
 * Two solvers fill this field: the graph geodesic (labeling/geodesic_graph.hpp, Dijkstra on the
 * element edges and diagonals) and the Fast Iterative Method eikonal solver
 * (labeling/geodesic_fim.hpp). Both start from InitializeSources, exchange values between ranks
 * with ReduceShared*, and turn the vertex field into element labels with LabelElementCentroids /
 * LabelElementsByVote. This class owns everything they have in common.
 *
 * **Vertex numbering.** The vertices are numbered by an order-1 continuous (H1) finite element
 * space on the mesh: its local degrees of freedom ("dofs", the nodal unknowns of the space) are
 * exactly the mesh vertices, and MFEM numbers them like the vertices: local dof v is local mesh
 * vertex v (fespace.cpp:3536-3544, `dofs.Append(V[i]*nv + j)` with nv = 1 dof per vertex). The
 * space is used for one thing only: its GroupCommunicator, MFEM's object that knows which local
 * dofs are copies of the same vertex on other ranks ("shared" vertices) and can combine and
 * redistribute their values (see ReduceShared).
 *
 * **Curved meshes.** Only the element vertices (corners) are used: a path runs in straight lines
 * between vertices, and a point inside an element is reached in a straight line from one of the
 * element's vertices. For a curved (high-order) mesh this is a geometric approximation of the
 * same order as replacing each element by its straight-sided version. Vertex coordinates of a
 * curved mesh are read from the mesh Nodes (the vertex dofs of a continuous Nodes space), not
 * from Mesh::GetVertex, which MFEM documents as possibly stale when Nodes exist (mesh.hpp:1426).
 * Discontinuous Nodes (periodic meshes) are rejected: Stage 1e does not combine geodesic distance
 * with periodicity.
 *
 * **Identical coordinates on every copy.** Both solvers compare sums of edge lengths exactly, so
 * every rank must compute the same bits for the same vertex. A ParMesh built from a serial mesh
 * copies the coordinates bit for bit, but nothing guarantees it in general (for example a mesh
 * refined after it was distributed may compute a new vertex as an average in a different order on
 * each rank). The constructor therefore broadcasts the owning rank's coordinates to every copy of
 * a shared vertex (GroupCommunicator::Bcast, one array per coordinate), so Coordinates() is
 * identical on all copies by construction. (Mesh::GetVertex is not changed; the seed snapping of
 * InitializeSources, which uses it, relies on the usual bitwise copy.)
 *
 * **Connectivity tables (CSR).** "CSR" (compressed sparse row) stores a list of variable-length
 * lists in two flat arrays: `offsets` (one more entry than lists) and `items`; list i is
 * `items[offsets[i] .. offsets[i+1]-1]`. Two tables are kept, as mfem::Array<int> so device
 * kernels can read them (Read()):
 *   - element -> vertices: ElementVertexOffsets() (size NE+1), ElementVertices(); the vertices of
 *     element e in MFEM's element vertex order (Mesh::GetElementVertices);
 *   - vertex -> elements: VertexElementOffsets() (size NV+1), VertexElements(); the local
 *     elements that contain vertex v, in increasing element order.
 * NE and NV are this rank's local element and vertex counts; indices are 0-based and local.
 *
 * **Shared vertices and determinism.** A vertex on a partition boundary exists on every rank whose
 * elements touch it. After a solver has updated its local copies, ReduceShared() makes all copies
 * equal to the lexicographic minimum of (T, label) over the copies: the smallest T, and among
 * copies with that T the smallest seed index. "Lexicographic" means compared like words in a
 * dictionary: first by T, then by label. Since MFEM's group reduction is a plain element-wise
 * minimum, the pair is reduced in two steps (see ReduceShared). The result does not depend on
 * the order of the ranks or on which rank owns the vertex, so a solver built on it can be made
 * independent of the rank count.
 *
 * **Where it runs.** Construction, source initialization and the rank exchanges run on the host
 * (MPI). The point labeling (LabelPoints and the element labelings built on it) is an mfem::forall
 * kernel (MFEM's parallel loop: a GPU kernel when the mfem::Device has a CUDA or HIP backend, a
 * host loop otherwise) that reads the CSR tables, coordinates and field through Read(), so it runs
 * wherever the mfem::Device runs kernels. Solvers may update Distance() / Label() on the device
 * (ReadWrite()); ReduceShared* moves them to the host for the exchange (HostReadWrite()).
 *
 * **Typical use (a solver):**
 * @code
 *   VertexField field(pmesh);                                  // collective
 *   SourceReport src = field.InitializeSources(seeds, a_i);    // collective, T/label at sources
 *   do {
 *       ... update field.Distance() / field.Label() from the local elements ...
 *   } while (field.ReduceShared());                            // collective, any change?
 *   mfem::Array<int> element_seed;
 *   field.LabelElementCentroids(element_seed);                 // local, kernel
 * @endcode
 */

namespace exaconstit::microstructure {

/**
 * @brief Where each seed starts its distance field (result of VertexField::InitializeSources).
 *
 * All per-seed vectors have one entry per seed (size N, seed index 0-based) and are identical on
 * every rank, except source_element_local, which is rank-local.
 */
struct SourceReport {
    /**
     * @brief Element key (mesh/element_key.hpp) of each seed's source element: the element that
     * contains the seed (smallest key when the seed lies on a face, edge or vertex shared by
     * several elements), or, for a seed outside the material, an element around the mesh vertex
     * nearest to the seed (nearest vertex: ties by smallest VertexKey; then the smallest element
     * key among the elements sharing that vertex).
     */
    std::vector<std::uint64_t> source_element_key;
    /**
     * @brief 0-based local index of the source element on the rank that owns it, -1 on every
     * other rank. Exactly one rank holds a non-negative entry per seed.
     */
    std::vector<int> source_element_local;
    /** @brief 1 for a seed that lies in no element and was moved ("snapped") to the nearest one. */
    std::vector<char> seed_outside;
    /** @brief Number of seeds with seed_outside = 1 (same on every rank). */
    int num_seeds_outside = 0;
};

/**
 * @brief Per-vertex distance and seed label on one rank's mesh partition, plus the vertex
 * numbering, connectivity tables and rank exchange shared by the geodesic solvers (see the file
 * comment).
 *
 * Host object. Not copyable (it owns an MFEM finite element space). The ParMesh must outlive the
 * VertexField and must not change while it is used.
 */
class VertexField {
public:
    /**
     * @brief Builds the vertex numbering (order-1 H1 space), the CSR tables, the vertex
     * coordinates and the shared/owned markers, and sets T = +infinity, label = -1 everywhere.
     *
     * **Collective** over the mesh communicator (the ParFiniteElementSpace constructor exchanges
     * data between ranks). A rank with no elements takes part with NV = 0.
     *
     * @param pmesh conforming 3D ParMesh (any element types; the vote sampling additionally needs
     *        a single hex or tet geometry). Its Nodes, if any, must be a continuous (H1) space.
     * @throws std::runtime_error on every rank if the mesh is nonconforming (hanging vertices
     *         from local refinement: their values are not plain copies of one vertex), not 3D, or
     *         has discontinuous Nodes, or if the order-1 space does not have exactly one dof per
     *         local vertex on some rank.
     */
    explicit VertexField(mfem::ParMesh& pmesh);
    ~VertexField();
    VertexField(const VertexField&) = delete;
    VertexField& operator=(const VertexField&) = delete;

    /** @brief The mesh the field lives on. */
    mfem::ParMesh& Mesh() const {
        return m_pmesh;
    }
    /** @brief The mesh communicator (ParMesh::GetComm()); every collective call uses it. */
    MPI_Comm Comm() const;
    /** @brief Number of local vertices NV (= local mesh vertices = local H1 dofs). */
    int NumVertices() const {
        return m_num_vertices;
    }
    /** @brief Number of local elements NE. */
    int NumElements() const {
        return m_num_elements;
    }

    /** @brief Element -> vertex CSR offsets, size NE+1 (see the file comment). */
    const mfem::Array<int>& ElementVertexOffsets() const {
        return m_elem_vert_offsets;
    }
    /** @brief Element -> vertex CSR items: the local vertex ids of each element. */
    const mfem::Array<int>& ElementVertices() const {
        return m_elem_verts;
    }
    /** @brief Vertex -> element CSR offsets, size NV+1. */
    const mfem::Array<int>& VertexElementOffsets() const {
        return m_vert_elem_offsets;
    }
    /** @brief Vertex -> element CSR items: the local elements containing each vertex. */
    const mfem::Array<int>& VertexElements() const {
        return m_vert_elems;
    }
    /**
     * @brief Vertex coordinates, size 3*NV, layout `xyz[3*v + d]` (d = 0, 1, 2 for x, y, z; mesh
     * length units). Bitwise identical on every rank that holds a copy of the vertex.
     */
    const mfem::Vector& Coordinates() const {
        return m_coords;
    }
    /**
     * @brief 1 for a vertex that also exists on another rank (it appears in a GroupCommunicator
     * group other than the rank-local group 0), else 0. Size NV.
     */
    const mfem::Array<int>& SharedVertexMarker() const {
        return m_shared;
    }
    /**
     * @brief 1 on exactly one rank per vertex (the rank owning its "true dof", i.e. the group
     * master), else 0. Size NV. Use it to count vertices globally without double counting.
     */
    const mfem::Array<int>& OwnedVertexMarker() const {
        return m_owned;
    }

    /** @brief T(v): distance from the nearest seed per local vertex, +infinity = not reached. */
    mfem::Vector& Distance() {
        return m_T;
    }
    /** @brief Read-only T(v). */
    const mfem::Vector& Distance() const {
        return m_T;
    }
    /** @brief label(v): 0-based seed index per local vertex, -1 = none. */
    mfem::Array<int>& Label() {
        return m_label;
    }
    /** @brief Read-only label(v). */
    const mfem::Array<int>& Label() const {
        return m_label;
    }
    /**
     * @brief Initial (source) values from the last InitializeSources: T0(v) = |v - s_i| - a_i for
     * the vertices of each seed's source element (lexicographic minimum over seeds), +infinity
     * elsewhere. Identical on all copies of a vertex. A solver keeps these vertices fixed or uses
     * them as boundary values; the graph labeler uses them to tell which vertices start a path.
     */
    const mfem::Vector& SourceDistance() const {
        return m_source_T;
    }
    /** @brief Seed index of SourceDistance() per vertex, -1 where SourceDistance() is +infinity. */
    const mfem::Array<int>& SourceLabel() const {
        return m_source_label;
    }

    /**
     * @brief Sets T = +infinity and label = -1 on every vertex, and clears the source values and
     * the per-element seed lists.
     * Not collective.
     */
    void Reset();

    /**
     * @brief Places the seeds on the mesh and sets the starting values of the field.
     *
     * 1. Source element of seed i: the element that contains s_i (ContainingElementKeys of
     *    seeding/seed_element_locator.hpp: BVH plus the exact point-in-element test, smallest
     *    element key on ties, reduced over ranks). For a seed in no element: the mesh vertex
     *    nearest to s_i (NearestVertex: smallest squared distance, ties by smallest VertexKey,
     *    over all ranks), then the smallest key among the elements of every rank that contain
     *    that vertex (MPI_Allreduce(MIN)).
     * 2. Every vertex v of seed i's source element gets the candidate (|v - s_i| - a_i, i); a
     *    vertex touched by several seeds keeps the lexicographic minimum (smallest value, then
     *    smallest seed index).
     * 3. T and label are set to these values (+infinity / -1 elsewhere), ReduceShared() makes all
     *    copies agree, and the result is stored as SourceDistance() / SourceLabel().
     * 4. Each source element remembers its seeds (position and a_i) on the rank that owns it, so
     *    LabelPoints can use the straight line from the seed inside that element.
     * Previous field values are discarded.
     *
     * For the Voronoi metric a_i = 0 and T0 is the straight-line distance; for the Additive metric
     * the grain of seed i is the set where (distance - a_i) is smallest, so the offset is applied
     * once, at the source. Collective; host.
     *
     * @param seeds            replicated seed set, N >= 1, finite positions (mesh length units).
     * @param additive_weights a_i per seed (length units), size N, or empty for all zero.
     * @return the source elements (see SourceReport).
     * @throws std::invalid_argument (on every rank, the inputs being replicated) for an empty seed
     *         set, a non-finite position or weight, or a weight array of the wrong size;
     *         std::runtime_error on every rank if the mesh has no element on any rank.
     */
    SourceReport InitializeSources(const SeedSet& seeds,
                                   const std::vector<double>& additive_weights);

    /**
     * @brief Makes all copies of every shared vertex equal to the lexicographic minimum of
     * (T, label) over the copies.
     *
     * Steps (MFEM's GroupCommunicator, communication.hpp:534 Reduce, :435 Bcast; the reduce
     * gathers each group's copies on the group's master rank and applies the operation there,
     * Bcast then sends the master's value back to every copy; the Min operation is
     * communication.cpp:1548):
     *  1. Min-reduce plus Bcast of T: every copy holds the smallest T.
     *  2. On each rank, label(v) = INT_MAX wherever the local T differed from that smallest T
     *     (this copy lost and must not contribute its label).
     *  3. Min-reduce plus Bcast of the labels: the smallest label among the copies that had the
     *     smallest T.
     * Unshared vertices are not touched.
     *
     * **Collective** (MPI_Allreduce for the return value). Host (moves T and label to the host).
     *
     * @param changed optional, resized to NV: changed[v] = 1 where T(v) or label(v) changed on
     *        this rank, else 0 (a solver restarts its local sweep from these vertices).
     * @return true if any vertex changed on any rank (same value on every rank).
     */
    bool ReduceShared(mfem::Array<int>* changed = nullptr);

    /**
     * @brief Step 1 of ReduceShared only: every copy of a shared vertex gets the smallest T over
     * the copies; labels are left as they are. For solvers that settle T first and labels later.
     * Collective. Host.
     * @param changed optional, resized to NV: 1 where T(v) changed on this rank.
     * @return true if any T changed on any rank.
     */
    bool ReduceSharedDistance(mfem::Array<int>* changed = nullptr);

    /**
     * @brief Min-reduce plus Bcast of the labels only (T is not used or changed): every copy of a
     * shared vertex gets the smallest label over the copies. Collective. Host.
     * @param changed optional, resized to NV: 1 where label(v) changed on this rank.
     * @return true if any label changed on any rank.
     */
    bool ReduceSharedLabels(mfem::Array<int>* changed = nullptr);

    /**
     * @brief Min-reduce plus Bcast of an arbitrary per-vertex array (not the field's own T or
     * label): every copy of a shared vertex gets the smallest value over the copies. Unshared
     * vertices keep their value. Used by solvers that need a per-vertex quantity other than T to
     * agree across ranks (the FIM label pass reduces the smallest recomputed update value).
     *
     * **Collective** (every rank must call it, also ranks without vertices). Host (moves `values`
     * to the host, HostReadWrite()). The same GroupCommunicator Reduce(Min) + Bcast pair as
     * ReduceShared.
     *
     * @param values per-vertex values, size NV (any finite or infinite doubles, no NaN).
     * @throws std::invalid_argument if values.Size() != NV (thrown on that rank only, before any
     *         communication; callers pass arrays sized from NumVertices(), so all ranks agree).
     */
    void ReduceSharedMinimum(mfem::Vector& values) const;

    /**
     * @brief Labels arbitrary points of a range of local elements from the vertex field.
     *
     * Point i belongs to local element e = elem_begin + i / points_per_element. It is reached in
     * a straight line from one of e's vertices, or, when e is the source element of seed j,
     * directly from the seed. Its distance and label are the lexicographic minimum of
     *   (T(v) + |p - v|, label(v))   over the vertices v of e, and
     *   (|p - s_j| - a_j, j)          over the seeds j whose source element is e,
     * i.e. the smallest arrival distance, then the smallest seed index. The direct term makes two
     * seeds in the same or neighboring elements compare by straight lines instead of by detours
     * through the vertices. A point whose element has no reached vertex and no seed gets distance
     * +infinity and label -1.
     *
     * Not collective (element-local). An mfem::forall kernel over the points (runs where the
     * mfem::Device runs kernels); the field must be consistent across ranks first (after a
     * ReduceShared that returned false) for the result to be partition independent.
     *
     * @param elem_begin         first 0-based local element.
     * @param elem_end           one past the last; 0 <= elem_begin <= elem_end <= NE.
     * @param points_per_element points per element, >= 1 (unused when the range is empty).
     * @param x, y, z            point coordinates, (elem_end - elem_begin) * points_per_element
     *                           entries each, element-major: entry (e - elem_begin) *
     *                           points_per_element + q is point q of element e (the
     *                           QuadraturePointSamples layout of mesh/mesh_sampler.hpp).
     * @param label              resized to the number of points: 0-based seed index or -1.
     * @param distance           optional, resized likewise: the arrival distance.
     * @throws std::invalid_argument for an invalid range, points_per_element < 1 with a
     *         non-empty range, or coordinate arrays of the wrong size.
     */
    void LabelPoints(int elem_begin,
                     int elem_end,
                     int points_per_element,
                     const mfem::Vector& x,
                     const mfem::Vector& y,
                     const mfem::Vector& z,
                     mfem::Array<int>& label,
                     mfem::Vector* distance = nullptr) const;

    /**
     * @brief Element label = LabelPoints at the element's volume centroid (ComputeElementSamples
     * of mesh/mesh_sampler.hpp, the same centroid the Euclidean centroid labeler uses).
     *
     * Not collective. Host setup plus device kernels.
     *
     * @param element_label    resized to NE: 0-based seed index, -1 for an unreached element.
     * @param element_distance optional, resized to NE: arrival distance at the centroid.
     */
    void LabelElementCentroids(mfem::Array<int>& element_label,
                               mfem::Vector* element_distance = nullptr) const;

    /**
     * @brief Element label by a volume-weighted vote of subcell points, the geodesic counterpart
     * of LabelElementsByQuadratureVote (labeling/quadrature_vote.hpp).
     *
     * The voting points are VoteRule(geometry, levels) evaluated by a QuadraturePointSampler with
     * weights w_q detJ_q; every point is labeled by LabelPoints; each element goes to the label
     * with the largest total weight, exact ties to the lowest seed index (totals summed in
     * increasing point order, so identical on every partition). Points labeled -1 do not vote;
     * an element none of whose points is reached gets -1.
     *
     * Not collective. Host setup plus device kernels, in chunks of whole elements.
     *
     * @param levels               subdivision levels 0..3 (8^levels points per element).
     * @param max_points_per_chunk upper bound on points processed at once (>= 1).
     * @param element_label        resized to NE: winning seed index or -1.
     * @return number of local elements whose voting points carried more than one label.
     * @throws std::invalid_argument for levels outside 0..3, max_points_per_chunk < 1 or an
     *         element geometry other than hex/tet; std::runtime_error for a mixed-geometry mesh.
     */
    int LabelElementsByVote(int levels,
                            int max_points_per_chunk,
                            mfem::Array<int>& element_label) const;

private:
    /** @brief Shared implementation of the three ReduceShared variants. */
    bool ReduceSharedImpl(bool reduce_T, bool reduce_labels, mfem::Array<int>* changed);

    mfem::ParMesh& m_pmesh;                          ///< the mesh (not owned)
    std::unique_ptr<mfem::H1_FECollection> m_fec;    ///< order-1 H1 collection
    std::unique_ptr<mfem::ParFiniteElementSpace> m_fes; ///< vertex numbering + GroupCommunicator
    int m_num_vertices = 0;                          ///< NV
    int m_num_elements = 0;                          ///< NE
    mfem::Array<int> m_elem_vert_offsets;            ///< element -> vertex CSR offsets (NE+1)
    mfem::Array<int> m_elem_verts;                   ///< element -> vertex CSR items
    mfem::Array<int> m_vert_elem_offsets;            ///< vertex -> element CSR offsets (NV+1)
    mfem::Array<int> m_vert_elems;                   ///< vertex -> element CSR items
    mfem::Vector m_coords;                           ///< xyz[3*v + d]
    mfem::Array<int> m_shared;                       ///< 1 = vertex also on another rank
    mfem::Array<int> m_owned;                        ///< 1 = this rank is the vertex's master
    mfem::Vector m_T;                                ///< T(v)
    mfem::Array<int> m_label;                        ///< label(v)
    mfem::Vector m_source_T;                         ///< T0(v)
    mfem::Array<int> m_source_label;                 ///< seed of T0(v)
    mfem::Array<int> m_elem_src_offsets;             ///< element -> own seeds CSR offsets (NE+1)
    mfem::Array<int> m_elem_src_seeds;               ///< seed index of each entry
    mfem::Vector m_elem_src_data;                    ///< per entry: x, y, z, a_j (4 doubles)
};

} // namespace exaconstit::microstructure
