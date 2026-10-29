/*
 * Implementation of the per-node distance field shared by the geodesic labelers
 * (vertex_field.hpp).
 *
 * Terms used in this file:
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it.
 * - dof (degree of freedom): one nodal unknown of a finite element space. Corner field: the
 *   order-1 H1 space, one dof per mesh vertex, numbered like the vertices. Nodal field: the
 *   scalar dofs of the mesh's Nodes space, i.e. its nodal points.
 * - Field node: a point that carries T and a label (a corner or a Nodes dof).
 * - Cell: a straight-sided piece of an element spanned by field nodes (the element itself for
 *   the corner field, a lattice sub-hex or sub-tet for the nodal field; see FieldCellView).
 * - Lattice: the nodal points of one high-order element indexed by (i, j, k) in its reference
 *   coordinates, 0 <= i, j, k <= p (tet: i + j + k <= p).
 * - GroupCommunicator: MFEM's object that knows, for every dof on a partition boundary, which
 *   other ranks hold a copy ("group" = set of ranks sharing some dofs; group 0 is this rank
 *   alone). One rank of each group is its "master".
 * - CSR (compressed sparse row): a list of lists stored as offsets + items (see the header).
 * - Lexicographic minimum of pairs (T, label): the pair with the smallest T, and among pairs with
 *   that T the smallest label.
 * - BVH (bounding volume hierarchy): a tree of nested bounding boxes that lets us skip field
 *   nodes that are provably too far from a seed (axom::spin::BVH), used to snap outside seeds.
 * - mfem::forall: MFEM's parallel loop; a GPU kernel when the mfem::Device has a CUDA or HIP
 *   backend, a plain host loop otherwise. MFEM_HOST_DEVICE marks code callable inside it.
 */
#include "microstructure/labeling/vertex_field.hpp"

#include "microstructure/labeling/quadrature_vote.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/mesh/straight_sided.hpp"
#include "microstructure/seeding/seed_element_locator.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <math.h> // unqualified sqrt in device code (C math library)
#include <stdexcept>
#include <string>
#include <vector>

#include "axom/core.hpp"
#include "axom/primal/geometry/BoundingBox.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/operators/squared_distance.hpp"
#include "axom/spin/BVH.hpp"

namespace exaconstit::microstructure {

namespace {

/** @brief +infinity, the "not reached" distance. */
constexpr double INF = std::numeric_limits<double>::infinity();

/** @brief "No key" marker of the nearest-node search (larger than any real key). */
constexpr std::uint64_t NO_KEY = std::numeric_limits<std::uint64_t>::max();

/** @brief A 3D point of the node search (Axom primal type). */
using BvhPoint = axom::primal::Point<double, 3>;
/** @brief An axis-aligned 3D box of the node search (one zero-size box per field node). */
using BvhBox = axom::primal::BoundingBox<double, 3>;
/** @brief The host (sequential) BVH used to find the field node nearest to an outside seed. */
using HostBVH = axom::spin::BVH<3, axom::SEQ_EXEC, double>;

/**
 * @brief Squared distance between point p and local node v of the coordinate table
 * xyz[3*v + d]. Written out in a fixed order (dx^2 + dy^2 + dz^2) so the same (point, node)
 * pair gives the same bits on every rank. Host and device.
 */
MFEM_HOST_DEVICE inline double
SquaredDistanceToVertex(const double* xyz, int v, double px, double py, double pz) {
    const double dx = px - xyz[3 * v + 0];
    const double dy = py - xyz[3 * v + 1];
    const double dz = pz - xyz[3 * v + 2];
    return dx * dx + dy * dy + dz * dz;
}

/**
 * @brief y = S p for the row-major 3x3 stretch S of VertexField::SetMetricStretch, written out
 * in a fixed order. Host and device.
 */
MFEM_HOST_DEVICE inline void
MapToMetric(const double* stretch, double px, double py, double pz, double* y) {
    y[0] = stretch[0] * px + stretch[1] * py + stretch[2] * pz;
    y[1] = stretch[3] * px + stretch[4] * py + stretch[5] * pz;
    y[2] = stretch[6] * px + stretch[7] * py + stretch[8] * pz;
}

/**
 * @brief The vote over per-point labels for one chunk of elements (one mfem::forall iteration
 * per element); the same rule as quadrature_vote.cpp's VoteChunk, but on labels that are already
 * seed indices.
 *
 * For each label that has not appeared at an earlier point of the element, its total weight is
 * summed over the points q..nq-1 in increasing order (so the sum is the same on every rank and
 * device) and compared with the best total so far: the larger total wins, an exactly equal total
 * goes to the lower label. Points labeled -1 (unreached) do not vote. Needs no scratch memory, so
 * it runs unchanged on a GPU. A named function (not a lambda inside another lambda) because CUDA
 * restricts device lambdas defined inside generic lambdas.
 *
 * @param chunk_begin   first local element of the chunk.
 * @param chunk_elems   elements in the chunk (> 0).
 * @param nq            points per element.
 * @param point_label   label per chunk point (size chunk_elems * nq).
 * @param weight        w_q detJ_q per chunk point.
 * @param element_label output at chunk_begin .. chunk_begin + chunk_elems - 1.
 * @param element_split output: 1 where the element's points carried more than one label.
 */
void VoteOnPointLabels(int chunk_begin,
                       int chunk_elems,
                       int nq,
                       const mfem::Array<int>& point_label,
                       const mfem::Vector& weight,
                       mfem::Array<int>& element_label,
                       mfem::Array<int>& element_split) {
    const int* lab = point_label.Read();
    const double* w = weight.Read();
    // ReadWrite: only this chunk's range of the whole-mesh arrays is written
    int* out = element_label.ReadWrite();
    int* split_out = element_split.ReadWrite();

    mfem::forall(chunk_elems, [=] MFEM_HOST_DEVICE(int local) {
        const int base = local * nq;
        int winner = -1;
        double winner_weight = 0.0;
        int first_label = -1;
        int split = 0;
        for (int q = 0; q < nq; ++q) {
            const int label = lab[base + q];
            if (label < 0) {
                continue; // unreached point: no vote
            }
            if (first_label < 0) {
                first_label = label;
            } else if (label != first_label) {
                split = 1;
            }
            // Count each label once: skip it if an earlier point already carried it
            bool seen_before = false;
            for (int p = 0; p < q; ++p) {
                if (lab[base + p] == label) {
                    seen_before = true;
                    break;
                }
            }
            if (seen_before) {
                continue;
            }
            // Total weight of this label, in increasing point order
            double total = 0.0;
            for (int p = q; p < nq; ++p) {
                if (lab[base + p] == label) {
                    total += w[base + p];
                }
            }
            // Larger total wins; exact tie -> lower seed index
            if (winner < 0 || total > winner_weight || (total == winner_weight && label < winner)) {
                winner = label;
                winner_weight = total;
            }
        }
        out[chunk_begin + local] = winner;
        split_out[chunk_begin + local] = split;
    });
}

/**
 * @brief Throws std::invalid_argument for invalid seed or weight data (identical on every rank,
 * the inputs being replicated, so every rank throws together without communication).
 */
void ValidateSeeds(const SeedSet& seeds, const std::vector<double>& additive_weights) {
    const std::size_t n = seeds.Size();
    if (n == 0) {
        throw std::invalid_argument("VertexField::InitializeSources: the seed set is empty");
    }
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("VertexField::InitializeSources: more than 2^31 - 1 seeds");
    }
    if (seeds.y.size() != n || seeds.z.size() != n) {
        throw std::invalid_argument("VertexField::InitializeSources: seed x, y, z sizes differ");
    }
    if (!additive_weights.empty() && additive_weights.size() != n) {
        throw std::invalid_argument("VertexField::InitializeSources: additive_weights must be "
                                    "empty or have one entry per seed");
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(seeds.x[i]) || !std::isfinite(seeds.y[i]) ||
            !std::isfinite(seeds.z[i])) {
            throw std::invalid_argument("VertexField::InitializeSources: seed " +
                                        std::to_string(i) + " has a non-finite position");
        }
        if (!additive_weights.empty() && !std::isfinite(additive_weights[i])) {
            throw std::invalid_argument("VertexField::InitializeSources: seed " +
                                        std::to_string(i) + " has a non-finite weight");
        }
    }
}

/**
 * @brief Fills the cell-type tables of a nodal-field view (FieldCellView::type_offset).
 *
 * Hex: one type, the 8 corners of a lattice cube in MFEM's hex vertex order (reference corners
 * (0,0,0), (1,0,0), (1,1,0), (0,1,0), then the same at z = 1).
 *
 * Tet: the Bey decomposition of the lattice (see the header): with anchor (i, j, k),
 *   type 0, up tet:    (0,0,0) (1,0,0) (0,1,0) (0,0,1);
 *   types 1-4, the octahedron with corners A = (1,0,0), B = (0,1,1), P = (0,1,0), Q = (1,1,0),
 *     R = (1,0,1), S = (0,0,1), cut around its diagonal A-B into (A,B,Q,P), (A,B,R,Q),
 *     (A,B,S,R), (A,B,P,S) (P, Q, R, S is the ring of corners around the diagonal: P is opposite
 *     R and Q opposite S);
 *   type 5, down tet:  (1,1,0) (1,0,1) (1,1,1) (0,1,1).
 * The corner order of every type gives a positive volume (det[c1-c0, c2-c0, c3-c0] = +1 in
 * lattice units, the orientation of MFEM's reference tet), so sub-cell volumes can be summed.
 * In a lattice cube with all 8 corners inside the tet these are the corner tet below the plane
 * i+j+k = s+1, the slab between s+1 and s+2 (an octahedron, 4/6 of the cube) and the corner tet
 * above s+2: 1/6 + 4/6 + 1/6 = the cube. A type placed at an anchor is a cell exactly when all its
 * corners are lattice points (every type has, per direction, a corner with offset 0, so this also
 * implies a non-negative anchor). Which of the three octahedron diagonals is used does not matter
 * for correctness; the faces on the element boundary are the same up/down triangles either way.
 */
void FillCellTypes(FieldCellView& view) {
    static const int hex[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    static const int tet[6][4][3] = {{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},  // up
                                     {{1, 0, 0}, {0, 1, 1}, {1, 1, 0}, {0, 1, 0}},  // A B Q P
                                     {{1, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}},  // A B R Q
                                     {{1, 0, 0}, {0, 1, 1}, {0, 0, 1}, {1, 0, 1}},  // A B S R
                                     {{1, 0, 0}, {0, 1, 1}, {0, 1, 0}, {0, 0, 1}},  // A B P S
                                     {{1, 1, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}}; // down
    if (view.tet == 0) {
        view.num_types = 1;
        view.nodes_per_cell = 8;
        for (int s = 0; s < 8; ++s) {
            for (int d = 0; d < 3; ++d) {
                view.type_offset[0][s][d] = hex[s][d];
            }
        }
    } else {
        view.num_types = 6;
        view.nodes_per_cell = 4;
        for (int t = 0; t < 6; ++t) {
            for (int s = 0; s < 4; ++s) {
                for (int d = 0; d < 3; ++d) {
                    view.type_offset[t][s][d] = tet[t][s][d];
                }
            }
        }
    }
}

/**
 * @brief The first cell of element e (in type, then corner-slot order) that has node v as a
 * corner, or -1. Used to give a snapped seed a source cell. Host only.
 */
int FirstCellWithNode(const FieldCellView& view, int v, int e) {
    if (view.curved == 0) {
        return 0;
    }
    const int lattice = view.IncidenceLattice(v, e);
    if (lattice < 0) {
        return -1;
    }
    const int lattice_size = view.stride * view.stride * view.stride;
    const int li = lattice % view.stride;
    const int lj = (lattice / view.stride) % view.stride;
    const int lk = lattice / (view.stride * view.stride);
    for (int t = 0; t < view.num_types; ++t) {
        for (int s = 0; s < view.nodes_per_cell; ++s) {
            // Anchor that puts corner s of type t on v; a cell if all corners exist
            const int a = li - view.type_offset[t][s][0];
            const int b = lj - view.type_offset[t][s][1];
            const int c = lk - view.type_offset[t][s][2];
            bool inside = true;
            for (int q = 0; q < view.nodes_per_cell && inside; ++q) {
                inside = view.ValidLattice(a + view.type_offset[t][q][0],
                                           b + view.type_offset[t][q][1],
                                           c + view.type_offset[t][q][2]);
            }
            if (inside) {
                return t * lattice_size + view.Pack(a, b, c);
            }
        }
    }
    return -1;
}

/**
 * @brief Reference coordinates of physical point (px, py, pz) in local element e of `mesh`:
 * mfem::InverseElementTransformation (eltrans.hpp:199-400) with the closest reference lattice
 * point as the initial guess and Newton steps projected back onto the element
 * (ClosestRefNode + NewtonElementProject, the combination MFEM calls the most reliable,
 * eltrans.hpp:285-290). A point outside the element (e.g. the centroid of a strongly curved
 * element) ends on the element boundary, which is what the cell search wants. Host only.
 */
mfem::IntegrationPoint ReferencePoint(mfem::ParMesh& mesh, int e, double px, double py, double pz) {
    mfem::ElementTransformation* trans = mesh.GetElementTransformation(e);
    mfem::InverseElementTransformation inverse(trans);
    inverse.SetInitialGuessType(mfem::InverseElementTransformation::ClosestRefNode);
    inverse.SetSolverType(mfem::InverseElementTransformation::NewtonElementProject);
    inverse.SetMaxIter(32);
    mfem::Vector point(3);
    point(0) = px;
    point(1) = py;
    point(2) = pz;
    mfem::IntegrationPoint ip;
    ip.Init(0);
    inverse.Transform(point, ip);
    return ip;
}

/**
 * @brief Result of NearestFieldNodes, per query: this rank's local index of the winning node when
 * this rank holds a copy of it, else -1 (rank-local; several ranks hold copies of a shared node).
 */
struct NearestNodeResult {
    std::vector<int> local_node; ///< per query: local index of the winner's copy here, or -1
};

/**
 * @brief For each query point, the field node nearest to it over all ranks: smallest squared
 * distance, ties by smallest VertexKey (seeding/seed_element_locator.hpp) of the node's
 * coordinates; returns, on every rank holding a copy of the winner, that copy's local index.
 *
 * This is NearestVertex of seed_element_locator.cpp applied to the field's own (owner-broadcast)
 * coordinates instead of Mesh::GetVertex: the same local search (a host BVH over zero-size boxes,
 * opened while "squared distance to the box <= best so far", so every exact tie reaches the leaf,
 * which keeps the lexicographic minimum of (d^2, key)), the same arithmetic and the same
 * reduction (MIN of d^2, then MIN of the keys at that d^2). For a corner field of a mesh without
 * Nodes the coordinates equal Mesh::GetVertex bit for bit, so the winner is exactly
 * NearestVertex's. A rank holds a copy of the winner exactly when its own local best has the
 * global (d^2, key): copies carry the same (owner-broadcast) coordinates, hence the same d^2 and
 * key, and nothing on this rank can beat the global minimum; so no lookup by position (and no
 * mesh-sized search table) is needed. With `node_used` (element mask) only nodes flagged 1 are
 * candidates; the flag is identical on all copies, so copies still agree. Collective (two
 * MPI_Allreduce calls). Host only.
 */
NearestNodeResult NearestFieldNodes(MPI_Comm comm,
                                    const mfem::Vector& coords,
                                    int num_nodes,
                                    const std::vector<double>& qx,
                                    const std::vector<double>& qy,
                                    const std::vector<double>& qz,
                                    const int* node_used) {
    const std::size_t nq = qx.size();
    const double* xyz = coords.HostRead();
    std::vector<double> best_d2(nq, DBL_MAX);
    std::vector<std::uint64_t> best_key(nq, NO_KEY);
    std::vector<int> best_node(nq, -1);

    // Local search over this rank's (used) nodes; candidate c is node candidate_node[c]
    std::vector<int> candidate_node;
    for (int v = 0; v < num_nodes; ++v) {
        if (node_used == nullptr || node_used[v] != 0) {
            candidate_node.push_back(v);
        }
    }
    const int num_candidates = static_cast<int>(candidate_node.size());
    if (num_candidates > 0 && nq > 0) {
        std::vector<std::uint64_t> keys(static_cast<std::size_t>(num_nodes));
        std::vector<BvhBox> boxes(static_cast<std::size_t>(num_candidates));
        for (int c = 0; c < num_candidates; ++c) {
            const int v = candidate_node[static_cast<std::size_t>(c)];
            keys[static_cast<std::size_t>(v)] = VertexKey(
                xyz[3 * v], xyz[3 * v + 1], xyz[3 * v + 2]);
            boxes[static_cast<std::size_t>(c)] = BvhBox(
                BvhPoint{xyz[3 * v], xyz[3 * v + 1], xyz[3 * v + 2]});
        }
        HostBVH bvh;
        bvh.initialize(boxes.data(), static_cast<axom::IndexType>(num_candidates));
        const auto traverser = bvh.getTraverser();
        for (std::size_t q = 0; q < nq; ++q) {
            double d2_best = DBL_MAX;
            std::uint64_t key_best = NO_KEY;
            int node_best = -1;
            // Leaf: exact squared distance, then the (d^2, key) order
            auto leaf_action = [&](std::int32_t leaf, const std::int32_t* leaf_nodes) {
                const int v = candidate_node[static_cast<std::size_t>(leaf_nodes[leaf])];
                const double dx = xyz[3 * v + 0] - qx[q];
                const double dy = xyz[3 * v + 1] - qy[q];
                const double dz = xyz[3 * v + 2] - qz[q];
                const double d2 = dx * dx + dy * dy + dz * dz;
                const std::uint64_t key = keys[static_cast<std::size_t>(v)];
                if (d2 < d2_best || (d2 == d2_best && key < key_best)) {
                    d2_best = d2;
                    key_best = key;
                    node_best = v;
                }
            };
            // Predicate: open a box unless it is provably farther than the best so far
            auto predicate = [&](const BvhPoint& query, const BvhBox& box) -> bool {
                return axom::primal::squared_distance(query, box) <= d2_best;
            };
            traverser.traverse_tree(BvhPoint{qx[q], qy[q], qz[q]}, leaf_action, predicate);
            best_d2[q] = d2_best;
            best_key[q] = key_best;
            best_node[q] = node_best;
        }
    }

    // Global (d^2, key) minimum, then the winner's coordinates
    const int n = static_cast<int>(nq);
    std::vector<double> global_d2(best_d2);
    if (n > 0) {
        MPI_Allreduce(MPI_IN_PLACE, global_d2.data(), n, MPI_DOUBLE, MPI_MIN, comm);
    }
    std::vector<std::uint64_t> key(best_key);
    for (std::size_t q = 0; q < nq; ++q) {
        if (best_d2[q] != global_d2[q]) {
            key[q] = NO_KEY; // another rank is nearer
        }
    }
    const std::vector<std::uint64_t> local_key(key);
    if (n > 0) {
        MPI_Allreduce(MPI_IN_PLACE, key.data(), n, MPI_UINT64_T, MPI_MIN, comm);
    }
    NearestNodeResult out;
    out.local_node.assign(nq, -1);
    for (std::size_t q = 0; q < nq; ++q) {
        if (key[q] != NO_KEY && local_key[q] == key[q]) {
            out.local_node[q] = best_node[q];
        }
    }
    return out;
}

} // namespace

/*
 * Constructor
 *
 * Steps:
 *  1. Collective checks (3D, conforming, continuous Nodes, Nodes present on all ranks or none),
 *     reduced so all ranks throw together.
 *  2. Choice of the field (FieldNodes): CORNERS -> corner field; MESH_NODES -> nodal field if the
 *     mesh has Nodes; AUTO -> nodal field if the mesh has Nodes and IsStraightSided
 *     (mesh/straight_sided.hpp, collective, the same answer on every rank) says it is curved.
 *     Every straight-sided mesh therefore takes the corner path.
 *  3. BuildCornerField or BuildNodalField (numbering, tables, coordinates, markers). With an
 *     element mask the node -> element table lists active elements only (BuildNodeToElementTable).
 *  3b. NodeUsed: 1 for nodes with an active element on some rank (collective minimum over the
 *     copies); all 1 without a mask, since every node belongs to an element.
 *  4. Field arrays at "not reached".
 */
VertexField::VertexField(mfem::ParMesh& pmesh, FieldNodes field_nodes, const ElementMask* mask)
    : m_pmesh(pmesh) {
    // Step 1: checks that are the same on every rank or reduced
    const MPI_Comm comm = pmesh.GetComm();
    const bool not_3d = pmesh.Dimension() != 3 || pmesh.SpaceDimension() != 3;
    ThrowIfAnyRank<std::runtime_error>(
        comm, not_3d, "VertexField: only 3D volume meshes are supported");
    ThrowIfAnyRank<std::runtime_error>(
        comm,
        pmesh.Nonconforming(),
        "VertexField: nonconforming meshes (hanging vertices) are not supported");
    mfem::GridFunction* nodes = pmesh.GetNodes();
    const bool bad_nodes = nodes != nullptr && nodes->FESpace()->FEColl()->GetContType() !=
                                                   mfem::FiniteElementCollection::CONTINUOUS;
    ThrowIfAnyRank<std::runtime_error>(
        comm,
        bad_nodes,
        "VertexField: discontinuous mesh Nodes (periodic meshes) are not supported");
    const bool has_nodes = AnyRank(nodes != nullptr, comm);
    ThrowIfAnyRank<std::runtime_error>(
        comm, has_nodes && nodes == nullptr, "VertexField: the mesh has Nodes on some ranks only");

    // Step 2: which field (the straightness of the active elements only), after the mask check
    m_num_elements = pmesh.GetNE();
    const bool bad_mask = mask != nullptr && mask->active.Size() != 0 &&
                          mask->active.Size() != m_num_elements;
    ThrowIfAnyRank<std::invalid_argument>(comm,
                                          bad_mask,
                                          "VertexField: the element mask needs one entry per "
                                          "local element");
    if (mask != nullptr) {
        m_mask.active = mask->active;
    }
    bool curved = false;
    if (field_nodes == FieldNodes::MESH_NODES) {
        curved = has_nodes;
    } else if (field_nodes == FieldNodes::AUTO && has_nodes) {
        curved = !IsStraightSided(pmesh, comm, &m_mask);
    }
    m_curved = curved;

    // Step 3: numbering, tables, coordinates and markers
    if (m_curved) {
        BuildNodalField(*nodes);
    } else {
        BuildCornerField(nodes);
    }

    // Step 3b: nodes in use (of an active element on some rank): 1 on a copy with a local active
    // element, then the maximum over the copies (as the minimum of 1 - used). Without a mask on
    // any rank every node belongs to an element and the exchange is skipped (a rank without
    // elements has no mask even when the others have one, hence the reduction)
    m_node_used.SetSize(m_num_vertices);
    if (!AnyRank(HasMask(), comm)) {
        m_node_used = 1;
    } else {
        const int* offsets = m_vert_elem_offsets.HostRead();
        mfem::Vector unused(m_num_vertices);
        for (int v = 0; v < m_num_vertices; ++v) {
            unused(v) = (offsets[v + 1] > offsets[v]) ? 0.0 : 1.0;
        }
        ReduceSharedMinimum(unused);
        for (int v = 0; v < m_num_vertices; ++v) {
            m_node_used[v] = (unused(v) == 0.0) ? 1 : 0;
        }
    }

    // Step 4: field arrays, all "not reached"
    m_distance.SetSize(m_num_vertices);
    m_distance.UseDevice(true);
    m_source_distance.SetSize(m_num_vertices);
    m_source_distance.UseDevice(true);
    m_label.SetSize(m_num_vertices);
    m_source_label.SetSize(m_num_vertices);
    Reset();
}

/* Destructor: the owned GroupCommunicator is freed by its unique_ptr; the mesh is not touched. */
VertexField::~VertexField() = default;

/*
 * BuildCornerField (field node = mesh vertex; no finite element space is built)
 *
 * Steps:
 *  1. Field node v = mesh vertex v (NV = Mesh::GetNV()).
 *  2. Element -> vertex CSR from Mesh::GetElementVertices (the element's vertices in element
 *     order: what GetElementDofs of an order-1 H1 space returns, since its dof v is vertex v,
 *     fespace.cpp:3536-3544), then vertex -> element CSR (BuildNodeToElementTable).
 *  3. Vertex coordinates: Mesh::GetVertex for a mesh without Nodes; for a mesh with Nodes the
 *     values of the vertex dofs of the (continuous) Nodes space, entry DofToVDof(v, d), since
 *     MFEM may leave the vertex array stale for high-order meshes (mesh.hpp:1426).
 *  4. The exchange communicator over the mesh's group topology (ParMesh::gtopo): row g >= 1 of
 *     its group table lists the local vertices ParMesh::GroupVertex(g, j), j = 0 .. in that order
 *     (the same order on every rank of the group), exactly the vertex block of
 *     ParFiniteElementSpace::GetGroupComm (pfespace.cpp:392-415) for an order-1 space; row 0
 *     (unshared) is empty; Finalize() allocates the buffers (communication.cpp:415). The same
 *     construction as mesh/vertex_coordinates.cpp. Then the owner's coordinates on every copy
 *     (BroadcastCoordinates).
 *  5. Shared markers (MarkSharedNodes) and owned markers: 1 when this rank is the master of the
 *     vertex's group (GroupTopology::IAmMaster; group 0, this rank alone, is always mastered
 *     here), the rule GetLocalTDofNumber applies to an H1 space (pfespace.cpp:1720-1739).
 * Memory: the two CSR tables and the per-vertex arrays; the communicator's tables are
 * proportional to the shared vertices.
 */
void VertexField::BuildCornerField(const mfem::GridFunction* nodes) {
    m_order = 1;

    // Step 1: one field node per mesh vertex
    m_num_vertices = m_pmesh.GetNV();

    // Step 2a: element -> vertex CSR
    m_elem_vert_offsets.SetSize(m_num_elements + 1);
    m_elem_vert_offsets[0] = 0;
    mfem::Array<int> verts;
    for (int e = 0; e < m_num_elements; ++e) {
        m_pmesh.GetElementVertices(e, verts);
        m_elem_vert_offsets[e + 1] = m_elem_vert_offsets[e] + verts.Size();
    }
    m_elem_verts.SetSize(m_elem_vert_offsets[m_num_elements]);
    for (int e = 0; e < m_num_elements; ++e) {
        m_pmesh.GetElementVertices(e, verts);
        for (int k = 0; k < verts.Size(); ++k) {
            m_elem_verts[m_elem_vert_offsets[e] + k] = verts[k];
        }
    }

    // Step 2b: vertex -> element CSR
    BuildNodeToElementTable();

    // Step 3: vertex coordinates xyz[3*v + d], then the owner's values on every copy
    m_coords.SetSize(3 * m_num_vertices);
    m_coords.UseDevice(true);
    double* xyz = m_coords.HostWrite();
    if (nodes == nullptr) {
        for (int v = 0; v < m_num_vertices; ++v) {
            const double* p = m_pmesh.GetVertex(v);
            for (int d = 0; d < 3; ++d) {
                xyz[3 * v + d] = p[d];
            }
        }
    } else {
        const mfem::FiniteElementSpace* nfes = nodes->FESpace();
        const double* nodal = nodes->HostRead();
        for (int v = 0; v < m_num_vertices; ++v) {
            for (int d = 0; d < 3; ++d) {
                xyz[3 * v + d] = nodal[nfes->DofToVDof(v, d)];
            }
        }
    }

    // Step 4: shared-vertex communicator over the mesh groups, then the owner's coordinates
    m_owned_gcomm = std::make_unique<mfem::GroupCommunicator>(m_pmesh.gtopo);
    mfem::Table& group_vertices = m_owned_gcomm->GroupLDofTable();
    const int num_groups = m_pmesh.GetNGroups();
    int num_entries = 0;
    for (int g = 1; g < num_groups; ++g) {
        num_entries += m_pmesh.GroupNVertices(g);
    }
    group_vertices.SetDims(num_groups, num_entries);
    int* row_start = group_vertices.GetI();
    int* entry = group_vertices.GetJ();
    row_start[0] = 0;
    int next = 0;
    std::vector<int> vertex_group(static_cast<std::size_t>(m_num_vertices), 0);
    for (int g = 1; g < num_groups; ++g) {
        row_start[g] = next;
        for (int j = 0; j < m_pmesh.GroupNVertices(g); ++j) {
            const int v = m_pmesh.GroupVertex(g, j);
            entry[next++] = v;
            vertex_group[static_cast<std::size_t>(v)] = g;
        }
    }
    row_start[num_groups] = next;
    m_owned_gcomm->Finalize();
    m_gcomm = m_owned_gcomm.get();
    BroadcastCoordinates();

    // Step 5: shared and owned markers
    MarkSharedNodes();
    m_owned.SetSize(m_num_vertices);
    for (int v = 0; v < m_num_vertices; ++v) {
        m_owned[v] = m_pmesh.gtopo.IAmMaster(vertex_group[static_cast<std::size_t>(v)]) ? 1 : 0;
    }
}

/*
 * BuildNodalField
 *
 * Steps:
 *  1. Checks (collective): the Nodes space is a fixed-order ParFiniteElementSpace (so it has a
 *     group topology for the exchange), and every active element on every rank has the same
 *     geometry, hex or tet (MPI_Allreduce MIN/MAX of the geometry code over ranks with active
 *     elements). Inactive elements may have another geometry: their rows of the element -> node
 *     table are not checked, they are absent from the node -> element table, and the labeling
 *     kernels skip them.
 *  2. Lattice tables from the order-p finite element of that geometry (identical on every rank,
 *     so their checks throw on every rank together):
 *       lattice_to_local[Pack(i,j,k)] = lex[o], o = position of (i,j,k) in lexicographic order;
 *       local_to_lattice = its inverse; ref_nodes = FiniteElement::GetNodes (fe_base.hpp:476);
 *       axis_points (hex) = reference coordinate of lattice index a along each axis.
 *     Checked: one-to-one, and reference coordinates strictly increasing along i, j and k.
 *  3. Element -> node CSR: a no-copy view of the Nodes space's element -> dof Table
 *     (Array::MakeRef on its I and J Memory, array.hpp:281). The table is const in MFEM's
 *     interface; MakeRef only needs a non-const Memory handle to create the alias and the arrays
 *     are never written. Each row must have exactly the element's dof count, entries in
 *     [0, NDofs).
 *  4. Node -> element CSR (BuildNodeToElementTable).
 *  5. Coordinates: Nodes value at DofToVDof(i, d) for a nodal basis; for a non-nodal (Bernstein)
 *     basis the element map evaluated at the node's reference point in the first local element
 *     holding it (ElementTransformation::Transform). Then the owner's value on every copy, through
 *     ScalarGroupComm (pfespace.cpp:1129-1141, GetGroupComm(gc, 0) = scalar dofs, finalized at
 *     pfespace.cpp:543).
 *  6. Shared markers, and owned markers from GetLocalTDofNumber of the dof's first vector
 *     component (ldof_group / ldof_ltdof are per vector dof, pfespace.cpp:2236-2257).
 */
void VertexField::BuildNodalField(mfem::GridFunction& nodes) {
    const MPI_Comm comm = m_pmesh.GetComm();

    // Step 1: the space and the geometry
    m_node_fes = dynamic_cast<mfem::ParFiniteElementSpace*>(nodes.FESpace());
    ThrowIfAnyRank<std::runtime_error>(
        comm,
        m_node_fes == nullptr,
        "VertexField: the nodal field needs mesh Nodes in a ParFiniteElementSpace");
    ThrowIfAnyRank<std::runtime_error>(comm,
                                       m_node_fes->IsVariableOrder(),
                                       "VertexField: variable-order mesh Nodes are not supported");
    int geom_range[2] = {INT_MAX, INT_MAX}; // {min, -max} so one MIN reduction gives both
    for (int e = 0; e < m_num_elements; ++e) {
        if (!ElementActive(e)) {
            continue; // inactive parts may have any geometry: they are never labeled
        }
        const int g = static_cast<int>(m_pmesh.GetElementGeometry(e));
        geom_range[0] = std::min(geom_range[0], g);
        geom_range[1] = std::min(geom_range[1], -g);
    }
    MPI_Allreduce(MPI_IN_PLACE, geom_range, 2, MPI_INT, MPI_MIN, comm);
    const bool no_elements = geom_range[0] == INT_MAX;
    const int geom_code = no_elements ? static_cast<int>(mfem::Geometry::CUBE) : geom_range[0];
    const bool supported = no_elements || (geom_range[0] == -geom_range[1] &&
                                           (geom_code == mfem::Geometry::CUBE ||
                                            geom_code == mfem::Geometry::TETRAHEDRON));
    if (!supported) {
        // Same reduced values on every rank: every rank throws
        throw std::runtime_error("VertexField: the nodal (curved) field needs active elements "
                                 "of only hexes or only tets");
    }
    const auto geom = static_cast<mfem::Geometry::Type>(geom_code);
    m_tet = (geom == mfem::Geometry::TETRAHEDRON) ? 1 : 0;
    const mfem::FiniteElement* fe = m_node_fes->FEColl()->FiniteElementForGeometry(geom);
    const int p = fe->GetOrder();
    const int ndof_el = fe->GetDof();
    m_order = p;

    // Step 2: lattice <-> native index tables (lexicographic map of the element)
    const mfem::Array<int>* lex = nullptr;
    if (const auto* nfe = dynamic_cast<const mfem::NodalFiniteElement*>(fe)) {
        lex = &nfe->GetLexicographicOrdering();
    } else if (const auto* tfe = dynamic_cast<const mfem::TensorBasisElement*>(fe)) {
        lex = &tfe->GetDofMap();
    } else if (const auto* pfe = dynamic_cast<const mfem::H1Pos_TetrahedronElement*>(fe)) {
        lex = &pfe->GetDofMap();
    }
    if (lex == nullptr) {
        throw std::runtime_error("VertexField: no lexicographic node ordering for the mesh "
                                 "Nodes element");
    }
    const int stride = p + 1;
    m_lattice_to_local.SetSize(stride * stride * stride);
    m_lattice_to_local = -1;
    m_local_to_lattice.SetSize(ndof_el);
    m_local_to_lattice = -1;
    int o = 0;
    bool one_to_one = true;
    for (int k = 0; k <= p; ++k) {
        for (int j = 0; j <= p; ++j) {
            for (int i = 0; i <= p; ++i) {
                if (m_tet != 0 && i + j + k > p) {
                    continue;
                }
                const int local = (lex->Size() > 0 && o < lex->Size()) ? (*lex)[o] : o;
                const int packed = i + stride * (j + stride * k);
                if (local < 0 || local >= ndof_el || m_local_to_lattice[local] != -1) {
                    one_to_one = false;
                } else {
                    m_lattice_to_local[packed] = local;
                    m_local_to_lattice[local] = packed;
                }
                ++o;
            }
        }
    }
    if (!one_to_one || o != ndof_el) {
        throw std::runtime_error("VertexField: the lexicographic node ordering of the mesh "
                                 "Nodes element is not a one-to-one lattice map");
    }
    const mfem::IntegrationRule& ref = fe->GetNodes();
    m_ref_nodes.SetSize(3 * ndof_el);
    m_ref_nodes.UseDevice(true);
    {
        double* r = m_ref_nodes.HostWrite();
        for (int l = 0; l < ndof_el; ++l) {
            r[3 * l + 0] = ref.IntPoint(l).x;
            r[3 * l + 1] = ref.IntPoint(l).y;
            r[3 * l + 2] = ref.IntPoint(l).z;
        }
    }
    // Monotone reference coordinates along every lattice direction (catches a wrong ordering)
    {
        const double* r = m_ref_nodes.HostRead();
        bool monotone = true;
        for (int k = 0; k <= p; ++k) {
            for (int j = 0; j <= p; ++j) {
                for (int i = 0; i <= p; ++i) {
                    const int here = m_lattice_to_local[i + stride * (j + stride * k)];
                    if (here < 0) {
                        continue;
                    }
                    const int step[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                    for (int d = 0; d < 3; ++d) {
                        const int ii = i + step[d][0];
                        const int jj = j + step[d][1];
                        const int kk = k + step[d][2];
                        if (ii > p || jj > p || kk > p) {
                            continue;
                        }
                        const int next = m_lattice_to_local[ii + stride * (jj + stride * kk)];
                        if (next >= 0 && !(r[3 * next + d] > r[3 * here + d])) {
                            monotone = false;
                        }
                    }
                }
            }
        }
        if (!monotone) {
            throw std::runtime_error("VertexField: the reference nodes of the mesh Nodes "
                                     "element do not follow its lexicographic ordering");
        }
    }
    m_axis_points.SetSize(3 * stride);
    m_axis_points.UseDevice(true);
    {
        const double* r = m_ref_nodes.HostRead();
        double* axis = m_axis_points.HostWrite();
        for (int a = 0; a <= p; ++a) {
            axis[3 * a + 0] = r[3 * m_lattice_to_local[a] + 0];
            axis[3 * a + 1] = r[3 * m_lattice_to_local[stride * a] + 1];
            axis[3 * a + 2] = r[3 * m_lattice_to_local[stride * stride * a] + 2];
        }
    }

    // Step 3: element -> node CSR as a view of MFEM's element -> dof table
    m_num_vertices = m_node_fes->GetNDofs();
    bool bad_table = false;
    if (m_num_elements > 0) {
        auto& table = const_cast<mfem::Table&>(m_node_fes->GetElementToDofTable());
        bad_table = table.Size() != m_num_elements;
        if (!bad_table) {
            m_elem_vert_offsets.MakeRef(table.GetIMemory(), 0, m_num_elements + 1);
            m_elem_verts.MakeRef(table.GetJMemory(), 0, table.Size_of_connections());
            const int* offsets = m_elem_vert_offsets.HostRead();
            const int* items = m_elem_verts.HostRead();
            for (int e = 0; e < m_num_elements && !bad_table; ++e) {
                if (!ElementActive(e)) {
                    continue;
                }
                bad_table = offsets[e + 1] - offsets[e] != ndof_el;
                for (int k = offsets[e]; k < offsets[e + 1] && !bad_table; ++k) {
                    bad_table = items[k] < 0 || items[k] >= m_num_vertices;
                }
            }
        }
    } else {
        m_elem_vert_offsets.SetSize(1);
        m_elem_vert_offsets[0] = 0;
        m_elem_verts.SetSize(0);
    }
    ThrowIfAnyRank<std::runtime_error>(
        comm, bad_table, "VertexField: unexpected element dof table of the mesh Nodes space");

    // Step 4: node -> element CSR, and the lattice point of the node of every incidence
    BuildNodeToElementTable();
    {
        const int* ve_offsets = m_vert_elem_offsets.HostRead();
        const int* ve_items = m_vert_elems.HostRead();
        const int* ev_offsets = m_elem_vert_offsets.HostRead();
        const int* ev_items = m_elem_verts.HostRead();
        m_vert_elem_lattice.SetSize(m_vert_elems.Size());
        int* lattice = m_vert_elem_lattice.HostWrite();
        for (int v = 0; v < m_num_vertices; ++v) {
            for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
                const int e = ve_items[k];
                lattice[k] = -1;
                for (int j = ev_offsets[e]; j < ev_offsets[e + 1]; ++j) {
                    if (ev_items[j] == v) {
                        lattice[k] = m_local_to_lattice[j - ev_offsets[e]];
                    }
                }
            }
        }
    }

    // Step 5: coordinates of the nodal points, then the owner's values on every copy
    m_coords.SetSize(3 * m_num_vertices);
    m_coords.UseDevice(true);
    {
        double* xyz = m_coords.HostWrite();
        const double* nodal = nodes.HostRead();
        for (int v = 0; v < m_num_vertices; ++v) {
            for (int d = 0; d < 3; ++d) {
                xyz[3 * v + d] = nodal[m_node_fes->DofToVDof(v, d)];
            }
        }
        const bool nodal_basis = dynamic_cast<const mfem::NodalFiniteElement*>(fe) != nullptr;
        if (!nodal_basis) {
            // Control points are not on the geometry: evaluate the map at the reference node
            const int* ve_offsets = m_vert_elem_offsets.HostRead();
            const int* ve_items = m_vert_elems.HostRead();
            const int* ev_offsets = m_elem_vert_offsets.HostRead();
            const int* ev_items = m_elem_verts.HostRead();
            mfem::Vector x(3);
            for (int v = 0; v < m_num_vertices; ++v) {
                if (ve_offsets[v + 1] == ve_offsets[v]) {
                    continue;
                }
                const int e = ve_items[ve_offsets[v]];
                int local = -1;
                for (int k = ev_offsets[e]; k < ev_offsets[e + 1]; ++k) {
                    if (ev_items[k] == v) {
                        local = k - ev_offsets[e];
                    }
                }
                mfem::ElementTransformation* trans = m_pmesh.GetElementTransformation(e);
                trans->Transform(ref.IntPoint(local), x);
                for (int d = 0; d < 3; ++d) {
                    xyz[3 * v + d] = x(d);
                }
            }
        }
    }
    m_owned_gcomm.reset(m_node_fes->ScalarGroupComm());
    m_gcomm = m_owned_gcomm.get();
    BroadcastCoordinates();

    // Step 6: shared and owned markers
    MarkSharedNodes();
    m_owned.SetSize(m_num_vertices);
    for (int v = 0; v < m_num_vertices; ++v) {
        m_owned[v] = (m_node_fes->GetLocalTDofNumber(m_node_fes->DofToVDof(v, 0)) >= 0) ? 1 : 0;
    }
}

/*
 * BuildNodeToElementTable: the transpose of the element -> node CSR by counting, a prefix sum and
 * a fill in increasing element order (so each node's elements are sorted).
 */
void VertexField::BuildNodeToElementTable() {
    const int* ev_offsets = m_elem_vert_offsets.HostRead();
    const int* ev_items = m_elem_verts.HostRead();
    m_vert_elem_offsets.SetSize(m_num_vertices + 1);
    m_vert_elem_offsets = 0;
    int* offsets = m_vert_elem_offsets.HostReadWrite();
    // Only active elements are listed (every element without a mask)
    for (int e = 0; e < m_num_elements; ++e) {
        if (!ElementActive(e)) {
            continue;
        }
        for (int k = ev_offsets[e]; k < ev_offsets[e + 1]; ++k) {
            offsets[ev_items[k] + 1] += 1;
        }
    }
    for (int v = 0; v < m_num_vertices; ++v) {
        offsets[v + 1] += offsets[v];
    }
    m_vert_elems.SetSize(offsets[m_num_vertices]);
    int* items = m_vert_elems.HostWrite();
    std::vector<int> fill(static_cast<std::size_t>(m_num_vertices), 0);
    for (int e = 0; e < m_num_elements; ++e) {
        if (!ElementActive(e)) {
            continue;
        }
        for (int k = ev_offsets[e]; k < ev_offsets[e + 1]; ++k) {
            const int v = ev_items[k];
            items[offsets[v] + fill[static_cast<std::size_t>(v)]++] = e;
        }
    }
}

/*
 * BroadcastCoordinates: the group master's coordinates are sent to every copy of a shared node
 * (GroupCommunicator::Bcast with layout 0, one entry per local dof, communication.hpp:435), one
 * array per coordinate, so all copies hold the same bits even if the mesh did not guarantee it.
 * Unshared nodes are not touched by Bcast.
 */
void VertexField::BroadcastCoordinates() {
    double* xyz = m_coords.HostReadWrite();
    std::vector<double> component(static_cast<std::size_t>(m_num_vertices));
    for (int d = 0; d < 3; ++d) {
        for (int v = 0; v < m_num_vertices; ++v) {
            component[static_cast<std::size_t>(v)] = xyz[3 * v + d];
        }
        m_gcomm->Bcast(component.data());
        for (int v = 0; v < m_num_vertices; ++v) {
            xyz[3 * v + d] = component[static_cast<std::size_t>(v)];
        }
    }
}

/*
 * SetMetricStretch: y = S x per node, in one host loop (MapToMetric), so every copy of a node,
 * whose physical coordinates are bitwise equal after BroadcastCoordinates, gets the same bits.
 * The identity clears the metric (no copy).
 */
void VertexField::SetMetricStretch(const std::array<double, 9>& stretch) {
    const std::array<double, 9> identity = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    m_stretch = stretch;
    m_has_metric = (stretch != identity);
    if (!m_has_metric) {
        m_metric_coords.Destroy();
        return;
    }
    m_metric_coords.SetSize(3 * m_num_vertices);
    m_metric_coords.UseDevice(true);
    const double* xyz = m_coords.HostRead();
    double* out = m_metric_coords.HostWrite();
    for (int v = 0; v < m_num_vertices; ++v) {
        MapToMetric(m_stretch.data(), xyz[3 * v + 0], xyz[3 * v + 1], xyz[3 * v + 2], &out[3 * v]);
    }
}

/*
 * ExchangeSharedPayload
 *
 * Neighbor exchange over the communicator's group topology (lproc = local index of a rank in the
 * topology, 0 = this rank): for every neighbor lproc p, the groups containing p, sorted by
 * (master rank, group number on the master), which both ranks compute identically for the groups
 * they share; the message holds the payloads of the group's nodes in GroupLDofTable row order
 * (the order GroupCommunicator relies on, identical on every rank of the group). Nonblocking
 * sends and receives with one tag, then a wait for all. Each shared node belongs to one group, so
 * it receives exactly (group size - 1) payloads.
 */
void VertexField::ExchangeSharedPayload(int width,
                                        const std::vector<double>& send,
                                        std::vector<double>& recv,
                                        std::vector<int>& offsets) const {
    if (width < 1) {
        throw std::invalid_argument("VertexField::ExchangeSharedPayload: width must be >= 1");
    }
    if (send.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(m_num_vertices)) {
        throw std::invalid_argument("VertexField::ExchangeSharedPayload: send must have width * NV "
                                    "entries");
    }
    const mfem::GroupTopology& gt = m_gcomm->GetGroupTopology();
    const mfem::Table& group_ldof = m_gcomm->GroupLDofTable();
    const int num_groups = gt.NGroups();
    const int num_lprocs = gt.GetNumNeighbors();

    // Receive layout: (group size - 1) payloads per shared node
    offsets.assign(static_cast<std::size_t>(m_num_vertices) + 1, 0);
    for (int g = 1; g < num_groups && g < group_ldof.Size(); ++g) {
        const int copies = gt.GetGroupSize(g) - 1;
        const int* row = group_ldof.GetRow(g);
        for (int j = 0; j < group_ldof.RowSize(g); ++j) {
            offsets[static_cast<std::size_t>(row[j]) + 1] += copies;
        }
    }
    for (int v = 0; v < m_num_vertices; ++v) {
        offsets[static_cast<std::size_t>(v) + 1] += offsets[static_cast<std::size_t>(v)];
    }
    recv.assign(static_cast<std::size_t>(offsets.back()) * static_cast<std::size_t>(width), 0.0);
    std::vector<int> fill(static_cast<std::size_t>(m_num_vertices), 0);

    // Groups shared with each neighbor, in the canonical order
    std::vector<std::vector<int>> groups_of(static_cast<std::size_t>(num_lprocs));
    for (int g = 1; g < num_groups && g < group_ldof.Size(); ++g) {
        const int* members = gt.GetGroup(g);
        for (int q = 0; q < gt.GetGroupSize(g); ++q) {
            if (members[q] != 0) {
                groups_of[static_cast<std::size_t>(members[q])].push_back(g);
            }
        }
    }
    const auto key = [&](int g) {
        return std::make_pair(gt.GetGroupMasterRank(g), gt.GetGroupMasterGroup(g));
    };
    std::vector<std::vector<double>> out(static_cast<std::size_t>(num_lprocs));
    std::vector<std::vector<double>> in(static_cast<std::size_t>(num_lprocs));
    std::vector<MPI_Request> requests;
    const MPI_Comm comm = Comm();
    constexpr int tag = 4711;
    for (int p = 1; p < num_lprocs; ++p) {
        auto& groups = groups_of[static_cast<std::size_t>(p)];
        std::sort(groups.begin(), groups.end(), [&](int a, int b) {
            return key(a) < key(b);
        });
        std::size_t count = 0;
        for (const int g : groups) {
            count += static_cast<std::size_t>(group_ldof.RowSize(g));
        }
        auto& buf = out[static_cast<std::size_t>(p)];
        buf.reserve(count * static_cast<std::size_t>(width));
        for (const int g : groups) {
            const int* row = group_ldof.GetRow(g);
            for (int j = 0; j < group_ldof.RowSize(g); ++j) {
                const double* src = &send[static_cast<std::size_t>(row[j]) * width];
                buf.insert(buf.end(), src, src + width);
            }
        }
        in[static_cast<std::size_t>(p)].resize(buf.size());
        const int rank = gt.GetNeighborRank(p);
        requests.emplace_back();
        MPI_Irecv(in[static_cast<std::size_t>(p)].data(),
                  static_cast<int>(buf.size()),
                  MPI_DOUBLE,
                  rank,
                  tag,
                  comm,
                  &requests.back());
        requests.emplace_back();
        MPI_Isend(buf.data(),
                  static_cast<int>(buf.size()),
                  MPI_DOUBLE,
                  rank,
                  tag,
                  comm,
                  &requests.back());
    }
    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
    // Unpack in the same order (neighbors in increasing lproc)
    for (int p = 1; p < num_lprocs; ++p) {
        const auto& buf = in[static_cast<std::size_t>(p)];
        std::size_t pos = 0;
        for (const int g : groups_of[static_cast<std::size_t>(p)]) {
            const int* row = group_ldof.GetRow(g);
            for (int j = 0; j < group_ldof.RowSize(g); ++j) {
                const int v = row[j];
                const auto vs = static_cast<std::size_t>(v);
                const auto slot = static_cast<std::size_t>(offsets[vs] + fill[vs]++);
                std::copy(&buf[pos], &buf[pos] + width, &recv[slot * width]);
                pos += static_cast<std::size_t>(width);
            }
        }
    }
}

/*
 * ExchangeSharedLists
 *
 * The groups shared with each neighbor in the order of ExchangeSharedPayload. Per neighbor the
 * send buffer holds, node by node, the length of the node's list and its entries. The buffer
 * lengths are exchanged first (one int per neighbor), then the buffers. Unpacking walks the
 * received buffer in the same group order; two passes over the received buffers (count, then
 * copy) build the receive CSR.
 */
void VertexField::ExchangeSharedLists(const std::vector<int>& send_offsets,
                                      const std::vector<double>& send_items,
                                      std::vector<int>& recv_offsets,
                                      std::vector<double>& recv_items) const {
    bool bad = send_offsets.size() != static_cast<std::size_t>(m_num_vertices) + 1 ||
               send_offsets.front() != 0 ||
               static_cast<std::size_t>(send_offsets.back()) != send_items.size();
    for (std::size_t v = 0; !bad && v + 1 < send_offsets.size(); ++v) {
        bad = send_offsets[v + 1] < send_offsets[v];
    }
    if (bad) {
        throw std::invalid_argument("VertexField::ExchangeSharedLists: send_offsets must have NV + "
                                    "1 non-decreasing entries from 0 to the number of items");
    }
    const mfem::GroupTopology& gt = m_gcomm->GetGroupTopology();
    const mfem::Table& group_ldof = m_gcomm->GroupLDofTable();
    const int num_groups = gt.NGroups();
    const int num_lprocs = gt.GetNumNeighbors();

    std::vector<std::vector<int>> groups_of(static_cast<std::size_t>(num_lprocs));
    for (int g = 1; g < num_groups && g < group_ldof.Size(); ++g) {
        const int* members = gt.GetGroup(g);
        for (int q = 0; q < gt.GetGroupSize(g); ++q) {
            if (members[q] != 0) {
                groups_of[static_cast<std::size_t>(members[q])].push_back(g);
            }
        }
    }
    const auto key = [&](int g) {
        return std::make_pair(gt.GetGroupMasterRank(g), gt.GetGroupMasterGroup(g));
    };
    std::vector<std::vector<double>> out(static_cast<std::size_t>(num_lprocs));
    std::vector<std::vector<double>> in(static_cast<std::size_t>(num_lprocs));
    std::vector<int> out_size(static_cast<std::size_t>(num_lprocs), 0);
    std::vector<int> in_size(static_cast<std::size_t>(num_lprocs), 0);
    for (int p = 1; p < num_lprocs; ++p) {
        auto& groups = groups_of[static_cast<std::size_t>(p)];
        std::sort(groups.begin(), groups.end(), [&](int a, int b) {
            return key(a) < key(b);
        });
        auto& buf = out[static_cast<std::size_t>(p)];
        for (const int g : groups) {
            const int* row = group_ldof.GetRow(g);
            for (int j = 0; j < group_ldof.RowSize(g); ++j) {
                const auto v = static_cast<std::size_t>(row[j]);
                buf.push_back(static_cast<double>(send_offsets[v + 1] - send_offsets[v]));
                buf.insert(buf.end(),
                           send_items.begin() + send_offsets[v],
                           send_items.begin() + send_offsets[v + 1]);
            }
        }
        out_size[static_cast<std::size_t>(p)] = static_cast<int>(buf.size());
    }
    const MPI_Comm comm = Comm();
    constexpr int size_tag = 4712;
    constexpr int data_tag = 4713;
    std::vector<MPI_Request> requests;
    for (int p = 1; p < num_lprocs; ++p) {
        const int rank = gt.GetNeighborRank(p);
        requests.emplace_back();
        MPI_Irecv(&in_size[static_cast<std::size_t>(p)],
                  1,
                  MPI_INT,
                  rank,
                  size_tag,
                  comm,
                  &requests.back());
        requests.emplace_back();
        MPI_Isend(&out_size[static_cast<std::size_t>(p)],
                  1,
                  MPI_INT,
                  rank,
                  size_tag,
                  comm,
                  &requests.back());
    }
    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
    requests.clear();
    for (int p = 1; p < num_lprocs; ++p) {
        const int rank = gt.GetNeighborRank(p);
        auto& buf = in[static_cast<std::size_t>(p)];
        buf.resize(static_cast<std::size_t>(in_size[static_cast<std::size_t>(p)]));
        requests.emplace_back();
        MPI_Irecv(buf.data(),
                  static_cast<int>(buf.size()),
                  MPI_DOUBLE,
                  rank,
                  data_tag,
                  comm,
                  &requests.back());
        requests.emplace_back();
        MPI_Isend(out[static_cast<std::size_t>(p)].data(),
                  out_size[static_cast<std::size_t>(p)],
                  MPI_DOUBLE,
                  rank,
                  data_tag,
                  comm,
                  &requests.back());
    }
    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }

    // Pass 1: received entries per node; pass 2: copy
    recv_offsets.assign(static_cast<std::size_t>(m_num_vertices) + 1, 0);
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<int> fill;
        if (pass == 1) {
            for (int v = 0; v < m_num_vertices; ++v) {
                recv_offsets[static_cast<std::size_t>(v) + 1] +=
                    recv_offsets[static_cast<std::size_t>(v)];
            }
            recv_items.assign(static_cast<std::size_t>(recv_offsets.back()), 0.0);
            fill.assign(recv_offsets.begin(), recv_offsets.end() - 1);
        }
        for (int p = 1; p < num_lprocs; ++p) {
            const auto& buf = in[static_cast<std::size_t>(p)];
            std::size_t pos = 0;
            for (const int g : groups_of[static_cast<std::size_t>(p)]) {
                const int* row = group_ldof.GetRow(g);
                for (int j = 0; j < group_ldof.RowSize(g); ++j) {
                    const auto v = static_cast<std::size_t>(row[j]);
                    const int count = static_cast<int>(buf[pos++]);
                    if (pass == 0) {
                        recv_offsets[v + 1] += count;
                    } else {
                        std::copy(buf.data() + pos,
                                  buf.data() + pos + count,
                                  recv_items.data() + fill[v]);
                        fill[v] += count;
                    }
                    pos += static_cast<std::size_t>(count);
                }
            }
        }
    }
}

/*
 * SetSeedLists: copies of the lists and weights (host write, read by the labeling kernels).
 */
void VertexField::SetSeedLists(int capacity,
                               const mfem::Array<int>& seeds,
                               const mfem::Vector& dist,
                               const std::vector<double>& weights,
                               const std::vector<double>& seed_stretch) {
    const std::size_t size = static_cast<std::size_t>(capacity) *
                             static_cast<std::size_t>(m_num_vertices);
    if (capacity < 1 || static_cast<std::size_t>(seeds.Size()) != size ||
        static_cast<std::size_t>(dist.Size()) != size || weights.empty()) {
        throw std::invalid_argument("VertexField::SetSeedLists: capacity must be >= 1, the lists "
                                    "must have capacity * NV entries and the weights one per "
                                    "seed");
    }
    m_list_capacity = capacity;
    m_list_seed.SetSize(seeds.Size());
    m_list_seed.HostWrite();
    m_list_seed = seeds;
    m_list_dist.SetSize(dist.Size());
    m_list_dist.UseDevice(true);
    m_list_dist = dist;
    m_list_weight.SetSize(static_cast<int>(weights.size()));
    m_list_weight.UseDevice(true);
    double* w = m_list_weight.HostWrite();
    for (std::size_t i = 0; i < weights.size(); ++i) {
        w[i] = weights[i];
    }
    if (!seed_stretch.empty() && (seed_stretch.size() != 9 * weights.size() || m_has_metric)) {
        throw std::invalid_argument("VertexField::SetSeedLists: seed_stretch must have 9 entries "
                                    "per seed, and the field no metric of its own");
    }
    m_list_stretch.SetSize(static_cast<int>(seed_stretch.size()));
    m_list_stretch.UseDevice(true);
    if (!seed_stretch.empty()) {
        double* st = m_list_stretch.HostWrite();
        for (std::size_t k = 0; k < seed_stretch.size(); ++k) {
            st[k] = seed_stretch[k];
        }
    }
}

/*
 * MarkSharedNodes: the GroupCommunicator's group -> dof table lists, in rows 1.., the dofs shared
 * with other ranks (communication.hpp:291); those nodes are marked 1.
 */
void VertexField::MarkSharedNodes() {
    m_shared.SetSize(m_num_vertices);
    m_shared = 0;
    const mfem::Table& group_ldof = m_gcomm->GroupLDofTable();
    for (int g = 1; g < group_ldof.Size(); ++g) {
        const int* row = group_ldof.GetRow(g);
        for (int k = 0; k < group_ldof.RowSize(g); ++k) {
            m_shared[row[k]] = 1;
        }
    }
}

/* Comm: the mesh communicator (ParMesh::GetComm), used by every collective member. */
MPI_Comm VertexField::Comm() const {
    return m_pmesh.GetComm();
}

/*
 * CellView / HostCellView: the plain view of FieldCellView with device (Read) or host (HostRead)
 * pointers into this object's arrays, plus the cell-type tables for the nodal field.
 */
FieldCellView VertexField::CellView() const {
    FieldCellView view;
    view.curved = m_curved ? 1 : 0;
    view.order = m_order;
    view.stride = m_order + 1;
    view.tet = m_tet;
    view.elem_offsets = m_elem_vert_offsets.Read();
    view.elem_nodes = m_elem_verts.Read();
    if (m_curved) {
        FillCellTypes(view);
        view.lattice_to_local = m_lattice_to_local.Read();
        view.local_to_lattice = m_local_to_lattice.Read();
        view.axis_points = m_axis_points.Read();
        view.incidence_lattice = m_vert_elem_lattice.Read();
        view.ref_nodes = m_ref_nodes.Read();
    }
    return view;
}

FieldCellView VertexField::HostCellView() const {
    FieldCellView view;
    view.curved = m_curved ? 1 : 0;
    view.order = m_order;
    view.stride = m_order + 1;
    view.tet = m_tet;
    view.elem_offsets = m_elem_vert_offsets.HostRead();
    view.elem_nodes = m_elem_verts.HostRead();
    if (m_curved) {
        FillCellTypes(view);
        view.lattice_to_local = m_lattice_to_local.HostRead();
        view.local_to_lattice = m_local_to_lattice.HostRead();
        view.axis_points = m_axis_points.HostRead();
        view.incidence_lattice = m_vert_elem_lattice.HostRead();
        view.ref_nodes = m_ref_nodes.HostRead();
    }
    return view;
}

/*
 * Reset: host fill of T = T0 = +inf and label = label0 = -1, and empty per-element seed lists
 * (all CSR offsets 0).
 */
void VertexField::Reset() {
    m_elem_src_offsets.SetSize(m_num_elements + 1);
    m_elem_src_offsets = 0;
    m_elem_src_seeds.SetSize(0);
    m_elem_src_cell.SetSize(0);
    m_elem_src_data.SetSize(0);
    double* t = m_distance.HostWrite();
    double* t0 = m_source_distance.HostWrite();
    int* lab = m_label.HostWrite();
    int* lab0 = m_source_label.HostWrite();
    for (int v = 0; v < m_num_vertices; ++v) {
        t[v] = INF;
        t0[v] = INF;
        lab[v] = -1;
        lab0[v] = -1;
    }
}

/*
 * InitializeSources
 *
 * Steps:
 *  1. Validate the replicated inputs (no communication needed: every rank throws together).
 *  2. Containment: ContainingElementKeys (seeding/seed_element_locator.hpp) gives, per seed, the
 *     smallest key of an element containing it over all ranks (BVH over the element boxes, the
 *     exact point-in-element test, MPI_Allreduce(MIN)); UINT64_MAX = "in no element".
 *  3. Snapping (only for seeds in no element): NearestFieldNodes gives the nearest field node
 *     over all ranks (smallest squared distance, ties by smallest VertexKey of the
 *     owner-broadcast Coordinates()) and, on every rank holding a copy, its local index. Each
 *     such rank takes the smallest key of the local elements containing it, and
 *     MPI_Allreduce(MIN) gives the source element.
 *  4. On the owning rank of each source element: the local index, the source cell (corner field:
 *     0; nodal field: LocateCell at the seed's reference coordinates, or for a snapped seed the
 *     first cell having the node as a corner), the per-element seed list (with the cell, for the
 *     direct term of LabelPoints), and (|v - s_i| - a_i, i) on the cell's corners, keeping the
 *     lexicographic minimum. Seeds are processed in increasing index order with a strict "<" on
 *     the value, so equal values keep the smaller seed index. ReduceShared spreads the values.
 * Cost: one BVH query per seed, plus a node BVH and lookups when seeds are outside; one element
 * inversion per seed for the nodal field.
 */
SourceReport VertexField::InitializeSources(const SeedSet& seeds,
                                            const std::vector<double>& additive_weights,
                                            const MaterialContext* context) {
    // Step 1: inputs and an active element somewhere
    ValidateSeeds(seeds, additive_weights);
    const MPI_Comm comm = Comm();
    long long global_elements = 0;
    for (int e = 0; e < m_num_elements; ++e) {
        global_elements += ElementActive(e) ? 1 : 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &global_elements, 1, MPI_LONG_LONG, MPI_SUM, comm);
    if (global_elements == 0) {
        throw std::runtime_error("VertexField::InitializeSources: the mesh has no elements");
    }
    const int n = static_cast<int>(seeds.Size());
    const std::uint64_t none = std::numeric_limits<std::uint64_t>::max();
    const std::vector<std::uint64_t> keys = ComputeElementKeys(m_pmesh);
    const FieldCellView view = HostCellView();

    SourceReport report;
    report.source_element_local.assign(static_cast<std::size_t>(n), -1);
    report.source_cell_local.assign(static_cast<std::size_t>(n), -1);
    report.seed_outside.assign(static_cast<std::size_t>(n), 0);

    // Step 2: smallest key of an element containing each seed (collective)
    // (only active elements: the shared locator of the context, or one built over the field's
    // mask)
    MaterialContext own_context;
    own_context.mask = MaskOrNull();
    if (context != nullptr) {
        own_context.locator = context->locator;
    }
    report.source_element_key = ContainingElementKeys(m_pmesh, comm, seeds, &keys, &own_context);

    // Step 3: seeds in no element snap to an element around the nearest field node
    std::vector<int> outside;
    for (int i = 0; i < n; ++i) {
        if (report.source_element_key[static_cast<std::size_t>(i)] == none) {
            outside.push_back(i);
            report.seed_outside[static_cast<std::size_t>(i)] = 1;
        }
    }
    report.num_seeds_outside = static_cast<int>(outside.size());
    std::vector<int> snapped_node(static_cast<std::size_t>(n), -1); // local copy, if any
    if (!outside.empty()) {
        // Step 3a: nearest field node over all ranks (collective; same list on every rank)
        const int num_out = static_cast<int>(outside.size());
        std::vector<double> qx(static_cast<std::size_t>(num_out));
        std::vector<double> qy(static_cast<std::size_t>(num_out));
        std::vector<double> qz(static_cast<std::size_t>(num_out));
        for (int j = 0; j < num_out; ++j) {
            const auto s = static_cast<std::size_t>(outside[static_cast<std::size_t>(j)]);
            qx[static_cast<std::size_t>(j)] = seeds.x[s];
            qy[static_cast<std::size_t>(j)] = seeds.y[s];
            qz[static_cast<std::size_t>(j)] = seeds.z[s];
        }
        const NearestNodeResult nearest = NearestFieldNodes(comm,
                                                            m_coords,
                                                            m_num_vertices,
                                                            qx,
                                                            qy,
                                                            qz,
                                                            HasMask() ? m_node_used.HostRead()
                                                                      : nullptr);
        // Step 3b: this rank's copy of that node (if any), then its elements' keys
        std::vector<std::uint64_t> snap_key(static_cast<std::size_t>(num_out), none);
        for (int j = 0; j < num_out; ++j) {
            const int v = nearest.local_node[static_cast<std::size_t>(j)];
            if (v < 0) {
                continue; // the node is not on this rank
            }
            snapped_node[static_cast<std::size_t>(outside[static_cast<std::size_t>(j)])] = v;
            for (int k = m_vert_elem_offsets[v]; k < m_vert_elem_offsets[v + 1]; ++k) {
                const std::uint64_t key = keys[static_cast<std::size_t>(m_vert_elems[k])];
                snap_key[static_cast<std::size_t>(j)] = std::min(
                    snap_key[static_cast<std::size_t>(j)], key);
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, snap_key.data(), num_out, MPI_UINT64_T, MPI_MIN, comm);
        for (int j = 0; j < num_out; ++j) {
            report.source_element_key[static_cast<std::size_t>(
                outside[static_cast<std::size_t>(j)])] = snap_key[static_cast<std::size_t>(j)];
        }
    }

    // Step 4a: local index of each source element on its owning rank. Element keys are unique
    // within a mesh (test_fragments checks this), so at most one local element matches.
    if (m_num_elements > 0) {
        std::vector<std::pair<std::uint64_t, int>> sorted_keys(
            static_cast<std::size_t>(m_num_elements));
        for (int e = 0; e < m_num_elements; ++e) {
            sorted_keys[static_cast<std::size_t>(e)] = {keys[static_cast<std::size_t>(e)], e};
        }
        std::sort(sorted_keys.begin(), sorted_keys.end());
        for (int i = 0; i < n; ++i) {
            const std::uint64_t key = report.source_element_key[static_cast<std::size_t>(i)];
            const auto it = std::lower_bound(
                sorted_keys.begin(), sorted_keys.end(), std::make_pair(key, -1));
            if (it != sorted_keys.end() && it->first == key) {
                report.source_element_local[static_cast<std::size_t>(i)] = it->second;
            }
        }
    }

    // Step 4b: source cell inside each local source element
    for (int i = 0; i < n; ++i) {
        const auto s = static_cast<std::size_t>(i);
        const int e = report.source_element_local[s];
        if (e < 0) {
            continue;
        }
        int cell = 0;
        if (m_curved) {
            if (report.seed_outside[s] != 0) {
                cell = FirstCellWithNode(view, snapped_node[s], e);
            } else {
                const mfem::IntegrationPoint ip = ReferencePoint(
                    m_pmesh, e, seeds.x[s], seeds.y[s], seeds.z[s]);
                cell = view.LocateCell(ip.x, ip.y, ip.z);
            }
        }
        report.source_cell_local[s] = cell;
    }

    // Step 4c: per-element seed lists (CSR over the local elements, seeds in increasing index)
    Reset();
    {
        int* offsets = m_elem_src_offsets.HostReadWrite();
        for (int i = 0; i < n; ++i) {
            const int e = report.source_element_local[static_cast<std::size_t>(i)];
            if (e >= 0) {
                offsets[e + 1] += 1;
            }
        }
        for (int e = 0; e < m_num_elements; ++e) {
            offsets[e + 1] += offsets[e];
        }
        const int num_entries = offsets[m_num_elements];
        m_elem_src_seeds.SetSize(num_entries);
        m_elem_src_cell.SetSize(num_entries);
        m_elem_src_data.SetSize(4 * num_entries);
        m_elem_src_data.UseDevice(true);
        int* entry_seed = m_elem_src_seeds.HostWrite();
        int* entry_cell = m_elem_src_cell.HostWrite();
        double* entry_data = m_elem_src_data.HostWrite();
        std::vector<int> fill(static_cast<std::size_t>(m_num_elements), 0);
        for (int i = 0; i < n; ++i) {
            const auto s = static_cast<std::size_t>(i);
            const int e = report.source_element_local[s];
            if (e < 0) {
                continue;
            }
            const int k = offsets[e] + fill[static_cast<std::size_t>(e)]++;
            entry_seed[k] = i;
            entry_cell[k] = report.source_cell_local[s];
            // Seed position in the metric (y = S s; the plain position without a metric)
            double ys[3] = {seeds.x[s], seeds.y[s], seeds.z[s]};
            if (m_has_metric) {
                MapToMetric(m_stretch.data(), seeds.x[s], seeds.y[s], seeds.z[s], ys);
            }
            entry_data[4 * k + 0] = ys[0];
            entry_data[4 * k + 1] = ys[1];
            entry_data[4 * k + 2] = ys[2];
            entry_data[4 * k + 3] = additive_weights.empty() ? 0.0 : additive_weights[s];
        }
    }

    // Step 4d: source values at the corners of each local source cell
    {
        // Distances in the metric: mapped nodes and mapped seed positions
        const double* xyz = MetricCoordinates().HostRead();
        double* t = m_distance.HostReadWrite();
        int* lab = m_label.HostReadWrite();
        int corners[FieldCellView::MAX_CELL_NODES];
        for (int i = 0; i < n; ++i) {
            const auto s = static_cast<std::size_t>(i);
            const int e = report.source_element_local[s];
            if (e < 0) {
                continue;
            }
            const double a = additive_weights.empty() ? 0.0 : additive_weights[s];
            double ys[3] = {seeds.x[s], seeds.y[s], seeds.z[s]};
            if (m_has_metric) {
                MapToMetric(m_stretch.data(), seeds.x[s], seeds.y[s], seeds.z[s], ys);
            }
            const int nc = view.CellNodes(e, report.source_cell_local[s], corners);
            for (int k = 0; k < nc; ++k) {
                const int v = corners[k];
                const double value = std::sqrt(
                                         SquaredDistanceToVertex(xyz, v, ys[0], ys[1], ys[2])) -
                                     a;
                // Lexicographic minimum: seeds come in increasing index, so "<" keeps the lower
                // index on equal values
                if (value < t[v]) {
                    t[v] = value;
                    lab[v] = i;
                }
            }
        }
    }

    // Step 4e: make all copies agree, then remember the result as the source values
    ReduceShared();
    m_source_distance = m_distance;
    m_source_label = m_label;
    return report;
}

/*
 * ReduceSharedImpl
 *
 * Host implementation of the three exchange variants (see ReduceShared in the header for the
 * lexicographic steps). GroupCommunicator::Reduce(ptr, Min) followed by Bcast(ptr) is MFEM's own
 * "all-reduce over the copies of each shared dof" (pfespace.cpp:1150-1152 uses the same pair with
 * BitOR); Bcast(ptr) uses layout 0, i.e. the array has one entry per local dof
 * (communication.hpp:435). On a single rank, or a rank without shared dofs, both calls return at
 * once (communication.cpp:1268). The before/after comparison is bitwise (+inf == +inf), and the
 * global "anything changed?" is an MPI_Allreduce (AnyRank).
 */
bool VertexField::ReduceSharedImpl(bool reduce_T, bool reduce_labels, mfem::Array<int>* changed) {
    const mfem::GroupCommunicator& gcomm = *m_gcomm;
    double* t = m_distance.HostReadWrite();
    int* lab = m_label.HostReadWrite();
    const std::vector<double> t_before(t, t + m_num_vertices);
    const std::vector<int> lab_before(lab, lab + m_num_vertices);

    if (reduce_T) {
        // Step 1: every copy gets the smallest T
        gcomm.Reduce<double>(t, mfem::GroupCommunicator::Min<double>);
        gcomm.Bcast(t);
        // Step 2 (lexicographic variant): copies that lost on T must not contribute a label
        if (reduce_labels) {
            for (int v = 0; v < m_num_vertices; ++v) {
                if (m_shared[v] != 0 && t[v] != t_before[static_cast<std::size_t>(v)]) {
                    lab[v] = INT_MAX;
                }
            }
        }
    }
    if (reduce_labels) {
        // Step 3: smallest label among the remaining copies
        gcomm.Reduce<int>(lab, mfem::GroupCommunicator::Min<int>);
        gcomm.Bcast(lab);
    }

    // Changed nodes on this rank, and on any rank
    bool local_change = false;
    if (changed != nullptr) {
        changed->SetSize(m_num_vertices);
    }
    for (int v = 0; v < m_num_vertices; ++v) {
        const auto s = static_cast<std::size_t>(v);
        const bool c = (t[v] != t_before[s]) || (lab[v] != lab_before[s]);
        local_change = local_change || c;
        if (changed != nullptr) {
            (*changed)[v] = c ? 1 : 0;
        }
    }
    return AnyRank(local_change, Comm());
}

/*
 * ReduceShared: the lexicographic (T, label) exchange, i.e. all three steps of ReduceSharedImpl
 * (T minimum, labels of losing copies set to INT_MAX, label minimum). Collective.
 */
bool VertexField::ReduceShared(mfem::Array<int>* changed) {
    return ReduceSharedImpl(true, true, changed);
}

/*
 * ReduceSharedDistance: step 1 of ReduceSharedImpl only (T minimum over the copies); labels are
 * left as they are, for solvers that settle T before the labels. Collective.
 */
bool VertexField::ReduceSharedDistance(mfem::Array<int>* changed) {
    return ReduceSharedImpl(true, false, changed);
}

/*
 * ReduceSharedLabels: step 3 of ReduceSharedImpl only (label minimum over the copies, T not
 * used), for the label rounds once T has converged. Collective.
 */
bool VertexField::ReduceSharedLabels(mfem::Array<int>* changed) {
    return ReduceSharedImpl(false, true, changed);
}

/*
 * ReduceSharedMinimum
 *
 * The step-1 exchange of ReduceShared applied to a caller's array: GroupCommunicator::Reduce with
 * the Min operation gathers each group's copies on the group master and keeps the smallest
 * (communication.hpp:534, communication.cpp Min), and Bcast(ptr) (layout 0: one entry per local
 * dof, communication.hpp:435) sends it back to every copy. Both calls return at once on a rank
 * without shared dofs, so ranks with no nodes take part without special handling.
 */
void VertexField::ReduceSharedMinimum(mfem::Vector& values) const {
    if (values.Size() != m_num_vertices) {
        throw std::invalid_argument("VertexField::ReduceSharedMinimum: values must have one "
                                    "entry per local vertex");
    }
    double* data = values.HostReadWrite();
    m_gcomm->Reduce<double>(data, mfem::GroupCommunicator::Min<double>);
    m_gcomm->Bcast(data);
}

/*
 * PointCells: one ReferencePoint (inverse element map, host) per point, then the sub-cell holding
 * that reference point (FieldCellView::LocateCell on the host view); the points of element e are
 * (e - elem_begin) * points_per_element + q, as in LabelPoints.
 */
void VertexField::PointCells(int elem_begin,
                             int elem_end,
                             int points_per_element,
                             const mfem::Vector& x,
                             const mfem::Vector& y,
                             const mfem::Vector& z,
                             mfem::Array<int>& cells) const {
    const int num_points = (elem_end - elem_begin) * points_per_element;
    cells.SetSize(num_points);
    const double* px = x.HostRead();
    const double* py = y.HostRead();
    const double* pz = z.HostRead();
    int* out = cells.HostWrite();
    const FieldCellView view = HostCellView();
    for (int i = 0; i < num_points; ++i) {
        const int e = elem_begin + i / points_per_element;
        if (!ElementActive(e)) {
            out[i] = 0; // not labeled (LabelPointsImpl skips inactive elements)
            continue;
        }
        const mfem::IntegrationPoint ip = ReferencePoint(m_pmesh, e, px[i], py[i], pz[i]);
        out[i] = view.LocateCell(ip.x, ip.y, ip.z);
    }
}

/*
 * LabelPoints: argument checks, then (nodal field) the sub-cell of every point, found by
 * inverting the element maps (PointCells), then the kernel (LabelPointsImpl).
 */
void VertexField::LabelPoints(int elem_begin,
                              int elem_end,
                              int points_per_element,
                              const mfem::Vector& x,
                              const mfem::Vector& y,
                              const mfem::Vector& z,
                              mfem::Array<int>& label,
                              mfem::Vector* distance) const {
    // Argument checks
    if (elem_begin < 0 || elem_end < elem_begin || elem_end > m_num_elements) {
        throw std::invalid_argument("VertexField::LabelPoints: invalid element range");
    }
    const int num_elems = elem_end - elem_begin;
    if (num_elems > 0 && points_per_element < 1) {
        throw std::invalid_argument("VertexField::LabelPoints: points_per_element must be >= 1");
    }
    const long long num_points_ll = static_cast<long long>(num_elems) *
                                    static_cast<long long>(num_elems > 0 ? points_per_element : 0);
    if (num_points_ll > std::numeric_limits<int>::max() / 3) {
        throw std::invalid_argument("VertexField::LabelPoints: too many points for one call");
    }
    const int num_points = static_cast<int>(num_points_ll);
    if (x.Size() != num_points || y.Size() != num_points || z.Size() != num_points) {
        throw std::invalid_argument("VertexField::LabelPoints: coordinate arrays must have "
                                    "(elem_end - elem_begin) * points_per_element entries");
    }
    if (m_curved && num_points > 0) {
        mfem::Array<int> cells;
        PointCells(elem_begin, elem_end, points_per_element, x, y, z, cells);
        LabelPointsImpl(
            elem_begin, elem_end, points_per_element, x, y, z, &cells, false, label, distance);
    } else {
        LabelPointsImpl(
            elem_begin, elem_end, points_per_element, x, y, z, nullptr, false, label, distance);
    }
}

/*
 * LabelPointsImpl
 *
 * One mfem::forall iteration per point: find the point's element from its index and its cell
 * (given: per point, or per rule point when all elements share one rule; cell 0, the element
 * itself, for the corner field), loop over the cell's corners (FieldCellView::CellNodes) and keep
 * the lexicographic
 * minimum of (T(v) + |p - v|, label(v)), then over the element's own seeds whose source cell is
 * this cell (CSR m_elem_src_*) of the direct term (|p - s_j| - a_j, j). For the corner field the
 * corners are the element's vertices in element order and every seed of the element has cell 0.
 * With seed lists (SetSeedLists) the kernel instead takes, over the corners' list entries
 * (T_i(v) + |p - v|) and the direct seeds (|p - s_i|), the seed with the smallest d_i^2 - w_i
 * (ties: lowest index), measuring lengths with each seed's own stretch when per-seed tensors are
 * set. The square root is the C library sqrt (unqualified, from
 * math.h), which is correctly rounded and available in device code. +infinity is captured by
 * value (a host constant), and inf + finite = inf, so unreached nodes never win against a reached
 * one.
 */
void VertexField::LabelPointsImpl(int elem_begin,
                                  int elem_end,
                                  int points_per_element,
                                  const mfem::Vector& x,
                                  const mfem::Vector& y,
                                  const mfem::Vector& z,
                                  const mfem::Array<int>* cells,
                                  bool cells_repeat,
                                  mfem::Array<int>& label,
                                  mfem::Vector* distance) const {
    const int num_points = (elem_end - elem_begin) *
                           (elem_end > elem_begin ? points_per_element : 0);
    label.SetSize(num_points);
    if (distance != nullptr) {
        distance->SetSize(num_points);
        distance->UseDevice(true);
    }
    if (num_points == 0) {
        return;
    }

    // Device views of the inputs
    const double* px = x.Read();
    const double* py = y.Read();
    const double* pz = z.Read();
    const FieldCellView view = CellView();
    const int* cl = (cells != nullptr) ? cells->Read() : nullptr;
    const bool repeats = cells_repeat;
    // Distances in the metric (VertexField::SetMetricStretch): mapped nodes, seeds stored
    // mapped, and each point mapped below
    const double* xyz = MetricCoordinates().Read();
    const bool has_metric = m_has_metric;
    double stretch[9];
    for (int k = 0; k < 9; ++k) {
        stretch[k] = m_stretch[static_cast<std::size_t>(k)];
    }
    const double* t = m_distance.Read();
    const int* lab = m_label.Read();
    // Seed lists (SetSeedLists): Laguerre / weighted Quadric rule over the corners' entries
    const int cap = m_list_capacity;
    const int* lseed = (cap > 0) ? m_list_seed.Read() : nullptr;
    const double* ldist = (cap > 0) ? m_list_dist.Read() : nullptr;
    const double* lweight = (cap > 0) ? m_list_weight.Read() : nullptr;
    const double* lstretch = (cap > 0 && m_list_stretch.Size() > 0) ? m_list_stretch.Read()
                                                                    : nullptr;
    const int* src_offsets = m_elem_src_offsets.Read();
    const int* src_seeds = m_elem_src_seeds.Read();
    const int* src_cell = m_elem_src_cell.Read();
    const double* src_data = m_elem_src_data.Read();
    int* out_label = label.Write();
    double* out_dist = (distance != nullptr) ? distance->Write() : nullptr;
    const int npe = points_per_element;
    const double inf = INF;
    const int* active = ElementMaskData(MaskOrNull(), true);

    mfem::forall(num_points, [=] MFEM_HOST_DEVICE(int i) {
        const int e = elem_begin + i / npe;
        // Inactive elements: label -1 (the cell tables need not fit their geometry)
        if (active != nullptr && active[e] == 0) {
            out_label[i] = -1;
            if (out_dist != nullptr) {
                out_dist[i] = inf;
            }
            return;
        }
        // The cell holding the point (0 = the element for the corner field)
        const int cell = (cl != nullptr) ? cl[repeats ? i % npe : i] : 0;
        int corners[FieldCellView::MAX_CELL_NODES];
        const int nc = view.CellNodes(e, cell, corners);
        double p[3] = {px[i], py[i], pz[i]};
        if (has_metric) {
            MapToMetric(stretch, px[i], py[i], pz[i], p);
        }
        if (cap > 0) {
            // Smallest d_i^2 - w_i, then smallest seed; d_i through a corner's entry or directly
            double best_f = inf;
            double best_d = inf;
            int best_seed = -1;
            const auto consider = [&](int seed, double d) {
                const double f = d * d - lweight[seed];
                if (f < best_f || (f == best_f && seed < best_seed)) {
                    best_f = f;
                    best_d = d;
                    best_seed = seed;
                }
            };
            // |r| of a difference vector, with seed s's own stretch for per-seed tensors
            const auto length = [&](int s, double dx, double dy, double dz) {
                if (lstretch != nullptr) {
                    const double r[3] = {dx, dy, dz};
                    double y[3];
                    MapToMetric(&lstretch[9 * s], r[0], r[1], r[2], y);
                    return sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
                }
                return sqrt(dx * dx + dy * dy + dz * dz);
            };
            for (int k = 0; k < nc; ++k) {
                const int v = corners[k];
                const double dx = p[0] - xyz[3 * v + 0];
                const double dy = p[1] - xyz[3 * v + 1];
                const double dz = p[2] - xyz[3 * v + 2];
                const double r = (lstretch == nullptr)
                                     ? sqrt(SquaredDistanceToVertex(xyz, v, p[0], p[1], p[2]))
                                     : 0.0;
                for (int q = 0; q < cap; ++q) {
                    const int seed = lseed[cap * v + q];
                    if (seed >= 0) {
                        const double rs = (lstretch == nullptr) ? r : length(seed, dx, dy, dz);
                        consider(seed, ldist[cap * v + q] + rs);
                    }
                }
            }
            for (int k = src_offsets[e]; k < src_offsets[e + 1]; ++k) {
                if (src_cell[k] != cell) {
                    continue;
                }
                const double dx = p[0] - src_data[4 * k + 0];
                const double dy = p[1] - src_data[4 * k + 1];
                const double dz = p[2] - src_data[4 * k + 2];
                consider(src_seeds[k], length(src_seeds[k], dx, dy, dz));
            }
            out_label[i] = best_seed;
            if (out_dist != nullptr) {
                out_dist[i] = best_d;
            }
            return;
        }
        double best = inf;
        int best_label = -1;
        for (int k = 0; k < nc; ++k) {
            const int v = corners[k];
            const double cand = t[v] + sqrt(SquaredDistanceToVertex(xyz, v, p[0], p[1], p[2]));
            const int cand_label = lab[v];
            // Lexicographic minimum of (distance, seed index); unreached nodes (inf, -1)
            // never replace the initial (inf, -1)
            if (cand < best || (cand == best && cand < inf && cand_label < best_label)) {
                best = cand;
                best_label = cand_label;
            }
        }
        // Direct term from the seeds whose source cell is this cell
        for (int k = src_offsets[e]; k < src_offsets[e + 1]; ++k) {
            if (src_cell[k] != cell) {
                continue;
            }
            const double dx = p[0] - src_data[4 * k + 0];
            const double dy = p[1] - src_data[4 * k + 1];
            const double dz = p[2] - src_data[4 * k + 2];
            const double cand = sqrt(dx * dx + dy * dy + dz * dz) - src_data[4 * k + 3];
            const int cand_label = src_seeds[k];
            if (cand < best || (cand == best && cand_label < best_label)) {
                best = cand;
                best_label = cand_label;
            }
        }
        out_label[i] = best_label;
        if (out_dist != nullptr) {
            out_dist[i] = best;
        }
    });
}

/*
 * LabelElementCentroids: ComputeElementSamples gives the volume centroid of every local element
 * (one point per element), which LabelPoints labels. Nothing to do on a rank without elements.
 */
void VertexField::LabelElementCentroids(mfem::Array<int>& element_label,
                                        mfem::Vector* element_distance) const {
    if (m_num_elements == 0) {
        element_label.SetSize(0);
        if (element_distance != nullptr) {
            element_distance->SetSize(0);
        }
        return;
    }
    ElementSamples samples;
    ComputeElementSamples(m_pmesh, -1, samples);
    LabelPoints(
        0, m_num_elements, 1, samples.cx, samples.cy, samples.cz, element_label, element_distance);
    // Inactive elements: placeholder -1 (not material)
    SetInactiveValues(MaskOrNull(), element_label, -1);
}

/*
 * LabelElementsByVote
 *
 * Same chunked structure as LabelElementsByQuadratureVote: one QuadraturePointSampler for the
 * voting rule (node data gathered once), then per chunk of whole elements: the points and their
 * weights w_q detJ_q (Sample), a label per point (LabelPointsImpl; for the nodal field the rule's
 * reference points, identical for every element, give the cells without any inversion) and the
 * per-element vote (VoteOnPointLabels). Memory grows with the chunk, not with the whole mesh.
 */
int VertexField::LabelElementsByVote(int levels,
                                     int max_points_per_chunk,
                                     mfem::Array<int>& element_label) const {
    // Argument checks (VoteRule checks levels and the geometry)
    if (levels < 0 || levels > 3) {
        throw std::invalid_argument("VertexField::LabelElementsByVote: levels must be 0..3");
    }
    if (max_points_per_chunk < 1) {
        throw std::invalid_argument(
            "VertexField::LabelElementsByVote: max_points_per_chunk must be >= 1");
    }
    element_label.SetSize(m_num_elements);
    if (m_num_elements == 0) {
        return 0;
    }
    if (m_pmesh.GetNumGeometries(3) > 1) {
        throw std::runtime_error(
            "VertexField::LabelElementsByVote: mixed element geometries are not supported");
    }

    // Sampler for the voting rule, and the chunk size in whole elements
    const mfem::IntegrationRule& rule = VoteRule(m_pmesh.GetElementGeometry(0), levels);
    const QuadraturePointSampler sampler(m_pmesh, rule);
    const int nq = sampler.NumPointsPerElement();
    const int chunk_elems = std::min(m_num_elements, std::max(1, max_points_per_chunk / nq));

    // Sub-cell of each rule point (nodal field only): it depends only on the point's reference
    // coordinates, so it is located once here instead of once per element
    mfem::Array<int> rule_cells;
    if (m_curved) {
        const FieldCellView view = HostCellView();
        rule_cells.SetSize(nq);
        for (int q = 0; q < nq; ++q) {
            const mfem::IntegrationPoint& ip = rule.IntPoint(q);
            rule_cells[q] = view.LocateCell(ip.x, ip.y, ip.z);
        }
    }

    // Chunks: points, point labels, vote
    mfem::Array<int> element_split(m_num_elements);
    QuadraturePointSamples points;
    mfem::Array<int> point_label;
    for (int begin = 0; begin < m_num_elements; begin += chunk_elems) {
        const int end = std::min(m_num_elements, begin + chunk_elems);
        sampler.Sample(begin, end, points);
        LabelPointsImpl(begin,
                        end,
                        nq,
                        points.x,
                        points.y,
                        points.z,
                        m_curved ? &rule_cells : nullptr,
                        true,
                        point_label,
                        nullptr);
        VoteOnPointLabels(
            begin, end - begin, nq, point_label, points.weight, element_label, element_split);
    }

    // Inactive elements: placeholder -1, never split
    SetInactiveValues(MaskOrNull(), element_label, -1);
    SetInactiveValues(MaskOrNull(), element_split, 0);

    // Split count on the host
    int num_split = 0;
    const int* split = element_split.HostRead();
    for (int e = 0; e < m_num_elements; ++e) {
        num_split += split[e];
    }
    return num_split;
}

} // namespace exaconstit::microstructure
