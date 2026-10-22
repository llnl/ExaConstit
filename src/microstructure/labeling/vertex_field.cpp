/*
 * Implementation of the per-vertex distance field shared by the geodesic labelers
 * (vertex_field.hpp).
 *
 * Terms used in this file:
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it.
 * - dof (degree of freedom): one nodal unknown of a finite element space. For the order-1 H1
 *   space used here, one dof per mesh vertex, numbered like the vertices.
 * - GroupCommunicator: MFEM's object that knows, for every dof on a partition boundary, which
 *   other ranks hold a copy ("group" = set of ranks sharing some dofs; group 0 is this rank
 *   alone). One rank of each group is its "master".
 * - CSR (compressed sparse row): a list of lists stored as offsets + items (see the header).
 * - Lexicographic minimum of pairs (T, label): the pair with the smallest T, and among pairs with
 *   that T the smallest label.
 * - mfem::forall: MFEM's parallel loop; a GPU kernel when the mfem::Device has a CUDA or HIP
 *   backend, a plain host loop otherwise. MFEM_HOST_DEVICE marks code callable inside it.
 */
#include "microstructure/labeling/vertex_field.hpp"

#include "microstructure/labeling/quadrature_vote.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/seeding/seed_element_locator.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <math.h> // unqualified sqrt in device code (C math library)

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief +infinity, the "not reached" distance. */
constexpr double INF = std::numeric_limits<double>::infinity();

/**
 * @brief Squared distance between point p and local vertex v of the coordinate table
 * xyz[3*v + d]. Written out in a fixed order (dx^2 + dy^2 + dz^2) so the same (point, vertex)
 * pair gives the same bits on every rank. Host and device.
 */
MFEM_HOST_DEVICE inline double SquaredDistanceToVertex(const double* xyz, int v, double px,
                                                       double py, double pz) {
    const double dx = px - xyz[3 * v + 0];
    const double dy = py - xyz[3 * v + 1];
    const double dz = pz - xyz[3 * v + 2];
    return dx * dx + dy * dy + dz * dz;
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
void VoteOnPointLabels(int chunk_begin, int chunk_elems, int nq,
                       const mfem::Array<int>& point_label, const mfem::Vector& weight,
                       mfem::Array<int>& element_label, mfem::Array<int>& element_split) {
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
            if (winner < 0 || total > winner_weight ||
                (total == winner_weight && label < winner)) {
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

} // namespace

/*
 * Constructor
 *
 * Steps:
 *  1. Collective checks (3D, conforming, continuous Nodes), reduced so all ranks throw together.
 *  2. The order-1 H1 space (collective constructor). Its local size must equal the local vertex
 *     count: for a conforming mesh MFEM numbers vertex dofs first, dof v = vertex v
 *     (fespace.cpp:3536-3544).
 *  3. Element -> vertex CSR from FiniteElementSpace::GetElementDofs (= the element's vertices
 *     for order 1), then vertex -> element CSR by counting and filling (a transpose).
 *  4. Vertex coordinates: Mesh::GetVertex for a straight-sided mesh; for a mesh with Nodes the
 *     values of the vertex dofs of the (continuous) Nodes space, entry DofToVDof(v, d), since
 *     MFEM may leave the vertex array stale for high-order meshes (mesh.hpp:1426). Then the
 *     group master's coordinates are broadcast to every copy of a shared vertex
 *     (GroupCommunicator::Bcast with layout 0, one entry per local dof, communication.hpp:435),
 *     one array per coordinate, so all copies hold the same bits even if the mesh did not
 *     guarantee it (see the header). Unshared vertices are not touched by Bcast.
 *  5. Shared/owned markers from the GroupCommunicator's group -> dof table (rows 1.. list the dofs
 *     shared with other ranks; communication.hpp:291) and GetLocalTDofNumber (>= 0 only on the
 *     group master, pfespace.cpp:1720-1739).
 */
VertexField::VertexField(mfem::ParMesh& pmesh) : m_pmesh(pmesh) {
    // Step 1: checks that are the same on every rank or reduced
    const MPI_Comm comm = pmesh.GetComm();
    const bool not_3d = pmesh.Dimension() != 3 || pmesh.SpaceDimension() != 3;
    ThrowIfAnyRank<std::runtime_error>(comm, not_3d,
                                       "VertexField: only 3D volume meshes are supported");
    ThrowIfAnyRank<std::runtime_error>(comm, pmesh.Nonconforming(),
                   "VertexField: nonconforming meshes (hanging vertices) are not supported");
    const mfem::GridFunction* nodes = pmesh.GetNodes();
    const bool bad_nodes =
        nodes != nullptr && nodes->FESpace()->FEColl()->GetContType() !=
                                mfem::FiniteElementCollection::CONTINUOUS;
    ThrowIfAnyRank<std::runtime_error>(comm, bad_nodes,
                   "VertexField: discontinuous mesh Nodes (periodic meshes) are not supported");

    // Step 2: vertex numbering through the order-1 H1 space
    m_fec = std::make_unique<mfem::H1_FECollection>(1, 3);
    m_fes = std::make_unique<mfem::ParFiniteElementSpace>(&pmesh, m_fec.get());
    m_num_vertices = pmesh.GetNV();
    m_num_elements = pmesh.GetNE();
    ThrowIfAnyRank<std::runtime_error>(comm, m_fes->GetVSize() != m_num_vertices,
                   "VertexField: the order-1 H1 space does not have one dof per vertex");

    // Step 3a: element -> vertex CSR
    m_elem_vert_offsets.SetSize(m_num_elements + 1);
    m_elem_vert_offsets[0] = 0;
    mfem::Array<int> dofs;
    for (int e = 0; e < m_num_elements; ++e) {
        m_fes->GetElementDofs(e, dofs);
        m_elem_vert_offsets[e + 1] = m_elem_vert_offsets[e] + dofs.Size();
    }
    m_elem_verts.SetSize(m_elem_vert_offsets[m_num_elements]);
    bool bad_dof = false;
    for (int e = 0; e < m_num_elements; ++e) {
        m_fes->GetElementDofs(e, dofs);
        for (int k = 0; k < dofs.Size(); ++k) {
            // H1 dofs carry no sign; a negative entry would mean an unexpected space
            bad_dof = bad_dof || dofs[k] < 0 || dofs[k] >= m_num_vertices;
            m_elem_verts[m_elem_vert_offsets[e] + k] = dofs[k];
        }
    }
    ThrowIfAnyRank<std::runtime_error>(comm, bad_dof,
                                       "VertexField: unexpected element dof numbering");

    // Step 3b: vertex -> element CSR (count, prefix sum, fill in increasing element order)
    m_vert_elem_offsets.SetSize(m_num_vertices + 1);
    m_vert_elem_offsets = 0;
    for (int k = 0; k < m_elem_verts.Size(); ++k) {
        m_vert_elem_offsets[m_elem_verts[k] + 1] += 1;
    }
    for (int v = 0; v < m_num_vertices; ++v) {
        m_vert_elem_offsets[v + 1] += m_vert_elem_offsets[v];
    }
    m_vert_elems.SetSize(m_vert_elem_offsets[m_num_vertices]);
    std::vector<int> fill(static_cast<std::size_t>(m_num_vertices), 0);
    for (int e = 0; e < m_num_elements; ++e) {
        for (int k = m_elem_vert_offsets[e]; k < m_elem_vert_offsets[e + 1]; ++k) {
            const int v = m_elem_verts[k];
            m_vert_elems[m_vert_elem_offsets[v] + fill[static_cast<std::size_t>(v)]++] = e;
        }
    }

    // Step 4: vertex coordinates xyz[3*v + d]
    m_coords.SetSize(3 * m_num_vertices);
    m_coords.UseDevice(true);
    double* xyz = m_coords.HostWrite();
    if (nodes == nullptr) {
        for (int v = 0; v < m_num_vertices; ++v) {
            const double* p = pmesh.GetVertex(v);
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
    // Step 4b: the owner's coordinates on every copy (one per-vertex array per coordinate,
    // because Bcast works on arrays with one entry per local dof)
    {
        const mfem::GroupCommunicator& gcomm = m_fes->GroupComm();
        std::vector<double> component(static_cast<std::size_t>(m_num_vertices));
        for (int d = 0; d < 3; ++d) {
            for (int v = 0; v < m_num_vertices; ++v) {
                component[static_cast<std::size_t>(v)] = xyz[3 * v + d];
            }
            gcomm.Bcast(component.data());
            for (int v = 0; v < m_num_vertices; ++v) {
                xyz[3 * v + d] = component[static_cast<std::size_t>(v)];
            }
        }
    }

    // Step 5: shared and owned markers
    m_shared.SetSize(m_num_vertices);
    m_shared = 0;
    const mfem::Table& group_ldof = m_fes->GroupComm().GroupLDofTable();
    for (int g = 1; g < group_ldof.Size(); ++g) {
        const int* row = group_ldof.GetRow(g);
        for (int k = 0; k < group_ldof.RowSize(g); ++k) {
            m_shared[row[k]] = 1;
        }
    }
    m_owned.SetSize(m_num_vertices);
    for (int v = 0; v < m_num_vertices; ++v) {
        m_owned[v] = (m_fes->GetLocalTDofNumber(v) >= 0) ? 1 : 0;
    }

    // Field arrays, all "not reached"
    m_T.SetSize(m_num_vertices);
    m_T.UseDevice(true);
    m_source_T.SetSize(m_num_vertices);
    m_source_T.UseDevice(true);
    m_label.SetSize(m_num_vertices);
    m_source_label.SetSize(m_num_vertices);
    Reset();
}

VertexField::~VertexField() = default;

MPI_Comm VertexField::Comm() const {
    return m_pmesh.GetComm();
}

/*
 * Reset: host fill of T = T0 = +inf and label = label0 = -1, and empty per-element seed lists
 * (all CSR offsets 0).
 */
void VertexField::Reset() {
    m_elem_src_offsets.SetSize(m_num_elements + 1);
    m_elem_src_offsets = 0;
    m_elem_src_seeds.SetSize(0);
    m_elem_src_data.SetSize(0);
    double* t = m_T.HostWrite();
    double* t0 = m_source_T.HostWrite();
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
 *  3. Snapping (only for seeds in no element): NearestVertex gives the nearest mesh vertex over
 *     all ranks (smallest squared distance, ties by smallest VertexKey) and its exact
 *     coordinates. Each rank finds its local copies of that vertex by exact position
 *     (Mesh::GetVertex, the coordinates NearestVertex searched), takes the smallest key of the
 *     local elements containing it, and MPI_Allreduce(MIN) gives the source element.
 *  4. The owning rank of each source element writes (|v - s_i| - a_i, i) into its vertices,
 *     keeping the lexicographic minimum; ReduceShared spreads the values to the other copies.
 *     Seeds are processed in increasing index order with a strict "<" on the value, so equal
 *     values keep the smaller seed index. The owning rank also records the seed in the element's
 *     seed list for the direct term of LabelPoints.
 * Cost: one BVH query per seed, plus one per snapped seed and a position lookup.
 */
SourceReport VertexField::InitializeSources(const SeedSet& seeds,
                                            const std::vector<double>& additive_weights) {
    // Step 1: inputs and an element somewhere
    ValidateSeeds(seeds, additive_weights);
    const MPI_Comm comm = Comm();
    long long global_elements = m_num_elements;
    MPI_Allreduce(MPI_IN_PLACE, &global_elements, 1, MPI_LONG_LONG, MPI_SUM, comm);
    if (global_elements == 0) {
        throw std::runtime_error("VertexField::InitializeSources: the mesh has no elements");
    }
    const int n = static_cast<int>(seeds.Size());
    const std::uint64_t none = std::numeric_limits<std::uint64_t>::max();
    const std::vector<std::uint64_t> keys = ComputeElementKeys(m_pmesh);

    SourceReport report;
    report.source_element_local.assign(static_cast<std::size_t>(n), -1);
    report.seed_outside.assign(static_cast<std::size_t>(n), 0);

    // Step 2: smallest key of an element containing each seed (collective)
    report.source_element_key = ContainingElementKeys(m_pmesh, comm, seeds, &keys);

    // Step 3: seeds in no element snap to an element around the nearest vertex
    std::vector<int> outside;
    for (int i = 0; i < n; ++i) {
        if (report.source_element_key[static_cast<std::size_t>(i)] == none) {
            outside.push_back(i);
            report.seed_outside[static_cast<std::size_t>(i)] = 1;
        }
    }
    report.num_seeds_outside = static_cast<int>(outside.size());
    if (!outside.empty()) {
        // Step 3a: nearest vertex over all ranks (collective; same list on every rank)
        const NearestVertexResult nearest = NearestVertex(m_pmesh, comm, seeds, outside);
        // Step 3b: local copies of that vertex by exact position, then their elements' keys
        std::map<std::array<double, 3>, int> local_vertex;
        for (int v = 0; v < m_num_vertices; ++v) {
            const double* p = m_pmesh.GetVertex(v);
            local_vertex.emplace(std::array<double, 3> {p[0], p[1], p[2]}, v);
        }
        const int num_out = static_cast<int>(outside.size());
        std::vector<std::uint64_t> snap_key(static_cast<std::size_t>(num_out), none);
        for (int j = 0; j < num_out; ++j) {
            const auto it = local_vertex.find(nearest.position[static_cast<std::size_t>(j)]);
            if (it == local_vertex.end()) {
                continue; // the vertex is not on this rank
            }
            const int v = it->second;
            for (int k = m_vert_elem_offsets[v]; k < m_vert_elem_offsets[v + 1]; ++k) {
                const std::uint64_t key = keys[static_cast<std::size_t>(m_vert_elems[k])];
                snap_key[static_cast<std::size_t>(j)] =
                    std::min(snap_key[static_cast<std::size_t>(j)], key);
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
            const auto it = std::lower_bound(sorted_keys.begin(), sorted_keys.end(),
                                             std::make_pair(key, -1));
            if (it != sorted_keys.end() && it->first == key) {
                report.source_element_local[static_cast<std::size_t>(i)] = it->second;
            }
        }
    }

    // Step 4b: per-element seed lists (CSR over the local elements, seeds in increasing index)
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
        m_elem_src_data.SetSize(4 * num_entries);
        m_elem_src_data.UseDevice(true);
        int* entry_seed = m_elem_src_seeds.HostWrite();
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
            entry_data[4 * k + 0] = seeds.x[s];
            entry_data[4 * k + 1] = seeds.y[s];
            entry_data[4 * k + 2] = seeds.z[s];
            entry_data[4 * k + 3] = additive_weights.empty() ? 0.0 : additive_weights[s];
        }
    }

    // Step 4c: source values at the vertices of each local source element
    {
        const double* xyz = m_coords.HostRead();
        double* t = m_T.HostReadWrite();
        int* lab = m_label.HostReadWrite();
        for (int i = 0; i < n; ++i) {
            const auto s = static_cast<std::size_t>(i);
            const int e = report.source_element_local[s];
            if (e < 0) {
                continue;
            }
            const double a = additive_weights.empty() ? 0.0 : additive_weights[s];
            for (int k = m_elem_vert_offsets[e]; k < m_elem_vert_offsets[e + 1]; ++k) {
                const int v = m_elem_verts[k];
                const double value =
                    std::sqrt(SquaredDistanceToVertex(xyz, v, seeds.x[s], seeds.y[s],
                                                      seeds.z[s])) - a;
                // Lexicographic minimum: seeds come in increasing index, so "<" keeps the lower
                // index on equal values
                if (value < t[v]) {
                    t[v] = value;
                    lab[v] = i;
                }
            }
        }
    }

    // Step 4d: make all copies agree, then remember the result as the source values
    ReduceShared();
    m_source_T = m_T;
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
bool VertexField::ReduceSharedImpl(bool reduce_T, bool reduce_labels,
                                   mfem::Array<int>* changed) {
    const mfem::GroupCommunicator& gcomm = m_fes->GroupComm();
    double* t = m_T.HostReadWrite();
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

    // Changed vertices on this rank, and on any rank
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
 * without shared dofs, so ranks with no vertices take part without special handling.
 */
void VertexField::ReduceSharedMinimum(mfem::Vector& values) const {
    if (values.Size() != m_num_vertices) {
        throw std::invalid_argument("VertexField::ReduceSharedMinimum: values must have one "
                                    "entry per local vertex");
    }
    const mfem::GroupCommunicator& gcomm = m_fes->GroupComm();
    double* data = values.HostReadWrite();
    gcomm.Reduce<double>(data, mfem::GroupCommunicator::Min<double>);
    gcomm.Bcast(data);
}

/*
 * LabelPoints
 *
 * One mfem::forall iteration per point: find the point's element from its index, loop over the
 * element's vertices through the CSR table, and keep the lexicographic minimum of
 * (T(v) + |p - v|, label(v)), then over the element's own seeds (source element only, CSR
 * m_elem_src_*) of the direct term (|p - s_j| - a_j, j). The square root is the C library sqrt
 * (unqualified, from math.h), which is correctly rounded and available in device code. +infinity
 * is captured by value (a host constant), and inf + finite = inf, so unreached vertices never win
 * against a reached one.
 */
void VertexField::LabelPoints(int elem_begin, int elem_end, int points_per_element,
                              const mfem::Vector& x, const mfem::Vector& y, const mfem::Vector& z,
                              mfem::Array<int>& label, mfem::Vector* distance) const {
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
    if (num_points_ll > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("VertexField::LabelPoints: too many points for one call");
    }
    const int num_points = static_cast<int>(num_points_ll);
    if (x.Size() != num_points || y.Size() != num_points || z.Size() != num_points) {
        throw std::invalid_argument("VertexField::LabelPoints: coordinate arrays must have "
                                    "(elem_end - elem_begin) * points_per_element entries");
    }
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
    const int* offsets = m_elem_vert_offsets.Read();
    const int* verts = m_elem_verts.Read();
    const double* xyz = m_coords.Read();
    const double* t = m_T.Read();
    const int* lab = m_label.Read();
    const int* src_offsets = m_elem_src_offsets.Read();
    const int* src_seeds = m_elem_src_seeds.Read();
    const double* src_data = m_elem_src_data.Read();
    int* out_label = label.Write();
    double* out_dist = (distance != nullptr) ? distance->Write() : nullptr;
    const int npe = points_per_element;
    const double inf = INF;

    mfem::forall(num_points, [=] MFEM_HOST_DEVICE(int i) {
        const int e = elem_begin + i / npe;
        double best = inf;
        int best_label = -1;
        for (int k = offsets[e]; k < offsets[e + 1]; ++k) {
            const int v = verts[k];
            const double cand = t[v] + sqrt(SquaredDistanceToVertex(xyz, v, px[i], py[i], pz[i]));
            const int cand_label = lab[v];
            // Lexicographic minimum of (distance, seed index); unreached vertices (inf, -1)
            // never replace the initial (inf, -1)
            if (cand < best || (cand == best && cand < inf && cand_label < best_label)) {
                best = cand;
                best_label = cand_label;
            }
        }
        // Direct term from the seeds whose source element is e (none outside source elements)
        for (int k = src_offsets[e]; k < src_offsets[e + 1]; ++k) {
            const double dx = px[i] - src_data[4 * k + 0];
            const double dy = py[i] - src_data[4 * k + 1];
            const double dz = pz[i] - src_data[4 * k + 2];
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
    LabelPoints(0, m_num_elements, 1, samples.cx, samples.cy, samples.cz, element_label,
                element_distance);
}

/*
 * LabelElementsByVote
 *
 * Same chunked structure as LabelElementsByQuadratureVote: one QuadraturePointSampler for the
 * voting rule (node data gathered once), then per chunk of whole elements: the points and their
 * weights w_q detJ_q (Sample), a label per point (LabelPoints) and the per-element vote
 * (VoteOnPointLabels). Memory grows with the chunk, not with the whole mesh.
 */
int VertexField::LabelElementsByVote(int levels, int max_points_per_chunk,
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
    const QuadraturePointSampler sampler(m_pmesh,
                                         VoteRule(m_pmesh.GetElementGeometry(0), levels));
    const int nq = sampler.NumPointsPerElement();
    const int chunk_elems = std::min(m_num_elements, std::max(1, max_points_per_chunk / nq));

    // Chunks: points, point labels, vote
    mfem::Array<int> element_split(m_num_elements);
    QuadraturePointSamples points;
    mfem::Array<int> point_label;
    for (int begin = 0; begin < m_num_elements; begin += chunk_elems) {
        const int end = std::min(m_num_elements, begin + chunk_elems);
        sampler.Sample(begin, end, points);
        LabelPoints(begin, end, nq, points.x, points.y, points.z, point_label);
        VoteOnPointLabels(begin, end - begin, nq, point_label, points.weight, element_label,
                          element_split);
    }

    // Split count on the host
    int num_split = 0;
    const int* split = element_split.HostRead();
    for (int e = 0; e < m_num_elements; ++e) {
        num_split += split[e];
    }
    return num_split;
}

} // namespace exaconstit::microstructure
