/*
 * Node label sets, boundary surfaces and constraint counts (node_labels.hpp).
 */
#include "microstructure/conformal/node_labels.hpp"

#include "microstructure/util/collective_errors.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace exaconstit::microstructure {

namespace {

constexpr double PI = 3.14159265358979323846;

/** @brief Root of i in a union-find forest with path halving. Host. */
int FindRoot(std::vector<int>& parent, int i) {
    while (parent[static_cast<std::size_t>(i)] != i) {
        const auto is = static_cast<std::size_t>(i);
        parent[is] = parent[static_cast<std::size_t>(parent[is])];
        i = parent[is];
    }
    return i;
}

/**
 * @brief Number of groups of the unit normals (3 doubles each) connected by steps between
 * normals whose dot product is >= cos_angle. Connected components do not depend on the order of
 * the normals. Host.
 */
int CountSurfaces(const std::vector<double>& normals, double cos_angle) {
    const int m = static_cast<int>(normals.size() / 3);
    std::vector<int> parent(static_cast<std::size_t>(m));
    std::iota(parent.begin(), parent.end(), 0);
    int groups = m;
    for (int a = 0; a < m; ++a) {
        const double* na = &normals[3 * static_cast<std::size_t>(a)];
        for (int b = a + 1; b < m; ++b) {
            const double* nb = &normals[3 * static_cast<std::size_t>(b)];
            if (na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2] >= cos_angle) {
                const int ra = FindRoot(parent, a);
                const int rb = FindRoot(parent, b);
                if (ra != rb) {
                    parent[static_cast<std::size_t>(std::max(ra, rb))] = std::min(ra, rb);
                    --groups;
                }
            }
        }
    }
    return groups;
}

/** @brief Appends normal n to list unless the same bits are already there. Host. */
void AddDistinctNormal(std::vector<double>& list, const double n[3]) {
    for (std::size_t k = 0; k < list.size(); k += 3) {
        if (list[k] == n[0] && list[k + 1] == n[1] && list[k + 2] == n[2]) {
            return;
        }
    }
    list.insert(list.end(), n, n + 3);
}

} // namespace

/*
 * Constructor: input checks (collective, before the collective VertexField constructor), the
 * node numbering, then label sets, boundary surfaces and the summary.
 */
NodeLabels::NodeLabels(mfem::ParMesh& pmesh, const mfem::Array<int>& element_label,
                       const NodeLabelOptions& options) {
    const MPI_Comm comm = pmesh.GetComm();
    ThrowIfAnyRank(comm, element_label.Size() != pmesh.GetNE(),
                   "NodeLabels: element_label needs one entry per local element");
    bool bad_label = false;
    for (int e = 0; e < element_label.Size(); ++e) {
        bad_label = bad_label || element_label[e] < 1;
    }
    ThrowIfAnyRank(comm, bad_label, "NodeLabels: element labels must be >= 1");
    const double angle = options.feature_angle_degrees;
    ThrowIfAnyRank(comm, !(angle > 0.0 && angle < 90.0),
                   "NodeLabels: feature_angle_degrees must be in (0, 90)");

    m_field = std::make_unique<VertexField>(pmesh, options.nodes);
    BuildLabelSets(element_label);
    BuildBoundarySurfaces(angle);
    BuildSummary();
}

NodeLabels::~NodeLabels() = default;

JunctionKind NodeLabels::Kind(int v) const {
    const int n = NumLabels(v);
    return static_cast<JunctionKind>(std::min(n, NUM_JUNCTION_KINDS) - 1);
}

/*
 * BuildLabelSets
 *  1. Local sets: the labels of the elements in the node -> element table (complete: the field
 *     has no element mask), sorted and made distinct.
 *  2. width = the largest local set over all ranks; every node sends its set padded with 0 (not a
 *     label) to the other copies.
 *  3. Each copy merges its own and the received sets (sorted union) into the CSR arrays.
 */
void NodeLabels::BuildLabelSets(const mfem::Array<int>& element_label) {
    const VertexField& field = *m_field;
    const int nv = field.NumVertices();
    const int* ve_off = field.VertexElementOffsets().HostRead();
    const int* ve = field.VertexElements().HostRead();
    const int* labels = element_label.HostRead();

    // Step 1
    std::vector<std::vector<int>> sets(static_cast<std::size_t>(nv));
    int width = 1;
    for (int v = 0; v < nv; ++v) {
        auto& set = sets[static_cast<std::size_t>(v)];
        for (int k = ve_off[v]; k < ve_off[v + 1]; ++k) {
            set.push_back(labels[ve[k]]);
        }
        std::sort(set.begin(), set.end());
        set.erase(std::unique(set.begin(), set.end()), set.end());
        width = std::max(width, static_cast<int>(set.size()));
    }

    // Step 2
    MPI_Allreduce(MPI_IN_PLACE, &width, 1, MPI_INT, MPI_MAX, field.Comm());
    const auto w = static_cast<std::size_t>(width);
    std::vector<double> send(w * static_cast<std::size_t>(nv), 0.0);
    for (int v = 0; v < nv; ++v) {
        const auto& set = sets[static_cast<std::size_t>(v)];
        std::copy(set.begin(), set.end(), send.begin() + static_cast<std::ptrdiff_t>(w * v));
    }
    std::vector<double> recv;
    std::vector<int> recv_off;
    field.ExchangeSharedPayload(width, send, recv, recv_off);

    // Step 3
    m_label_offsets.SetSize(nv + 1);
    m_label_offsets[0] = 0;
    std::vector<int> items;
    for (int v = 0; v < nv; ++v) {
        auto& set = sets[static_cast<std::size_t>(v)];
        for (int c = recv_off[static_cast<std::size_t>(v)];
             c < recv_off[static_cast<std::size_t>(v) + 1]; ++c) {
            for (std::size_t s = 0; s < w; ++s) {
                const double label = recv[w * static_cast<std::size_t>(c) + s];
                if (label != 0.0) {
                    set.push_back(static_cast<int>(label));
                }
            }
        }
        std::sort(set.begin(), set.end());
        set.erase(std::unique(set.begin(), set.end()), set.end());
        items.insert(items.end(), set.begin(), set.end());
        m_label_offsets[v + 1] = static_cast<int>(items.size());
    }
    m_labels.SetSize(static_cast<int>(items.size()));
    std::copy(items.begin(), items.end(), m_labels.begin());
}

/*
 * BuildBoundarySurfaces
 *  1. Boundary faces: local faces with one element that are not shared with another rank
 *     (ParMesh::GetSharedFace lists the shared ones).
 *  2. Unit normal at the face center: CalcOrtho of the face transformation's Jacobian (Nodes when
 *     the mesh has them), which points out of the face's element. Each node of the face (its
 *     vertices for the corner field, its Nodes dofs for the nodal field) records the normal
 *     unless the same bits are already recorded.
 *  3. Exchange: width 1 + 3 * (largest local list over all ranks); entry 0 is the count.
 *  4. Surfaces = connected components of all normals of the node (CountSurfaces).
 */
void NodeLabels::BuildBoundarySurfaces(double feature_angle_degrees) {
    const VertexField& field = *m_field;
    mfem::ParMesh& pmesh = field.Mesh();
    const int nv = field.NumVertices();

    // Step 1
    std::vector<char> shared(static_cast<std::size_t>(pmesh.GetNumFaces()), 0);
    for (int i = 0; i < pmesh.GetNSharedFaces(); ++i) {
        shared[static_cast<std::size_t>(pmesh.GetSharedFace(i))] = 1;
    }

    // Step 2
    std::vector<std::vector<double>> normals(static_cast<std::size_t>(nv));
    const mfem::FiniteElementSpace* node_fes =
        field.UsesMeshNodes() ? pmesh.GetNodes()->FESpace() : nullptr;
    mfem::IsoparametricTransformation trans;
    mfem::Array<int> face_nodes;
    mfem::Vector ortho(3);
    for (int f = 0; f < pmesh.GetNumFaces(); ++f) {
        int e1 = -1;
        int e2 = -1;
        pmesh.GetFaceElements(f, &e1, &e2);
        if (e2 >= 0 || shared[static_cast<std::size_t>(f)] != 0) {
            continue;
        }
        pmesh.GetFaceTransformation(f, &trans);
        const mfem::IntegrationPoint& center =
            mfem::Geometries.GetCenter(pmesh.GetFaceGeometry(f));
        trans.SetIntPoint(&center);
        mfem::CalcOrtho(trans.Jacobian(), ortho);
        const double len = ortho.Norml2();
        const double n[3] = {ortho(0) / len, ortho(1) / len, ortho(2) / len};
        if (node_fes != nullptr) {
            node_fes->GetFaceDofs(f, face_nodes);
        } else {
            pmesh.GetFaceVertices(f, face_nodes);
        }
        for (int j = 0; j < face_nodes.Size(); ++j) {
            const int node = (face_nodes[j] >= 0) ? face_nodes[j] : -1 - face_nodes[j];
            AddDistinctNormal(normals[static_cast<std::size_t>(node)], n);
        }
    }

    // Step 3
    int most = 0;
    for (const auto& list : normals) {
        most = std::max(most, static_cast<int>(list.size() / 3));
    }
    MPI_Allreduce(MPI_IN_PLACE, &most, 1, MPI_INT, MPI_MAX, field.Comm());
    const int width = 1 + 3 * most;
    const auto w = static_cast<std::size_t>(width);
    std::vector<double> send(w * static_cast<std::size_t>(nv), 0.0);
    for (int v = 0; v < nv; ++v) {
        const auto& list = normals[static_cast<std::size_t>(v)];
        send[w * static_cast<std::size_t>(v)] = static_cast<double>(list.size() / 3);
        std::copy(list.begin(), list.end(),
                  send.begin() + static_cast<std::ptrdiff_t>(w * v + 1));
    }
    std::vector<double> recv;
    std::vector<int> recv_off;
    field.ExchangeSharedPayload(width, send, recv, recv_off);

    // Step 4
    const double cos_angle = std::cos(feature_angle_degrees * PI / 180.0);
    m_boundary_surfaces.SetSize(nv);
    for (int v = 0; v < nv; ++v) {
        auto& list = normals[static_cast<std::size_t>(v)];
        for (int c = recv_off[static_cast<std::size_t>(v)];
             c < recv_off[static_cast<std::size_t>(v) + 1]; ++c) {
            const double* payload = &recv[w * static_cast<std::size_t>(c)];
            const int count = static_cast<int>(payload[0]);
            for (int k = 0; k < count; ++k) {
                AddDistinctNormal(list, payload + 1 + 3 * k);
            }
        }
        m_boundary_surfaces[v] = CountSurfaces(list, cos_angle);
    }
}

/*
 * BuildSummary: counts over the owned nodes (each node once), summed exactly as int64.
 */
void NodeLabels::BuildSummary() {
    const VertexField& field = *m_field;
    const int* owned = field.OwnedVertexMarker().HostRead();
    constexpr int num_counts = NUM_JUNCTION_KINDS * NUM_BOUNDARY_CONSTRAINTS;
    std::int64_t counts[num_counts + 2] = {};
    int max_labels = 0;
    for (int v = 0; v < field.NumVertices(); ++v) {
        max_labels = std::max(max_labels, NumLabels(v));
        if (owned[v] == 0) {
            continue;
        }
        ++counts[static_cast<int>(Kind(v)) * NUM_BOUNDARY_CONSTRAINTS + BoundaryConstraints(v)];
        counts[num_counts] += OverConstrained(v) ? 1 : 0;
        ++counts[num_counts + 1];
    }
    MPI_Allreduce(MPI_IN_PLACE, counts, num_counts + 2, MPI_INT64_T, MPI_SUM, field.Comm());
    MPI_Allreduce(MPI_IN_PLACE, &max_labels, 1, MPI_INT, MPI_MAX, field.Comm());
    for (int k = 0; k < NUM_JUNCTION_KINDS; ++k) {
        for (int b = 0; b < NUM_BOUNDARY_CONSTRAINTS; ++b) {
            m_summary.count[static_cast<std::size_t>(k)][static_cast<std::size_t>(b)] =
                counts[k * NUM_BOUNDARY_CONSTRAINTS + b];
        }
    }
    m_summary.over_constrained = counts[num_counts];
    m_summary.num_nodes = counts[num_counts + 1];
    m_summary.max_labels = max_labels;
}

} // namespace exaconstit::microstructure
