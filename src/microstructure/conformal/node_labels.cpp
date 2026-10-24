/*
 * Node label sets, boundary surfaces and constraint counts (node_labels.hpp).
 */
#include "microstructure/conformal/node_labels.hpp"

#include "microstructure/util/collective_errors.hpp"

#include <algorithm>
#include <array>
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

/** @brief One boundary surface at a node: its normal and whether it is flat there. */
struct Surface {
    double normal[3];
    bool planar;
};

/**
 * @brief Groups the unit normals (3 doubles each, sorted lexicographically) into surfaces:
 * connected components of steps between normals whose dot product is >= cos_angle (components do
 * not depend on the order of the normals). A surface is planar when all its normals agree to
 * 1e-12 (then its normal is its first normal, so exact axis normals stay exact); otherwise its
 * normal is the normalized sum in sorted order. Surfaces are listed in the order of their first
 * normal. Host.
 */
std::vector<Surface> GroupSurfaces(const std::vector<double>& normals, double cos_angle) {
    const int m = static_cast<int>(normals.size() / 3);
    std::vector<int> parent(static_cast<std::size_t>(m));
    std::iota(parent.begin(), parent.end(), 0);
    const auto dot = [&](int a, int b) {
        const double* na = &normals[3 * static_cast<std::size_t>(a)];
        const double* nb = &normals[3 * static_cast<std::size_t>(b)];
        return na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2];
    };
    for (int a = 0; a < m; ++a) {
        for (int b = a + 1; b < m; ++b) {
            if (dot(a, b) >= cos_angle) {
                const int ra = FindRoot(parent, a);
                const int rb = FindRoot(parent, b);
                if (ra != rb) {
                    parent[static_cast<std::size_t>(std::max(ra, rb))] = std::min(ra, rb);
                }
            }
        }
    }
    std::vector<Surface> surfaces;
    for (int root = 0; root < m; ++root) {
        if (FindRoot(parent, root) != root) {
            continue;
        }
        Surface s{{0.0, 0.0, 0.0}, true};
        for (int a = root; a < m; ++a) {
            if (FindRoot(parent, a) != root) {
                continue;
            }
            s.planar = s.planar && dot(root, a) >= 1.0 - 1e-12;
            for (int d = 0; d < 3; ++d) {
                s.normal[d] += normals[3 * static_cast<std::size_t>(a) + d];
            }
        }
        if (s.planar) {
            std::copy(&normals[3 * static_cast<std::size_t>(root)],
                      &normals[3 * static_cast<std::size_t>(root)] + 3, s.normal);
        } else {
            const double len = std::sqrt(s.normal[0] * s.normal[0] + s.normal[1] * s.normal[1] +
                                         s.normal[2] * s.normal[2]);
            for (double& c : s.normal) {
                c /= len;
            }
        }
        surfaces.push_back(s);
    }
    return surfaces;
}

/** @brief Sorts a list of 3-vectors lexicographically (a canonical order on every copy). Host. */
void SortNormals(std::vector<double>& list) {
    const std::size_t m = list.size() / 3;
    std::vector<std::array<double, 3>> v(m);
    for (std::size_t k = 0; k < m; ++k) {
        v[k] = {list[3 * k], list[3 * k + 1], list[3 * k + 2]};
    }
    std::sort(v.begin(), v.end());
    for (std::size_t k = 0; k < m; ++k) {
        std::copy(v[k].begin(), v[k].end(), &list[3 * k]);
    }
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
 *  2. The sets go to the other copies (VertexField::ExchangeSharedLists).
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
    std::vector<int> send_off(static_cast<std::size_t>(nv) + 1, 0);
    std::vector<double> send;
    for (int v = 0; v < nv; ++v) {
        auto& set = sets[static_cast<std::size_t>(v)];
        for (int k = ve_off[v]; k < ve_off[v + 1]; ++k) {
            set.push_back(labels[ve[k]]);
        }
        std::sort(set.begin(), set.end());
        set.erase(std::unique(set.begin(), set.end()), set.end());
        send.insert(send.end(), set.begin(), set.end());
        send_off[static_cast<std::size_t>(v) + 1] = static_cast<int>(send.size());
    }

    // Step 2
    std::vector<double> recv;
    std::vector<int> recv_off;
    field.ExchangeSharedLists(send_off, send, recv_off, recv);

    // Step 3
    m_label_offsets.SetSize(nv + 1);
    m_label_offsets[0] = 0;
    std::vector<int> items;
    for (int v = 0; v < nv; ++v) {
        auto& set = sets[static_cast<std::size_t>(v)];
        for (int k = recv_off[static_cast<std::size_t>(v)];
             k < recv_off[static_cast<std::size_t>(v) + 1]; ++k) {
            set.push_back(static_cast<int>(recv[static_cast<std::size_t>(k)]));
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
 *  3. The lists go to the other copies (VertexField::ExchangeSharedLists).
 *  4. Surfaces = connected components of all normals of the node, sorted first so every copy
 *     sees the same order (GroupSurfaces), stored with their normal and planarity.
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
    std::vector<int> send_off(static_cast<std::size_t>(nv) + 1, 0);
    std::vector<double> send;
    for (int v = 0; v < nv; ++v) {
        const auto& list = normals[static_cast<std::size_t>(v)];
        send.insert(send.end(), list.begin(), list.end());
        send_off[static_cast<std::size_t>(v) + 1] = static_cast<int>(send.size());
    }
    std::vector<double> recv;
    std::vector<int> recv_off;
    field.ExchangeSharedLists(send_off, send, recv_off, recv);

    // Step 4
    const double cos_angle = std::cos(feature_angle_degrees * PI / 180.0);
    m_boundary_surfaces.SetSize(nv);
    m_surface_offsets.SetSize(nv + 1);
    m_surface_offsets[0] = 0;
    std::vector<double> normals_out;
    std::vector<int> planar_out;
    for (int v = 0; v < nv; ++v) {
        auto& list = normals[static_cast<std::size_t>(v)];
        for (int k = recv_off[static_cast<std::size_t>(v)];
             k < recv_off[static_cast<std::size_t>(v) + 1]; k += 3) {
            AddDistinctNormal(list, &recv[static_cast<std::size_t>(k)]);
        }
        SortNormals(list);
        const std::vector<Surface> surfaces = GroupSurfaces(list, cos_angle);
        m_boundary_surfaces[v] = static_cast<int>(surfaces.size());
        for (const Surface& s : surfaces) {
            normals_out.insert(normals_out.end(), s.normal, s.normal + 3);
            planar_out.push_back(s.planar ? 1 : 0);
        }
        m_surface_offsets[v + 1] = static_cast<int>(planar_out.size());
    }
    m_surface_normals.SetSize(static_cast<int>(normals_out.size()));
    std::copy(normals_out.begin(), normals_out.end(), m_surface_normals.begin());
    m_surface_planar.SetSize(static_cast<int>(planar_out.size()));
    std::copy(planar_out.begin(), planar_out.end(), m_surface_planar.begin());
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
