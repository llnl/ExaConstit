/*
 * Node label sets, boundary surfaces and constraint counts (node_labels.hpp).
 */
#include "microstructure/conformal/node_labels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/util/collective_errors.hpp"

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
                      &normals[3 * static_cast<std::size_t>(root)] + 3,
                      s.normal);
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
namespace {

/** @brief Collective check of the element labels: one per local element, all >= 1. */
void CheckElementLabels(const mfem::ParMesh& pmesh, const mfem::Array<int>& element_label) {
    const MPI_Comm comm = pmesh.GetComm();
    ThrowIfAnyRank(comm,
                   element_label.Size() != pmesh.GetNE(),
                   "NodeLabels: element_label needs one entry per local element");
    bool bad_label = false;
    for (int e = 0; e < element_label.Size(); ++e) {
        bad_label = bad_label || element_label[e] < 1;
    }
    ThrowIfAnyRank(comm, bad_label, "NodeLabels: element labels must be >= 1");
}

} // namespace

NodeLabels::NodeLabels(mfem::ParMesh& pmesh,
                       const mfem::Array<int>& element_label,
                       const NodeLabelOptions& options) {
    const MPI_Comm comm = pmesh.GetComm();
    CheckElementLabels(pmesh, element_label);
    const double angle = options.feature_angle_degrees;
    ThrowIfAnyRank(comm,
                   !(angle > 0.0 && angle < 90.0),
                   "NodeLabels: feature_angle_degrees must be in (0, 90)");

    m_feature_cosine = std::cos(angle * PI / 180.0);
    m_fixed_labels = options.fixed_labels;
    ThrowIfAnyRank(comm,
                   !std::is_sorted(m_fixed_labels.begin(), m_fixed_labels.end()),
                   "NodeLabels: fixed_labels must be ascending");

    m_field = std::make_unique<VertexField>(pmesh, options.nodes);
    m_element_label = element_label;
    m_adjacency = std::make_unique<MeshAdjacency>(pmesh);
    m_keys = ComputeElementKeys(pmesh);
    m_adjacency->ExchangeKeys(m_keys, m_ghost_keys);
    FindSurfaceFaces();
    BuildLabelSets(element_label);
    BuildBoundarySurfaces(angle);
    FindPinches();
    BuildSummary();
}

bool NodeLabels::IsFixedLabel(int l) const {
    return std::binary_search(m_fixed_labels.begin(), m_fixed_labels.end(), l);
}

/*
 * FindSurfaceFaces
 *  1. Outer boundary faces: one element and not shared with another rank.
 *  2. With fixed labels: interface faces from the face adjacency (ghost labels exchanged,
 *     MeshAdjacency, collective): a face between labels a != b with at least one fixed is counted
 *     by the side of the non-fixed label, or of the smaller label when both are fixed, and only
 *     when that side is a local element (a shared face is therefore counted on one rank).
 */
void NodeLabels::FindSurfaceFaces() {
    mfem::ParMesh& pmesh = m_field->Mesh();
    const int nf = pmesh.GetNumFaces();

    // Step 1
    std::vector<char> shared(static_cast<std::size_t>(nf), 0);
    for (int i = 0; i < pmesh.GetNSharedFaces(); ++i) {
        shared[static_cast<std::size_t>(pmesh.GetSharedFace(i))] = 1;
    }
    m_surface_faces.assign(static_cast<std::size_t>(nf), 0);
    for (int f = 0; f < nf; ++f) {
        int e1 = -1;
        int e2 = -1;
        pmesh.GetFaceElements(f, &e1, &e2);
        m_surface_faces[static_cast<std::size_t>(f)] = (e2 < 0 &&
                                                        shared[static_cast<std::size_t>(f)] == 0)
                                                           ? 1
                                                           : 0;
    }

    // Step 2
    m_interface_faces.clear();
    if (m_fixed_labels.empty()) {
        return;
    }
    const MeshAdjacency& adj = *m_adjacency;
    mfem::Array<int> ghost;
    adj.ExchangeLabels(m_element_label, ghost);
    const int* off = adj.Offsets().HostRead();
    const int* nbr = adj.Neighbors().HostRead();
    const int* face = adj.FaceIndices().HostRead();
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        const int a = m_element_label[e];
        for (int k = off[e]; k < off[e + 1]; ++k) {
            const int n = nbr[k];
            const int b = adj.IsGhost(n) ? ghost[n - adj.NumElements()] : m_element_label[n];
            const bool fa = IsFixedLabel(a);
            const bool fb = IsFixedLabel(b);
            if (a == b || !(fa || fb)) {
                continue;
            }
            if ((!fa && fb) || (fa && fb && a < b)) {
                m_surface_faces[static_cast<std::size_t>(face[k])] = 1;
                m_interface_faces.emplace_back(face[k], e);
            }
        }
    }
}

NodeLabels::~NodeLabels() = default;

/*
 * SetElementLabels: the boundary surfaces do not depend on the labels, so only the label sets and
 * the summary are rebuilt.
 */
void NodeLabels::SetElementLabels(const mfem::Array<int>& element_label) {
    CheckElementLabels(m_field->Mesh(), element_label);
    bool moved_fixed = false;
    for (int e = 0; e < element_label.Size(); ++e) {
        moved_fixed = moved_fixed ||
                      (element_label[e] != m_element_label[e] &&
                       (IsFixedLabel(element_label[e]) || IsFixedLabel(m_element_label[e])));
    }
    ThrowIfAnyRank(m_field->Comm(),
                   moved_fixed,
                   "NodeLabels::SetElementLabels: elements of fixed labels cannot change");
    m_element_label = element_label;
    BuildLabelSets(element_label);
    FindPinches();
    BuildSummary();
}

JunctionKind NodeLabels::Kind(int v) const {
    const int n = NumLabels(v);
    return static_cast<JunctionKind>(std::min(n, NUM_JUNCTION_KINDS) - 1);
}

/*
 * BuildLabelSets
 *  1. Local counts: per node, (label, number of local elements with that label) over the node ->
 *     element table (complete: the field has no element mask), sorted by label.
 *  2. The (label, count) pairs go to the other copies (VertexField::ExchangeSharedLists).
 *  3. Each copy adds its own and the received counts per label (a sum of integers does not
 *     depend on the order) into the CSR arrays.
 */
void NodeLabels::BuildLabelSets(const mfem::Array<int>& element_label) {
    const VertexField& field = *m_field;
    const int nv = field.NumVertices();
    const int* ve_off = field.VertexElementOffsets().HostRead();
    const int* ve = field.VertexElements().HostRead();
    const int* labels = element_label.HostRead();

    // Step 1
    std::vector<std::vector<std::pair<int, int>>> counts(static_cast<std::size_t>(nv));
    const auto add = [](std::vector<std::pair<int, int>>& list, int label, int count) {
        auto it = std::lower_bound(list.begin(), list.end(), std::make_pair(label, 0));
        if (it != list.end() && it->first == label) {
            it->second += count;
        } else {
            list.insert(it, {label, count});
        }
    };
    std::vector<int> send_off(static_cast<std::size_t>(nv) + 1, 0);
    std::vector<double> send;
    for (int v = 0; v < nv; ++v) {
        auto& list = counts[static_cast<std::size_t>(v)];
        for (int k = ve_off[v]; k < ve_off[v + 1]; ++k) {
            add(list, labels[ve[k]], 1);
        }
        for (const auto& [label, count] : list) {
            send.push_back(static_cast<double>(label));
            send.push_back(static_cast<double>(count));
        }
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
    std::vector<int> item_counts;
    for (int v = 0; v < nv; ++v) {
        auto& list = counts[static_cast<std::size_t>(v)];
        for (int k = recv_off[static_cast<std::size_t>(v)];
             k < recv_off[static_cast<std::size_t>(v) + 1];
             k += 2) {
            add(list,
                static_cast<int>(recv[static_cast<std::size_t>(k)]),
                static_cast<int>(recv[static_cast<std::size_t>(k) + 1]));
        }
        // Fixed labels leave the set when the node has another label (their interfaces are
        // boundary surfaces); a node with fixed labels only keeps the smallest
        bool other = false;
        for (const auto& entry : list) {
            other = other || !IsFixedLabel(entry.first);
        }
        for (const auto& [label, count] : list) {
            if (IsFixedLabel(label) && (other || label != list.front().first)) {
                continue;
            }
            items.push_back(label);
            item_counts.push_back(count);
        }
        m_label_offsets[v + 1] = static_cast<int>(items.size());
    }
    m_labels.SetSize(static_cast<int>(items.size()));
    std::copy(items.begin(), items.end(), m_labels.begin());
    m_label_counts.SetSize(static_cast<int>(item_counts.size()));
    std::copy(item_counts.begin(), item_counts.end(), m_label_counts.begin());
}

/*
 * BuildBoundarySurfaces
 *  1. Boundary surface faces (SurfaceFaces(): outer boundary faces and the interface faces of
 *     fixed labels counted by this rank, FindSurfaceFaces).
 *  2. Unit normal at the face center: CalcOrtho of the face transformation's Jacobian (Nodes when
 *     the mesh has them), which points out of the face's first element (for an interface face:
 *     out of the element that counts it). Each node of the face (its
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

    // Step 1 (interface faces point out of the element that counts them)
    std::vector<int> out_of(static_cast<std::size_t>(pmesh.GetNumFaces()), -1);
    for (const auto& [f, e] : m_interface_faces) {
        out_of[static_cast<std::size_t>(f)] = e;
    }

    // Step 2
    std::vector<std::vector<double>> normals(static_cast<std::size_t>(nv));
    const mfem::FiniteElementSpace* node_fes = field.UsesMeshNodes() ? pmesh.GetNodes()->FESpace()
                                                                     : nullptr;
    mfem::IsoparametricTransformation trans;
    mfem::Array<int> face_nodes;
    mfem::Vector ortho(3);
    for (int f = 0; f < pmesh.GetNumFaces(); ++f) {
        if (m_surface_faces[static_cast<std::size_t>(f)] == 0) {
            continue;
        }
        int e1 = -1;
        int e2 = -1;
        pmesh.GetFaceElements(f, &e1, &e2);
        pmesh.GetFaceTransformation(f, &trans);
        const mfem::IntegrationPoint& center = mfem::Geometries.GetCenter(pmesh.GetFaceGeometry(f));
        trans.SetIntPoint(&center);
        mfem::CalcOrtho(trans.Jacobian(), ortho); // points out of e1
        const int from = out_of[static_cast<std::size_t>(f)];
        if (from >= 0 && from != e1) {
            ortho.Neg();
        }
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
             k < recv_off[static_cast<std::size_t>(v) + 1];
             k += 3) {
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
 * FindPinches
 *  1. Ghost labels (MeshAdjacency::ExchangeLabels).
 *  2. Per node, from every local element containing it: an entry (key, label) and, for each face
 *     neighbor through a face containing the node, a pair (key, neighbor key); keys as two exact
 *     32-bit halves in doubles. Fixed labels are left out.
 *  3. The lists go to the other copies (VertexField::ExchangeSharedLists).
 *  4. Per node: distinct elements, union of the pairs whose two elements have the same label,
 *     and the node is pinched when some label has two or more groups.
 */
void NodeLabels::FindPinches() {
    const VertexField& field = *m_field;
    mfem::ParMesh& pmesh = field.Mesh();
    const MeshAdjacency& adj = *m_adjacency;
    const int nv = field.NumVertices();

    // Step 1
    mfem::Array<int> ghost;
    adj.ExchangeLabels(m_element_label, ghost);
    const auto label_of = [&](int n) {
        return adj.IsGhost(n) ? ghost[n - adj.NumElements()] : m_element_label[n];
    };
    const auto key_of = [&](int n) {
        return adj.IsGhost(n) ? m_ghost_keys[static_cast<std::size_t>(n - adj.NumElements())]
                              : m_keys[static_cast<std::size_t>(n)];
    };

    // Step 2: entries per node, 4 doubles each: (kind, key hi, key lo, label or other key)
    const mfem::FiniteElementSpace* node_fes = field.UsesMeshNodes() ? pmesh.GetNodes()->FESpace()
                                                                     : nullptr;
    const int* en_off = field.ElementVertexOffsets().HostRead();
    const int* en = field.ElementVertices().HostRead();
    const int* off = adj.Offsets().HostRead();
    const int* nbr = adj.Neighbors().HostRead();
    const int* face = adj.FaceIndices().HostRead();
    std::vector<std::vector<double>> lists(static_cast<std::size_t>(nv));
    const auto split = [](std::uint64_t k, double& hi, double& lo) {
        hi = static_cast<double>(k >> 32);
        lo = static_cast<double>(k & 0xffffffffULL);
    };
    mfem::Array<int> face_nodes;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        const int a = m_element_label[e];
        if (IsFixedLabel(a)) {
            continue;
        }
        double kh, kl;
        split(m_keys[static_cast<std::size_t>(e)], kh, kl);
        for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
            auto& list = lists[static_cast<std::size_t>(en[k])];
            list.insert(list.end(), {0.0, kh, kl, static_cast<double>(a)});
        }
        for (int j = off[e]; j < off[e + 1]; ++j) {
            const int n = nbr[j];
            if (label_of(n) != a) {
                continue; // only same-label pairs join groups
            }
            if (node_fes != nullptr) {
                node_fes->GetFaceDofs(face[j], face_nodes);
            } else {
                pmesh.GetFaceVertices(face[j], face_nodes);
            }
            const std::uint64_t other = key_of(n);
            double oh, ol;
            split(other, oh, ol);
            for (int q = 0; q < face_nodes.Size(); ++q) {
                const int node = (face_nodes[q] >= 0) ? face_nodes[q] : -1 - face_nodes[q];
                // A pair: kind 1 (this key) directly followed by kind 2 (the neighbor's key)
                auto& list = lists[static_cast<std::size_t>(node)];
                list.insert(list.end(), {1.0, kh, kl, 0.0, 2.0, oh, ol, 0.0});
            }
        }
    }

    // Step 3
    std::vector<int> send_off(static_cast<std::size_t>(nv) + 1, 0);
    std::vector<double> send;
    for (int v = 0; v < nv; ++v) {
        const auto& list = lists[static_cast<std::size_t>(v)];
        send.insert(send.end(), list.begin(), list.end());
        send_off[static_cast<std::size_t>(v) + 1] = static_cast<int>(send.size());
    }
    std::vector<int> recv_off;
    std::vector<double> recv;
    field.ExchangeSharedLists(send_off, send, recv_off, recv);

    // Step 4
    m_pinched.SetSize(nv);
    const int* owned = field.OwnedVertexMarker().HostRead();
    std::int64_t count = 0;
    for (int v = 0; v < nv; ++v) {
        const auto vs = static_cast<std::size_t>(v);
        std::map<std::pair<double, double>, int> label; // element key -> label
        std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> pairs;
        const auto scan = [&](const std::vector<double>& items, int begin, int end) {
            for (int k = begin; k < end; k += 4) {
                const double* it = &items[static_cast<std::size_t>(k)];
                if (it[0] == 0.0) {
                    label[{it[1], it[2]}] = static_cast<int>(it[3]);
                } else if (it[0] == 1.0) {
                    pairs.push_back({{it[1], it[2]}, {it[5], it[6]}});
                    k += 4; // the kind-2 entry that follows
                }
            }
        };
        scan(send, send_off[vs], send_off[vs + 1]);
        scan(recv, recv_off[vs], recv_off[vs + 1]);
        // Union-find over the elements around the node
        std::map<std::pair<double, double>, std::pair<double, double>> parent;
        for (const auto& entry : label) {
            parent[entry.first] = entry.first;
        }
        const auto find = [&](std::pair<double, double> x) {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        };
        for (const auto& [a, b] : pairs) {
            if (parent.count(a) == 0 || parent.count(b) == 0) {
                continue;
            }
            const auto ra = find(a);
            const auto rb = find(b);
            if (ra != rb) {
                parent[std::max(ra, rb)] = std::min(ra, rb);
            }
        }
        std::map<int, std::set<std::pair<double, double>>> groups;
        for (const auto& entry : label) {
            groups[entry.second].insert(find(entry.first));
        }
        bool pinched = false;
        for (const auto& g : groups) {
            pinched = pinched || g.second.size() > 1;
        }
        m_pinched[v] = pinched ? 1 : 0;
        count += (pinched && owned[v] != 0) ? 1 : 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT64_T, MPI_SUM, field.Comm());
    m_num_pinched = count;
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
