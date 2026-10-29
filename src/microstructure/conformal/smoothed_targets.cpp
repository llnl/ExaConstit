/*
 * Smoothed staircase positions of the grain boundary nodes, by Taubin smoothing on the node
 * lattice (smoothed_targets.hpp).
 */
#include "microstructure/conformal/smoothed_targets.hpp"

#include "microstructure/conformal/boundary_projector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief Taubin's shrinking step (a plain averaging step) for order-1 Nodes. */
constexpr double TAUBIN_LAMBDA = 0.5;
/** @brief Taubin's inflating step for order-1 Nodes; |mu| slightly larger than lambda so curved
 * boundaries keep their size (Taubin 1995). */
constexpr double TAUBIN_MU = -0.53;

/**
 * @brief Pairs of nodal points of the element that are neighbors on its node lattice: their
 * reference positions differ by one lattice step (1 / order) along an edge direction of the
 * reference element (the axes for a hex; the axes and their differences for a tet). For order 1
 * these are the element edges; for order 2 (uniform lattice) the edges of the sub-elements.
 * Indices are element-local dof indices (the order of FiniteElement::GetNodes).
 */
std::vector<std::pair<int, int>> LatticeNeighbors(const mfem::FiniteElement& fe) {
    const mfem::IntegrationRule& ref = fe.GetNodes();
    const double p = static_cast<double>(fe.GetOrder());
    const bool tet = fe.GetGeomType() == mfem::Geometry::TETRAHEDRON;
    std::vector<std::pair<int, int>> pairs;
    for (int a = 0; a < ref.GetNPoints(); ++a) {
        for (int b = a + 1; b < ref.GetNPoints(); ++b) {
            const mfem::IntegrationPoint& ia = ref.IntPoint(a);
            const mfem::IntegrationPoint& ib = ref.IntPoint(b);
            const double d[3] = {p * (ib.x - ia.x), p * (ib.y - ia.y), p * (ib.z - ia.z)};
            int ones = 0;
            int zeros = 0;
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                ones += (std::fabs(std::fabs(d[k]) - 1.0) < 1e-9) ? 1 : 0;
                zeros += (std::fabs(d[k]) < 1e-9) ? 1 : 0;
                sum += d[k];
            }
            // One axis step, or (tets) one step along a difference of two axes
            if ((ones == 1 && zeros == 2) ||
                (tet && ones == 2 && zeros == 1 && std::fabs(sum) < 1e-9)) {
                pairs.emplace_back(a, b);
            }
        }
    }
    return pairs;
}

/** @brief True when every label of v is a label of u (both label sets are sorted, so
 * std::includes applies). */
bool HasLabelsOf(const NodeLabels& labels, int u, int v) {
    const int* off = labels.LabelOffsets().HostRead();
    const int* items = labels.Labels().HostRead();
    return std::includes(items + off[u], items + off[u + 1], items + off[v], items + off[v + 1]);
}

/** @brief True when u lies on every boundary surface of v: for each surface of v, u has a
 * surface whose normal is within the feature angle. */
bool HasSurfacesOf(const NodeLabels& labels, int u, int v) {
    const int* off = labels.SurfaceOffsets().HostRead();
    const double* n = labels.SurfaceNormals().HostRead();
    for (int a = off[v]; a < off[v + 1]; ++a) {
        bool found = false;
        for (int b = off[u]; b < off[u + 1] && !found; ++b) {
            const double dot = n[3 * a] * n[3 * b] + n[3 * a + 1] * n[3 * b + 1] +
                               n[3 * a + 2] * n[3 * b + 2];
            found = dot >= labels.FeatureCosine();
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

} // namespace

/*
 * SmoothGrainBoundaries
 *  1. Grain boundary nodes (file comment) and their local neighbors through element edges.
 *  2. Steps: each rank lists (input position, current position) of the known neighbors of every
 *     grain boundary node, the lists are exchanged, each copy merges, removes duplicates by input
 *     position, sorts, averages, takes the lambda or mu step and projects back on the surfaces.
 *  3. Move limit.
 */
void SmoothGrainBoundaries(const NodeLabels& labels,
                           const SurfacePatches& patches,
                           const mfem::Vector& h,
                           const SmoothingParameters& params,
                           mfem::Vector& smoothed,
                           std::vector<char>& moved) {
    const VertexField& field = labels.Field();
    mfem::ParMesh& pmesh = field.Mesh();
    const int nv = labels.NumNodes();
    const double* x0 = field.Coordinates().HostRead();

    // Step 1: which nodes move, and each one's neighbors on the node lattice of its local
    // elements (the neighbors on other ranks arrive through the exchange in step 2)
    moved.assign(static_cast<std::size_t>(nv), 0);
    for (int v = 0; v < nv; ++v) {
        const int n = labels.NumLabels(v);
        const bool curved_ok = CurvedPatchesReady(labels, patches, v);
        moved[static_cast<std::size_t>(v)] = (n >= 2 && n <= 3 && !labels.OverConstrained(v) &&
                                              curved_ok)
                                                 ? 1
                                                 : 0;
    }
    const int* en_off = field.ElementVertexOffsets().HostRead();
    const int* en = field.ElementVertices().HostRead();
    // The element's nodal points: the Nodes space, or the vertices (order 1) without Nodes
    const mfem::FiniteElementSpace* fes = (pmesh.GetNodes() != nullptr)
                                              ? pmesh.GetNodes()->FESpace()
                                              : nullptr;
    const mfem::H1_FECollection linear(1, 3);
    std::vector<std::vector<int>> nbrs(static_cast<std::size_t>(nv));
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        const mfem::FiniteElement& fe = (fes != nullptr) ? *fes->GetFE(e)
                                                         : *linear.FiniteElementForGeometry(
                                                               pmesh.GetElementBaseGeometry(e));
        for (const auto& [ka, kb] : LatticeNeighbors(fe)) {
            const int a = en[en_off[e] + ka];
            const int b = en[en_off[e] + kb];
            for (const auto& [v, u] : {std::pair<int, int>{a, b}, std::pair<int, int>{b, a}}) {
                if (moved[static_cast<std::size_t>(v)] != 0 && HasLabelsOf(labels, u, v) &&
                    HasSurfacesOf(labels, u, v)) {
                    nbrs[static_cast<std::size_t>(v)].push_back(u);
                }
            }
        }
    }
    for (auto& list : nbrs) {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }

    // Step 2: the smoothing steps. p = current positions (starting at the input positions).
    // Each list entry is 6 doubles: a neighbor's input position (its identity) and its current
    // position.
    std::vector<double> p(x0, x0 + 3 * static_cast<std::size_t>(nv));
    std::vector<int> send_off(static_cast<std::size_t>(nv) + 1, 0);
    std::vector<double> send;
    std::vector<int> recv_off;
    std::vector<double> recv;
    // Both the smoothing length (sqrt(steps) nodal spacings) and Taubin's pass band
    // k_pb = 1 / lambda + 1 / mu (wavelengths longer than about 2 pi / sqrt(k_pb) spacings are
    // kept) are in units of the nodal spacing, element size / order. For order p the steps are
    // multiplied by p^2 and the pass band divided by p^2, so both stay the same in element sizes.
    const int order = field.LatticeOrder();
    const int steps = params.steps * order * order;
    const double pass_band = (1.0 / TAUBIN_LAMBDA + 1.0 / TAUBIN_MU) / (order * order);
    const double mu = 1.0 / (pass_band - 1.0 / TAUBIN_LAMBDA);
    for (int step = 0; step < 2 * steps; ++step) {
        const double w = (step % 2 == 0) ? TAUBIN_LAMBDA : mu;
        send.clear();
        for (int v = 0; v < nv; ++v) {
            for (const int u : nbrs[static_cast<std::size_t>(v)]) {
                send.insert(send.end(), x0 + 3 * u, x0 + 3 * u + 3);
                send.insert(send.end(),
                            &p[3 * static_cast<std::size_t>(u)],
                            &p[3 * static_cast<std::size_t>(u)] + 3);
            }
            send_off[static_cast<std::size_t>(v) + 1] = static_cast<int>(send.size());
        }
        field.ExchangeSharedLists(send_off, send, recv_off, recv);
        std::vector<double> next(p);
        for (int v = 0; v < nv; ++v) {
            if (moved[static_cast<std::size_t>(v)] == 0) {
                continue;
            }
            const auto vs = static_cast<std::size_t>(v);
            std::vector<std::array<double, 6>> list;
            const auto append = [&list](const std::vector<double>& items, int begin, int end) {
                for (int k = begin; k < end; k += 6) {
                    std::array<double, 6> item;
                    const double* src = &items[static_cast<std::size_t>(k)];
                    std::copy(src, src + 6, item.begin());
                    list.push_back(item);
                }
            };
            append(send, send_off[vs], send_off[vs + 1]);
            append(recv, recv_off[vs], recv_off[vs + 1]);
            if (list.empty()) {
                continue;
            }
            const auto key_less = [](const std::array<double, 6>& a,
                                     const std::array<double, 6>& b) {
                return std::lexicographical_compare(
                    a.begin(), a.begin() + 3, b.begin(), b.begin() + 3);
            };
            const auto key_equal = [](const std::array<double, 6>& a,
                                      const std::array<double, 6>& b) {
                return std::equal(a.begin(), a.begin() + 3, b.begin());
            };
            std::sort(list.begin(), list.end(), key_less);
            list.erase(std::unique(list.begin(), list.end(), key_equal), list.end());
            double mean[3] = {0.0, 0.0, 0.0};
            for (const auto& item : list) {
                for (int d = 0; d < 3; ++d) {
                    mean[d] += item[3 + static_cast<std::size_t>(d)];
                }
            }
            double* y = &next[3 * static_cast<std::size_t>(v)];
            for (int d = 0; d < 3; ++d) {
                y[d] += w * (mean[d] / static_cast<double>(list.size()) - y[d]);
            }
            ProjectOnNodeSurfaces(
                labels, patches, v, x0 + 3 * v, 1e-10 * h(v), SURFACE_PROJECTION_ITERATIONS, y);
        }
        p.swap(next);
    }

    // Step 3: move limit, scaling a long move back toward the input position (and onto the
    // node's surfaces again)
    smoothed.SetSize(3 * nv);
    for (int v = 0; v < nv; ++v) {
        double* y = &p[3 * static_cast<std::size_t>(v)];
        const double* x = x0 + 3 * v;
        const double dist = std::sqrt((y[0] - x[0]) * (y[0] - x[0]) +
                                      (y[1] - x[1]) * (y[1] - x[1]) +
                                      (y[2] - x[2]) * (y[2] - x[2]));
        const double limit = params.max_move * h(v) * field.LatticeOrder(); // element sizes
        if (moved[static_cast<std::size_t>(v)] != 0 && dist > limit) {
            for (int d = 0; d < 3; ++d) {
                y[d] = x[d] + (y[d] - x[d]) * (limit / dist);
            }
            ProjectOnNodeSurfaces(
                labels, patches, v, x, 1e-10 * h(v), SURFACE_PROJECTION_ITERATIONS, y);
        }
        for (int d = 0; d < 3; ++d) {
            smoothed(3 * v + d) = y[d];
        }
    }
}

} // namespace exaconstit::microstructure
