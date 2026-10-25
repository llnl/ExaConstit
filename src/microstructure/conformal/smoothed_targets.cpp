/*
 * Smoothed staircase positions of grain boundary nodes (smoothed_targets.hpp).
 */
#include "microstructure/conformal/smoothed_targets.hpp"

#include "microstructure/conformal/boundary_projector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace exaconstit::microstructure {

namespace {

constexpr double TAUBIN_LAMBDA = 0.5;
constexpr double TAUBIN_MU = -0.53;

/** @brief Index d when n is exactly +-e_d, else -1. */
int AxisOf(const double n[3]) {
    int zeros = 0;
    int axis = -1;
    for (int d = 0; d < 3; ++d) {
        if (n[d] == 0.0) {
            ++zeros;
        } else {
            axis = d;
        }
    }
    return (zeros == 2) ? axis : -1;
}

/** @brief True when every label of v is a label of u (both ascending). */
bool HasLabelsOf(const NodeLabels& labels, int u, int v) {
    const int* off = labels.LabelOffsets().HostRead();
    const int* items = labels.Labels().HostRead();
    return std::includes(items + off[u], items + off[u + 1], items + off[v], items + off[v + 1]);
}

/** @brief True when u lies on every boundary surface of v (normals within the feature angle). */
bool HasSurfacesOf(const NodeLabels& labels, int u, int v) {
    const int* off = labels.SurfaceOffsets().HostRead();
    const double* n = labels.SurfaceNormals().HostRead();
    for (int a = off[v]; a < off[v + 1]; ++a) {
        bool found = false;
        for (int b = off[u]; b < off[u + 1] && !found; ++b) {
            const double dot =
                n[3 * a] * n[3 * b] + n[3 * a + 1] * n[3 * b + 1] + n[3 * a + 2] * n[3 * b + 2];
            found = dot >= labels.FeatureCosine();
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Puts y back on the boundary surfaces of v: the planes through x0 (one or two; the
 * least-norm correction, then box coordinates restored bit for bit), or the closest point of its
 * curved patch.
 */
void ProjectOnSurfaces(const NodeLabels& labels, const SurfacePatches& patches, int v,
                       const double* x0, double y[3]) {
    const int* off = labels.SurfaceOffsets().HostRead();
    const double* nrm = labels.SurfaceNormals().HostRead();
    const int* planar = labels.SurfacePlanar().HostRead();
    const int num = off[v + 1] - off[v];
    if (num == 0) {
        return;
    }
    if (planar[off[v]] == 0) { // one curved surface (eligibility)
        double p[3];
        double pn[3];
        bool interior = false;
        patches.ClosestPoint(v, y, p, pn, interior);
        std::copy(p, p + 3, y);
        return;
    }
    // y - N^T (N N^T)^-1 (N (y - x0)) for unit normals n_0 (, n_1)
    const double* n0 = nrm + 3 * off[v];
    double r[2] = {0.0, 0.0};
    for (int d = 0; d < 3; ++d) {
        r[0] += n0[d] * (y[d] - x0[d]);
    }
    double a[2] = {r[0], 0.0};
    const double* n1 = (num == 2) ? n0 + 3 : nullptr;
    if (n1 != nullptr) {
        double g = 0.0;
        for (int d = 0; d < 3; ++d) {
            r[1] += n1[d] * (y[d] - x0[d]);
            g += n0[d] * n1[d];
        }
        const double det = 1.0 - g * g;
        a[0] = (r[0] - g * r[1]) / det;
        a[1] = (r[1] - g * r[0]) / det;
    }
    for (int d = 0; d < 3; ++d) {
        y[d] -= a[0] * n0[d] + ((n1 != nullptr) ? a[1] * n1[d] : 0.0);
    }
    for (int s = off[v]; s < off[v + 1]; ++s) {
        const int axis = AxisOf(nrm + 3 * s);
        if (axis >= 0) {
            y[axis] = x0[axis];
        }
    }
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
void SmoothGrainBoundaries(const NodeLabels& labels, const SurfacePatches& patches,
                           const mfem::Vector& h, const SmoothingParameters& params,
                           mfem::Vector& smoothed, std::vector<char>& moved) {
    const VertexField& field = labels.Field();
    mfem::ParMesh& pmesh = field.Mesh();
    const int nv = labels.NumNodes();
    const double* x0 = field.Coordinates().HostRead();
    const int* surf_off = labels.SurfaceOffsets().HostRead();
    const int* planar = labels.SurfacePlanar().HostRead();

    // Step 1
    moved.assign(static_cast<std::size_t>(nv), 0);
    for (int v = 0; v < nv; ++v) {
        const int n = labels.NumLabels(v);
        const int surfaces = surf_off[v + 1] - surf_off[v];
        bool curved = false;
        for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
            curved = curved || planar[s] == 0;
        }
        const bool curved_ok = !curved || (surfaces == 1 && patches.NumTriangles(v) > 0);
        moved[static_cast<std::size_t>(v)] =
            (n >= 2 && n <= 3 && !labels.OverConstrained(v) && curved_ok) ? 1 : 0;
    }
    const int* en_off = field.ElementVertexOffsets().HostRead();
    const int* en = field.ElementVertices().HostRead();
    std::vector<std::vector<int>> nbrs(static_cast<std::size_t>(nv));
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        const mfem::Element* el = pmesh.GetElement(e);
        for (int k = 0; k < el->GetNEdges(); ++k) {
            const int* ev = el->GetEdgeVertices(k);
            const int a = en[en_off[e] + ev[0]];
            const int b = en[en_off[e] + ev[1]];
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

    // Step 2
    std::vector<double> p(x0, x0 + 3 * static_cast<std::size_t>(nv));
    std::vector<int> send_off(static_cast<std::size_t>(nv) + 1, 0);
    std::vector<double> send;
    std::vector<int> recv_off;
    std::vector<double> recv;
    for (int step = 0; step < 2 * params.steps; ++step) {
        const double w = (step % 2 == 0) ? TAUBIN_LAMBDA : TAUBIN_MU;
        send.clear();
        for (int v = 0; v < nv; ++v) {
            for (const int u : nbrs[static_cast<std::size_t>(v)]) {
                send.insert(send.end(), x0 + 3 * u, x0 + 3 * u + 3);
                send.insert(send.end(), &p[3 * static_cast<std::size_t>(u)],
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
                return std::lexicographical_compare(a.begin(), a.begin() + 3, b.begin(),
                                                    b.begin() + 3);
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
            ProjectOnSurfaces(labels, patches, v, x0 + 3 * v, y);
        }
        p.swap(next);
    }

    // Step 3
    smoothed.SetSize(3 * nv);
    for (int v = 0; v < nv; ++v) {
        double* y = &p[3 * static_cast<std::size_t>(v)];
        const double* x = x0 + 3 * v;
        const double dist = std::sqrt((y[0] - x[0]) * (y[0] - x[0]) +
                                      (y[1] - x[1]) * (y[1] - x[1]) +
                                      (y[2] - x[2]) * (y[2] - x[2]));
        const double limit = params.max_move * h(v);
        if (moved[static_cast<std::size_t>(v)] != 0 && dist > limit) {
            for (int d = 0; d < 3; ++d) {
                y[d] = x[d] + (y[d] - x[d]) * (limit / dist);
            }
            ProjectOnSurfaces(labels, patches, v, x, y);
        }
        for (int d = 0; d < 3; ++d) {
            smoothed(3 * v + d) = y[d];
        }
    }
}

} // namespace exaconstit::microstructure
