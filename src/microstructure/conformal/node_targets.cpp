/*
 * Targets on the analytic grain boundaries and their validation (node_targets.hpp).
 */
#include "microstructure/conformal/node_targets.hpp"

#include "microstructure/conformal/boundary_projector.hpp"
#include "microstructure/conformal/manifold_projector.hpp"
#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "axom/core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

namespace {

/** @brief Index d when n is exactly +-e_d (two zero components), else -1. Host. */
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

/** @brief Per-node work of the target computation before the foreign-seed check. */
struct Candidate {
    int node;
    int first_seed; ///< offset of the node's seeds in the exclusion CSR
    int num_seeds;
};

} // namespace

/*
 * Constructor
 *  1. Collective input checks.
 *  2. Mesh size h per node (ComputeMeshSize).
 *  3. Per node: status from the label set and surfaces; for the others the equations
 *     (grain boundaries to the first label's seed, flat boundary planes), the projection (for a
 *     node on one curved surface alternated with closest points on its patch,
 *     boundary_projector.hpp), the axis-plane coordinates restored bit for bit, the residual and
 *     distance checks.
 *  4. Foreign-seed check of the remaining candidates in one batched exact seed search that
 *     excludes each node's own seeds.
 *  5. Summary.
 */
NodeTargets::NodeTargets(const NodeLabels& labels, MetricType metric, const SeedSet& seeds,
                         const MetricSetup& setup, const std::vector<int>& grain_seed,
                         const NodeTargetOptions& options)
    : m_labels(labels) {
    const VertexField& field = labels.Field();
    const MPI_Comm comm = field.Comm();
    const int nv = labels.NumNodes();
    const int num_seeds = static_cast<int>(seeds.Size());

    // Step 1
    ThrowIfAnyRank(comm,
                   !(options.max_move > 0.0) || !(options.tolerance > 0.0) ||
                       options.max_iterations < 1 || options.num_bins < 1,
                   "NodeTargets: max_move and tolerance must be > 0, max_iterations and num_bins "
                   ">= 1");
    ThrowIfAnyRank(comm, num_seeds < 1, "NodeTargets: the seed set is empty");
    bool bad_seed = false;
    for (const int s : grain_seed) {
        bad_seed = bad_seed || s < -1 || s >= num_seeds;
    }
    ThrowIfAnyRank(comm, bad_seed, "NodeTargets: grain_seed holds a seed index out of range");
    const int* items = labels.Labels().HostRead();
    bool bad_label = false;
    for (int k = 0; k < labels.Labels().Size(); ++k) {
        bad_label = bad_label || items[k] > static_cast<int>(grain_seed.size());
    }
    ThrowIfAnyRank(comm, bad_label, "NodeTargets: a node label has no entry in grain_seed");

    // Step 2
    ComputeMeshSize();

    m_targets.SetSize(3 * nv);
    m_status.SetSize(nv);
    m_reason.SetSize(nv);
    const double* xyz = field.Coordinates().HostRead();
    double* target = m_targets.HostWrite();
    std::copy(xyz, xyz + 3 * nv, target);
    m_status = static_cast<int>(TargetStatus::NONE);
    m_reason = static_cast<int>(TargetReason::NONE);
    const auto set_status = [&](int v, TargetStatus s, TargetReason r) {
        m_status[v] = static_cast<int>(s);
        m_reason[v] = static_cast<int>(r);
    };

    const int* off = labels.LabelOffsets().HostRead();
    const int* surf_off = labels.SurfaceOffsets().HostRead();
    const double* surf_normal = labels.SurfaceNormals().HostRead();
    const int* surf_planar = labels.SurfacePlanar().HostRead();

    // Nodes on one curved boundary surface get the triangle patch of their one-ring (the same
    // selection on every copy: it depends only on the labels, surfaces and grain_seed)
    const auto analytic_node = [&](int v) {
        for (int k = off[v]; k < off[v + 1]; ++k) {
            if (grain_seed[static_cast<std::size_t>(items[k] - 1)] < 0) {
                return false;
            }
        }
        return true;
    };
    std::vector<char> on_curved(static_cast<std::size_t>(nv), 0);
    for (int v = 0; v < nv; ++v) {
        on_curved[static_cast<std::size_t>(v)] =
            (labels.NumLabels(v) > 1 && !labels.OverConstrained(v) &&
             labels.BoundarySurfaces(v) == 1 && surf_planar[surf_off[v]] == 0 && analytic_node(v))
                ? 1
                : 0;
    }
    const SurfacePatches patches(field, on_curved);

    DispatchMetric(metric, [&](auto metric_tag) {
        using Metric = decltype(metric_tag);
        SeedIndex<axom::SEQ_EXEC, Metric> index;
        index.Build(seeds, setup, options.num_bins,
                    axom::execution_space<axom::SEQ_EXEC>::allocatorID());
        const Metric& f = index.GetMetric();

        // Step 3
        std::vector<Candidate> candidates;
        std::vector<int> own_seeds;
        for (int v = 0; v < nv; ++v) {
            const int n = labels.NumLabels(v);
            if (n == 1) {
                continue;
            }
            if (labels.OverConstrained(v)) {
                set_status(v, TargetStatus::OVER_CONSTRAINED, TargetReason::NONE);
                continue;
            }
            int node_seeds[MANIFOLD_MAX_ROWS + 1];
            bool analytic = true;
            for (int k = 0; k < n; ++k) {
                node_seeds[k] = grain_seed[static_cast<std::size_t>(items[off[v] + k] - 1)];
                analytic = analytic && node_seeds[k] >= 0;
            }
            if (!analytic) {
                set_status(v, TargetStatus::HELD, TargetReason::NON_ANALYTIC);
                continue;
            }
            bool curved = false;
            for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
                curved = curved || surf_planar[s] == 0;
            }
            if (curved && on_curved[static_cast<std::size_t>(v)] == 0) {
                set_status(v, TargetStatus::HELD, TargetReason::CURVED_FEATURE);
                continue;
            }
            bool same = false;
            for (int a = 0; a < n; ++a) {
                for (int b = a + 1; b < n; ++b) {
                    same = same || node_seeds[a] == node_seeds[b];
                }
            }
            if (same) {
                set_status(v, TargetStatus::FAKE_JUNCTION, TargetReason::SAME_SEED);
                continue;
            }

            // Equations: grain boundaries, then the boundary planes through x
            const double* x = xyz + 3 * v;
            ManifoldRow rows[MANIFOLD_MAX_ROWS];
            int num_rows = 0;
            for (int k = 1; k < n; ++k) {
                rows[num_rows].kind = ManifoldRow::GRAIN_BOUNDARY;
                rows[num_rows].seed_a = node_seeds[0];
                rows[num_rows].seed_b = node_seeds[k];
                ++num_rows;
            }
            for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
                ManifoldRow& r = rows[num_rows++];
                r.kind = ManifoldRow::PLANE;
                const double* nrm = surf_normal + 3 * s;
                r.normal[0] = nrm[0];
                r.normal[1] = nrm[1];
                r.normal[2] = nrm[2];
                r.offset = nrm[0] * x[0] + nrm[1] * x[1] + nrm[2] * x[2];
            }

            const double h = m_h(v);
            const double tol = options.tolerance * h;
            double y[3];
            ProjectionResult result = ProjectionResult::NOT_CONVERGED;
            if (!curved) {
                result = ProjectOntoManifold(f, rows, num_rows, x, tol, options.max_iterations, y);
            } else {
                // Curved surface (the last row): alternate between the closest point p of the
                // current iterate on the patch, the plane through p (the triangle's plane, or the
                // plane normal to y - p when p is on an edge or corner of the patch and y is off
                // the surface), and the projection of x on the grain boundaries and that plane
                std::copy(x, x + 3, y);
                ManifoldRow& plane = rows[num_rows - 1];
                for (int it = 0; it < options.max_iterations; ++it) {
                    double p[3];
                    double nrm[3];
                    bool interior = false;
                    const double dist = patches.ClosestPoint(v, y, p, nrm, interior);
                    if (!interior && dist > tol) {
                        for (int d = 0; d < 3; ++d) {
                            nrm[d] = (y[d] - p[d]) / dist;
                        }
                    }
                    std::copy(nrm, nrm + 3, plane.normal);
                    plane.offset = nrm[0] * p[0] + nrm[1] * p[1] + nrm[2] * p[2];
                    double y_new[3];
                    result = ProjectOntoManifold(f, rows, num_rows, x, tol,
                                                 options.max_iterations, y_new);
                    if (result != ProjectionResult::CONVERGED) {
                        break;
                    }
                    const double step2 = (y_new[0] - y[0]) * (y_new[0] - y[0]) +
                                         (y_new[1] - y[1]) * (y_new[1] - y[1]) +
                                         (y_new[2] - y[2]) * (y_new[2] - y[2]);
                    std::copy(y_new, y_new + 3, y);
                    result = ProjectionResult::NOT_CONVERGED;
                    if (step2 <= tol * tol && dist <= tol) {
                        result = ProjectionResult::CONVERGED;
                        break;
                    }
                }
                if (result == ProjectionResult::CONVERGED) {
                    double p[3];
                    double nrm[3];
                    bool interior = false;
                    if (patches.ClosestPoint(v, y, p, nrm, interior) > tol) {
                        result = ProjectionResult::NOT_CONVERGED;
                    }
                }
            }
            if (result == ProjectionResult::SINGULAR) {
                set_status(v, TargetStatus::FAKE_JUNCTION, TargetReason::SINGULAR);
                continue;
            }
            if (result == ProjectionResult::NOT_CONVERGED) {
                set_status(v, TargetStatus::FAKE_JUNCTION, TargetReason::NOT_CONVERGED);
                continue;
            }
            // Axis planes keep the coordinate bit for bit
            for (int k = n - 1; k < num_rows; ++k) {
                const int axis = AxisOf(rows[k].normal);
                if (axis >= 0) {
                    y[axis] = x[axis];
                }
            }
            bool residual_ok = true;
            for (int k = 0; k < num_rows; ++k) {
                double g[3];
                const double phi = EvalManifoldRow(f, rows[k], y, g);
                const double glen = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
                residual_ok = residual_ok && std::fabs(phi) <= options.tolerance * h * glen;
            }
            if (!residual_ok) {
                set_status(v, TargetStatus::FAKE_JUNCTION, TargetReason::RESIDUAL);
                continue;
            }
            const double move = std::sqrt((y[0] - x[0]) * (y[0] - x[0]) +
                                          (y[1] - x[1]) * (y[1] - x[1]) +
                                          (y[2] - x[2]) * (y[2] - x[2]));
            if (move > options.max_move * h) {
                set_status(v, TargetStatus::FAKE_JUNCTION, TargetReason::TOO_FAR);
                continue;
            }
            std::copy(y, y + 3, target + 3 * v);
            candidates.push_back({v, static_cast<int>(own_seeds.size()), n});
            own_seeds.insert(own_seeds.end(), node_seeds, node_seeds + n);
        }

        // Step 4: best seed outside each node's own seeds, at its target
        const int nc = static_cast<int>(candidates.size());
        mfem::Vector qx(nc), qy(nc), qz(nc);
        mfem::Array<int> ex_off(nc + 1);
        mfem::Array<int> ex_ids(static_cast<int>(own_seeds.size()));
        ex_off[0] = 0;
        for (int c = 0; c < nc; ++c) {
            const int v = candidates[static_cast<std::size_t>(c)].node;
            qx(c) = target[3 * v];
            qy(c) = target[3 * v + 1];
            qz(c) = target[3 * v + 2];
            ex_off[c + 1] = ex_off[c] + candidates[static_cast<std::size_t>(c)].num_seeds;
        }
        std::copy(own_seeds.begin(), own_seeds.end(), ex_ids.begin());
        mfem::Array<BestPair> best;
        if (nc > 0) {
            index.FindBestExcluding(qx, qy, qz, ex_off, ex_ids, best);
        }
        const BestPair* best_host = (nc > 0) ? best.HostRead() : nullptr;
        for (int c = 0; c < nc; ++c) {
            const Candidate& cand = candidates[static_cast<std::size_t>(c)];
            const int v = cand.node;
            const double* y = target + 3 * v;
            double largest = -HUGE_VAL;
            double grad_max = 0.0;
            for (int k = 0; k < cand.num_seeds; ++k) {
                const int s = own_seeds[static_cast<std::size_t>(cand.first_seed + k)];
                largest = std::max(largest, f.Eval(s, y[0], y[1], y[2]));
                double g[3];
                f.Grad(s, y[0], y[1], y[2], g);
                grad_max = std::max(grad_max, std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]));
            }
            const double delta = options.tolerance * m_h(v) * grad_max;
            if (best_host[c].id1 >= 0 && best_host[c].f1 < largest - delta) {
                std::copy(xyz + 3 * v, xyz + 3 * v + 3, target + 3 * v);
                set_status(v, TargetStatus::FAKE_JUNCTION, TargetReason::FOREIGN_SEED);
            } else {
                set_status(v, TargetStatus::FIT, TargetReason::NONE);
            }
        }
    });

    // Step 5
    BuildSummary();
}

/*
 * ComputeMeshSize: per element Mesh::GetElementSize(e, 0) / p (mesh.cpp: |det J|^(1/3) at the
 * element center, with J relative to the regular reference element, Geometries.JacToPerfJac: the
 * edge length of the regular cube or tetrahedron of the same volume), minimum over the local
 * elements of each node, then the minimum over the copies (exact, so every copy has the same h).
 */
void NodeTargets::ComputeMeshSize() {
    const VertexField& field = m_labels.Field();
    mfem::ParMesh& pmesh = field.Mesh();
    const int nv = field.NumVertices();
    const int ne = pmesh.GetNE();
    std::vector<double> size(static_cast<std::size_t>(ne));
    const double p = static_cast<double>(field.LatticeOrder());
    for (int e = 0; e < ne; ++e) {
        size[static_cast<std::size_t>(e)] = pmesh.GetElementSize(e, 0) / p;
    }
    const int* ve_off = field.VertexElementOffsets().HostRead();
    const int* ve = field.VertexElements().HostRead();
    m_h.SetSize(nv);
    for (int v = 0; v < nv; ++v) {
        double h = HUGE_VAL;
        for (int k = ve_off[v]; k < ve_off[v + 1]; ++k) {
            h = std::min(h, size[static_cast<std::size_t>(ve[k])]);
        }
        m_h(v) = h;
    }
    field.ReduceSharedMinimum(m_h);
}

/*
 * BuildSummary: counts over the owned nodes (int64 sums) and the largest move ratio (maximum).
 */
void NodeTargets::BuildSummary() {
    const VertexField& field = m_labels.Field();
    const int* owned = field.OwnedVertexMarker().HostRead();
    const double* xyz = field.Coordinates().HostRead();
    const double* target = m_targets.HostRead();
    std::int64_t counts[NUM_TARGET_STATUSES + NUM_TARGET_REASONS] = {};
    double ratio = 0.0;
    for (int v = 0; v < field.NumVertices(); ++v) {
        if (owned[v] == 0) {
            continue;
        }
        ++counts[m_status[v]];
        ++counts[NUM_TARGET_STATUSES + m_reason[v]];
        if (Status(v) == TargetStatus::FIT) {
            double d2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                d2 += (target[3 * v + d] - xyz[3 * v + d]) * (target[3 * v + d] - xyz[3 * v + d]);
            }
            ratio = std::max(ratio, std::sqrt(d2) / m_h(v));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, counts, NUM_TARGET_STATUSES + NUM_TARGET_REASONS, MPI_INT64_T,
                  MPI_SUM, field.Comm());
    MPI_Allreduce(MPI_IN_PLACE, &ratio, 1, MPI_DOUBLE, MPI_MAX, field.Comm());
    for (int k = 0; k < NUM_TARGET_STATUSES; ++k) {
        m_summary.status[static_cast<std::size_t>(k)] = counts[k];
    }
    for (int k = 0; k < NUM_TARGET_REASONS; ++k) {
        m_summary.reason[static_cast<std::size_t>(k)] = counts[NUM_TARGET_STATUSES + k];
    }
    m_summary.max_move_ratio = ratio;
}

} // namespace exaconstit::microstructure
