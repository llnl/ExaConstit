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

/** @brief A query that passed the projection checks and waits for the foreign-seed check. */
struct Candidate {
    int query;
    int first_seed; ///< offset of the node's seeds in the exclusion CSR
    int num_seeds;
};

} // namespace

/** @brief Type-erased evaluator: one instantiation per metric functor. */
class NodeTargets::Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual void Evaluate(const std::vector<NodeLabelQuery>& queries,
                          std::vector<NodeTargetResult>& out) const = 0;
};

namespace {

/**
 * @brief The classification rules only (NodeTargetOptions::analytic = false): NONE,
 * OVER_CONSTRAINED, HELD (CURVED_FEATURE), and HELD (NON_ANALYTIC) for every other grain
 * boundary node.
 */
class NonAnalyticEvaluator final : public NodeTargets::Evaluator {
public:
    NonAnalyticEvaluator(const NodeLabels& labels, const SurfacePatches& patches)
        : m_labels(labels), m_patches(patches) {}

    void Evaluate(const std::vector<NodeLabelQuery>& queries,
                  std::vector<NodeTargetResult>& out) const override {
        const int* surf_off = m_labels.SurfaceOffsets().HostRead();
        const int* surf_planar = m_labels.SurfacePlanar().HostRead();
        out.assign(queries.size(), NodeTargetResult{});
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const NodeLabelQuery& q = queries[i];
            const int v = q.node;
            NodeTargetResult& r = out[i];
            const double* x = (q.position != nullptr)
                                  ? q.position
                                  : m_labels.Field().Coordinates().HostRead() + 3 * v;
            std::copy(x, x + 3, r.target);
            bool curved = false;
            for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
                curved = curved || surf_planar[s] == 0;
            }
            if (q.num_labels <= 1) {
                r.status = TargetStatus::NONE;
            } else if (q.num_labels - 1 + m_labels.BoundaryConstraints(v) > 3) {
                r.status = TargetStatus::OVER_CONSTRAINED;
            } else if (curved &&
                       (m_labels.BoundarySurfaces(v) != 1 || m_patches.NumTriangles(v) == 0)) {
                r.status = TargetStatus::HELD;
                r.reason = TargetReason::CURVED_FEATURE;
            } else {
                r.status = TargetStatus::HELD;
                r.reason = TargetReason::NON_ANALYTIC;
            }
        }
    }

private:
    const NodeLabels& m_labels;
    const SurfacePatches& m_patches;
};

/** @brief True when a node with this result may take its smoothed target instead. */
bool SmoothedFallback(const NodeTargetResult& r) {
    return r.status == TargetStatus::FAKE_JUNCTION ||
           (r.status == TargetStatus::HELD && r.reason == TargetReason::NON_ANALYTIC);
}

/**
 * @brief The rules of the file comment for a batch of queries: classification, equations,
 * projection (with the patch alternation on curved surfaces), axis coordinates, residual and
 * distance checks, then one batched foreign-seed search. Holds the seed index (built once).
 */
template <typename Metric>
class EvaluatorImpl final : public NodeTargets::Evaluator {
public:
    EvaluatorImpl(const NodeLabels& labels, const SurfacePatches& patches, const mfem::Vector& h,
                  const SeedSet& seeds, const MetricSetup& setup,
                  const std::vector<int>& grain_seed, const NodeTargetOptions& options)
        : m_labels(labels), m_patches(patches), m_h(h), m_grain_seed(grain_seed),
          m_options(options) {
        m_index.Build(seeds, setup, options.num_bins,
                      axom::execution_space<axom::SEQ_EXEC>::allocatorID());
    }

    void Evaluate(const std::vector<NodeLabelQuery>& queries,
                  std::vector<NodeTargetResult>& out) const override;

private:
    /** @brief Steps before the foreign check for one query; true = candidate (FIT so far). */
    bool Project(const NodeLabelQuery& q, int* node_seeds, NodeTargetResult& r) const;

    const NodeLabels& m_labels;
    const SurfacePatches& m_patches;
    const mfem::Vector& m_h;
    const std::vector<int>& m_grain_seed;
    NodeTargetOptions m_options;
    SeedIndex<axom::SEQ_EXEC, Metric> m_index;
};

template <typename Metric>
bool EvaluatorImpl<Metric>::Project(const NodeLabelQuery& q, int* node_seeds,
                                    NodeTargetResult& r) const {
    const int v = q.node;
    const int n = q.num_labels;
    const double* x0 = m_labels.Field().Coordinates().HostRead() + 3 * v;
    const double* x = (q.position != nullptr) ? q.position : x0;
    std::copy(x, x + 3, r.target);
    const auto set = [&](TargetStatus s, TargetReason why) {
        r.status = s;
        r.reason = why;
        return false;
    };
    if (n <= 1) {
        return set(TargetStatus::NONE, TargetReason::NONE);
    }
    if (n - 1 + m_labels.BoundaryConstraints(v) > 3) {
        return set(TargetStatus::OVER_CONSTRAINED, TargetReason::NONE);
    }
    for (int k = 0; k < n; ++k) {
        node_seeds[k] = m_grain_seed[static_cast<std::size_t>(q.labels[k] - 1)];
        if (node_seeds[k] < 0) {
            return set(TargetStatus::HELD, TargetReason::NON_ANALYTIC);
        }
    }
    const int* surf_off = m_labels.SurfaceOffsets().HostRead();
    const double* surf_normal = m_labels.SurfaceNormals().HostRead();
    const int* surf_planar = m_labels.SurfacePlanar().HostRead();
    bool curved = false;
    for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
        curved = curved || surf_planar[s] == 0;
    }
    if (curved && (m_labels.BoundarySurfaces(v) != 1 || m_patches.NumTriangles(v) == 0)) {
        return set(TargetStatus::HELD, TargetReason::CURVED_FEATURE);
    }
    for (int a = 0; a < n; ++a) {
        for (int b = a + 1; b < n; ++b) {
            if (node_seeds[a] == node_seeds[b]) {
                return set(TargetStatus::FAKE_JUNCTION, TargetReason::SAME_SEED);
            }
        }
    }

    // Equations: grain boundaries, then the boundary planes through x
    ManifoldRow rows[MANIFOLD_MAX_ROWS];
    int num_rows = 0;
    for (int k = 1; k < n; ++k) {
        rows[num_rows].kind = ManifoldRow::GRAIN_BOUNDARY;
        rows[num_rows].seed_a = node_seeds[0];
        rows[num_rows].seed_b = node_seeds[k];
        ++num_rows;
    }
    for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
        ManifoldRow& row = rows[num_rows++];
        row.kind = ManifoldRow::PLANE;
        const double* nrm = surf_normal + 3 * s;
        std::copy(nrm, nrm + 3, row.normal);
        row.offset = nrm[0] * x0[0] + nrm[1] * x0[1] + nrm[2] * x0[2];
    }

    const Metric& f = m_index.GetMetric();
    const double h = m_h(v);
    const double tol = m_options.tolerance * h;
    double y[3];
    ProjectionResult result = ProjectionResult::NOT_CONVERGED;
    if (!curved) {
        result = ProjectOntoManifold(f, rows, num_rows, x, tol, m_options.max_iterations, y);
    } else {
        // Curved surface (the last row): alternate between the closest point p of the current
        // iterate on the patch, the plane through p (the triangle's plane, or the plane normal to
        // y - p when p is on an edge or corner of the patch and y is off the surface), and the
        // projection of x on the grain boundaries and that plane
        std::copy(x, x + 3, y);
        ManifoldRow& plane = rows[num_rows - 1];
        for (int it = 0; it < m_options.max_iterations; ++it) {
            double p[3];
            double nrm[3];
            bool interior = false;
            const double dist = m_patches.ClosestPoint(v, y, p, nrm, interior);
            if (!interior && dist > tol) {
                for (int d = 0; d < 3; ++d) {
                    nrm[d] = (y[d] - p[d]) / dist;
                }
            }
            std::copy(nrm, nrm + 3, plane.normal);
            plane.offset = nrm[0] * p[0] + nrm[1] * p[1] + nrm[2] * p[2];
            double y_new[3];
            result =
                ProjectOntoManifold(f, rows, num_rows, x, tol, m_options.max_iterations, y_new);
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
            if (m_patches.ClosestPoint(v, y, p, nrm, interior) > tol) {
                result = ProjectionResult::NOT_CONVERGED;
            }
        }
    }
    if (result == ProjectionResult::SINGULAR) {
        return set(TargetStatus::FAKE_JUNCTION, TargetReason::SINGULAR);
    }
    if (result == ProjectionResult::NOT_CONVERGED) {
        return set(TargetStatus::FAKE_JUNCTION, TargetReason::NOT_CONVERGED);
    }
    // Axis planes keep the coordinate bit for bit
    for (int k = n - 1; k < num_rows; ++k) {
        const int axis = AxisOf(rows[k].normal);
        if (axis >= 0) {
            y[axis] = x0[axis];
        }
    }
    for (int k = 0; k < num_rows; ++k) {
        double g[3];
        const double phi = EvalManifoldRow(f, rows[k], y, g);
        const double glen = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (!(std::fabs(phi) <= tol * glen)) {
            return set(TargetStatus::FAKE_JUNCTION, TargetReason::RESIDUAL);
        }
    }
    const double move = std::sqrt((y[0] - x[0]) * (y[0] - x[0]) + (y[1] - x[1]) * (y[1] - x[1]) +
                                  (y[2] - x[2]) * (y[2] - x[2]));
    // The move limit is in element sizes (h times the order of the Nodes)
    if (move > m_options.max_move * h * m_labels.Field().LatticeOrder()) {
        return set(TargetStatus::FAKE_JUNCTION, TargetReason::TOO_FAR);
    }
    std::copy(y, y + 3, r.target);
    r.status = TargetStatus::FIT;
    r.reason = TargetReason::NONE;
    return true;
}

template <typename Metric>
void EvaluatorImpl<Metric>::Evaluate(const std::vector<NodeLabelQuery>& queries,
                                     std::vector<NodeTargetResult>& out) const {
    const int nq = static_cast<int>(queries.size());
    out.assign(queries.size(), NodeTargetResult{});
    std::vector<Candidate> candidates;
    std::vector<int> own_seeds;
    for (int i = 0; i < nq; ++i) {
        int node_seeds[NODE_QUERY_LABELS];
        if (Project(queries[static_cast<std::size_t>(i)], node_seeds,
                    out[static_cast<std::size_t>(i)])) {
            const int n = queries[static_cast<std::size_t>(i)].num_labels;
            candidates.push_back({i, static_cast<int>(own_seeds.size()), n});
            own_seeds.insert(own_seeds.end(), node_seeds, node_seeds + n);
        }
    }

    // Best seed outside each candidate's own seeds, at its target
    const int nc = static_cast<int>(candidates.size());
    if (nc == 0) {
        return;
    }
    mfem::Vector qx(nc), qy(nc), qz(nc);
    mfem::Array<int> ex_off(nc + 1);
    mfem::Array<int> ex_ids(static_cast<int>(own_seeds.size()));
    ex_off[0] = 0;
    for (int c = 0; c < nc; ++c) {
        const auto query = static_cast<std::size_t>(candidates[static_cast<std::size_t>(c)].query);
        const double* y = out[query].target;
        qx(c) = y[0];
        qy(c) = y[1];
        qz(c) = y[2];
        ex_off[c + 1] = ex_off[c] + candidates[static_cast<std::size_t>(c)].num_seeds;
    }
    std::copy(own_seeds.begin(), own_seeds.end(), ex_ids.begin());
    mfem::Array<BestPair> best;
    m_index.FindBestExcluding(qx, qy, qz, ex_off, ex_ids, best);
    const BestPair* best_host = best.HostRead();
    const Metric& f = m_index.GetMetric();
    for (int c = 0; c < nc; ++c) {
        const Candidate& cand = candidates[static_cast<std::size_t>(c)];
        NodeTargetResult& r = out[static_cast<std::size_t>(cand.query)];
        const int v = queries[static_cast<std::size_t>(cand.query)].node;
        const double* y = r.target;
        double largest = -HUGE_VAL;
        double grad_max = 0.0;
        for (int k = 0; k < cand.num_seeds; ++k) {
            const int s = own_seeds[static_cast<std::size_t>(cand.first_seed + k)];
            largest = std::max(largest, f.Eval(s, y[0], y[1], y[2]));
            double g[3];
            f.Grad(s, y[0], y[1], y[2], g);
            grad_max = std::max(grad_max, std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]));
        }
        const double delta = m_options.tolerance * m_h(v) * grad_max;
        if (best_host[c].id1 >= 0 && best_host[c].f1 < largest - delta) {
            const NodeLabelQuery& q = queries[static_cast<std::size_t>(cand.query)];
            const double* x = (q.position != nullptr)
                                  ? q.position
                                  : m_labels.Field().Coordinates().HostRead() + 3 * v;
            std::copy(x, x + 3, r.target);
            r.status = TargetStatus::FAKE_JUNCTION;
            r.reason = TargetReason::FOREIGN_SEED;
        }
    }
}

} // namespace

/*
 * Constructor
 *  1. Collective input checks.
 *  2. Mesh size h per node (ComputeMeshSize).
 *  3. Patches of the nodes on exactly one curved surface (SurfacePatches, collective).
 *  4. The evaluator of the metric (seed index; classification only without analytic targets),
 *     the smoothed staircase when asked for, then Evaluate on every node's own label set.
 *  5. Summary.
 */
NodeTargets::NodeTargets(const NodeLabels& labels, MetricType metric, const SeedSet& seeds,
                         const MetricSetup& setup, const std::vector<int>& grain_seed,
                         const NodeTargetOptions& options)
    : m_labels(labels), m_options(options) {
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
    ThrowIfAnyRank(comm, options.analytic && num_seeds < 1, "NodeTargets: the seed set is empty");
    ThrowIfAnyRank(comm, options.smoothed && (options.smoothing.steps < 0 ||
                                              !(options.smoothing.max_move > 0.0)),
                   "NodeTargets: smoothing steps must be >= 0 and max_move > 0");
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

    // Step 3 (the selection depends only on the surfaces, identical on every copy)
    const int* surf_off = labels.SurfaceOffsets().HostRead();
    const int* surf_planar = labels.SurfacePlanar().HostRead();
    std::vector<char> on_curved(static_cast<std::size_t>(nv), 0);
    for (int v = 0; v < nv; ++v) {
        on_curved[static_cast<std::size_t>(v)] =
            (labels.BoundarySurfaces(v) == 1 && surf_planar[surf_off[v]] == 0) ? 1 : 0;
    }
    m_patches = std::make_unique<SurfacePatches>(field, on_curved, &labels.SurfaceFaces());

    // Step 4
    if (options.analytic) {
        DispatchMetric(metric, [&](auto metric_tag) {
            using Metric = decltype(metric_tag);
            m_evaluator = std::make_unique<EvaluatorImpl<Metric>>(labels, *m_patches, m_h, seeds,
                                                                  setup, grain_seed, options);
        });
    } else {
        m_evaluator = std::make_unique<NonAnalyticEvaluator>(labels, *m_patches);
    }
    // Smoothed staircase (collective) before Evaluate, which falls back to it
    m_smoothed.assign(static_cast<std::size_t>(nv), 0);
    if (options.smoothed) {
        SmoothGrainBoundaries(labels, *m_patches, m_h, options.smoothing, m_smoothed_target,
                              m_smoothed);
    }
    const int* off = labels.LabelOffsets().HostRead();
    std::vector<NodeLabelQuery> queries(static_cast<std::size_t>(nv));
    for (int v = 0; v < nv; ++v) {
        NodeLabelQuery& q = queries[static_cast<std::size_t>(v)];
        q.node = v;
        q.num_labels = labels.NumLabels(v);
        for (int k = 0; k < std::min(q.num_labels, NODE_QUERY_LABELS); ++k) {
            q.labels[k] = items[off[v] + k];
        }
    }
    const std::vector<NodeTargetResult> results = Evaluate(queries);
    m_targets.SetSize(3 * nv);
    m_status.SetSize(nv);
    m_reason.SetSize(nv);
    double* target = m_targets.HostWrite();
    for (int v = 0; v < nv; ++v) {
        const NodeTargetResult& r = results[static_cast<std::size_t>(v)];
        std::copy(r.target, r.target + 3, target + 3 * v);
        m_status[v] = static_cast<int>(r.status);
        m_reason[v] = static_cast<int>(r.reason);
    }
    if (options.smoothed) {
        // Smoothed(v): FIT with the smoothed target (analytic FIT nodes keep theirs)
        const double* st = m_smoothed_target.HostRead();
        for (int v = 0; v < nv; ++v) {
            const auto vs = static_cast<std::size_t>(v);
            m_smoothed[vs] = (m_smoothed[vs] != 0 && Status(v) == TargetStatus::FIT &&
                              std::equal(st + 3 * v, st + 3 * v + 3, target + 3 * v))
                                 ? 1
                                 : 0;
        }
    }

    // Step 5
    BuildSummary();
}

NodeTargets::~NodeTargets() = default;

std::vector<NodeTargetResult> NodeTargets::Evaluate(
    const std::vector<NodeLabelQuery>& queries) const {
    std::vector<NodeTargetResult> out;
    m_evaluator->Evaluate(queries, out);
    if (!m_options.smoothed) {
        return out;
    }
    // Smoothed fallback, for the node's own label set only
    const int* off = m_labels.LabelOffsets().HostRead();
    const int* items = m_labels.Labels().HostRead();
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const NodeLabelQuery& q = queries[i];
        const int v = q.node;
        if (m_smoothed[static_cast<std::size_t>(v)] == 0 || !SmoothedFallback(out[i]) ||
            q.num_labels != m_labels.NumLabels(v) ||
            !std::equal(items + off[v], items + off[v] + std::min(q.num_labels, NODE_QUERY_LABELS),
                        q.labels)) {
            continue;
        }
        out[i].status = TargetStatus::FIT;
        out[i].reason = TargetReason::NONE;
        std::copy(m_smoothed_target.HostRead() + 3 * v, m_smoothed_target.HostRead() + 3 * v + 3,
                  out[i].target);
    }
    return out;
}

/*
 * ComputeMeshSize: per element Mesh::GetElementSize(e, 0) / p (mesh.cpp: |det J|^(1/3) at the
 * element center, with J relative to the regular reference element, Geometries.JacToPerfJac: the
 * edge length of the regular cube or tetrahedron of the same volume), the mean over all elements
 * of each node: the local sizes go to the other copies, and every copy sorts all of them and
 * averages, so every copy has the same h. (The mean rather than the smallest: on an unstructured
 * mesh one sliver next to a node would make its h, and so its spring and move limit, tiny.)
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
    {
        // Mean over all elements of the node (every element belongs to one rank): the local
        // sizes go to the other copies; each copy sorts all sizes and averages them
        std::vector<int> send_off(static_cast<std::size_t>(nv) + 1, 0);
        std::vector<double> send;
        for (int v = 0; v < nv; ++v) {
            for (int k = ve_off[v]; k < ve_off[v + 1]; ++k) {
                send.push_back(size[static_cast<std::size_t>(ve[k])]);
            }
            send_off[static_cast<std::size_t>(v) + 1] = static_cast<int>(send.size());
        }
        std::vector<int> recv_off;
        std::vector<double> recv;
        field.ExchangeSharedLists(send_off, send, recv_off, recv);
        for (int v = 0; v < nv; ++v) {
            const auto vs = static_cast<std::size_t>(v);
            std::vector<double> all(send.begin() + send_off[vs], send.begin() + send_off[vs + 1]);
            all.insert(all.end(), recv.begin() + recv_off[vs], recv.begin() + recv_off[vs + 1]);
            std::sort(all.begin(), all.end());
            double sum = 0.0;
            for (const double x : all) {
                sum += x;
            }
            m_h(v) = sum / static_cast<double>(all.size());
        }
    }
}

/*
 * BuildSummary: counts over the owned nodes (int64 sums) and the largest move ratio (maximum).
 */
void NodeTargets::BuildSummary() {
    const VertexField& field = m_labels.Field();
    const int* owned = field.OwnedVertexMarker().HostRead();
    const double* xyz = field.Coordinates().HostRead();
    const double* target = m_targets.HostRead();
    std::int64_t counts[NUM_TARGET_STATUSES + NUM_TARGET_REASONS + 1] = {};
    double ratio = 0.0;
    for (int v = 0; v < field.NumVertices(); ++v) {
        if (owned[v] == 0) {
            continue;
        }
        ++counts[m_status[v]];
        ++counts[NUM_TARGET_STATUSES + m_reason[v]];
        counts[NUM_TARGET_STATUSES + NUM_TARGET_REASONS] += Smoothed(v) ? 1 : 0;
        if (Status(v) == TargetStatus::FIT) {
            double d2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                d2 += (target[3 * v + d] - xyz[3 * v + d]) * (target[3 * v + d] - xyz[3 * v + d]);
            }
            ratio = std::max(ratio, std::sqrt(d2) / m_h(v));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, counts, NUM_TARGET_STATUSES + NUM_TARGET_REASONS + 1,
                  MPI_INT64_T, MPI_SUM, field.Comm());
    MPI_Allreduce(MPI_IN_PLACE, &ratio, 1, MPI_DOUBLE, MPI_MAX, field.Comm());
    for (int k = 0; k < NUM_TARGET_STATUSES; ++k) {
        m_summary.status[static_cast<std::size_t>(k)] = counts[k];
    }
    for (int k = 0; k < NUM_TARGET_REASONS; ++k) {
        m_summary.reason[static_cast<std::size_t>(k)] = counts[NUM_TARGET_STATUSES + k];
    }
    m_summary.max_move_ratio = ratio;
    m_summary.smoothed = counts[NUM_TARGET_STATUSES + NUM_TARGET_REASONS];
}

} // namespace exaconstit::microstructure
