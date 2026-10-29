/*
 * Targets of the grain boundary nodes on the exact grain boundaries, their checks, and the
 * fallbacks (smoothed staircase, holds) (node_targets.hpp).
 *
 * The seed search and the projection are compiled once per metric (Voronoi, Laguerre, ...): the
 * class NodeTargets::Evaluator is an interface with one implementation per metric type
 * (EvaluatorImpl<Metric>), chosen at run time from MetricType, so NodeTargets itself does not
 * depend on the metric type.
 */
#include "microstructure/conformal/node_targets.hpp"

#include "microstructure/conformal/boundary_projector.hpp"
#include "microstructure/conformal/manifold_projector.hpp"
#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/util/collective_errors.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "axom/core.hpp"

namespace exaconstit::microstructure {

namespace {

/** @brief The axis d when n points exactly along +-e_d (two components exactly zero, as for the
 * faces of an axis-aligned box), else -1. Host. */
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
    int query;      ///< index of the query
    int first_seed; ///< where the node's own seeds start in the list of seeds to exclude
    int num_seeds;  ///< number of the node's own seeds
};

} // namespace

/** @brief Evaluates queries for one metric (see the file comment): EvaluatorImpl<Metric> with
 * exact boundaries, NonAnalyticEvaluator without. */
class NodeTargets::Evaluator {
public:
    /** @brief Virtual, so the implementations are destroyed through this interface. */
    virtual ~Evaluator() = default;
    /** @brief The rules of node_targets.hpp for every query, without the smoothed fallback
     * (NodeTargets::Evaluate adds it). out[i] answers queries[i]. Host, local. */
    virtual void Evaluate(const std::vector<NodeLabelQuery>& queries,
                          std::vector<NodeTargetResult>& out) const = 0;
    /** @brief NodeTargets::NearbySeeds; this default gives empty lists (no exact boundaries). */
    virtual void NearbySeeds(const std::vector<int>& nodes,
                             int max_seeds,
                             double reach,
                             std::vector<int>& offsets,
                             std::vector<int>& seeds) const {
        (void)max_seeds;
        (void)reach;
        offsets.assign(nodes.size() + 1, 0);
        seeds.clear();
    }
};

namespace {

/**
 * @brief The classification rules only (NodeTargetOptions::analytic = false): NONE,
 * OVER_CONSTRAINED, HELD (CURVED_FEATURE), and HELD (NON_ANALYTIC) for every other grain
 * boundary node.
 */
class NonAnalyticEvaluator final : public NodeTargets::Evaluator {
public:
    /** @brief Keeps references to the node labels and patches (both outlive the evaluator). */
    NonAnalyticEvaluator(const NodeLabels& labels, const SurfacePatches& patches)
        : m_labels(labels), m_patches(patches) {}

    /** @brief The classification of each query; every target is the node's own position. */
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
            } else if (curved && !CurvedPatchesReady(m_labels, m_patches, v)) {
                r.status = TargetStatus::HELD;
                r.reason = TargetReason::CURVED_FEATURE;
            } else {
                r.status = TargetStatus::HELD;
                r.reason = TargetReason::NON_ANALYTIC;
            }
        }
    }

private:
    const NodeLabels& m_labels;      ///< node labels and boundary surfaces
    const SurfacePatches& m_patches; ///< patches of the nodes on curved surfaces
};

/** @brief True when a node with this result may take its smoothed target instead: no valid
 * exact target (FAKE_JUNCTION) or no exact boundary at all (HELD for NON_ANALYTIC). The other
 * holds (thin grains, pinches, curved features) keep the node in place. */
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
    /** @brief Keeps references to the labels, patches and mesh sizes (they outlive the
     * evaluator), copies grain_seed and the options, and builds the seed index (SeedIndex, on the
     * host). */
    EvaluatorImpl(const NodeLabels& labels,
                  const SurfacePatches& patches,
                  const mfem::Vector& h,
                  const SeedSet& seeds,
                  const MetricSetup& setup,
                  const std::vector<int>& grain_seed,
                  const NodeTargetOptions& options)
        : m_labels(labels), m_patches(patches), m_h(h), m_grain_seed(grain_seed),
          m_options(options) {
        m_index.Build(
            seeds, setup, options.num_bins, axom::execution_space<axom::SEQ_EXEC>::allocatorID());
    }

    /** @brief See NodeTargets::Evaluator::Evaluate. */
    void Evaluate(const std::vector<NodeLabelQuery>& queries,
                  std::vector<NodeTargetResult>& out) const override;
    /** @brief See NodeTargets::NearbySeeds. */
    void NearbySeeds(const std::vector<int>& nodes,
                     int max_seeds,
                     double reach,
                     std::vector<int>& offsets,
                     std::vector<int>& seeds) const override;

private:
    /** @brief Every rule for one query except the foreign-seed check, which is batched over all
     * queries. Returns true when the query is FIT so far (a candidate); node_seeds receives the
     * seeds of its labels (NODE_QUERY_LABELS entries). */
    bool Project(const NodeLabelQuery& q, int* node_seeds, NodeTargetResult& r) const;

    const NodeLabels& m_labels;                ///< node labels and boundary surfaces
    const SurfacePatches& m_patches;           ///< patches of the nodes on curved surfaces
    const mfem::Vector& m_h;                   ///< mesh size h per node
    const std::vector<int> m_grain_seed;       ///< a copy (the caller's may be a temporary)
    NodeTargetOptions m_options;               ///< move limit, tolerances
    SeedIndex<axom::SEQ_EXEC, Metric> m_index; ///< exact closest-seed search
};

/*
 * EvaluatorImpl::Project: the rules of node_targets.hpp in order, stopping at the first that
 * decides the status: one label (NONE), too many constraints (OVER_CONSTRAINED), a label without
 * a seed (HELD), a curved surface without patch (HELD), two labels of one seed (FAKE_JUNCTION);
 * then the equations (grain boundaries with the first label's seed, the planes of the boundary
 * surfaces through the input position), the projection (with the patch alternation on curved
 * surfaces), the exact box coordinates, and the residual and move checks.
 */
template <typename Metric>
bool EvaluatorImpl<Metric>::Project(const NodeLabelQuery& q,
                                    int* node_seeds,
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
    if (!CurvedPatchesReady(m_labels, m_patches, v)) {
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
        // Curved surfaces (their rows): alternate between the closest point p of the current
        // iterate on each curved surface's patch, the plane through p (the triangle's plane, or
        // the plane normal to y - p when p is on an edge or corner of the patch and y is off the
        // surface), and the projection of x on the grain boundaries and those planes. On a
        // feature edge of a curved boundary (two surfaces) this converges to the point where the
        // grain boundary meets the edge.
        std::copy(x, x + 3, y);
        for (int it = 0; it < m_options.max_iterations; ++it) {
            double dist_max = 0.0;
            for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
                if (surf_planar[s] != 0) {
                    continue;
                }
                ManifoldRow& plane = rows[n - 1 + (s - surf_off[v])];
                double p[3];
                double nrm[3];
                bool interior = false;
                const double dist = m_patches.ClosestPoint(v, y, p, nrm, interior, s - surf_off[v]);
                if (!interior && dist > tol) {
                    for (int d = 0; d < 3; ++d) {
                        nrm[d] = (y[d] - p[d]) / dist;
                    }
                }
                std::copy(nrm, nrm + 3, plane.normal);
                plane.offset = nrm[0] * p[0] + nrm[1] * p[1] + nrm[2] * p[2];
                dist_max = std::max(dist_max, dist);
            }
            double y_new[3];
            result = ProjectOntoManifold(
                f, rows, num_rows, x, tol, m_options.max_iterations, y_new);
            if (result != ProjectionResult::CONVERGED) {
                break;
            }
            const double step2 = (y_new[0] - y[0]) * (y_new[0] - y[0]) +
                                 (y_new[1] - y[1]) * (y_new[1] - y[1]) +
                                 (y_new[2] - y[2]) * (y_new[2] - y[2]);
            std::copy(y_new, y_new + 3, y);
            result = ProjectionResult::NOT_CONVERGED;
            if (step2 <= tol * tol && dist_max <= tol) {
                result = ProjectionResult::CONVERGED;
                break;
            }
        }
        for (int s = surf_off[v]; s < surf_off[v + 1] && result == ProjectionResult::CONVERGED;
             ++s) {
            double p[3];
            double nrm[3];
            bool interior = false;
            if (surf_planar[s] == 0 &&
                m_patches.ClosestPoint(v, y, p, nrm, interior, s - surf_off[v]) > tol) {
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

/*
 * EvaluatorImpl::Evaluate: Project for every query, then one batched search for the closest seed
 * other than each candidate's own seeds at its target (SeedIndex::FindBestExcluding). A
 * candidate whose foreign seed is closer than its own seeds there (by more than the tolerance)
 * becomes a FAKE_JUNCTION (FOREIGN_SEED) and keeps its position.
 */
template <typename Metric>
void EvaluatorImpl<Metric>::Evaluate(const std::vector<NodeLabelQuery>& queries,
                                     std::vector<NodeTargetResult>& out) const {
    const int nq = static_cast<int>(queries.size());
    out.assign(queries.size(), NodeTargetResult{});
    std::vector<Candidate> candidates;
    std::vector<int> own_seeds;
    for (int i = 0; i < nq; ++i) {
        int node_seeds[NODE_QUERY_LABELS];
        if (Project(queries[static_cast<std::size_t>(i)],
                    node_seeds,
                    out[static_cast<std::size_t>(i)])) {
            const int n = queries[static_cast<std::size_t>(i)].num_labels;
            candidates.push_back({i, static_cast<int>(own_seeds.size()), n});
            own_seeds.insert(own_seeds.end(), node_seeds, node_seeds + n);
        }
    }

    // Best seed outside each candidate's own seeds, at its target. The seeds to exclude for
    // candidate c are ex_ids[ex_off[c]] .. ex_ids[ex_off[c + 1] - 1].
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

/*
 * NearbySeeds: max_seeds rounds of FindBestExcluding (each round excludes the seeds found so far),
 * then the estimate of the distance from x to the boundary of the best seed s_1 and seed s_k,
 * (f_k - f_1) / |grad f_k - grad f_1| at x (exact for the Voronoi and Laguerre metrics, whose f_k -
 * f_1 is linear; a first-order estimate otherwise), kept when it is at most `reach` element sizes.
 */
template <typename Metric>
void EvaluatorImpl<Metric>::NearbySeeds(const std::vector<int>& nodes,
                                        int max_seeds,
                                        double reach,
                                        std::vector<int>& offsets,
                                        std::vector<int>& seeds) const {
    const int nq = static_cast<int>(nodes.size());
    offsets.assign(nodes.size() + 1, 0);
    seeds.clear();
    if (nq == 0 || max_seeds < 1) {
        return;
    }
    const double* xyz = m_labels.Field().Coordinates().HostRead();
    mfem::Vector qx(nq), qy(nq), qz(nq);
    for (int i = 0; i < nq; ++i) {
        const double* x = xyz + 3 * nodes[static_cast<std::size_t>(i)];
        qx(i) = x[0];
        qy(i) = x[1];
        qz(i) = x[2];
    }
    // found[i * max_seeds + r]: seed of round r at query i (-1 = none)
    std::vector<int> found(static_cast<std::size_t>(nq) * static_cast<std::size_t>(max_seeds), -1);
    for (int r = 0; r < max_seeds; ++r) {
        mfem::Array<int> ex_off(nq + 1);
        std::vector<int> ex;
        ex_off[0] = 0;
        for (int i = 0; i < nq; ++i) {
            for (int k = 0; k < r; ++k) {
                const int s = found[static_cast<std::size_t>(i * max_seeds + k)];
                if (s >= 0) {
                    ex.push_back(s);
                }
            }
            ex_off[i + 1] = static_cast<int>(ex.size());
        }
        mfem::Array<int> ex_ids(std::max<int>(1, static_cast<int>(ex.size())));
        std::copy(ex.begin(), ex.end(), ex_ids.begin());
        mfem::Array<BestPair> best;
        m_index.FindBestExcluding(qx, qy, qz, ex_off, ex_ids, best);
        const BestPair* b = best.HostRead();
        for (int i = 0; i < nq; ++i) {
            found[static_cast<std::size_t>(i * max_seeds + r)] = b[i].id1;
        }
    }
    const Metric& f = m_index.GetMetric();
    const int p = m_labels.Field().LatticeOrder();
    for (int i = 0; i < nq; ++i) {
        const int v = nodes[static_cast<std::size_t>(i)];
        const double* x = xyz + 3 * v;
        const int s1 = found[static_cast<std::size_t>(i * max_seeds)];
        if (s1 >= 0) {
            seeds.push_back(s1);
            const double f1 = f.Eval(s1, x[0], x[1], x[2]);
            double g1[3];
            f.Grad(s1, x[0], x[1], x[2], g1);
            for (int r = 1; r < max_seeds; ++r) {
                const int sk = found[static_cast<std::size_t>(i * max_seeds + r)];
                if (sk < 0) {
                    break;
                }
                double gk[3];
                f.Grad(sk, x[0], x[1], x[2], gk);
                const double dg = std::sqrt((gk[0] - g1[0]) * (gk[0] - g1[0]) +
                                            (gk[1] - g1[1]) * (gk[1] - g1[1]) +
                                            (gk[2] - g1[2]) * (gk[2] - g1[2]));
                const double gap = f.Eval(sk, x[0], x[1], x[2]) - f1;
                if (gap <= reach * m_h(v) * p * dg) {
                    seeds.push_back(sk);
                }
            }
        }
        offsets[static_cast<std::size_t>(i) + 1] = static_cast<int>(seeds.size());
    }
}

} // namespace

/*
 * Constructor
 *  1. Collective input checks.
 *  2. Mesh size h per node (ComputeMeshSize).
 *  3. Patches of the nodes on curved boundary surfaces, with enough face rings for the largest
 *     move (MakeCurvedPatches, PatchRingsFor; collective).
 *  4. The evaluator of the metric (seed index; classification only without analytic targets),
 *     the smoothed staircase when asked for, then Evaluate on every node's own label set.
 *  5. Summary.
 */
NodeTargets::NodeTargets(const NodeLabels& labels,
                         MetricType metric,
                         const SeedSet& seeds,
                         const MetricSetup& setup,
                         const std::vector<int>& grain_seed,
                         const NodeTargetOptions& options)
    : m_labels(labels), m_options(options) {
    const VertexField& field = labels.Field();
    const MPI_Comm comm = field.Comm();
    const int nv = labels.NumNodes();
    const int num_seeds = static_cast<int>(seeds.Size());

    // Step 1: input checks (collective)
    ThrowIfAnyRank(comm,
                   !(options.max_move > 0.0) || !(options.tolerance > 0.0) ||
                       options.max_iterations < 1 || options.num_bins < 1,
                   "NodeTargets: max_move and tolerance must be > 0, max_iterations and num_bins "
                   ">= 1");
    ThrowIfAnyRank(comm, options.analytic && num_seeds < 1, "NodeTargets: the seed set is empty");
    ThrowIfAnyRank(comm,
                   options.smoothed &&
                       (options.smoothing.steps < 0 || !(options.smoothing.max_move > 0.0)),
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

    // Step 2: mesh size per node
    ComputeMeshSize();

    // Step 3: patches (the selection depends only on the surfaces, identical on every copy)
    const double reach = std::max(options.max_move,
                                  options.smoothed ? options.smoothing.max_move : 0.0);
    m_patch_rings = PatchRingsFor(reach, field.LatticeOrder());
    m_patches = MakeCurvedPatches(labels, m_patch_rings);

    // Step 4: the evaluator, the smoothed staircase, then every node's own label set; pinched
    // and thin-grain nodes are held afterwards, and Smoothed(v) is settled last
    if (options.analytic) {
        DispatchMetric(metric, [&](auto metric_tag) {
            using Metric = decltype(metric_tag);
            m_evaluator = std::make_unique<EvaluatorImpl<Metric>>(
                labels, *m_patches, m_h, seeds, setup, grain_seed, options);
        });
    } else {
        m_evaluator = std::make_unique<NonAnalyticEvaluator>(labels, *m_patches);
    }
    // Smoothed staircase (collective) before Evaluate, which falls back to it
    m_smoothed.assign(static_cast<std::size_t>(nv), 0);
    if (options.smoothed) {
        SmoothGrainBoundaries(
            labels, *m_patches, m_h, options.smoothing, m_smoothed_target, m_smoothed);
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
    // Pinches: held, or bad for label repair (over-constrained nodes stay over-constrained)
    {
        const double* xyz = field.Coordinates().HostRead();
        for (int v = 0; v < nv; ++v) {
            if (labels.Pinched(v) && labels.NumLabels(v) >= 2 &&
                Status(v) != TargetStatus::OVER_CONSTRAINED) {
                m_status[v] = static_cast<int>(options.hold_pinches ? TargetStatus::HELD
                                                                    : TargetStatus::FAKE_JUNCTION);
                m_reason[v] = static_cast<int>(TargetReason::PINCH);
                std::copy(xyz + 3 * v, xyz + 3 * v + 3, target + 3 * v);
            }
        }
    }
    // Under-resolved grains: interior nodes per grain (owned, one non-fixed label), summed
    if (options.hold_under_resolved) {
        int max_label = 0;
        for (int k = 0; k < labels.Labels().Size(); ++k) {
            max_label = std::max(max_label, items[k]);
        }
        MPI_Allreduce(MPI_IN_PLACE, &max_label, 1, MPI_INT, MPI_MAX, comm);
        std::vector<std::int64_t> interior(static_cast<std::size_t>(max_label) + 1, 0);
        std::vector<std::int64_t> present(static_cast<std::size_t>(max_label) + 1, 0);
        const int* owned = field.OwnedVertexMarker().HostRead();
        for (int v = 0; v < nv; ++v) {
            for (int k = off[v]; k < off[v + 1]; ++k) {
                present[static_cast<std::size_t>(items[k])] = 1;
            }
            if (owned[v] != 0 && labels.NumLabels(v) == 1) {
                ++interior[static_cast<std::size_t>(items[off[v]])];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, interior.data(), max_label + 1, MPI_INT64_T, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, present.data(), max_label + 1, MPI_INT64_T, MPI_MAX, comm);
        std::vector<char> thin(static_cast<std::size_t>(max_label) + 1, 0);
        for (int g = 1; g <= max_label; ++g) {
            const auto gs = static_cast<std::size_t>(g);
            thin[gs] = (present[gs] != 0 && interior[gs] == 0 && !labels.IsFixedLabel(g)) ? 1 : 0;
            m_summary.under_resolved_grains += thin[gs];
        }
        const double* xyz = field.Coordinates().HostRead();
        for (int v = 0; v < nv; ++v) {
            if (labels.NumLabels(v) < 2) {
                continue;
            }
            bool any = false;
            for (int k = off[v]; k < off[v + 1]; ++k) {
                any = any || thin[static_cast<std::size_t>(items[k])] != 0;
            }
            if (any) {
                m_status[v] = static_cast<int>(TargetStatus::HELD);
                m_reason[v] = static_cast<int>(TargetReason::UNDER_RESOLVED);
                std::copy(xyz + 3 * v, xyz + 3 * v + 3, target + 3 * v);
            }
        }
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

    // Step 5: summary
    BuildSummary();
}

/* Destructor: defined here, where Evaluator and SurfacePatches are complete types. */
NodeTargets::~NodeTargets() = default;

/* EvaluateAnalytic: the evaluator alone, without the smoothed fallback of Evaluate. */
std::vector<NodeTargetResult>
NodeTargets::EvaluateAnalytic(const std::vector<NodeLabelQuery>& queries) const {
    std::vector<NodeTargetResult> out;
    m_evaluator->Evaluate(queries, out);
    return out;
}

/* NearbySeeds: forwarded to the evaluator of the metric. */
void NodeTargets::NearbySeeds(const std::vector<int>& nodes,
                              int max_seeds,
                              double reach,
                              std::vector<int>& offsets,
                              std::vector<int>& seeds) const {
    m_evaluator->NearbySeeds(nodes, max_seeds, reach, offsets, seeds);
}

/*
 * Evaluate: the evaluator's answers, then the smoothed fallback: a query for the node's own label
 * set that has a smoothed target and no valid exact one (SmoothedFallback) becomes FIT with the
 * smoothed target. Queries for other label sets (previews of label repair) never get it, since
 * the smoothed staircase was computed for the current labels only.
 */
std::vector<NodeTargetResult>
NodeTargets::Evaluate(const std::vector<NodeLabelQuery>& queries) const {
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
            !std::equal(items + off[v],
                        items + off[v] + std::min(q.num_labels, NODE_QUERY_LABELS),
                        q.labels)) {
            continue;
        }
        out[i].status = TargetStatus::FIT;
        out[i].reason = TargetReason::NONE;
        std::copy(m_smoothed_target.HostRead() + 3 * v,
                  m_smoothed_target.HostRead() + 3 * v + 3,
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
    MPI_Allreduce(MPI_IN_PLACE,
                  counts,
                  NUM_TARGET_STATUSES + NUM_TARGET_REASONS + 1,
                  MPI_INT64_T,
                  MPI_SUM,
                  field.Comm());
    MPI_Allreduce(MPI_IN_PLACE, &ratio, 1, MPI_DOUBLE, MPI_MAX, field.Comm());
    for (int k = 0; k < NUM_TARGET_STATUSES; ++k) {
        m_summary.status[static_cast<std::size_t>(k)] = counts[k];
    }
    for (int k = 0; k < NUM_TARGET_REASONS; ++k) {
        m_summary.reason[static_cast<std::size_t>(k)] = counts[NUM_TARGET_STATUSES + k];
    }
    m_summary.max_move_ratio = ratio; // under_resolved_grains was set by the constructor
    m_summary.smoothed = counts[NUM_TARGET_STATUSES + NUM_TARGET_REASONS];
}

} // namespace exaconstit::microstructure
