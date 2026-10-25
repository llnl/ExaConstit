#pragma once

#include "microstructure/conformal/node_labels.hpp"
#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <array>
#include <memory>
#include <cstdint>
#include <vector>

/**
 * @file node_targets.hpp
 * @brief Stage 2b of conformal fitting: for every node on a grain boundary, the position on the
 * analytic grain boundary it should move to (its *target*), and a check that the target makes
 * sense.
 *
 * **Input.** The node label sets (node_labels.hpp) and a description of the analytic grain
 * boundaries: the metric, the seeds the labeling used (after Lloyd moves, with the final
 * weights; GenerationReport::seeds) and, per grain, the seed it grew from
 * (GenerationReport::grain_seed; -1 for a grain whose boundary is not a function of the seeds:
 * voxel-map and passthrough grains, or every grain of a geodesic labeling, whose grains follow
 * paths inside the material rather than the metric).
 *
 * **Target of a node** with labels l_0 < l_1 < ... (seeds s(l)):
 *   - the equations f_{s(l_0)} = f_{s(l_k)}, k >= 1 (n - 1 grain constraints), and for every flat
 *     boundary surface at the node (NodeLabels::SurfacePlanar) the plane n . y = n . x;
 *   - the target is the closest point of x on all of them (manifold_projector.hpp);
 *   - on a boundary plane whose normal is a coordinate axis (the faces, edges and corners of a box)
 *     the target keeps that coordinate of x bit for bit. Box boundary conditions find their nodes
 *     with an absolute 1e-12 test on that coordinate, so it must not move by a rounding error.
 *   - a node on one curved boundary surface (not flat at the node) must stay on the mesh's own
 *     boundary faces around it (its patch, boundary_projector.hpp). The plane row is then the
 *     plane of the patch triangle closest to the current iterate, and the projection is repeated
 *     from x until the iterate stops moving and lies on the patch (within `tolerance * h`). On a
 *     faceted surface this ends after a few changes of triangle. Where the closest point is on an
 *     edge or corner of the patch and the iterate is off the surface, the plane normal to the
 *     offset is used instead, which pulls the iterate back onto the edge.
 *
 * **Validation** (a failure marks a *fake junction*: the mesh shows grains meeting at the node,
 * but their analytic boundaries do not meet near it; label repair, Stage 2c, handles these):
 *   1. the projection converged and every equation holds to `tolerance * h` in distance;
 *   2. the node moves at most `max_move * h`;
 *   3. no other seed is closer than the node's own seeds at the target: with F = the largest
 *      f_{s(l)}(y) over the node's seeds, every other seed k has f_k(y) >= F - delta (delta =
 *      `tolerance * h` times the largest |grad f_{s(l)}(y)|). Otherwise the target lies inside
 *      another grain's cell (the seed search SeedIndex::FindBestExcluding, exact).
 * h is the node's mesh size: the smallest element size / p over the elements that contain it
 * (element size = the edge length of the regular cube or tetrahedron with the element's volume,
 * mfem::Mesh::GetElementSize(e, 0); p = order of the Nodes, 1 without Nodes), the same on every
 * copy (minimum over the copies).
 *
 * Nodes on exactly one curved surface get their triangle patch whatever their labels, so
 * Evaluate can also answer for label sets the node does not have yet.
 *
 * **Status of every node** (TargetStatus; the target of every node that is not FIT is x itself):
 *   - NONE: one label (inside a grain, or on the outer boundary only);
 *   - FIT: a valid target;
 *   - OVER_CONSTRAINED: more than 3 constraints (NodeLabels::OverConstrained);
 *   - HELD: a label without analytic boundary (grain_seed = -1), or a curved boundary surface that
 *     meets another boundary surface at the node (a feature edge or corner of a curved boundary
 *     is held fixed);
 *   - FAKE_JUNCTION: validation failed, or two labels grew from the same seed (TargetReason says
 *     which).
 *
 * **Determinism.** A target depends only on the node's coordinates (identical on every copy), its
 * label set and surfaces (identical on every copy) and the replicated seeds, so all copies compute
 * the same bits and the result does not depend on the partition. Only h and the summary need
 * communication.
 *
 * **Where it runs.** Host (Stage 2 is CPU-only); the seed search uses axom::SEQ_EXEC.
 */

namespace exaconstit::microstructure {

/** @brief Settings of NodeTargets. */
struct NodeTargetOptions {
    /** @brief Largest allowed move, in units of the node's mesh size h. */
    double max_move = 1.0;
    /** @brief Projection step and residual tolerance, in units of h. */
    double tolerance = 1e-10;
    /** @brief Gauss-Newton steps allowed per node. */
    int max_iterations = 25;
    /** @brief Parameter bins of the seed search (performance only). */
    int num_bins = 8;
};

/** @brief What happens to a node (see the file comment). */
enum class TargetStatus : int {
    NONE = 0,
    FIT = 1,
    OVER_CONSTRAINED = 2,
    HELD = 3,
    FAKE_JUNCTION = 4
};
inline constexpr int NUM_TARGET_STATUSES = 5;

/** @brief Why a node is HELD or a FAKE_JUNCTION (NONE otherwise). */
enum class TargetReason : int {
    NONE = 0,
    NON_ANALYTIC = 1,    ///< HELD: a label with grain_seed = -1
    CURVED_FEATURE = 2,  ///< HELD: a curved boundary surface meets another surface at the node
    SAME_SEED = 3,       ///< two labels grew from the same seed
    SINGULAR = 4,        ///< dependent equations (e.g. a grain boundary parallel to a box face)
    NOT_CONVERGED = 5,   ///< the projection did not converge
    RESIDUAL = 6,        ///< an equation does not hold at the target
    TOO_FAR = 7,         ///< the target is more than max_move * h away
    FOREIGN_SEED = 8     ///< another seed is closer at the target
};
inline constexpr int NUM_TARGET_REASONS = 9;

/** @brief Global counts (each node once, on its owner). */
struct NodeTargetSummary {
    std::array<std::int64_t, NUM_TARGET_STATUSES> status{};
    std::array<std::int64_t, NUM_TARGET_REASONS> reason{};
    /** @brief Largest |target - x| / h over the FIT nodes. */
    double max_move_ratio = 0.0;
};

/**
 * @brief A label set to evaluate at a node: the node's own set (the constructor) or a
 * hypothetical one (label repair previews a flip, label_repair.hpp). Labels ascending and
 * distinct; only the first NODE_QUERY_LABELS are stored (a larger set is over-constrained
 * anyway, and num_labels keeps its size).
 */
inline constexpr int NODE_QUERY_LABELS = 4;
struct NodeLabelQuery {
    int node = -1;
    int num_labels = 0;
    int labels[NODE_QUERY_LABELS] = {};
    /**
     * @brief Where the node is (3 doubles), or nullptr = its original position
     * (VertexField::Coordinates). The fitter re-projects from the nodes' current positions so
     * that they can slide along the grain boundaries. The boundary planes still pass through
     * the original position (a free surface does not move).
     */
    const double* position = nullptr;
};

/** @brief The evaluation of a NodeLabelQuery. */
struct NodeTargetResult {
    TargetStatus status = TargetStatus::NONE;
    TargetReason reason = TargetReason::NONE;
    double target[3] = {0.0, 0.0, 0.0}; ///< the target when FIT, else the node's position
};

class SurfacePatches;

/** @brief Targets, statuses and mesh sizes of every node (see the file comment). */
class NodeTargets {
public:
    /**
     * @brief Computes the targets and validates them.
     *
     * **Collective** over the mesh communicator of `labels`.
     * @param labels     node label sets of the mesh (kept by reference: must outlive this).
     * @param metric     the metric of the labeling.
     * @param seeds      the seeds the labeling used (positions, weights, per-seed tensors).
     * @param setup      the global Quadric tensor / per-seed flag (as for SeedIndex::Build).
     * @param grain_seed grain g (1-based) grew from seed grain_seed[g - 1] (0-based), -1 = not
     *                   analytic. Must cover every label of `labels`.
     * @param options    tolerances.
     * @throws std::invalid_argument on every rank for a label outside grain_seed, a seed index out
     *         of range, an empty seed set or invalid options; SeedIndex::Build errors.
     */
    NodeTargets(const NodeLabels& labels, MetricType metric, const SeedSet& seeds,
                const MetricSetup& setup, const std::vector<int>& grain_seed,
                const NodeTargetOptions& options = {});
    ~NodeTargets();
    NodeTargets(const NodeTargets&) = delete;
    NodeTargets& operator=(const NodeTargets&) = delete;

    /**
     * @brief Status, reason and target of each query, with exactly the rules the constructor
     * applies to the nodes' own label sets (same mesh size, surfaces and patches). Every label
     * must be a grain of grain_seed. Not collective (local nodes only); host.
     */
    std::vector<NodeTargetResult> Evaluate(const std::vector<NodeLabelQuery>& queries) const;

    /** @brief Target positions, 3 per node (node v at 3 v); x itself unless FIT. */
    const mfem::Vector& Targets() const {
        return m_targets;
    }
    /** @brief Status of node v. */
    TargetStatus Status(int v) const {
        return static_cast<TargetStatus>(m_status[v]);
    }
    /** @brief Reason of node v (HELD and FAKE_JUNCTION; NONE otherwise). */
    TargetReason Reason(int v) const {
        return static_cast<TargetReason>(m_reason[v]);
    }
    /** @brief Mesh size h of node v (see the file comment). */
    double MeshSize(int v) const {
        return m_h(v);
    }
    /** @brief Global counts (same on every rank). */
    const NodeTargetSummary& Summary() const {
        return m_summary;
    }
    /** @brief The node labels the targets belong to. */
    const NodeLabels& Labels() const {
        return m_labels;
    }

    /** @brief Per-metric evaluator (seed search and projection), defined in node_targets.cpp. */
    class Evaluator;

private:
    void ComputeMeshSize();
    void BuildSummary();

    const NodeLabels& m_labels;
    std::unique_ptr<SurfacePatches> m_patches;
    std::unique_ptr<Evaluator> m_evaluator;
    mfem::Vector m_targets;
    mfem::Array<int> m_status;
    mfem::Array<int> m_reason;
    mfem::Vector m_h;
    NodeTargetSummary m_summary;
};

} // namespace exaconstit::microstructure
