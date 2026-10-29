#pragma once

#include "microstructure/conformal/node_labels.hpp"
#include "microstructure/conformal/smoothed_targets.hpp"
#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

/**
 * @file node_targets.hpp
 * @brief For every node on a grain boundary, the position on the exact (analytic) grain boundary
 * it should move to in the conformal fit (its *target*), and a check that the target makes sense.
 *
 * **Words used.** A *label* is the grain id of an element. f_s(y) is the metric value of seed s
 * at point y (metric_functors.hpp: the distance for Voronoi, the weighted or stretched distance
 * for the other metrics); a point belongs to the grain of the seed with the smallest f, so the
 * exact boundary between seeds a and b is the surface f_a = f_b.
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
 *     offset is used instead, which pulls the iterate back onto the edge. The patch has enough
 *     face rings to hold every target within the move limit (PatchRingsFor of the larger of
 *     max_move and the smoothing's move limit).
 *   - a node on a feature edge where a curved surface meets one other surface (the rim of a
 *     strut, where its wall meets a flat end) has a patch per surface and a plane row per
 *     surface; the rows of the curved surfaces are updated the same way, so the projection ends
 *     where the grain boundary meets the edge, and the node slides along the edge.
 *
 * **Checks of a target** (a failure marks a *fake junction*: the mesh shows grains meeting at the
 * node, but their exact boundaries do not meet near it; label repair, label_repair.hpp, handles
 * these):
 *   1. the projection converged and every equation holds to `tolerance * h` in distance;
 *   2. the node moves at most `max_move` element sizes (max_move * h * p, p the order of the
 *      Nodes: for curved meshes h is the spacing of the nodal points, a fraction of the element);
 *   3. no other seed is closer than the node's own seeds at the target: with F = the largest
 *      f_{s(l)}(y) over the node's seeds, every other seed k has f_k(y) >= F - delta (delta =
 *      `tolerance * h` times the largest |grad f_{s(l)}(y)|). Otherwise the target lies inside
 *      another grain's cell (the seed search SeedIndex::FindBestExcluding, exact).
 * h is the node's mesh size: the mean element size / p over the elements that contain it
 * (element size = the edge length of the regular cube or tetrahedron with the element's volume,
 * mfem::Mesh::GetElementSize(e, 0); p = order of the Nodes, 1 without Nodes), the same on every
 * copy (all sizes gathered and averaged in sorted order).
 *
 * Nodes on curved surfaces (one, or two on a feature edge) get their triangle patches whatever
 * their labels, so Evaluate can also answer for label sets the node does not have yet.
 *
 * **Non-analytic boundaries.** With NodeTargetOptions::analytic = false every grain boundary node
 * is HELD (NON_ANALYTIC) by the rules above. With NodeTargetOptions::smoothed the grain boundary
 * nodes that are HELD for a non-analytic label or are FAKE_JUNCTION get the position of the
 * smoothed staircase (smoothed_targets.hpp) as target and become FIT; Smoothed(v) tells which.
 * So a labeling without analytic boundaries (voxel map, geodesic) is fitted to its smoothed
 * staircase, and a labeling whose boundaries are analytic almost everywhere (Euclidean or
 * connected mode, with pieces moved by the cleanup or the connected relabeling) uses the
 * analytic boundary where it is valid and the smoothed one elsewhere. Evaluate gives a node its
 * smoothed target only for the node's own label set.
 *
 * **Under-resolved grains.** A grain without a single interior node (a node whose only label is
 * that grain), e.g. one element thick, cannot be fitted without squeezing its elements. With
 * NodeTargetOptions::hold_under_resolved (default) every grain boundary node with such a label is
 * HELD (UNDER_RESOLVED) in the constructor's result, and the grains are counted
 * (NodeTargetSummary::under_resolved_grains). Fixed labels (passthrough parts) are not grains here.
 *
 * **Pinches.** A grain boundary node where a grain pinches (NodeLabels::Pinched: the grain
 * touches itself only at the node or along an edge through it) cannot be fitted. It is HELD
 * (PINCH) with NodeTargetOptions::hold_pinches (default), or a FAKE_JUNCTION (PINCH) for label
 * repair to remove.
 *
 * **Status of every node** (TargetStatus; the target of every node that is not FIT is x itself):
 *   - NONE: one label (inside a grain, or on the outer boundary only);
 *   - FIT: a valid target;
 *   - OVER_CONSTRAINED: more than 3 constraints (NodeLabels::OverConstrained);
 *   - HELD: the node keeps its place: a label without analytic boundary (grain_seed = -1,
 *     NON_ANALYTIC), a curved boundary surface without patch triangles at the node
 *     (CURVED_FEATURE; corners with three surfaces are over-constrained instead), a grain too
 *     thin for the mesh (UNDER_RESOLVED) or a pinch (PINCH);
 *   - FAKE_JUNCTION: validation failed, or two labels grew from the same seed (TargetReason says
 *     which).
 *
 * **Determinism.** A target depends only on the node's coordinates (identical on every copy), its
 * label set and surfaces (identical on every copy) and the replicated seeds, so all copies compute
 * the same bits and the result does not depend on the partition. Only h and the summary need
 * communication.
 *
 * **Where it runs.** Host only (the conformal fit runs on the CPU); the seed search uses Axom's
 * sequential execution space (axom::SEQ_EXEC). The constructor is collective over the mesh
 * communicator; Evaluate, EvaluateAnalytic, NearbySeeds and the accessors are local.
 */

namespace exaconstit::microstructure {

/** @brief Settings of NodeTargets. */
struct NodeTargetOptions {
    /** @brief Largest allowed move, in element sizes (h times the order of the Nodes). */
    double max_move = 1.0;
    /** @brief Projection step and residual tolerance, in units of h. */
    double tolerance = 1e-10;
    /** @brief Most Gauss-Newton steps per node (manifold_projector.hpp). */
    int max_iterations = 25;
    /** @brief Number of bins of the seed search over seed parameters such as weights (affects
     * speed only, not results; SeedIndex::Build). */
    int num_bins = 8;
    /**
     * @brief Use the analytic grain boundaries of the seeds. false: no node gets an analytic
     * target (labels from a voxel map or a geodesic labeling, whose grains are not cells of the
     * metric); the seed set may then be empty.
     */
    bool analytic = true;
    /**
     * @brief Give grain boundary nodes without a valid analytic target (HELD for a non-analytic
     * label, or FAKE_JUNCTION) the smoothed staircase position instead (smoothed_targets.hpp).
     */
    bool smoothed = false;
    /** @brief Smoothing settings (with `smoothed`). */
    SmoothingParameters smoothing;
    /**
     * @brief Hold the grain boundary nodes of under-resolved grains: grains without a single
     * interior node (a node of that grain only), e.g. one element thick. Fitting them would
     * squeeze their elements; they keep their staircase (HELD, UNDER_RESOLVED) and are counted in
     * the summary.
     */
    bool hold_under_resolved = true;
    /**
     * @brief Pinched nodes (NodeLabels::Pinched): true = HELD (PINCH), they keep their place;
     * false = FAKE_JUNCTION (PINCH), a bad node that label repair works on (its passes).
     */
    bool hold_pinches = true;
};

/** @brief What happens to a node in the fit (see the file comment). */
enum class TargetStatus : int {
    NONE = 0,             ///< one label: inside a grain, or on the outer boundary only
    FIT = 1,              ///< has a valid target and is pulled toward it
    OVER_CONSTRAINED = 2, ///< more than 3 constraints: no point satisfies them all
    HELD = 3,             ///< keeps its place (TargetReason says why)
    FAKE_JUNCTION = 4     ///< its grains do not meet near it exactly (TargetReason says why)
};
/** @brief Number of TargetStatus values (size of NodeTargetSummary::status). */
inline constexpr int NUM_TARGET_STATUSES = 5;

/** @brief Why a node is HELD or a FAKE_JUNCTION (NONE otherwise). */
enum class TargetReason : int {
    NONE = 0,           ///< not HELD nor FAKE_JUNCTION
    NON_ANALYTIC = 1,   ///< HELD: a label with grain_seed = -1
    CURVED_FEATURE = 2, ///< HELD: a curved boundary surface has no patch triangles at the node
    SAME_SEED = 3,      ///< two labels grew from the same seed
    SINGULAR = 4,       ///< dependent equations (e.g. a grain boundary parallel to a box face)
    NOT_CONVERGED = 5,  ///< the projection did not converge
    RESIDUAL = 6,       ///< an equation does not hold at the target
    TOO_FAR = 7,        ///< the target is more than max_move element sizes away
    FOREIGN_SEED = 8,   ///< another seed is closer at the target
    UNDER_RESOLVED = 9, ///< HELD: a label of a grain without an interior node
    PINCH = 10          ///< a grain pinches at the node (NodeLabels::Pinched)
};
/** @brief Number of TargetReason values (size of NodeTargetSummary::reason). */
inline constexpr int NUM_TARGET_REASONS = 11;

/** @brief Node counts over all ranks, for the log (each node counted once, on its owner rank;
 * the same on every rank). */
struct NodeTargetSummary {
    /** @brief Nodes per TargetStatus (index = static_cast<int>(status)). */
    std::array<std::int64_t, NUM_TARGET_STATUSES> status{};
    /** @brief Nodes per TargetReason (index = static_cast<int>(reason)). */
    std::array<std::int64_t, NUM_TARGET_REASONS> reason{};
    /** @brief Largest |target - x| / h over the FIT nodes. */
    double max_move_ratio = 0.0;
    /** @brief FIT nodes whose target is the smoothed staircase (NodeTargetOptions::smoothed). */
    std::int64_t smoothed = 0;
    /** @brief Grains without an interior node (NodeTargetOptions::hold_under_resolved). */
    std::int64_t under_resolved_grains = 0;
};

/** @brief Most labels stored in a NodeLabelQuery: a larger set is over-constrained anyway. */
inline constexpr int NODE_QUERY_LABELS = 4;

/**
 * @brief A label set to evaluate at a node: the node's own set (the constructor) or a
 * hypothetical one (label repair previews switching an element, label_repair.hpp; the adjacency
 * check asks about pairs, adjacency_check.hpp).
 */
struct NodeLabelQuery {
    /** @brief The node (local node index of the NodeLabels). */
    int node = -1;
    /** @brief Size of the label set, also when it exceeds NODE_QUERY_LABELS. */
    int num_labels = 0;
    /** @brief The labels, ascending and distinct; only the first NODE_QUERY_LABELS are stored. */
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
    TargetStatus status = TargetStatus::NONE; ///< what would happen to the node
    TargetReason reason = TargetReason::NONE; ///< why, for HELD and FAKE_JUNCTION
    double target[3] = {0.0, 0.0, 0.0};       ///< the target when FIT, else the node's position
};

/** @brief Patches of boundary triangles (boundary_projector.hpp); declared here so this header
 * does not need to include it. */
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
     * @param setup      metric parameters of the seeds: the global Quadric tensor, or the flag
     *                   for per-seed tensors (as for SeedIndex::Build).
     * @param grain_seed grain g (1-based) grew from seed grain_seed[g - 1] (0-based), -1 = not
     *                   analytic. Must cover every label of `labels`.
     * @param options    move limit, tolerances, smoothed fallback and holds.
     * @throws std::invalid_argument on every rank for a label outside grain_seed, a seed index out
     *         of range, an empty seed set or invalid options; SeedIndex::Build errors.
     */
    NodeTargets(const NodeLabels& labels,
                MetricType metric,
                const SeedSet& seeds,
                const MetricSetup& setup,
                const std::vector<int>& grain_seed,
                const NodeTargetOptions& options = {});
    /** @brief Defined in the source file, where Evaluator and SurfacePatches are complete. */
    ~NodeTargets();
    /** @brief Not copyable: owns the evaluator and the patches, and refers to the labels. */
    NodeTargets(const NodeTargets&) = delete;
    /** @brief Not copyable. */
    NodeTargets& operator=(const NodeTargets&) = delete;

    /**
     * @brief Status, reason and target of each query, with exactly the rules the constructor
     * applies to the nodes' own label sets (same mesh size, surfaces and patches). Every label
     * must be a grain of grain_seed. Not collective (local nodes only); host.
     */
    std::vector<NodeTargetResult> Evaluate(const std::vector<NodeLabelQuery>& queries) const;

    /**
     * @brief Evaluate with the analytic rules only: no smoothed fallback, so FIT means a valid
     * analytic target (the adjacency check, adjacency_check.hpp). Not collective; host.
     */
    std::vector<NodeTargetResult>
    EvaluateAnalytic(const std::vector<NodeLabelQuery>& queries) const;

    /**
     * @brief For each listed node: the best seed at its original position, then the next best
     * seeds (up to `max_seeds` in all) whose analytic boundary with the best seed is estimated to
     * pass within `reach` element sizes of the node, estimated as (f_k - f_1) /
     * |grad f_k - grad f_1| at the node (exact for Voronoi and Laguerre, whose boundaries are
     * planes). The seeds of node i (closest first) are seeds[offsets[i]] .. seeds[offsets[i + 1]
     * - 1], 0-based seed indices; offsets has nodes.size() + 1 entries. Empty lists with
     * NodeTargetOptions::analytic = false. Not collective; host.
     */
    void NearbySeeds(const std::vector<int>& nodes,
                     int max_seeds,
                     double reach,
                     std::vector<int>& offsets,
                     std::vector<int>& seeds) const;

    /** @brief Face rings of the curved-surface patches: PatchRingsFor of the largest move a
     * target may need (max_move, or the smoothing's move limit when larger). */
    int PatchRings() const {
        return m_patch_rings;
    }
    /** @brief The options the targets were built with. */
    const NodeTargetOptions& Options() const {
        return m_options;
    }

    /** @brief Target positions, 3 doubles per node (node v at 3 v, 3 v + 1, 3 v + 2); the node's
     * own position unless FIT. */
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
    /** @brief True when node v is FIT with a smoothed (not analytic) target. */
    bool Smoothed(int v) const {
        return m_smoothed[static_cast<std::size_t>(v)] != 0;
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
    /** @brief The mesh size h of every node (see the file comment). Collective. */
    void ComputeMeshSize();
    /** @brief The global counts. Collective. */
    void BuildSummary();

    const NodeLabels& m_labels;                ///< the node labels (not owned)
    std::unique_ptr<SurfacePatches> m_patches; ///< patches of the nodes on curved surfaces
    int m_patch_rings = 1;                     ///< see PatchRings()
    std::unique_ptr<Evaluator> m_evaluator;    ///< seed search and projection for the metric
    NodeTargetOptions m_options;               ///< see Options()
    std::vector<char> m_smoothed;              ///< per node: FIT with the smoothed target
    mfem::Vector m_smoothed_target;            ///< smoothed staircase position, 3 per node
    mfem::Vector m_targets;                    ///< see Targets()
    mfem::Array<int> m_status;                 ///< per node: TargetStatus as int
    mfem::Array<int> m_reason;                 ///< per node: TargetReason as int
    mfem::Vector m_h;                          ///< per node: mesh size h
    NodeTargetSummary m_summary;               ///< see Summary()
};

} // namespace exaconstit::microstructure
