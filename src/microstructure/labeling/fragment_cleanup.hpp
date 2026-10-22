#pragma once

#include "microstructure/mesh/mesh_adjacency.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <string>
#include <vector>

/**
 * @file fragment_cleanup.hpp
 * @brief Finds grains whose elements form several disconnected pieces and repairs them; merges
 * grains that are too small.
 *
 * **Why.** The tessellation gives each element to the seed that is closest to the element's
 * centroid in straight-line (Euclidean) distance. On a non-convex part (a U-shaped bracket, a
 * lattice strut, a part with holes) one grain's region of space can cross a gap of empty space,
 * so its elements form two or more pieces that touch only through the void. Some metrics (per-
 * seed anisotropy, multiplicative weights) can also produce disconnected cells in a convex
 * domain. Each such piece is called a *fragment*: a maximal set of elements of one grain that are
 * connected to each other through shared faces. A healthy grain has exactly one fragment.
 *
 * **Fragment identity.** A fragment is named by the smallest element key among its elements
 * (mesh/element_key.hpp: a hash of the element's exact vertex coordinates). Element keys do not
 * depend on the mesh partition, so the name, and every decision below that uses it as a
 * tie-break, is the same for every rank count. (Global element numbers would change with the
 * partition.)
 *
 * **Finding the fragments** (LabelComponents). Two passes:
 * 1. *On each rank, union-find.* A union-find (or disjoint-set) structure keeps a forest in which
 *    every element points to a "parent" element; two elements belong to the same group exactly
 *    when following the parents from each ends at the same root. Joining two groups is a single
 *    pointer change between roots. Every local face whose two elements carry the same grain joins
 *    their groups, so afterwards each group is a piece of a fragment that is connected within
 *    this rank. Each group gets the smallest element key it contains as its provisional id.
 * 2. *Across ranks, min-propagation.* A fragment can continue on other ranks through shared
 *    faces. Repeatedly: every rank sends each element's current id to its face neighbors on other
 *    ranks (MeshAdjacency::ExchangeKeys); wherever a ghost element of the same grain has a
 *    smaller id, the local group takes that smaller id. A collective "did anything change on any
 *    rank?" test (MPI_Allreduce) ends the loop. Every id only decreases and is always the key of
 *    an element of the same fragment, so when nothing changes every element carries the smallest
 *    key of its whole fragment. The number of rounds is one more than the largest number of
 *    rank-boundary crossings on the path from a fragment's smallest-key element to any of its
 *    elements (a fragment that winds back and forth across a partition boundary needs more
 *    rounds than the number of ranks it touches).
 *
 * **Policies** (FragmentCleanupParameters):
 * - For each grain with several fragments one fragment is *kept*: the largest (most elements;
 *   ties: smallest fragment id), or the fragment containing a given *anchor* element (normally
 *   the element that contains the grain's seed), falling back to the largest when the anchor is
 *   not part of the grain. Under SPLIT_LARGE_MERGE_TINY only fragments that are not tiny can be
 *   kept (unless all are tiny): a seed lying in a tiny piece must not keep that piece while the
 *   grain's large body is split off with a derived id and a new orientation.
 * - SPLIT (natural for random seeds): every other fragment becomes a new grain. New grains are
 *   numbered after the existing ones, in order of (original grain, fragment id); the source id of
 *   the k-th fragment split off a grain is DeriveFragmentSourceId(source id of the grain, k), so
 *   a random orientation keyed by it is reproducible.
 * - MERGE (natural for measured data, where the grain list is fixed): every other fragment joins
 *   the face-neighboring grain with which it shares the most faces (ties: lowest grain id). Only
 *   neighbors that are themselves not about to move count ("stable" neighbors); a fragment
 *   surrounded only by moving fragments waits for the next round. A fragment with no neighboring
 *   grain at all (an isolated body of the mesh) is split off into a new grain. When a round makes
 *   no progress (every waiting fragment touches only waiting fragments), the largest waiting
 *   fragment alone merges into the neighboring grain with the most shared faces, waiting
 *   neighbors included; with a single piece moving, no two pieces can swap labels.
 * - SPLIT_LARGE_MERGE_TINY (the generator's default for random seeds): extra fragments with at
 *   least min_fragment_elements elements and at least min_fragment_fraction of their grain's
 *   elements are split off as by SPLIT; the smaller ("tiny") ones, typically single elements that
 *   touch their grain only along an edge because the labels are sampled at element centroids,
 *   are merged as by MERGE. Every fragment is classified once, at the first search, against the
 *   element count of its grain as labeled (classifying again after large pieces were split off
 *   would measure tiny pieces against a shrinking remainder and could split them). Large
 *   fragments are split first, then the tiny ones merge in rounds.
 * - NONE: fragments are counted and reported, nothing is changed.
 * - `min_grain_elements`: afterwards, grains with fewer elements merge into the stable
 *   face-neighboring grain with the most shared faces (ties: lowest grain id); the largest small
 *   grain is kept as it is when no small grain has a stable neighbor (for example when every
 *   grain is small).
 * Rounds repeat until nothing changes. Each round moves or settles at least one whole fragment or
 * grain, so the number of candidates strictly decreases and the loop terminates. `max_rounds` is
 * only a safety cap on the merge and small-grain rounds; the split rounds of
 * SPLIT_LARGE_MERGE_TINY are not counted (they are bounded by the number of large fragments).
 *
 * **Output.** Grain ids are renumbered compactly 1..N' in increasing order of their working id:
 * surviving original grains first (in their original order), then the split-off grains in order
 * of creation. Every count and table is reduced with exact integer sums and replicated on every
 * rank, so the result is identical for every rank count and mesh partition.
 *
 * **Failure modes of the element key.** Fragments are named by element keys (64-bit hashes of
 * the vertex coordinates, mesh/element_key.hpp). If two elements of one grain had the same key
 * (a hash collision, probability ~ NE^2 / 2^65, or two elements with identical vertices, i.e. a
 * duplicated element in a broken mesh), two of its fragments could get the same id and be
 * counted as one (their table records are summed), so that grain's fragments would not be split
 * or merged correctly. Different grains are never mixed up: fragments are always identified by
 * (grain label, id).
 *
 * **Where it runs and what it costs.** Host only (union-find is inherently sequential); data
 * exchange uses MFEM's face-neighbor machinery and MPI collectives. Per-grain tables are
 * replicated on every rank: every component search and every round does MPI_Allreduce calls over
 * arrays of one entry per grain label (the fragment table's MIN/MAX test, the label counts), so
 * the communication volume is O(number of labels x number of rounds); gathers scale with the
 * number of fragments that need a decision.
 */

namespace exaconstit::microstructure {

/** @brief What to do with the extra fragments of a disconnected grain. */
enum class FragmentPolicy {
    NONE,  ///< Count and report only
    SPLIT, ///< Each extra fragment becomes a new grain
    MERGE, ///< Each extra fragment joins the neighboring grain with the most shared faces
    /**
     * Large extra fragments become new grains (a grain really cut in two, e.g. by a void of a
     * lattice); tiny ones (FragmentCleanupParameters::min_fragment_elements /
     * min_fragment_fraction) merge like MERGE (discretization artifacts, e.g. an element that
     * touches the rest of its grain only along an edge). The generator's default for generated
     * seeds.
     */
    SPLIT_LARGE_MERGE_TINY
};

/** @brief Which fragment of a disconnected grain keeps the grain's identity. */
enum class KeepRule {
    LARGEST,     ///< Most elements; ties go to the smallest fragment id (smallest element key)
    SEED_ELEMENT ///< The fragment containing the grain's anchor element (normally the element
                 ///< that contains the seed); LARGEST when the anchor is not in the grain
};

/** @brief Options of CleanupFragments; identical on every rank. */
struct FragmentCleanupParameters {
    /** @brief Treatment of extra fragments. */
    FragmentPolicy policy = FragmentPolicy::SPLIT;
    /** @brief Which fragment is kept. */
    KeepRule keep = KeepRule::LARGEST;
    /**
     * @brief Only for KeepRule::SEED_ELEMENT: size num_grains, entry g-1 = element key
     * (ElementKey) of the anchor element of grain g. A key that matches no element of grain g
     * means "no anchor" for that grain (LARGEST is used).
     */
    std::vector<std::uint64_t> grain_anchor_key;
    /**
     * @brief SPLIT_LARGE_MERGE_TINY only: an extra fragment with fewer elements than this is
     * tiny (merged). Must be >= 0.
     */
    int min_fragment_elements = 8;
    /**
     * @brief SPLIT_LARGE_MERGE_TINY only: an extra fragment with fewer than this fraction of its
     * grain's elements (all fragments counted) is tiny too. Must be in [0, 1].
     */
    double min_fragment_fraction = 0.1;
    /**
     * @brief Grains with fewer elements than this merge into a neighbor. 0 and 1 disable the
     * step. Must be >= 0.
     */
    int min_grain_elements = 1;
    /**
     * @brief Safety cap on merge rounds per phase (each round has a few collective calls). The
     * loop provably ends before the cap for any valid input; exceeding it throws.
     */
    int max_rounds = 1000000;
};

/** @brief Counts describing one CleanupFragments run; identical on every rank. */
struct FragmentCleanupReport {
    int initial_grains = 0;               ///< num_grains on input
    int final_grains = 0;                 ///< compact grain count on output
    std::int64_t fragments_found = 0;     ///< extra fragments on input: sum over grains of
                                          ///< (number of fragments - 1)
    std::int64_t fragments_split = 0;     ///< fragments that became new grains
    std::int64_t fragments_merged = 0;    ///< fragments merged into a neighboring grain
    std::int64_t small_grains_merged = 0; ///< grains below min_grain_elements merged away
    std::int64_t small_grains_kept = 0;   ///< grains below min_grain_elements that could not
                                          ///< merge (no stable neighbor) and were kept
    int merge_rounds = 0;                 ///< rounds of fragment merging (MERGE, and the tiny
                                          ///< fragments of SPLIT_LARGE_MERGE_TINY)
    int small_grain_rounds = 0;           ///< rounds of small-grain merging
    int max_propagation_rounds = 0;       ///< most min-propagation rounds of one component search

    /** @brief One-paragraph human-readable summary (host only). */
    std::string Summary() const;
};

/** @brief Result of CleanupFragments. */
struct FragmentCleanupResult {
    /**
     * @brief Size: local NE. Compact grain id 1..num_grains of each local element (host
     * memory).
     */
    mfem::Array<int> element_grain;
    /** @brief Global grain count N' after cleanup (same on every rank). */
    int num_grains = 0;
    /**
     * @brief Size N'. grain_parent[g-1] = the input grain (1-based) that grain g came from:
     * itself for surviving grains, the grain it was split from for new grains. Use it to carry
     * per-grain data (seed index, orientation from a data source, phase) over to the new grains.
     */
    std::vector<int> grain_parent;
    /**
     * @brief Size N'. Source id of grain g: the input value for surviving grains,
     * DeriveFragmentSourceId(parent's source id, k) for the k-th fragment split off a grain.
     */
    std::vector<std::int64_t> grain_source_id;
    /** @brief Counts of what was done. */
    FragmentCleanupReport report;
};

/**
 * @brief Connected components (fragments) of the element labels through face adjacency.
 *
 * Every local element gets the id of its fragment: the smallest element key among the elements
 * connected to it through faces whose two elements have the same label (union-find on this rank,
 * then min-propagation across ranks; see the file comment).
 *
 * **Collective** over the mesh communicator (repeated neighbor exchanges and one MPI_Allreduce
 * per round). Host only.
 *
 * @param adjacency      face adjacency of the mesh.
 * @param element_label  size NE, label of each local element (any int values; equal labels are
 *                       joined). Read on the host.
 * @param element_key    size NE, ComputeElementKeys(adjacency.GetParMesh()).
 * @param component      output, resized to NE: fragment id of each local element.
 * @param ghost_component optional output, resized to NG: fragment id of each ghost element.
 * @return the number of exchange rounds used (>= 1).
 * @throws std::invalid_argument on every rank if an input size is wrong on any rank.
 */
int LabelComponents(const MeshAdjacency& adjacency,
                    const mfem::Array<int>& element_label,
                    const std::vector<std::uint64_t>& element_key,
                    std::vector<std::uint64_t>& component,
                    std::vector<std::uint64_t>* ghost_component = nullptr);

/**
 * @brief Source id of the k-th fragment split off a grain whose source id is `source_id`.
 *
 * Derivation: h = Mix64(Mix64(source_id XOR 0xD1B54A32D192ED03) + k), with the top bit set, read
 * as a signed 64-bit integer (Mix64 = the splitmix64 mixing function). The result is always
 * negative, so it never equals a generated seed's source id (generated ids are candidate indices,
 * >= 0); it depends only on (source_id, k), so it is reproducible on any rank count. Distinct
 * pairs collide only by a 63-bit hash collision.
 *
 * Host only, not collective.
 *
 * @param source_id the parent grain's source id (any value, including negative table ids).
 * @param ordinal   k >= 1: 1 for the first fragment split off this grain, 2 for the second, ...
 *                  (fragments of one round in order of increasing fragment id).
 */
std::int64_t DeriveFragmentSourceId(std::int64_t source_id, std::int64_t ordinal);

/**
 * @brief Splits or merges the disconnected fragments of every grain and merges grains with fewer
 * than `min_grain_elements` elements (see the file comment for the rules).
 *
 * **Collective** over the mesh communicator. Host only. The result is identical for every rank
 * count and partition of the same mesh.
 *
 * @param adjacency       face adjacency of the mesh (MeshAdjacency).
 * @param element_grain   size NE: grain id of each local element, in [1, num_grains] (e.g.
 *                        TessellationLabels::element_grain). Read with HostRead().
 * @param num_grains      number of grains N >= 0 (same on every rank).
 * @param grain_source_id size N: source id of each grain (index g-1), e.g.
 *                        MicrostructureResult::grain_source_id. May be empty when the caller
 *                        does not track source ids; grain g then uses source id g-1.
 * @param params          policy, keep rule, anchors, min_grain_elements.
 * @return relabeled elements, new grain count, per-grain parent and source ids, report.
 * @throws std::invalid_argument on every rank for invalid parameters (sizes, negative
 *         min_grain_elements or min_fragment_elements, min_fragment_fraction outside [0, 1]) or
 *         a grain id outside [1, num_grains] on any rank;
 *         std::runtime_error on every rank if max_rounds is exceeded.
 */
FragmentCleanupResult CleanupFragments(const MeshAdjacency& adjacency,
                                       const mfem::Array<int>& element_grain,
                                       int num_grains,
                                       const std::vector<std::int64_t>& grain_source_id,
                                       const FragmentCleanupParameters& params);

} // namespace exaconstit::microstructure
