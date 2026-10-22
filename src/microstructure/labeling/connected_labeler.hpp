#pragma once

#include "microstructure/labeling/connected_limits.hpp" // CONNECTED_MAX_ROUNDS_LIMIT
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/microstructure_options.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <mpi.h>
#include <vector>

/**
 * @file connected_labeler.hpp
 * @brief "Connected" distance mode: every grain keeps only the elements that are connected
 * through material to the element holding its seed.
 *
 * **The problem.** The tessellation labeler (tessellation_labeler.hpp) gives each element to the
 * seed with the smallest metric value at the element centroid, measured in straight lines. On a
 * lattice strut or a U-shaped part a seed's cell can reach across a void, so the same grain shows
 * up on two struts (or both arms of the U) that touch only through the gap. The fragment cleanup
 * (fragment_cleanup.hpp) can split or merge such pieces afterwards, but then a piece's boundary is
 * no longer a boundary between two seeds.
 *
 * **What this mode does.** It keeps the straight-line (metric) labels, but a seed may only keep
 * the elements that its own cell connects to the seed through material. An element that carries
 * seed s but lies in a piece of s's elements that does not contain s's *anchor element* (the
 * element holding the seed) is an *orphan*: it puts s on its own exclusion list and takes the
 * best seed that is not on the list (SeedIndex::FindBestExcluding). This repeats in rounds until
 * no orphan is left. Every grain boundary is still the bisector (or metric boundary) between two
 * seeds, so later stages can fit the boundaries analytically, while no grain jumps across a void.
 *
 * **A fixed greedy order, not an exact definition.** The rounds do not compute "the nearest seed
 * whose cell reaches the element through material" in the strict sense. Two choices make them a
 * deterministic greedy procedure instead:
 * - exclusions are permanent: once an element has dropped seed s, it never takes s again, even if
 *   a later round connects it to s's anchor;
 * - all orphans of a round are relabeled at the same time, each looking only at its own list.
 * Example on a bent strip of elements ... a | e1 | e2 | ... where `a` is the anchor of seed A:
 * e1 carries B and e2 carries A, and both are cut off (e1 from B's anchor, e2 from A's anchor
 * because e1 lies between them). In round 1, e1 drops B and takes its next seed, A; at the same
 * time e2 drops A and takes its next seed, C. After round 1, e1 joins A's piece, so e2 would now be
 * connected to A's anchor, but A is on e2's list for good and e2 stays with C. The strict
 * definition would give e2 to A. The result is still well defined (the same order on every rank
 * count, see below), connected to the seeds' anchors, and made only of metric boundaries; it can
 * differ from the strict definition in a few elements next to such chains.
 *
 * **Terms.**
 * - Element key: a 64-bit hash of an element's exact vertex coordinates (mesh/element_key.hpp),
 *   the same for every rank count and partition; used to name elements and break ties.
 * - Component (or fragment) of label s: a maximal set of elements labeled s that are connected
 *   through shared faces. LabelComponents (fragment_cleanup.hpp) names each component by the
 *   smallest element key it contains, which is partition independent.
 * - Material component: a component of the whole mesh (every element given the same label), i.e.
 *   a separate body of the part.
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it.
 *
 * **Anchors** (ConnectedAnchorKeys, computed once from the input labels). The anchor of seed s
 * is an element that carries label s in the input:
 * - the element that contains the seed point (point-in-element test, seeding/
 *   seed_element_locator.hpp; the smallest element key when the point lies on a face or edge
 *   shared by several elements), if that element is labeled s;
 * - otherwise the s-labeled element whose volume centroid is nearest to the seed (smallest key on
 *   a distance tie). This covers seeds outside the material, and seeds whose own element went to
 *   a neighbor. The second case is common even for plain Voronoi cells: a seed near a corner of
 *   its element can be farther from that element's centroid than a neighboring seed is, so the
 *   element is labeled by the neighbor. Anchoring such a seed to its containing element would make
 *   its whole grain orphans (a few percent of the seeds at about 100 elements per grain).
 * - A seed that labels no element at all has no anchor and no cell. That is a legitimate empty
 *   cell (a Voronoi seed coincident with a lower-index seed, or a Laguerre/Additive/Quadric seed
 *   dominated by its neighbors' weights) and is counted in seeds_without_cell.
 * The values are reduced over all ranks with MPI_Allreduce(MIN) on (squared distance, key), so
 * every rank knows the same anchor key for every seed. An anchor carries label s and lies in the
 * component of s that holds the anchor, so it is never an orphan and keeps label s in every round.
 *
 * Snapping of seeds outside the material differs between the distance modes on purpose: the
 * connected mode uses the nearest element *centroid* (of the seed's own elements), while the
 * geodesic mode (labeling/vertex_field.hpp) snaps to the elements around the nearest mesh *vertex*,
 * because its distance field lives on vertices.
 *
 * **Bodies without an anchor are exempt.** A material component that holds no seed's anchor
 * (an isolated body of the part whose elements belong to seeds anchored elsewhere) cannot be
 * connected to any seed. Its elements keep their straight-line labels and are counted
 * (ConnectedLabelReport::unanchored_component_elements); the driver's fragment cleanup then
 * splits or merges them.
 *
 * **Why the rounds end.** Let X(e) be the exclusion list of element e (empty at the start).
 * - In each round every orphan appends its current label to X(e). That label is never already on
 *   X(e): it is either the input label (X(e) is still empty) or a result of FindBestExcluding,
 *   which never returns a listed seed. So every list holds distinct seeds and grows by one in
 *   every round in which its element is an orphan, and sum_e |X(e)| <= NE x N.
 * - A round happens only when some orphan exists, and then the sum grows by at least 1, so there
 *   are at most NE x N rounds even without a cap; ConnectedLabelParameters::max_rounds caps them.
 *   The lists are stored only for elements that have been orphans and grow round by round, so the
 *   list memory is one int per list entry actually created plus one int per element.
 * - Settled elements stay settled: an element in the component of seed s that holds s's anchor is
 *   not an orphan and keeps label s; the other elements of that component keep label s too, so
 *   the component (and the path through it to the anchor) survives every later round. It can only
 *   grow, when orphans relabeled to s join it.
 * - An element whose list contains every seed (possible only when max_rounds >= N) has no
 *   candidate left; it keeps its label, takes part in no further round and is reported as
 *   unresolved.
 *
 * **Why the result does not depend on the rank count.** By induction over the rounds: if every
 * element has the same label and exclusion list for every partition at the start of a round, then
 * (1) LabelComponents gives the same component ids (smallest element key of the component),
 * (2) the per-seed table "component holding s's anchor, if the anchor carries label s" is reduced
 * with MPI_Allreduce(MIN) over values that only one rank (the anchor's owner) contributes, so it
 * is the same everywhere, (3) whether an element is an orphan depends only on its label, its
 * component id and that replicated table, and (4) its new label depends only on its own sample
 * points (centroid or voting points, computed from its own nodes), its own list and the
 * replicated seeds. So the labels and lists are again the same at the end of the round. Counts in
 * the report are exact integer sums over the ranks.
 *
 * **Sampling.** With SamplingMode::CENTROID an orphan is relabeled at its volume centroid. With
 * SamplingMode::QUADRATURE_VOTE every voting point of the element (quadrature_vote.hpp, VoteRule)
 * is searched with the element's exclusion list, and the element goes to the seed with the
 * largest total weight w_q detJ_q (ties: lowest seed index), as in the quadrature vote.
 *
 * **Scope** (the driver refuses other combinations): no periodic axes (an adjacency with periodic
 * face pairing is refused), no Lloyd/weight fitting, no voxel maps; the mesh is the active
 * material (AttributeDomain::ActiveMesh()).
 *
 * **Where it runs.** Component search and the round bookkeeping are host loops (the component
 * search uses union-find, a forest in which every element points to a parent element and two
 * elements are in the same group when their parent chains end at the same root; it is
 * sequential); the seed search (SeedIndex on the Axom execution space matching the mfem::Device),
 * the element samplers and the vote are device kernels.
 */

namespace exaconstit::microstructure {

/** @brief Options of ConstrainToConnectedCells; identical on every rank. */
struct ConnectedLabelParameters {
    /**
     * @brief Largest number of relabeling rounds, 0..CONNECTED_MAX_ROUNDS_LIMIT. Orphans left
     * after the last round keep their label and are counted as unresolved. 0 only counts the
     * orphans of the input labels. The exclusion lists grow with the rounds actually performed
     * (one int per orphan per round), not with this cap.
     */
    int max_rounds = 16;
    /** @brief How an orphan is relabeled: at its centroid, or by a vote of points inside it. */
    SamplingMode sampling = SamplingMode::CENTROID;
    /**
     * @brief Subdivision levels of the voting points for SamplingMode::QUADRATURE_VOTE, 0..3
     * (8^levels points per element; see QuadratureVoteParameters::levels). Ignored for centroid
     * sampling.
     */
    int vote_levels = 2;
};

/** @brief What ConstrainToConnectedCells did. Global counts, identical on every rank. */
struct ConnectedLabelReport {
    /** @brief Relabeling rounds performed (0 when the input had no orphan). */
    int rounds = 0;
    /** @brief Elements whose final label differs from their input label. */
    std::int64_t relabeled_elements = 0;
    /**
     * @brief Orphans left at the end: elements (outside exempt bodies) that are not connected to
     * their seed's anchor through elements of the same label, because max_rounds was reached or
     * every seed was excluded. They keep their last label; the fragment cleanup handles them.
     */
    std::int64_t unresolved_elements = 0;
    /**
     * @brief Elements of material components (separate bodies) that hold no seed's anchor. They
     * keep their input labels and never take part in the rounds.
     */
    std::int64_t unanchored_component_elements = 0;
    /**
     * @brief Seeds that label no element in the input, so they have no anchor and no cell (a
     * Voronoi seed coincident with a lower-index seed, or a weighted seed dominated by its
     * neighbors). A seed whose own element went to a neighbor is *not* counted: it is anchored
     * to its nearest element of its own label (see "Anchors" in the file comment).
     */
    int seeds_without_cell = 0;
    /**
     * @brief Size N (seed order), identical on every rank: the anchor element key of every seed
     * that the rounds used, i.e. ConnectedAnchorKeys of the input labels (UINT64_MAX for a seed
     * that labeled no element). Returned so a caller (the driver's fragment cleanup, keep rule
     * "seed") can use the same anchors without recomputing them.
     */
    std::vector<std::uint64_t> anchor_key;
};

/**
 * @brief A label-independent anchor element of every seed, as its element key: the element that
 * contains the seed (smallest key on ties), or for a seed outside the material the element whose
 * centroid is nearest (smallest key on ties). ConstrainToConnectedCells itself uses the
 * label-aware ConnectedAnchorKeys instead.
 *
 * Steps: ContainingElementKeys, then NearestElementCentroidKeys for the seeds no element contains
 * (both in seeding/seed_element_locator.hpp, with an Axom BVH per rank and a global
 * MPI_Allreduce(MIN) on (squared distance, key)).
 *
 * The result is per *seed*. The driver's SeedAnchorKeys (keep rule "seed" of the fragment
 * cleanup, per *grain*) can be expressed with it as anchor[grain_seed[g - 1]]; the two differ only
 * when a seed lies exactly on a face between elements of different grains (SeedAnchorKeys then
 * takes the smallest key among the elements of the seed's own grain) and for seeds outside the
 * material (SeedAnchorKeys has no anchor for them).
 *
 * **Collective** over `comm` (one MPI_Allreduce of N entries, plus two of (number of outside
 * seeds) entries). Host only. About log(NE) work per seed.
 *
 * @param pmesh       this rank's part of the mesh (3D). May have no local elements.
 * @param comm        communicator of the mesh.
 * @param seeds       replicated seeds (positions only), N >= 0.
 * @param element_key size NE: ComputeElementKeys(pmesh).
 * @param centroid_x, centroid_y, centroid_z size NE: element volume centroids
 *                    (ElementSamples::cx/cy/cz from ComputeElementSamples), read with HostRead().
 * @return size N, identical on every rank: the anchor key of each seed. UINT64_MAX only when the
 *         whole mesh has no element.
 * @throws std::invalid_argument on every rank if an input size is wrong on any rank.
 */
std::vector<std::uint64_t> SeedAnchorElementKeys(mfem::ParMesh& pmesh,
                                                 MPI_Comm comm,
                                                 const SeedSet& seeds,
                                                 const std::vector<std::uint64_t>& element_key,
                                                 const mfem::Vector& centroid_x,
                                                 const mfem::Vector& centroid_y,
                                                 const mfem::Vector& centroid_z);

/**
 * @brief The anchor element of every seed as used by ConstrainToConnectedCells (see "Anchors" in
 * the file comment): the containing element if it carries the seed's label in `element_seed`,
 * else the element of that label whose centroid is nearest to the seed (smallest key on ties),
 * else UINT64_MAX (the seed labels no element: no cell).
 *
 * The containing element comes from ContainingElementKeys (seeding/seed_element_locator.hpp); the
 * nearest labeled element is found in one pass over the local elements (each element is compared
 * with its own seed only, O(NE)), followed by a global MPI_Allreduce(MIN) on the squared distance
 * and then on the keys at exactly that distance. Also suitable for the fragment cleanup's keep rule
 * "seed" (per grain: anchor[grain_seed[g - 1]]), since the anchor always carries the grain's label.
 *
 * **Collective** over `comm`. Host only.
 *
 * @param pmesh        this rank's part of the mesh (3D). May have no local elements.
 * @param comm         communicator of the mesh.
 * @param seeds        replicated seeds (positions only), N >= 0.
 * @param element_seed size NE: 0-based seed label of every local element (HostRead).
 * @param element_key  size NE: ComputeElementKeys(pmesh).
 * @param centroid_x, centroid_y, centroid_z size NE: element volume centroids (HostRead).
 * @return size N, identical on every rank.
 * @throws std::invalid_argument on every rank if an input size is wrong or a label is outside
 *         [0, N) on any rank.
 */
std::vector<std::uint64_t> ConnectedAnchorKeys(mfem::ParMesh& pmesh,
                                               MPI_Comm comm,
                                               const SeedSet& seeds,
                                               const mfem::Array<int>& element_seed,
                                               const std::vector<std::uint64_t>& element_key,
                                               const mfem::Vector& centroid_x,
                                               const mfem::Vector& centroid_y,
                                               const mfem::Vector& centroid_z);

/**
 * @brief Restricts every seed's elements to those connected through material to its anchor
 * element (see the file comment for the rounds, the exemptions and the proofs).
 *
 * **Collective** over `comm` (component searches, neighbor exchanges and MPI_Allreduce calls in
 * every round). Host function launching device kernels. The result is identical for every rank
 * count and mesh partition.
 *
 * @param pmesh        this rank's part of the (active) mesh: 3D, one element geometry (hex or
 *                     tet for vote sampling). Non-const only for the element samplers; not
 *                     modified.
 * @param comm         communicator of the mesh (the adjacency's communicator).
 * @param adjacency    face adjacency of `pmesh` (MeshAdjacency), without periodic pairing.
 * @param seeds        replicated seed set, N >= 1 (positions; weight/shape as the metric uses
 *                     them).
 * @param tessellation the metric, num_bins and anisotropy (EffectiveGlobalTensor) of the labeling
 *                     that produced `element_seed`; distance, sampling and periodic are not read
 *                     (sampling comes from `params`).
 * @param element_seed size NE. In: 0-based nearest-seed label of every local element, from the
 *                     Euclidean labeler (LabelElementsWithSeeds or LabelElementsByQuadratureVote
 *                     with the same metric). Out: the constrained labels, still 0-based seed
 *                     indices (compact them with CompactSeedLabels). Read and written on the host.
 * @param params       rounds cap and sampling.
 * @return the report (global counts).
 * @throws std::invalid_argument on every rank for: an empty seed set, max_rounds outside
 *         0..CONNECTED_MAX_ROUNDS_LIMIT, vote_levels outside 0..3, an adjacency of another mesh
 *         or with periodic pairing, a wrong element_seed size or a label outside [0, N) on any
 *         rank; as SeedIndex::Build for invalid seed data.
 */
ConnectedLabelReport ConstrainToConnectedCells(mfem::ParMesh& pmesh,
                                               MPI_Comm comm,
                                               const MeshAdjacency& adjacency,
                                               const SeedSet& seeds,
                                               const TessellationOptions& tessellation,
                                               mfem::Array<int>& element_seed,
                                               const ConnectedLabelParameters& params);

} // namespace exaconstit::microstructure
