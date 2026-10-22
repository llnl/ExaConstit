#pragma once

#include "microstructure/io/voxel_map_io.hpp"
#include "microstructure/labeling/connected_labeler.hpp"
#include "microstructure/labeling/fragment_cleanup.hpp"
#include "microstructure/labeling/geodesic_common.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/microstructure_options.hpp"
#include "microstructure/microstructure_types.hpp"
#include "microstructure/morphology/lloyd.hpp"
#include "microstructure/morphology/weight_fit.hpp"
#include "microstructure/seeding/johnson_mehl.hpp"
#include "microstructure/seeding/seed_generators.hpp"

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

/**
 * @file microstructure_driver.hpp
 * @brief Entry point shared by the standalone tool and the mechanics hook.
 */

namespace exaconstit::microstructure {

/**
 * @brief What GenerateMicrostructure did, for logs and provenance. Identical on every rank
 * (all counts are global), except `adjacency`, which describes this rank's mesh partition, and
 * the per-rank voxel read counts `voxel_tiles_read` / `voxel_voxels_read`. Host data.
 *
 * Optional stages fill their sub-report and set the matching `used_*` / `ran_*` flag; the
 * sub-reports of stages that did not run keep their default values.
 */
struct GenerationReport {
    /** @brief The seed generator's report (requested/achieved seeds, candidates, fallbacks).
     * SeedingReport::Summary() gives a multi-line text for the log. For Johnson-Mehl seeds only
     * requested (num_nuclei), achieved, placement_used, candidates and the boxes are filled; the
     * details are in johnson_mehl. */
    SeedingReport seeding;
    /** @brief True when the seeds came from Johnson-Mehl nucleation (placement = johnson_mehl). */
    bool used_johnson_mehl = false;
    /** @brief The nucleation report (JohnsonMehlReport::Summary() gives the log text). */
    JohnsonMehlReport johnson_mehl;
    /** @brief Number of seeds that were labeled (= seeding.achieved). */
    std::int64_t num_seeds = 0;
    /**
     * @brief The seed set the labeling used (after any Lloyd moves and with the final weights):
     * positions in mesh units, source ids, weights and per-seed tensors when present. Seed i is
     * the seed of GenerationReport::grain_seed index i. Empty for voxel maps.
     */
    SeedSet seeds;
    /** @brief Material volume (fixed-point sum of the element volumes, length^3); 0 when no stage
     * needed it (size targets, polydisperse placement, a seed count from the mean grain size,
     * Johnson-Mehl). */
    double material_volume = 0.0;
    /**
     * @brief Target volume of every seed (size num_seeds, seed order; length^3) when size targets
     * were used ([Seeds.Size] distribution, polydisperse placement or fit_volumes); empty
     * otherwise. The targets sum to material_volume.
     */
    std::vector<double> seed_target_volume;
    /** @brief True when polydisperse RSA placed the seeds. */
    bool used_polydisperse_placement = false;
    /** @brief True when Lloyd relaxation ran (lloyd holds its history). */
    bool ran_lloyd = false;
    LloydReport lloyd;
    /** @brief True when a separate volume fit ran (weight_fit holds its history). In centroidal
     * power mode the refits are in lloyd.fits instead. */
    bool ran_weight_fit = false;
    WeightFitReport weight_fit;
    /** @brief Image layers and seed images of a periodic run (0 when nothing is periodic). */
    int periodic_layers = 0;
    int num_periodic_images = 0;
    /** @brief Periodic face pairing used by the fragment cleanup (zero when not periodic). */
    PeriodicFaceReport periodic_faces;
    /**
     * @brief The face adjacency of the `pmesh` passed to GenerateMicrostructure (the whole mesh,
     * with periodic face pairing when any axis is periodic), matching
     * MicrostructureResult::element_grain. Shared so callers (the tool's grain statistics) can
     * reuse it instead of repeating the face-neighbor exchange and the periodic gather. It is
     * valid only while that mesh lives and is not refined or repartitioned. Never null after a
     * successful call. Per-rank data (with `voxel_tiles_read` and `voxel_voxels_read`, the only
     * members that differ between ranks).
     *
     * Normally this is the adjacency the fragment cleanup used. With passthrough attributes the
     * cleanup ran on the sub-mesh of the active elements, which is destroyed before
     * GenerateMicrostructure returns, so the driver builds a separate adjacency of the whole mesh
     * for this field (one extra face-neighbor exchange); nothing that refers to the sub-mesh
     * leaves the driver.
     */
    std::shared_ptr<const MeshAdjacency> adjacency;
    /** @brief Quadrature vote: elements cut by a grain boundary (global; 0 for centroids). */
    std::int64_t num_split_elements = 0;
    /** @brief Grains right after labeling, before fragment cleanup: seeds that won at least one
     * element. */
    int num_labeled_grains = 0;
    /** @brief The fragment cleanup report (fragments found, split, merged; small grains). */
    FragmentCleanupReport cleanup;
    /** @brief Final number of grains (after cleanup). */
    int num_grains = 0;
    /**
     * @brief Size num_grains: 0-based index into the full seed set (seed order, the SeedSet the
     * labeling used, before any periodic images: seed-table rows first, then the filler seeds) of
     * the seed each final grain descends from. A grain split off by the cleanup has its parent
     * grain's seed. -1 for grains that did not grow from a seed (voxel-map grains, passthrough
     * grains).
     */
    std::vector<int> grain_seed;
    /** @brief Seeds that won no element (num_seeds - num_labeled_grains): seeds in parts of the
     * sampling domain too thin to hold an element centroid, coincident Voronoi seeds, or empty
     * weighted cells. */
    std::int64_t num_empty_seeds = 0;
    /**
     * @brief Seeds sharing their exact position with a lower-index seed. With Voronoi they never
     * win (ties go to the lower index); with weighted metrics a coincident seed with a larger
     * Laguerre/Additive weight or a different per-seed quadric tensor can win.
     */
    int num_coincident_seeds = 0;
    /** @brief Global number of elements of the labeled mesh. */
    std::int64_t num_elements = 0;

    // ---- Stage 1d: seed tables, ROI filler, voxel maps, phases and regions, passthrough ------

    /** @brief True when the seeds came from a seed table (Seeds.source = "file"). */
    bool used_seed_table = false;
    /** @brief Rows read from the seed table. */
    std::int64_t num_table_rows = 0;
    /** @brief Table rows dropped because their position lies outside the ROI box. */
    std::int64_t num_table_dropped = 0;
    /** @brief Table seeds used (rows read minus dropped rows), seed indices 0..num - 1. */
    std::int64_t num_table_seeds = 0;
    /** @brief Which optional columns the table had (io/seed_table_io.hpp SeedTable::has_*). */
    bool table_has_quat = false;
    bool table_has_phase = false;
    bool table_has_weight = false;
    bool table_has_volume = false;
    bool table_has_shape = false;
    /** @brief Unknown table columns that were ignored (lower case). */
    std::vector<std::string> table_ignored_columns;
    /** @brief True when Seeds.roi was given; the box (mesh units) follows. */
    bool used_roi = false;
    std::array<double, 3> roi_lo = {0.0, 0.0, 0.0};
    std::array<double, 3> roi_hi = {0.0, 0.0, 0.0};
    /** @brief With filler seeds: material volume strictly inside / outside the ROI box (fixed-point
     * quadrature sums, length^3; seeding/roi_filler.hpp). 0 without filler. */
    double roi_material_volume = 0.0;
    double outside_material_volume = 0.0;
    /** @brief True when random filler seeds were requested (fill_outside_roi = "random"). */
    bool used_filler = false;
    /** @brief Filler seeds requested (given, or the automatic count) and placed. */
    std::int64_t num_filler_requested = 0;
    std::int64_t num_filler_seeds = 0;
    /** @brief source_id of the first filler candidate: max table id + 1 (whole table). */
    std::int64_t filler_source_id_offset = 0;
    /** @brief The seed generator's report for the filler seeds. */
    SeedingReport filler_seeding;
    /** @brief Final grains that grew from filler seeds (MicrostructureResult::grain_filler). */
    int num_filler_grains = 0;

    /** @brief True when the grains came from a voxel map (Source.type = "voxel_map"). */
    bool used_voxel_map = false;
    /** @brief The voxel grid (dimensions, origin and spacing in mesh units, id dataset path). */
    VoxelGridInfo voxel_grid;
    /** @brief Elements filled from their neighbors (no voxel vote; global). */
    std::int64_t voxel_num_filled = 0;
    /** @brief Fill rounds that assigned at least one element. */
    int voxel_fill_rounds = 0;
    /** @brief Sample points outside the voxel grid (global). */
    std::int64_t voxel_num_outside_points = 0;
    /** @brief Voxel tiles and voxels read by this rank (local counts: per-rank data). */
    std::int64_t voxel_tiles_read = 0;
    std::int64_t voxel_voxels_read = 0;
    /** @brief Voxel tiles and voxels read by all ranks together (global sums). */
    std::int64_t voxel_total_tiles_read = 0;
    std::int64_t voxel_total_voxels_read = 0;
    /** @brief Size num_grains: voxel feature id each final grain descends from, -1 for grains not
     * read from a voxel map. */
    std::vector<std::int64_t> grain_feature;

    /** @brief Regions.mode = "phase_fractions": the phase ids and the volume fractions achieved
     * (element volumes, regions/region_assigner.hpp). Empty otherwise. */
    std::vector<int> phase_fraction_ids;
    std::vector<double> phase_fraction_achieved;

    /** @brief True when [Mesh] listed active or passthrough attributes (an AttributeDomain). */
    bool used_attribute_domain = false;
    /** @brief True when the active elements formed a sub-mesh (some passthrough attribute). */
    bool used_submesh = false;
    /** @brief Active and passthrough attributes (ascending), the region of each passthrough
     * attribute and its global element count; same order. */
    std::vector<int> active_attributes;
    std::vector<int> passthrough_attributes;
    std::vector<int> passthrough_regions;
    std::vector<std::int64_t> passthrough_element_counts;
    /** @brief Global number of active (tessellated or voxel-labeled) elements. */
    std::int64_t num_active_elements = 0;
    /** @brief Passthrough grains appended after the active grains (one per attribute). */
    int num_passthrough_grains = 0;

    // ---- Stage 1e: distance modes for parts with holes and lattices -------------------------

    /** @brief The distance mode of the labeling (Tessellation.distance; EUCLIDEAN for voxel
     * maps). */
    DistanceMode distance = DistanceMode::EUCLIDEAN;
    /** @brief True when the connected mode ran (distance = "connected"). */
    bool used_connected = false;
    /**
     * @brief The connected-mode report (labeling/connected_labeler.hpp): relabeling rounds,
     * elements relabeled, orphans left unresolved after connected_max_rounds (merged by the
     * cleanup), elements of bodies without any seed anchor (left to the cleanup), seeds whose
     * cell does not reach their own anchor element (seeds without a cell). Global counts.
     */
    ConnectedLabelReport connected;
    /** @brief True when the geodesic mode ran (distance = "geodesic"). */
    bool used_geodesic = false;
    /** @brief The geodesic solver used: "graph" or "fim" (empty when the mode did not run). */
    std::string geodesic_solver;
    /**
     * @brief The geodesic solver's report (labeling/geodesic_common.hpp, GeodesicReport): rounds
     * of the distance and label phases (partition dependent for "graph", independent for "fim"),
     * seeds outside the material, unreached vertices and elements, vote-split elements,
     * unlabeled vertices. Global counts.
     */
    GeodesicReport geodesic;
    /**
     * @brief Geodesic mode: active elements that no seed reaches through material (bodies of the
     * part without a seed; GeodesicReport::unreached_elements: none of their vertices has a
     * finite distance). They take their straight-line (Euclidean) nearest-seed label, and the
     * fragment cleanup then splits each such body off into a grain of its own (a body with no
     * face neighbor is always split, even by "merge"). 0 for the other modes.
     */
    std::int64_t num_unreached_elements = 0;
    /**
     * @brief Geodesic mode: active elements that are reached but were left without a label by
     * the solver (GeodesicReport::unlabeled_elements: their winning vertex had no label; zero
     * unless the mesh is degenerate, e.g. element edges shorter than the rounding error of the
     * distance). They also take their straight-line nearest-seed label. 0 for the other modes.
     */
    std::int64_t num_unlabeled_elements = 0;

    /** @brief Warnings about the data (e.g. ignored table columns), also printed by rank 0. */
    std::vector<std::string> warnings;
};

/**
 * @brief Labels the elements of `pmesh` with grains and assigns orientations and regions.
 *
 * Pipeline (Stages 1a-1e):
 *  1. Support checks: options this pipeline does not implement are refused before any work.
 *  2. Preflight of the whole mesh (passthrough parts included): the mesh must be 3D, have one
 *     element geometry, be conforming, have no inverted elements and no periodic
 *     (discontinuous) Nodes (CheckMesh, mesh/preflight.hpp).
 *  2a. Straight-sided geometry (mesh/straight_sided.hpp): the pipeline works in place on
 *     `pmesh` (no copy). A mesh whose Nodes describe straight-sided elements (any order, e.g.
 *     ExaConstit's mesh after Mesh::SetCurvature) is recognized collectively, so seeding uses
 *     exact corner boxes and no padding of the sampling box: it draws the same seeds as the same
 *     mesh without Nodes. Element volumes, centroids and face areas are still evaluated through
 *     the Nodes and may differ from the no-Nodes mesh in the last bits, so labels agree up to
 *     round-off (ties). Genuinely curved meshes keep the curved treatment.
 *  2b. Active part ([Mesh] active_attributes / passthrough_attributes): an AttributeDomain
 *     (mesh/attribute_domain.hpp); seeding, labeling and cleanup run on its ActiveMesh() (a
 *     sub-mesh of the active elements when there are passthrough attributes).
 *  Voxel maps ([Source] type = "voxel_map") replace steps 3-8: ReadVoxelGridInfo,
 *     LabelElementsFromVoxels (centroid or vote sampling, fill of unassigned elements) and
 *     CompactFeatureLabels (labeling/voxel_labeler.hpp), then the cleanup of step 9.
 *  3. Seeds, replicated identically on every rank. A seed table (Seeds.source = "file",
 *     io/seed_table_io.hpp): rows outside Seeds.roi dropped, plus random filler seeds outside the
 *     ROI (seeding/roi_filler.hpp) when fill_outside_roi = "random"; table columns supply
 *     orientations, phases, weights (weighted metrics), volumes ([Seeds.Size] from_source:
 *     targets summing to the material inside the ROI with filler, else to the whole material;
 *     filler seeds share the outside volume equally) and shape tensors (anisotropy kind
 *     per_seed). Otherwise Johnson-Mehl nucleation
 *     (seeding/johnson_mehl.hpp, Additive weights a_i = -v t_i), or GenerateSeeds
 *     (seeding/seed_generators.hpp: rejection, element-weighted or lattice; with
 *     [Seeds.Size] polydisperse_placement, polydisperse RSA with exclusion radii
 *     kappa r_i, largest first). The seed count comes from num_seeds or, when it is 0, from the
 *     [Seeds.Size] mean grain size. Size targets (morphology/size_targets.hpp) give every seed a
 *     target volume; with the Laguerre metric the seeds start with weights w_i = r_i^2 (even
 *     without fit_volumes), unless a seed table supplies weights, which are then kept. Filler
 *     seeds next to table weights get the mean table weight, or with size targets (Laguerre)
 *     their own r_j^2 plus the mean of (table weight - r_i^2) over the table rows, so a common
 *     offset of the table weights cannot empty them.
 *  4. (Periodic images are built inside the labeling, step 6.)
 *  5. Lloyd relaxation (morphology/lloyd.hpp, optionally centroidal power) and/or the volume fit
 *     (morphology/weight_fit.hpp), when requested.
 *  6-8. Labels and compact grain ids, by Tessellation.distance:
 *     - "euclidean": LabelTessellation (labeling/tessellation_labeler.hpp): centroid or
 *       quadrature-vote sampling, periodic seed images mapped back to their originals, grains
 *       numbered 1..N in seed order. The Quadric tensor follows [Tessellation] anisotropy
 *       (columnar / lamellar / tensor) unless the seeds carry per-seed tensors.
 *     - "connected": the straight-line labels above, then ConstrainToConnectedCells
 *       (labeling/connected_labeler.hpp; connected_max_rounds, the same sampling), so every grain
 *       keeps only the elements connected through material to its seed's anchor element, then
 *       CompactSeedLabels. Orphans left after the last round and bodies without any anchor keep
 *       their labels and are handled by the cleanup of step 9.
 *     - "geodesic": path-length labels from the solver Tessellation.geodesic_solver ("fim", the
 *       default: LabelGeodesicFim, labeling/geodesic_fim.hpp; "graph": LabelGeodesicGraph,
 *       labeling/geodesic_graph.hpp), with the Additive
 *       offsets a_i = seed weights for metric = "additive" (Johnson-Mehl included), then
 *       CompactSeedLabels. Elements no seed reaches (a body of the part without a seed) take
 *       their straight-line nearest-seed label and are split off by the cleanup
 *       (GenerationReport::num_unreached_elements). Paths connect elements through shared
 *       vertices, edges and faces, while the cleanup is face based: a piece attached to its grain
 *       only through an edge or vertex is a separate fragment there and is split or merged by the
 *       [Cleanup] fragments policy (orchestrator decision: node-sharing elements are connected in
 *       the conforming FE mesh, so the geodesic labels are kept as they are). With Johnson-Mehl
 *       placement the nucleation step still discards a candidate covered by an earlier grain's
 *       straight-line growth sphere, even across a void; only the labels use path lengths, and a
 *       warning is recorded in GenerationReport::warnings.
 *     Both non-Euclidean modes run on the active mesh, so paths and connections never pass
 *     through passthrough parts; they refuse periodic axes, fit_volumes, Lloyd and voxel maps.
 *  9. Fragment cleanup (labeling/fragment_cleanup.hpp) over the face adjacency of the mesh; with
 *     periodic axes the adjacency pairs matching faces on opposite periodic planes, so a grain
 *     that wraps around the box is one grain. Policy from [Cleanup] (auto = split large pieces
 *     and merge tiny ones for generated seeds, merge every extra piece for data-driven sources:
 *     seed tables and voxel maps), keep rule (largest piece or the piece containing the seed),
 *     min_grain_elements. The connected and geodesic modes always keep the piece holding the
 *     element the mode grew the grain from (connected: the seed's anchor element,
 *     ConnectedAnchorKeys; geodesic: the seed's source element,
 *     SourceReport::source_element_key), i.e. keep_component is effectively "seed" for them: a
 *     leftover piece (an unresolved orphan, a body without a seed) never takes over the seed's
 *     identity, even when it is larger.
 * 10. Orientations: one quaternion per final grain. Random orientations are keyed by the grain's
 *     source id: the seed's id (the feature id for voxel grains), or for a grain split off by the
 *     cleanup a derived id (DeriveFragmentSourceId), so every piece gets its own reproducible
 *     orientation; with Orientations.source = "from_source" measured orientations (table
 *     quaternions, voxel feature quaternions) are taken from the parent seed or feature,
 *     normalized to unit length and otherwise passed through unchanged (convention, sign and
 *     component order kept), and grains without one (filler grains and their pieces) get random
 *     ones. Every row of MicrostructureResult::grain_quat has norm 1 to machine precision
 *     (checked before returning).
 * 11. Phases: from the table / voxel features (filler grains: Regions.filler_phase), or synthetic
 *     phases with prescribed fractions (AssignPhaseFractions on fixed-point element volumes).
 * 12. Regions: "single" (every grain region 1), or AssignRegionsFromPhases
 *     (regions/region_assigner.hpp); grains of non-crystalline regions get the identity
 *     quaternion (1, 0, 0, 0).
 * 13. Passthrough: labels mapped back to the whole mesh (AttributeDomain::ToParent) and one grain
 *     appended per passthrough attribute (identity quaternion, its region, phase -1, source id
 *     -attribute). The phase regions and passthrough regions together must be exactly 1..R.
 *
 * Collective over `comm`: every rank of `comm` must call it with identical `options`. The result
 * does not depend on the number of ranks: element labels of the same element are identical for
 * any partition of the same mesh. Runs on the host; kernels inside (mfem::forall, and the Axom
 * BVH through DispatchAxomExec) run on the backend of the mfem::Device, which the caller must
 * have configured from `options.exec` with ConfigureMfemDevice (exec_dispatch.hpp).
 *
 * @param options validated options (MicrostructureOptions::validate() returned true).
 * @param pmesh   this rank's partition of the mesh. Its geometry and attributes are not
 *                modified, but MFEM builds caches on demand when element geometry is evaluated:
 *                the mesh keeps the GeometricFactors object (mesh.cpp:887-899), and the nodal
 *                FE space keeps a QuadratureInterpolator (its output layout set to byNODES) and
 *                an element restriction (mesh.cpp:15251-15261); the face adjacency also calls
 *                ParMesh::ExchangeFaceNbrData. Mesh::GetGeometricFactors is therefore a
 *                non-const member (mesh.hpp:1973), hence the non-const reference.
 * @param comm    communicator of `pmesh`.
 * @param report  optional; filled identically on every rank. Ranks may differ in whether they
 *                pass it (the global reductions for the report run either way).
 * @return per-local-element grain ids and the replicated per-grain tables (see
 *         MicrostructureResult).
 * @throws std::runtime_error if the mesh fails preflight, seeding fails (infeasible minimum
 *         distance, stall, empty mesh, no nucleus), or a seed table / voxel map cannot be read;
 *         std::invalid_argument for options this pipeline does not implement yet ("list"
 *         orientations), an unknown geodesic solver or a connected rounds cap outside 0..1000,
 *         combinations outside the Stage 1e scope (see TessellationOptions::distance), invalid
 *         combinations reaching the
 *         driver, data that the options need but the source lacks (e.g. from_source
 *         orientations without quaternions), and region ids that are not exactly 1..R. Every
 *         check is either made on identical data on all ranks or globally reduced, so all ranks
 *         throw together.
 */
MicrostructureResult GenerateMicrostructure(const MicrostructureOptions& options,
                                            mfem::ParMesh& pmesh,
                                            MPI_Comm comm,
                                            GenerationReport* report = nullptr);

} // namespace exaconstit::microstructure
