#pragma once

#include "microstructure/labeling/fragment_cleanup.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/microstructure_options.hpp"
#include "microstructure/microstructure_types.hpp"
#include "microstructure/morphology/lloyd.hpp"
#include "microstructure/morphology/weight_fit.hpp"
#include "microstructure/seeding/johnson_mehl.hpp"
#include "microstructure/seeding/seed_generators.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <memory>
#include <mpi.h>
#include <vector>

/**
 * @file microstructure_driver.hpp
 * @brief Entry point shared by the standalone tool and the mechanics hook.
 */

namespace exaconstit::microstructure {

/**
 * @brief What GenerateMicrostructure did, for logs and provenance. Identical on every rank
 * (all counts are global), except `adjacency`, which describes this rank's mesh partition.
 * Host data.
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
     * @brief The face adjacency of the mesh partition that the fragment cleanup used (with
     * periodic face pairing when any axis is periodic). Shared so callers (the tool's grain
     * statistics) can reuse it instead of repeating the face-neighbor exchange and the periodic
     * gather. It refers to the `pmesh` passed to GenerateMicrostructure and is valid only while
     * that mesh lives and is not refined or repartitioned. Never null after a successful call.
     * Per-rank data (the only member that differs between ranks).
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
     * labeling used, before any periodic images) of the seed each final grain descends from. A
     * grain split off by the cleanup has its parent grain's seed.
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
};

/**
 * @brief Labels the elements of `pmesh` with grains and assigns orientations and regions.
 *
 * Pipeline (Stages 1a-1c):
 *  1. Support checks: options this pipeline does not implement are refused before any work.
 *  2. Preflight: the mesh must be 3D, have one element geometry, be conforming, have no inverted
 *     elements and no periodic (discontinuous) Nodes (CheckMesh, mesh/preflight.hpp).
 *  3. Seeds, replicated identically on every rank: Johnson-Mehl nucleation
 *     (seeding/johnson_mehl.hpp, Additive weights a_i = -v t_i), or GenerateSeeds
 *     (seeding/seed_generators.hpp: rejection, element-weighted or lattice; with
 *     [Seeds.Size] polydisperse_placement, polydisperse RSA with exclusion radii
 *     kappa r_i, largest first). The seed count comes from num_seeds or, when it is 0, from the
 *     [Seeds.Size] mean grain size. Size targets (morphology/size_targets.hpp) give every seed a
 *     target volume; with the Laguerre metric the seeds start with weights w_i = r_i^2.
 *  4. (Periodic images are built inside the labeling, step 6.)
 *  5. Lloyd relaxation (morphology/lloyd.hpp, optionally centroidal power) and/or the volume fit
 *     (morphology/weight_fit.hpp), when requested.
 *  6-8. Labels and compact grain ids (LabelTessellation, labeling/tessellation_labeler.hpp):
 *     centroid or quadrature-vote sampling, periodic seed images mapped back to their originals,
 *     grains numbered 1..N in seed order. The Quadric tensor follows [Tessellation] anisotropy
 *     (columnar / lamellar / tensor) unless the seeds carry per-seed tensors.
 *  9. Fragment cleanup (labeling/fragment_cleanup.hpp) over the face adjacency of the mesh; with
 *     periodic axes the adjacency pairs matching faces on opposite periodic planes, so a grain
 *     that wraps around the box is one grain. Policy from [Cleanup] (auto = split large pieces
 *     and merge tiny ones for generated seeds), keep rule (largest piece or the piece containing
 *     the seed), min_grain_elements.
 * 10. Orientations: one quaternion per final grain. Random orientations are keyed by the grain's
 *     source id: the seed's id, or for a grain split off by the cleanup a derived id
 *     (DeriveFragmentSourceId), so every piece gets its own reproducible orientation;
 *     orientations from a data source are copied from the parent seed.
 * 11. Regions: one region (region 1 for every grain); Regions.mode = "single" only so far.
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
 * @throws std::runtime_error if the mesh fails preflight or seeding fails (infeasible minimum
 *         distance, stall, empty mesh, no nucleus); std::invalid_argument for options this
 *         pipeline does not implement yet (seed tables, connected/geodesic distance, "list"
 *         orientations, multi-region modes) and invalid combinations reaching the driver. Every
 *         check is either made on identical data on all ranks or globally reduced, so all ranks
 *         throw together.
 */
MicrostructureResult GenerateMicrostructure(const MicrostructureOptions& options,
                                            mfem::ParMesh& pmesh,
                                            MPI_Comm comm,
                                            GenerationReport* report = nullptr);

} // namespace exaconstit::microstructure
