#pragma once

#include "microstructure/microstructure_options.hpp"
#include "microstructure/microstructure_types.hpp"
#include "microstructure/seeding/seed_generators.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <mpi.h>

/**
 * @file microstructure_driver.hpp
 * @brief Entry point shared by the standalone tool and the mechanics hook.
 */

namespace exaconstit::microstructure {

/**
 * @brief What GenerateMicrostructure did, for logs and provenance. Identical on every rank
 * (all counts are global). Host data.
 */
struct GenerationReport {
    /** @brief The seed generator's report (requested/achieved seeds, candidates, fallbacks).
     * SeedingReport::Summary() gives a multi-line text for the log. */
    SeedingReport seeding;
    /** @brief Number of seeds that were labeled (= seeding.achieved). */
    std::int64_t num_seeds = 0;
    /** @brief Final number of grains: seeds that won at least one element. */
    int num_grains = 0;
    /** @brief Seeds that won no element (num_seeds - num_grains): seeds in parts of the sampling
     * domain too thin to hold an element centroid, coincident Voronoi seeds, or empty weighted
     * cells. */
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
 * Pipeline (Stage 1a):
 * 1. Preflight: the mesh must be 3D, have one element geometry, be conforming, have no inverted
 *    elements and no periodic (discontinuous) Nodes (CheckMesh, mesh/preflight.hpp).
 * 2. Seeds: GenerateSeeds (seeding/seed_generators.hpp) with the parameters of [Seeds]
 *    (MakeSeedingParameters), replicated identically on every rank.
 * 3. Labels: every local element gets the grain of the seed with the smallest metric value at
 *    its volume centroid; grains that received elements are numbered 1..N in seed order
 *    (LabelTessellation, labeling/tessellation_labeler.hpp). For the Quadric metric the global
 *    tensor Tessellation.global_tensor is used unless the seeds carry per-seed tensors
 *    (SeedSet::shape non-empty), which then silently replace it.
 * 4. Orientations: one quaternion per grain, random (keyed by the seed's source id) or passed
 *    through from the seed source (AssignOrientations, orientation/orientation_assigner.hpp).
 * 5. Regions: one region (region 1 for every grain); Regions.mode = "single" only so far.
 * Fragment cleanup (splitting grains cut into pieces, merging tiny grains) is not part of this
 * pipeline yet: Cleanup.fragments = "auto" or "none" and min_grain_elements <= 1 are accepted
 * and change nothing; explicit "split"/"merge" or min_grain_elements > 1 are rejected.
 *
 * Options this pipeline does not implement are refused before any work (before the preflight
 * and the collective seeding): see @throws. Unsupported seed placements (Johnson-Mehl, file) are
 * refused by the seed generator's own parameter check at the start of seeding, before any
 * communication.
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
 *                an element restriction (mesh.cpp:15251-15261). Mesh::GetGeometricFactors is
 *                therefore a non-const member (mesh.hpp:1973), hence the non-const reference.
 * @param comm    communicator of `pmesh`.
 * @param report  optional; filled identically on every rank. Ranks may differ in whether they
 *                pass it (the global reduction for the report runs either way).
 * @return per-local-element grain ids and the replicated per-grain tables (see
 *         MicrostructureResult).
 * @throws std::runtime_error if the mesh fails preflight or seeding fails (infeasible minimum
 *         distance, stall, empty mesh); std::invalid_argument for options this pipeline does not
 *         implement yet (seed tables, Johnson-Mehl seeds, quadrature voting, connected/geodesic
 *         distance, periodic grains, "list" orientations, multi-region modes, explicit fragment
 *         cleanup). Every check is either made on identical data on all ranks or globally
 *         reduced, so all ranks throw together.
 */
MicrostructureResult GenerateMicrostructure(const MicrostructureOptions& options,
                                            mfem::ParMesh& pmesh,
                                            MPI_Comm comm,
                                            GenerationReport* report = nullptr);

} // namespace exaconstit::microstructure
