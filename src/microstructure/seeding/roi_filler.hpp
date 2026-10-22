#pragma once

#include "microstructure/microstructure_types.hpp"
#include "microstructure/seeding/seed_generators.hpp"

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <mpi.h>

/**
 * @file roi_filler.hpp
 * @brief Random "filler" seeds for the material outside a measured region of interest (ROI).
 *
 * **Use case.** An HEDM (or similar) measurement often covers only part of the specimen: the
 * region of interest, an axis-aligned box. Its seeds come from a seed table
 * (io/seed_table_io.hpp, usually trimmed with KeepSeedsInBox). The rest of the mesh still needs
 * grains, so it is filled with random seeds that never fall inside the ROI box. By default the
 * filler grains get the same mean grain volume as the measured ones:
 *
 *     num_filler = round(num_roi_seeds * V_outside / V_roi),
 *
 * where V_roi is the material volume inside the box and V_outside the material volume outside it.
 *
 * **Volumes.** V_roi and V_outside are integrals of the indicator of the box over the material,
 * evaluated with the element quadrature points (mesh/mesh_sampler.hpp): a point q with weight
 * w_q detJ_q counts toward V_roi when it lies strictly inside the box, otherwise toward V_outside.
 * Elements cut by the box faces are therefore split approximately (to the resolution of their
 * quadrature points); elements entirely on one side are counted exactly. The sums use fixed-point
 * arithmetic (util/fixed_point.hpp: each term is rounded to an integer multiple of a tiny quantum
 * and the integers are added, which does not depend on the order of the additions), so both
 * volumes are bitwise identical for any number of ranks.
 *
 * **Seeds.** The filler seeds come from the default rejection sampler (seed_generators.hpp) with
 * the ROI box as its exclusion box, so a candidate strictly inside the box is never kept and the
 * result does not depend on the rank count. Their source ids start at source_id_offset (the
 * caller passes max table id + 1, so filler ids never collide with measured ones). The generator
 * is also told the outside fraction V_outside / (V_roi + V_outside), so its automatic stall cap
 * expects the right acceptance even when only a thin shell of material lies outside the box. With
 * min_distance_factor > 0 the filler seeds keep a minimum spacing
 * d = min_distance_factor * (V_outside / num_filler)^(1/3) from each other (not from the ROI
 * seeds).
 */

namespace exaconstit::microstructure {

/**
 * @brief Inputs of FillOutsideRoi. Lengths in mesh units.
 */
struct RoiFillerParameters {
    std::array<double, 3> lo{}; ///< ROI box lower corner (hi > lo on every axis)
    std::array<double, 3> hi{}; ///< ROI box upper corner
    /** @brief Number of filler seeds; 0 = automatic (same mean grain volume as the ROI). */
    std::int64_t num_filler = 0;
    /** @brief Key of the counter-based RNG of the filler positions. */
    std::uint64_t rng_seed = 0;
    /**
     * @brief source_id of the first filler candidate, in [0, 2^63 - 1 - 2^40]; usually max table
     * id + 1. Filler ids are offset + candidate index, with candidate indices below 2^40.
     */
    std::int64_t source_id_offset = 0;
    /** @brief RSA spacing factor among filler seeds (0 = none), relative to (V_out / N)^(1/3). */
    double min_distance_factor = 0.0;
};

/**
 * @brief Output of FillOutsideRoi, identical on every rank.
 */
struct RoiFillerResult {
    /** @brief Filler seeds (positions and source_id only), in increasing source_id order. */
    SeedSet filler;
    double roi_material_volume = 0.0;     ///< V_roi: material volume strictly inside the box
    double outside_material_volume = 0.0; ///< V_outside: the rest of the material volume
    std::int64_t num_requested = 0;       ///< filler seeds requested (given or automatic)
    /** @brief Report of the seed generator (all zero when no filler seed was requested). */
    SeedingReport report;
};

/**
 * @brief Measures the material inside and outside the ROI box and places filler seeds outside it.
 *
 * Steps: quadrature-point volumes V_roi and V_outside (fixed-point sums over all ranks); the
 * requested count (given, or the automatic formula above; 0 when V_outside = 0); then
 * GenerateSeeds with the ROI box as exclusion box and the id offset. No seeds are generated when
 * the count is 0.
 *
 * **Collective** over `comm`. Host function; the quadrature points are evaluated where the
 * mfem::Device runs kernels.
 *
 * @param local_mesh    this rank's part of the mesh (a ParMesh binds here; pass its
 *                      communicator). Not modified.
 * @param comm          communicator over which the mesh is distributed.
 * @param params        ROI box, count, RNG key, id offset, spacing; identical on every rank.
 * @param num_roi_seeds number of measured seeds in the ROI (>= 0; > 0 in automatic mode when the
 *                      outside volume is positive).
 * @return the filler seeds and the measured volumes.
 * @throws std::invalid_argument on every rank for invalid parameters (including a
 *         source_id_offset above 2^63 - 1 - 2^40, which leaves no room for the generator's
 *         candidate indices), or when the automatic count
 *         cannot be computed (V_roi = 0 or num_roi_seeds = 0 while V_outside > 0), or when filler
 *         seeds are requested but V_outside = 0; std::runtime_error (all ranks) from the sampler
 *         or the seed generator.
 */
RoiFillerResult FillOutsideRoi(mfem::Mesh& local_mesh,
                               MPI_Comm comm,
                               const RoiFillerParameters& params,
                               std::int64_t num_roi_seeds);

} // namespace exaconstit::microstructure
