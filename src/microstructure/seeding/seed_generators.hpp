#pragma once

#include "microstructure/microstructure_options.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

/**
 * @file seed_generators.hpp
 * @brief Seed placement on any hex or tet mesh: random points inside the material (rejection
 * sampling or element-volume-weighted sampling), optional minimum spacing (RSA) and surface
 * clearance, and regular lattices (cubic, BCC, FCC, HCP) clipped to the material.
 *
 * **The default "N random grains" path.** Rejection sampling in physical space: candidate point
 * k (k = 0, 1, 2, ...) is
 *
 *     c_k = lo + U_k (hi - lo)   (component-wise),
 *
 * where [lo, hi] is the sampling box and U_k is a uniform point of the unit cube computed by the
 * counter-based RNG (random numbers computed from (rng_seed, purpose, id), so they don't depend on
 * call order or rank count) with key (rng_seed, RngPurpose::SEED_POSITION, k). A candidate is
 * kept if it lies inside the material and thrown away otherwise ("rejection sampling"). Every
 * point of the material is equally likely, whatever the mesh shape (box, lattice, tomography),
 * because the candidates are uniform in the box and the material is a subset of the box.
 *
 * **Parallel test, identical answer everywhere.** Candidates are processed in batches. Each rank
 * tests the batch against its own elements (a BVH, i.e. a bounding volume hierarchy: a tree of
 * nested bounding boxes that lets a point skip every element whose box provably does not contain
 * it; then an exact point-in-element test, see material_point_locator.hpp) and sets bit i of a
 * bitmask (an array of
 * 64-bit words, one bit per candidate) when candidate i is inside one of its elements. An
 * MPI_Allreduce with the bitwise OR operation (MPI_BOR) then gives every rank the same mask: bit i
 * is set if candidate i is inside the material on any rank. All ranks then walk the batch in
 * candidate order and accept the same candidates, so the seeds are identical on every rank and do
 * not depend on the number of ranks or on how the mesh is partitioned. Uniform refinement of a
 * straight-sided mesh does not change the material, so ref_ser = 0 and 1 give the same seeds too,
 * except for candidates that lie within the inside-test tolerance of the outer surface: about
 * 1e-12 (relative barycentric tolerance) for straight-sided tets, and 1e-8 in reference
 * coordinates, i.e. about 1e-8 of an element size, for hexes and curved elements. The chance that
 * one of the accepted candidates falls in that thin layer is of the order of (layer thickness x
 * surface area / volume) per seed.
 *
 * **Collective** (used below): a function that every rank of the communicator must call, in the
 * same order, because it exchanges data between ranks (MPI_Allreduce and the like); a rank that
 * skips it or throws early leaves the others waiting forever.
 *
 * **Sampling box.** By default the global mesh bounding box (for curved elements, the union of the
 * padded element boxes, so bulges are reachable), "rounded outward": each corner is
 * moved out to a multiple of a power-of-two length about 1/64 to 1/128 of the box size. The
 * candidates then do not change when the bounding box moves by round-off (for example a curved
 * mesh whose box is estimated from sampled points), which keeps the seeds reproducible. A user
 * sampling box replaces it (seeds then fill the part of the material inside that box).
 *
 * **Minimum spacing (RSA).** With a minimum distance d > 0, a candidate inside the material is
 * also rejected if it is closer than d to an already accepted seed (random sequential adsorption,
 * tested with a hash grid, see min_distance_grid.hpp). *Polydisperse* RSA gives every seed slot
 * its own exclusion radius (SeedingParameters::rsa_radius) and requires non-overlapping exclusion
 * spheres instead (the guard below then uses the sum of the sphere volumes). Guards:
 * - feasibility: N (pi/6) d^3 <= 0.3 V is required, i.e. N non-overlapping spheres of diameter d
 *   may fill at most 30% of the volume V. Random sequential packing jams near 38%, and the last
 *   few percent before that take extremely many candidates, so larger requests are refused up
 *   front with an error;
 * - stall cap: after a maximum number of candidates generation stops, and the error (or, with
 *   allow_fewer_seeds, the report) says how many seeds were placed.
 *
 * **Element-volume-weighted fallback.** When the material fills less than 1e-4 of the sampling
 * box (a sparse lattice in a large box), rejection sampling would waste more than 10^4 candidates
 * per seed. Points are then generated element by element instead: every element emits points at
 * random "arrival times" whose rate is proportional to its volume (a Poisson process in space and
 * time), and the points are taken in order of arrival time. Ordering the points of such a process
 * by time gives independent, uniformly distributed points in the material, and since each element
 * draws from its own RNG stream keyed by its position (purpose ELEMENT_WEIGHTED_POSITION, id = a
 * hash of the element centroid), the result again does not depend on the rank count. (It does
 * depend on the elements, so ref_ser 0 and 1 give different seeds here.)
 *
 * **Lattices.** Seeds on a cubic, body-centered cubic, face-centered cubic or hexagonal
 * close-packed lattice whose spacing gives about num_seeds lattice points in the material volume,
 * optionally jittered, and clipped to the material.
 *
 * **Seed order and ids.** Seeds are stored in increasing candidate order, and
 * SeedSet::source_id of a seed is its candidate index: k for rejection sampling, the lattice
 * index for lattices, and the position in the arrival-time order for element-weighted sampling.
 * The id of a seed therefore does not change when other candidates are rejected (by RSA, the
 * clearance or a different num_seeds), so orientations keyed by source_id stay attached to the
 * same seed. For a box mesh without RSA every candidate is accepted and source_id = 0..N-1.
 */

namespace exaconstit::microstructure {

/**
 * @brief Everything GenerateSeeds needs. Build it from the options with MakeSeedingParameters and
 * adjust the fields that have no TOML key (the stall and batch controls max_candidates,
 * fallback_acceptance, max_batch_size) if needed. All lengths are in mesh length units.
 */
struct SeedingParameters {
    /** @brief REJECTION (default), ELEMENT_WEIGHTED or LATTICE. */
    SeedPlacement placement = SeedPlacement::REJECTION;
    /**
     * @brief Requested number of seeds N (1 .. 2^31 - 1). For lattices it only sets the spacing,
     * and the number of lattice points that fit in the material is returned.
     */
    std::int64_t num_seeds = 0;
    /** @brief Key of the counter-based RNG; the seeds depend only on it, the mesh and N. */
    std::uint64_t rng_seed = 0;
    /** @brief Absolute RSA minimum distance d (> 0 enables RSA and overrides the factor). */
    double min_distance = 0.0;
    /** @brief d = min_distance_factor * (V / N)^(1/3) when min_distance is 0 (0 disables RSA). */
    double min_distance_factor = 0.0;
    /**
     * @brief Polydisperse RSA: exclusion radius rho_j of the j-th accepted seed (acceptance order
     * = increasing candidate index), size num_seeds, every entry > 0 and finite. A candidate for
     * slot j is rejected when it is closer than rho_j + rho_i to an accepted seed i, i.e. when
     * the exclusion spheres would overlap. Sorting the radii from largest to smallest places the
     * largest grains first (the usual polydisperse RSA). Empty (default) = monodisperse RSA with
     * min_distance / min_distance_factor, which must both be 0 when this is used. Not for
     * lattices. Feasibility guard: sum_j (4/3) pi rho_j^3 <= 0.3 V. No TOML key: the driver fills
     * it from [Seeds.Size] (kappa x equivalent-sphere radius of the target volumes).
     */
    std::vector<double> rsa_radius;
    /** @brief Minimum distance of a seed from the material surface (0 disables the check). */
    double boundary_clearance = 0.0;
    /**
     * @brief "material": seeds only inside the elements (default); "bbox": anywhere in the
     * sampling box, which is then the exact (not rounded) mesh bounding box unless a user box is
     * given.
     */
    std::string sampling_domain = "material";
    /** @brief Use [sampling_box_lo, sampling_box_hi] instead of the mesh bounding box. */
    bool use_sampling_box = false;
    std::array<double, 3> sampling_box_lo = {0.0, 0.0, 0.0}; ///< user box lower corner
    std::array<double, 3> sampling_box_hi = {0.0, 0.0, 0.0}; ///< user box upper corner
    /**
     * @brief Periodic axes (Tessellation.periodic): the RSA distance is measured to the nearest
     * periodic image, with the exact mesh bounding box as the period.
     */
    std::array<bool, 3> periodic = {false, false, false};
    /** @brief LATTICE only: "cubic", "bcc", "fcc" or "hcp". */
    std::string lattice_type = "cubic";
    /**
     * @brief LATTICE only: each point moves by an independent uniform offset in
     * [-jitter a / 2, jitter a / 2) per axis, a = lattice parameter (>= 0).
     */
    double lattice_jitter = 0.0;
    /**
     * @brief When the stall cap is reached before N seeds are placed: false (default) throws
     * std::runtime_error; true returns the seeds placed so far and notes it in the report.
     */
    bool allow_fewer_seeds = false;
    /**
     * @brief Stall cap: maximum number of candidates to generate (rejection, lattice) or of
     * in-material points to draw (element-weighted). 0 selects
     * max(2^22, 1000 N / p), p = estimated fraction of candidates inside the material.
     */
    std::int64_t max_candidates = 0;
    /**
     * @brief Rejection sampling switches to element-weighted sampling when the fraction of
     * candidates inside the material is below this value (estimated from volumes up front, and
     * measured after 2^20 candidates). 0 disables the fallback.
     */
    double fallback_acceptance = 1.0e-4;
    /** @brief Largest batch of candidates tested at once (memory/performance only, 64..2^24). */
    std::int64_t max_batch_size = std::int64_t(1) << 20;
};

/**
 * @brief Copies the [Seeds] options (placement, num_seeds, rng_seed, min_distance_factor,
 * min_distance, boundary_clearance, sampling_domain, sampling_box, allow_fewer_seeds,
 * lattice_type, lattice_jitter) and Tessellation.periodic into SeedingParameters. Host only,
 * never throws.
 */
SeedingParameters MakeSeedingParameters(const MicrostructureOptions& options);

/**
 * @brief Checks a parameter set; called by GenerateSeeds. Host only, not collective.
 * @throws std::invalid_argument naming the first invalid field (N <= 0, unsupported placement,
 *         unknown sampling_domain or lattice_type, negative distances, empty user box, ...).
 */
void ValidateSeedingParameters(const SeedingParameters& params);

/**
 * @brief Moves the corners of [lo, hi] outward to multiples of q, a power of two between 1/128
 * and 1/64 of the largest edge (see the file comment). Host only; exact (power-of-two
 * arithmetic). A degenerate box gets a positive extent of at least q.
 */
void RoundBoxOutward(std::array<double, 3>& lo, std::array<double, 3>& hi);

/**
 * @brief What GenerateSeeds did: identical on every rank. Counts are global.
 */
struct SeedingReport {
    std::int64_t requested = 0;           ///< SeedingParameters::num_seeds
    std::int64_t achieved = 0;            ///< seeds returned
    SeedPlacement placement_used = SeedPlacement::REJECTION; ///< after any fallback
    bool used_fallback = false;           ///< rejection switched to element-weighted
    bool stalled = false;                 ///< the stall cap was reached before N seeds
    std::int64_t candidates_generated = 0;  ///< candidates/lattice points/arrivals examined
    std::int64_t candidates_in_domain = 0;  ///< of those, inside the material (or box)
    std::int64_t rejected_by_clearance = 0; ///< in domain but closer than the clearance
    std::int64_t rejected_by_distance = 0;  ///< in domain but closer than d to a seed (RSA)
    double material_volume = 0.0;         ///< V of the whole mesh (fixed-point sum, length^3)
    double min_distance = 0.0;            ///< d actually used (0: no RSA)
    double lattice_parameter = 0.0;       ///< a (lattices only)
    std::array<double, 3> bbox_lo = {0, 0, 0};     ///< exact global mesh bounding box
    std::array<double, 3> bbox_hi = {0, 0, 0};
    std::array<double, 3> sampling_lo = {0, 0, 0}; ///< box the candidates were drawn from
    std::array<double, 3> sampling_hi = {0, 0, 0};
    std::vector<std::string> notes;       ///< human-readable remarks (fallback, partial result)

    /** @brief Multi-line summary for the rank-0 log. Host only. */
    std::string Summary() const;
};

/**
 * @brief Generates seeds in the material described by the distributed mesh.
 *
 * **Collective** over `comm`: every rank calls it with its own partition (an empty partition is
 * allowed) and identical parameters, and every rank returns the identical SeedSet (positions and
 * source_id filled; the other arrays empty). Runs on the host (the per-element volume/centroid
 * sampler follows mfem::Device; everything else is host code).
 *
 * @param local_mesh this rank's part of the mesh (3D hex or tet; for a ParMesh pass it or use the
 *        overload below). Non-const only because the element sampler needs it; not modified.
 *        A serial mfem::Mesh is fine with a single-rank communicator; with several ranks each must
 *        hold a different piece, otherwise volumes are counted several times.
 * @param comm   communicator over which the mesh is distributed.
 * @param params seeding parameters (validated here).
 * @param report optional; filled on every rank.
 * @return the seeds, in increasing source_id (candidate) order.
 * @throws std::invalid_argument for invalid parameters (checked on identical data on every
 *         rank); std::runtime_error for an empty mesh, elements with volume <= 0, an infeasible
 *         minimum distance, a stall (unless allow_fewer_seeds), or a failure of the local setup
 *         (element boxes, element sampler, surface triangles) on any rank. Local setup failures
 *         are caught on the failing rank and broadcast, and every other condition is decided from
 *         globally reduced data, so all ranks throw together with the same message.
 */
SeedSet GenerateSeeds(mfem::Mesh& local_mesh, MPI_Comm comm, const SeedingParameters& params,
                      SeedingReport* report = nullptr);

/** @brief GenerateSeeds over the ParMesh's own communicator. Collective. */
SeedSet GenerateSeeds(mfem::ParMesh& pmesh, const SeedingParameters& params,
                      SeedingReport* report = nullptr);

} // namespace exaconstit::microstructure
