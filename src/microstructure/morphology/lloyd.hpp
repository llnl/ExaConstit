#pragma once

#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/labeling/cell_measures.hpp"
#include "microstructure/microstructure_types.hpp"
#include "microstructure/morphology/weight_fit.hpp"

#include "mfem.hpp"

#include <mpi.h>
#include <vector>

/**
 * @file lloyd.hpp
 * @brief Lloyd relaxation: moves every seed to the center of mass of its grain, repeatedly, so
 * grains become rounder and more uniform (a centroidal Voronoi tessellation), optionally
 * combined with weight fitting so grains also keep prescribed volumes (centroidal power mode).
 *
 * **Lloyd's algorithm.** Random (Poisson) seeds give grains of very uneven size and shape.
 * Lloyd's algorithm alternates two steps:
 * 1. *label*: give every sample point of the material to its nearest seed (build the cells);
 * 2. *move*: put every seed at the volume centroid of its cell.
 * Each step lowers (or keeps) the "CVT energy"
 *
 *     E = sum_i integral over cell i of |x - s_i|^2 dx,
 *
 * the polar moment of inertia of every grain about its own seed, summed over grains: step 2
 * because the centroid is the point about which a body's moment of inertia is smallest, step 1
 * because every point is moved to the seed nearest to it. (This is the same alternation as
 * k-means clustering.) A tessellation in which every seed already sits at its cell's centroid is
 * a centroidal Voronoi tessellation (CVT): equiaxed grains with a narrow size distribution, a
 * common idealization of recrystallized microstructures. The energy is evaluated with the same
 * samples as the cell measures, so with sampled cells the decrease holds in exact arithmetic. In
 * floating point the centroids come from fixed-point sums (each term rounded to a tiny quantum)
 * and the energy is itself a rounded fixed-point sum, so a step that has converged can raise the
 * reported energy by round-off, about 1e-16 relative; compare successive energies with a small
 * slack.
 *
 * **Non-convex material (lattices, notches).** The centroid of a cell that wraps around a hole or
 * a notch can lie outside the material, where a seed would be meaningless. After each move the
 * centroids are tested against the mesh (MaterialPointLocator, seeding/material_point_locator.hpp)
 * and an outside centroid is replaced by the sample point of that cell closest to it (the element
 * centroid when the samples are element centroids). Clamping can raise the energy slightly; the
 * report counts the clamped seeds.
 *
 * **Centroidal power mode.** With target volumes, each iteration first refits the Laguerre
 * weights for the current positions (morphology/weight_fit.hpp, warm-started from the previous
 * weights), then moves the seeds to the centroids of those power cells; a final refit after the
 * last move makes the returned weights match the returned positions. The result has both
 * prescribed volumes and seeds at their grain centers.
 *
 * **Determinism.** Centroids come from reproducible fixed-point sums (cell_measures.hpp), the
 * inside test is combined over ranks with a bitwise OR, and the clamping target is chosen by an
 * exact minimum with a coordinate tie-break, so every rank returns bitwise the same seeds for any
 * rank count.
 */

namespace exaconstit::microstructure {

/** @brief Controls of the Lloyd relaxation. Host data. */
struct LloydParameters {
    /** @brief Number of Lloyd steps (>= 0). */
    int iterations = 10;
    /**
     * @brief Metric of the cells: VORONOI, or LAGUERRE with the seeds' current weights.
     * Ignored in centroidal power mode (always LAGUERRE with refitted weights).
     */
    MetricType metric = MetricType::VORONOI;
    /** @brief Parameter bins of the seed search (performance only, >= 1). */
    int num_bins = 8;
    /** @brief Replace centroids outside the material by the closest sample of the cell. */
    bool clamp_to_material = true;
    /** @brief Alternate every Lloyd step with a weight refit (needs target volumes). */
    bool centroidal_power = false;
    /** @brief Controls of the refits in centroidal power mode. */
    WeightFitParameters weight_fit{};
};

/** @brief History of one relaxation. Host data, identical on every rank. */
struct LloydReport {
    /**
     * @brief CVT energy sum_q W_q |x_q - s_label(q)|^2 (length^5) of the cells before each step
     * and after the last one (size iterations + 1). Non-increasing for VORONOI without clamping.
     */
    std::vector<double> energy;
    /** @brief Largest seed displacement of each step (length). */
    std::vector<double> max_displacement;
    /** @brief Number of seeds clamped back into the material in each step. */
    std::vector<int> num_clamped;
    /** @brief Number of empty cells (seeds that did not move) in each step. */
    std::vector<int> num_empty;
    /** @brief Weight-fit reports of centroidal power mode (iterations + 1 refits), else empty. */
    std::vector<WeightFitReport> fits;
};

/**
 * @brief Checks the parameters.
 * @throws std::invalid_argument naming the first invalid field.
 */
void ValidateLloydParameters(const LloydParameters& params);

/**
 * @brief Runs `params.iterations` Lloyd steps (see the file comment) and returns the moved seeds.
 *
 * **Collective** over `comm`. Host function; labeling and accumulation run as kernels on the
 * mfem::Device backend, the inside test and the clamping search on the host.
 *
 * @param local_mesh    this rank's mesh partition (for the inside test); must be the mesh the
 *                      samples were taken from. Not modified.
 * @param comm          communicator over which the mesh and samples are distributed.
 * @param samples       this rank's weighted samples of the material (element or quadrature).
 * @param seeds         replicated input seeds. All fields are copied to the output; positions
 *                      change, and in centroidal power mode `weight` holds the fitted weights.
 *                      Seeds whose cell is empty stay where they are.
 * @param target_volume one target per seed for centroidal power mode (see FitWeights); ignored
 *                      otherwise (may be empty).
 * @param report        optional; receives the history.
 * @return the relaxed seeds (same order, same source ids).
 * @throws std::invalid_argument (on all ranks) for invalid parameters, missing targets in
 *         centroidal power mode, and the errors of LabelSamples / CellMeasureAccumulator /
 *         FitWeights.
 */
SeedSet RunLloyd(const mfem::Mesh& local_mesh, MPI_Comm comm, const WeightedSamples& samples,
                 const SeedSet& seeds, const LloydParameters& params,
                 const std::vector<double>& target_volume, LloydReport* report = nullptr);

} // namespace exaconstit::microstructure
