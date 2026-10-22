#pragma once

#include "microstructure/util/fixed_point.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <mpi.h>
#include <vector>

/**
 * @file cell_measures.hpp
 * @brief Volume, volume centroid and sample count of every grain cell ("cell measures"),
 * integrated over a set of weighted sample points of the mesh, with sums that are bitwise
 * identical for every MPI rank count.
 *
 * **What is measured.** A tessellation gives every point x of the material to one seed i (its
 * cell). Size control (morphology/weight_fit.hpp) needs the volume V_i of every cell, and Lloyd
 * relaxation (morphology/lloyd.hpp) needs its volume centroid c_i. Both are integrals over the
 * cell, evaluated with a quadrature rule over the mesh:
 *
 *     V_i = sum over samples q labeled i of W_q,
 *     c_i = (sum over samples q labeled i of W_q x_q) / V_i,
 *
 * where x_q is a sample point and W_q the volume it represents. Two kinds of samples are used:
 * - *element samples* (MakeElementWeightedSamples): one point per element, its volume centroid,
 *   with W = element volume. Labels of these points are exactly the element labels written to the
 *   output mesh, so V_i is then the volume of the grain as ExaConstit will see it. V_i changes in
 *   jumps of one whole element when a grain boundary moves.
 * - *quadrature samples* (MakeQuadratureWeightedSamples): every quadrature point of every element,
 *   with W = w_q detJ_q. V_i then changes in much smaller jumps (one quadrature point), which a
 *   weight-fitting solver needs to "feel" a small change of a weight: with element samples the
 *   sampled volume stays constant while a weight changes by a small amount and then jumps, so
 *   the solver sees a zero slope (a "plateau") and stalls; finer samples make the plateaus
 *   much narrower.
 *
 * **Reproducible sums.** Every global sum here uses fixed-point integers (util/fixed_point.hpp:
 * each term is rounded once to a multiple of a tiny quantum and stored as an int64 count of
 * quanta; integer sums do not depend on the order of the additions). The per-sample terms are
 * added with atomic adds inside an mfem::forall kernel (an atomic add is an addition that many
 * GPU threads can perform on the same memory location without losing updates), and the per-rank
 * totals are combined with MPI_Allreduce(MPI_SUM) over 64-bit integers. The result is bitwise the
 * same for any thread schedule and any number of ranks, so every rank holds identical cell
 * measures and every later decision based on them (a weight update, a seed move) is identical
 * too. The quanta are chosen from global bounds (MPI_MIN/MPI_MAX reductions, which are exact) so
 * that no sum can overflow; the resolution is about 2^-62 of the largest possible total, far
 * below any geometric tolerance.
 *
 * **Where it runs.** The per-sample kernels are mfem::forall loops on the backend of the
 * configured mfem::Device (host or GPU); results are small replicated host vectors (one entry
 * per seed). All functions that reduce over ranks are *collective*: every rank of the
 * communicator must call them, in the same order, because the ranks exchange data inside them.
 */

namespace exaconstit::microstructure {

/**
 * @brief Points that sample the material, each carrying the volume it represents.
 *
 * All four Vectors have the same size n (local to this rank); entry q describes sample q
 * (0-based). Coordinates are in mesh length units, `weight` in length^3, and the weights of all
 * samples of the mesh sum to the material volume. The data may be valid only on the device after
 * a GPU run; read it on the host with HostRead().
 */
struct WeightedSamples {
    /** @brief Sample coordinates, size n each. */
    mfem::Vector x, y, z;
    /** @brief Volume represented by each sample (element volume or w_q detJ_q), size n. */
    mfem::Vector weight;

    /** @brief Number of local samples n. Host. */
    int Size() const {
        return x.Size();
    }
};

/**
 * @brief One sample per local element: its volume centroid, weighted by its volume
 * (ComputeElementSamples, mesh/mesh_sampler.hpp, with the exact default quadrature order).
 *
 * Sample q is local element q, so labels of these samples are element labels. Not collective.
 * Host function launching mfem::forall kernels.
 *
 * @param mesh this rank's 3D mesh (a ParMesh binds here). Not modified.
 * @param out  overwritten; size = local element count.
 * @throws std::runtime_error from the element sampler for a mixed-geometry or non-3D mesh.
 */
void MakeElementWeightedSamples(mfem::Mesh& mesh, WeightedSamples& out);

/**
 * @brief Highest tetrahedron quadrature order whose MFEM v4.10 rule has only positive weights
 * (fem/intrules.cpp tabulates positive rules up to order 20; beyond that it falls back to
 * Grundmann-Moller rules, which contain negative weights).
 */
inline constexpr int MAX_POSITIVE_TET_QUAD_ORDER = 20;

/**
 * @brief Every quadrature point of every local element, weighted by w_q detJ_q
 * (ComputeQuadraturePointSamples, mesh/mesh_sampler.hpp).
 *
 * Sample (e * num_qp + q) is quadrature point q of local element e. Host function launching
 * mfem::forall kernels.
 *
 * **Collective when `mesh` is an mfem::ParMesh** (detected with dynamic_cast): the resolved order
 * and the element geometry are combined with MPI_Allreduce(MAX) over the ParMesh communicator, so
 * every rank samples with the same rule and the positivity check below gives the same answer on
 * every rank. This matters for variable-order Nodes, where DefaultSampleQuadOrder uses the
 * largest *local* element order, which can differ between ranks (and with the partition). For a
 * serial mfem::Mesh nothing is communicated.
 *
 * @param mesh       this rank's 3D mesh. Not modified.
 * @param quad_order order of mfem::IntRules.Get(geometry, order); a negative value selects
 *                   DefaultSampleQuadOrder (exact volumes and centroids). Higher orders give more,
 *                   smaller samples (a smoother volume response for weight fitting).
 * @param out        overwritten; size = local element count x points per element.
 * @throws std::runtime_error (on every rank for a ParMesh) if the element geometry's rule of the
 *         resolved order has a non-positive weight, i.e. a sample of negative volume. With MFEM
 *         v4.10 (ExaConstit's required version) hexahedron rules are always positive and
 *         tetrahedron rules are positive up to order MAX_POSITIVE_TET_QUAD_ORDER = 20; higher tet
 *         orders fall back to Grundmann-Moller rules with negative weights (reached by the default
 *         order 4p - 1 for node order p >= 6). Also from the sampler for a mixed-geometry or
 *         non-3D mesh.
 */
void MakeQuadratureWeightedSamples(mfem::Mesh& mesh, int quad_order, WeightedSamples& out);

/**
 * @brief Measures of every seed's cell. Host data, identical on every rank.
 *
 * Per-seed vectors have one entry per seed (0-based seed index, the order of the SeedSet).
 */
struct CellMeasures {
    /** @brief V_i: total sample weight labeled with seed i (length^3). 0 for an empty cell. */
    std::vector<double> volume;
    /**
     * @brief Volume centroid of cell i (length). Meaningful only where volume[i] > 0; set to 0
     * for empty cells.
     */
    std::vector<double> cx, cy, cz;
    /**
     * @brief Number of samples labeled with seed i (the element count for element samples).
     */
    std::vector<std::int64_t> sample_count;
    /** @brief Sum of all cell volumes, i.e. the material volume as seen by the samples. */
    double total_volume = 0.0;
    /** @brief Global number of samples. */
    std::int64_t total_samples = 0;
};

/**
 * @brief Reproducible per-cell and whole-domain integrals over one fixed set of samples.
 *
 * The constructor checks the samples and fixes the fixed-point scales once (it needs global
 * bounds, i.e. MPI reductions); the samples may then be relabeled and measured many times (one
 * call per solver iteration) without repeating that setup. Keeps a reference to the samples:
 * they must stay alive and unchanged while the accumulator is used.
 *
 * Every public function is **collective** over the communicator given to the constructor, and
 * every exception is thrown on all ranks together.
 */
class CellMeasureAccumulator {
public:
    /**
     * @brief Validates the samples and computes the fixed-point scales.
     *
     * Collective (MPI_Allreduce of the sample count, bounding box, largest weight and an error
     * flag).
     *
     * @param comm      communicator over which the samples are distributed.
     * @param samples   this rank's samples (may be empty on some ranks). Coordinates and weights
     *                  must be finite and weights >= 0.
     * @param num_seeds number of seeds N >= 1 (same on every rank); labels must be in [0, N).
     * @throws std::invalid_argument on all ranks if num_seeds < 1, any rank has a non-finite
     *         coordinate/weight or a negative weight, or the largest weight or weight x extent
     *         lies outside [1e-280, 1e280] (see Integrate).
     */
    CellMeasureAccumulator(MPI_Comm comm, const WeightedSamples& samples, int num_seeds);

    /**
     * @brief Volume, centroid and sample count of every cell for the given labels.
     *
     * Collective (an MPI_Allreduce over 5 N + 1 64-bit integers, the last one counting invalid
     * labels, plus one MPI_Allreduce(MAX) of the label-array size check). The
     * accumulation is an mfem::forall kernel with atomic adds.
     *
     * @param sample_seed size = local sample count; 0-based seed index of every sample.
     * @return the cell measures, bitwise identical on every rank and for every rank count.
     * @throws std::invalid_argument on all ranks if the label array has the wrong size or a
     *         label is outside [0, N) on any rank.
     */
    CellMeasures Measure(const mfem::Array<int>& sample_seed) const;

    /**
     * @brief Reproducible integral sum_q W_q v_q over all samples of all ranks.
     *
     * Collective (an MPI_Allreduce(MAX) of the size check, then two more: the largest |W_q v_q|
     * with an error flag to choose the quantum, then the integer sum).
     *
     * @param value size = local sample count; the integrand v at every sample (any units).
     * @return the integral, bitwise identical on every rank and for every rank count; its
     *         absolute resolution is about 2^-62 x (number of samples) x max |W_q v_q|.
     * @throws std::invalid_argument on all ranks if the size is wrong, a W_q v_q is not finite,
     *         or the largest nonzero |W_q v_q| lies outside [1e-280, 1e280] (the fixed-point
     *         quantum could not be represented; rescale the integrand).
     */
    double Integrate(const mfem::Vector& value) const;

    /** @brief Global number of samples. */
    std::int64_t TotalSamples() const {
        return m_total_samples;
    }
    /** @brief Largest sample weight W_q over all ranks (length^3). */
    double MaxSampleWeight() const {
        return m_max_weight;
    }
    /** @brief Number of seeds N the labels refer to. */
    int NumSeeds() const {
        return m_num_seeds;
    }

private:
    MPI_Comm m_comm;                    ///< communicator of the samples
    const WeightedSamples& m_samples;   ///< the samples (not owned)
    int m_num_seeds = 0;                ///< N
    std::int64_t m_total_samples = 0;   ///< global sample count
    double m_max_weight = 0.0;          ///< global max W_q
    double m_center[3] = {0, 0, 0};     ///< center of the global sample bounding box
    FixedPointScale m_volume_scale;     ///< quantum for W_q terms
    FixedPointScale m_moment_scale;     ///< quantum for W_q (x_q - center) terms
};

/**
 * @brief One-shot convenience: CellMeasureAccumulator(comm, samples, num_seeds).Measure(...).
 * Collective; same parameters and exceptions.
 */
CellMeasures ComputeCellMeasures(MPI_Comm comm, const WeightedSamples& samples,
                                 const mfem::Array<int>& sample_seed, int num_seeds);

} // namespace exaconstit::microstructure
