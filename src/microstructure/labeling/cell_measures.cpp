/*
 * Implementation of the cell measures (cell_measures.hpp): per-seed volume, centroid and sample
 * count, and whole-domain integrals, over weighted sample points, with reproducible sums.
 *
 * Terms used in this file:
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it (here: MPI_Allreduce, which combines one value per rank and gives the result
 *   to every rank).
 * - Fixed-point sum: every term is rounded once to a multiple of a small quantum and stored as
 *   the integer number of quanta (util/fixed_point.hpp). Integer addition is exact, so the total
 *   does not depend on the order of the additions, the thread schedule or the rank count.
 * - Atomic add (MFEM's AtomicAdd, general/backends.hpp:96; declared in the global namespace,
 *   not in mfem::): an addition to a memory location that
 *   many GPU or OpenMP threads may update at the same time without losing updates. On the GPU it
 *   is the hardware atomicAdd, which exists for `unsigned long long`, not for signed 64-bit
 *   integers; the accumulators are therefore unsigned. Unsigned addition wraps modulo 2^64, which
 *   is exactly two's-complement signed addition, so the signed fixed-point terms are cast to
 *   unsigned, summed, and the total is cast back. MPI_SUM on MPI_UNSIGNED_LONG_LONG wraps the
 *   same way. On the host, AtomicAdd is a real atomic only when MFEM is built with
 *   MFEM_USE_OPENMP (it then uses `#pragma omp atomic`); otherwise it is a plain `+=`. That is
 *   correct for serial host loops, but a RAJA-OpenMP device ("raja-omp") runs mfem::forall on
 *   several threads, so such builds need MFEM_USE_OPENMP (the ExaConstit install scripts set it
 *   together with RAJA OpenMP).
 */
#include "microstructure/labeling/cell_measures.hpp"

#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <float.h>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief The accumulator type of the kernels (see "Atomic add" in the file comment). */
using Accumulator = unsigned long long;

// ThrowIfAnyRank (std::invalid_argument with one message on every rank when any rank's check
// fails) comes from util/collective_errors.hpp.

/**
 * @brief Global sum of `count` accumulators: copies the device result to the host and adds the
 * per-rank values with MPI_Allreduce(MPI_SUM) on unsigned 64-bit integers. Collective.
 */
std::vector<Accumulator> AllreduceAccumulators(MPI_Comm comm,
                                              const mfem::Array<Accumulator>& acc) {
    const int count = acc.Size();
    std::vector<Accumulator> total(static_cast<std::size_t>(count), 0);
    const Accumulator* host = acc.HostRead();
    for (int k = 0; k < count; ++k) {
        total[static_cast<std::size_t>(k)] = host[k];
    }
    MPI_Allreduce(MPI_IN_PLACE, total.data(), count, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    return total;
}

/**
 * @brief Reinterprets a wrapped unsigned sum as the signed fixed-point total (two's complement;
 * the conversion is implementation-defined before C++20 but is two's complement on every
 * supported compiler, and C++20 makes it standard).
 */
std::int64_t ToSigned(Accumulator a) {
    return static_cast<std::int64_t>(a);
}

/**
 * @brief Throws std::invalid_argument unless `value` (a bound handed to FixedPointScale::ForRange)
 * lies in [1e-280, 1e280].
 *
 * ForRange computes 2^62 / (count x value) and rounds it to a power of two with frexp. For a
 * subnormal or extremely small value that quotient overflows to infinity (frexp of infinity is
 * unspecified), and for an extremely large value it underflows to zero; either would give a
 * meaningless quantum. Values in [1e-280, 1e280] keep the quotient finite and nonzero for any
 * sample count below 2^62. `value` is a global bound (identical on every rank), so all ranks throw
 * together.
 */
void RequireRepresentableScale(double value, const char* what) {
    if (!(value >= 1e-280) || !(value <= 1e280)) {
        throw std::invalid_argument(std::string("CellMeasureAccumulator: ") + what + " (" +
                                    std::to_string(value) +
                                    ") is outside [1e-280, 1e280]; the fixed-point sums cannot "
                                    "represent it. Rescale the mesh or the integrand.");
    }
}

/*
 * RequirePositiveSampleRule
 *
 * A quadrature point with a negative (or zero) weight would be a sample with negative volume: a
 * cell could then have a negative or non-monotone sampled volume, which breaks the concavity the
 * weight fit relies on. With MFEM v4.10 (required by ExaConstit) hexahedron rules are Gauss tensor
 * rules and always positive; tetrahedron rules are positive up to order
 * MAX_POSITIVE_TET_QUAD_ORDER (20), and higher orders fall back to Grundmann-Moller rules with
 * negative weights (fem/intrules.cpp, `default:` branch of TetrahedronIntegrationRule). Older MFEM
 * versions also had negative tet rules at orders 3, 4, 7 and 8, which the generic check below
 * catches. The caller passes the same (globally reduced) geometry and order on every rank, so
 * every rank throws together. Host only.
 */
void RequirePositiveSampleRule(mfem::Geometry::Type geometry, int order) {
    const std::string name = mfem::Geometry::Name[geometry];
    if (geometry == mfem::Geometry::TETRAHEDRON && order > MAX_POSITIVE_TET_QUAD_ORDER) {
        throw std::runtime_error(
            "MakeQuadratureWeightedSamples: tetrahedron quadrature order " +
            std::to_string(order) + " > " + std::to_string(MAX_POSITIVE_TET_QUAD_ORDER) +
            " is not available with positive weights (MFEM's positive tet rules stop at order " +
            std::to_string(MAX_POSITIVE_TET_QUAD_ORDER) +
            "); request a lower quad_order (the default 4p - 1 exceeds it for node order p >= 6)");
    }
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(geometry, order);
    for (int k = 0; k < rule.GetNPoints(); ++k) {
        if (!(rule.IntPoint(k).weight > 0.0)) {
            throw std::runtime_error(
                "MakeQuadratureWeightedSamples: MFEM's " + name + " quadrature rule of order " +
                std::to_string(order) +
                " has a non-positive weight; the microstructure generator needs MFEM v4.10 "
                "(all-positive simplex rules up to order " +
                std::to_string(MAX_POSITIVE_TET_QUAD_ORDER) + ")");
        }
    }
}

} // namespace

/*
 * MakeElementWeightedSamples / MakeQuadratureWeightedSamples
 *
 * Thin wrappers around the mesh sampler (mesh/mesh_sampler.hpp). The sampler's Vectors are moved
 * into the WeightedSamples with Vector::Swap (no copy); the data stays wherever the sampler wrote
 * it (device memory on a GPU run).
 */

void MakeElementWeightedSamples(mfem::Mesh& mesh, WeightedSamples& out) {
    ElementSamples element;
    ComputeElementSamples(mesh, -1, element);
    out.x.Swap(element.cx);
    out.y.Swap(element.cy);
    out.z.Swap(element.cz);
    out.weight.Swap(element.volume);
}

/*
 * MakeQuadratureWeightedSamples
 *
 * Steps:
 *  1. Resolve the order (DefaultSampleQuadOrder for a negative request) and note the element
 *     geometry (-1 on a rank without elements, which cannot know it).
 *  2. For a ParMesh, MPI_Allreduce(MAX) both: every rank then uses the same rule even when the
 *     local default orders differ (variable-order Nodes), so the samples, and with them every
 *     result, are independent of the partition, and the check in step 3 is rank-identical.
 *  3. Check that the rule has only positive weights (RequirePositiveSampleRule), then sample.
 */
void MakeQuadratureWeightedSamples(mfem::Mesh& mesh, int quad_order, WeightedSamples& out) {
    // Step 1: local order and geometry
    int resolved[2] = {quad_order, -1};
    if (quad_order < 0) {
        resolved[0] = DefaultSampleQuadOrder(mesh);
    }
    if (mesh.GetNE() > 0) {
        resolved[1] = static_cast<int>(mesh.GetElementBaseGeometry(0));
    }

    // Step 2: agree over the ranks of a distributed mesh
    auto* pmesh = dynamic_cast<mfem::ParMesh*>(&mesh);
    if (pmesh != nullptr) {
        MPI_Allreduce(MPI_IN_PLACE, resolved, 2, MPI_INT, MPI_MAX, pmesh->GetComm());
    }
    const int order = resolved[0];

    // Step 3: positivity check (skipped when no rank has elements: nothing is sampled)
    if (resolved[1] >= 0) {
        RequirePositiveSampleRule(static_cast<mfem::Geometry::Type>(resolved[1]), order);
    }
    QuadraturePointSamples qp;
    ComputeQuadraturePointSamples(mesh, order, 0, mesh.GetNE(), qp);
    out.x.Swap(qp.x);
    out.y.Swap(qp.y);
    out.z.Swap(qp.z);
    out.weight.Swap(qp.weight);
}

/*
 * CellMeasureAccumulator constructor
 *
 * The fixed-point quanta must be the same on every rank and large enough that no sum overflows,
 * so they are derived from global bounds:
 *  1. One host pass over the local samples (one-time setup, so a host loop over HostRead() data
 *     is acceptable): check sizes, finiteness and weight >= 0, and find the local bounding box
 *     and the largest weight.
 *  2. MPI_Allreduce: sample count (SUM), bounding box (MIN/MAX), largest weight (MAX) and the
 *     error flag (MAX). Minimum and maximum are exact operations, so every rank gets bitwise the
 *     same bounds.
 *  3. Volume terms W_q are at most max_weight; moment terms W_q (x_q - center) are at most
 *     max_weight x half the largest box extent (center = box center keeps them small).
 *     FixedPointScale::ForRange then picks the finest power-of-two quantum for which the sum of
 *     all samples' terms stays below 2^62. Per-cell sums are partial sums, so they are safe too.
 */
CellMeasureAccumulator::CellMeasureAccumulator(MPI_Comm comm, const WeightedSamples& samples,
                                               int num_seeds)
    : m_comm(comm), m_samples(samples), m_num_seeds(num_seeds) {
    if (num_seeds < 1) {
        throw std::invalid_argument("CellMeasureAccumulator: num_seeds must be >= 1");
    }
    // 5 accumulators per seed plus one error counter must fit an MPI count (int)
    if (num_seeds > (std::numeric_limits<int>::max() - 1) / 5) {
        throw std::invalid_argument("CellMeasureAccumulator: too many seeds");
    }

    // Step 1: local checks and bounds
    const int n = samples.Size();
    bool bad = samples.y.Size() != n || samples.z.Size() != n || samples.weight.Size() != n;
    double lo[3] = {DBL_MAX, DBL_MAX, DBL_MAX};
    double hi[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    double max_weight = 0.0;
    if (!bad && n > 0) {
        const double* coords[3] = {samples.x.HostRead(), samples.y.HostRead(),
                                   samples.z.HostRead()};
        const double* weight = samples.weight.HostRead();
        for (int q = 0; q < n && !bad; ++q) {
            for (int d = 0; d < 3; ++d) {
                const double c = coords[d][q];
                // !(|c| <= DBL_MAX) is true for NaN and for +-inf
                if (!(fabs(c) <= DBL_MAX)) {
                    bad = true;
                }
                lo[d] = fmin(lo[d], c);
                hi[d] = fmax(hi[d], c);
            }
            if (!(weight[q] >= 0.0) || !(weight[q] <= DBL_MAX)) {
                bad = true;
            }
            max_weight = fmax(max_weight, weight[q]);
        }
    }

    // Step 2: global reductions (all ranks reach them, also those with bad data)
    ThrowIfAnyRank(comm, bad,
                   "CellMeasureAccumulator: inconsistent sample array sizes, a non-finite sample "
                   "coordinate or weight, or a negative weight on at least one rank");
    long long local_count = n;
    long long global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, lo, 3, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, hi, 3, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &max_weight, 1, MPI_DOUBLE, MPI_MAX, comm);
    m_total_samples = global_count;
    m_max_weight = max_weight;
    if (global_count == 0) {
        return; // no samples anywhere: every measure is zero, the default scales are never used
    }

    // Step 3: fixed-point scales from the global bounds
    double half_extent = 0.0;
    for (int d = 0; d < 3; ++d) {
        m_center[d] = 0.5 * (lo[d] + hi[d]);
        half_extent = fmax(half_extent, 0.5 * (hi[d] - lo[d]));
    }
    // Degenerate cases (all weights zero, all samples at one point) still need positive bounds
    double weight_bound = 1.0;
    if (max_weight > 0.0) {
        weight_bound = max_weight;
    }
    double extent_bound = 1.0;
    if (half_extent > 0.0) {
        extent_bound = half_extent;
    }
    RequireRepresentableScale(weight_bound, "the largest sample weight");
    RequireRepresentableScale(weight_bound * extent_bound,
                              "the largest sample weight x half the sample extent");
    m_volume_scale = FixedPointScale::ForRange(0.0, weight_bound, global_count);
    m_moment_scale = FixedPointScale::ForRange(0.0, weight_bound * extent_bound, global_count);
}

/*
 * CellMeasureAccumulator::Measure
 *
 * Algorithm:
 *  1. Zero an accumulator array of 5 N + 1 unsigned 64-bit integers on the device. Layout:
 *     acc[5 i + 0] volume of cell i, acc[5 i + 1..3] first moments W (x - center) in x, y, z,
 *     acc[5 i + 4] sample count, acc[5 N] number of invalid labels.
 *  2. One mfem::forall over the samples: encode each term to fixed point and atomically add it to
 *     the accumulators of the sample's seed. The order in which threads add does not matter,
 *     because the additions are integer additions.
 *  3. MPI_Allreduce(MPI_SUM) of the whole array (AllreduceAccumulators): afterwards every rank
 *     holds the global integer sums.
 *  4. Decode on the host: V_i = volume sum x quantum, c_i = center + moment_i / V_i.
 * Scales and sample pointers are copied into local variables first: the kernel lambda captures
 * by value, and capturing `this` would hand a host pointer to a GPU kernel.
 */
CellMeasures CellMeasureAccumulator::Measure(const mfem::Array<int>& sample_seed) const {
    const int n = m_samples.Size();
    ThrowIfAnyRank(m_comm, sample_seed.Size() != n,
                   "CellMeasureAccumulator::Measure: label array size differs from the sample "
                   "count on at least one rank");

    const int num_seeds = m_num_seeds;
    const int num_acc = 5 * num_seeds + 1;

    // Step 1: zeroed accumulators where the kernels run
    mfem::Array<Accumulator> acc(num_acc);
    {
        Accumulator* acc_w = acc.Write();
        mfem::forall(num_acc, [=] MFEM_HOST_DEVICE(int k) { acc_w[k] = 0ULL; });
    }

    // Step 2: encode and accumulate every sample
    if (n > 0) {
        const FixedPointScale volume_scale = m_volume_scale;
        const FixedPointScale moment_scale = m_moment_scale;
        const double center_x = m_center[0];
        const double center_y = m_center[1];
        const double center_z = m_center[2];
        const double* x = m_samples.x.Read();
        const double* y = m_samples.y.Read();
        const double* z = m_samples.z.Read();
        const double* w = m_samples.weight.Read();
        const int* label = sample_seed.Read();
        Accumulator* sums = acc.ReadWrite();
        mfem::forall(n, [=] MFEM_HOST_DEVICE(int q) {
            const int s = label[q];
            if (s < 0 || s >= num_seeds) {
                AtomicAdd(sums[5 * num_seeds], 1ULL);
                return;
            }
            // Signed fixed-point terms, cast to unsigned for the wrap-around atomic add
            const Accumulator v = static_cast<Accumulator>(volume_scale.Encode(w[q]));
            const Accumulator mx =
                static_cast<Accumulator>(moment_scale.Encode(w[q] * (x[q] - center_x)));
            const Accumulator my =
                static_cast<Accumulator>(moment_scale.Encode(w[q] * (y[q] - center_y)));
            const Accumulator mz =
                static_cast<Accumulator>(moment_scale.Encode(w[q] * (z[q] - center_z)));
            AtomicAdd(sums[5 * s + 0], v);
            AtomicAdd(sums[5 * s + 1], mx);
            AtomicAdd(sums[5 * s + 2], my);
            AtomicAdd(sums[5 * s + 3], mz);
            AtomicAdd(sums[5 * s + 4], 1ULL);
        });
    }

    // Step 3: global integer sums, identical on every rank
    const std::vector<Accumulator> total = AllreduceAccumulators(m_comm, acc);
    if (total[static_cast<std::size_t>(5 * num_seeds)] != 0ULL) {
        throw std::invalid_argument("CellMeasureAccumulator::Measure: a sample label is outside "
                                    "[0, num_seeds) on at least one rank");
    }

    // Step 4: decode per seed (host; N entries)
    CellMeasures out;
    const auto num = static_cast<std::size_t>(num_seeds);
    out.volume.assign(num, 0.0);
    out.cx.assign(num, 0.0);
    out.cy.assign(num, 0.0);
    out.cz.assign(num, 0.0);
    out.sample_count.assign(num, 0);
    Accumulator volume_total = 0ULL;
    std::int64_t count_total = 0;
    for (std::size_t i = 0; i < num; ++i) {
        const std::int64_t volume_int = ToSigned(total[5 * i + 0]);
        volume_total += total[5 * i + 0];
        out.volume[i] = m_volume_scale.Decode(volume_int);
        out.sample_count[i] = ToSigned(total[5 * i + 4]);
        count_total += out.sample_count[i];
        if (out.volume[i] > 0.0) {
            out.cx[i] = m_center[0] + m_moment_scale.Decode(ToSigned(total[5 * i + 1])) /
                                          out.volume[i];
            out.cy[i] = m_center[1] + m_moment_scale.Decode(ToSigned(total[5 * i + 2])) /
                                          out.volume[i];
            out.cz[i] = m_center[2] + m_moment_scale.Decode(ToSigned(total[5 * i + 3])) /
                                          out.volume[i];
        }
    }
    // The total is the decoded integer sum of all cells, so it equals sum_i V_i up to one
    // rounding of the final decode, and it is bitwise the same on every rank
    out.total_volume = m_volume_scale.Decode(ToSigned(volume_total));
    out.total_samples = count_total;
    return out;
}

/*
 * CellMeasureAccumulator::Integrate
 *
 * The integrand changes from call to call (e.g. the metric value of every sample during weight
 * fitting), so the quantum is chosen per call:
 *  1. Kernel: t_q = W_q v_q, and count non-finite t_q with an atomic add.
 *  2. Largest |t_q| on this rank with Vector::Normlinf (an MFEM device reduction,
 *     linalg/vector.cpp:1028; fmax ignores NaN, which is why step 1 counts them separately), then
 *     one MPI_Allreduce(MPI_MAX) of {max |t|, error count}: maximum is exact, so all ranks agree.
 *  3. Check that the largest term is representable (RequireRepresentableScale), then the
 *     ForRange quantum for (global sample count) terms of that size; encode and atomically add
 *     every t_q into one accumulator; MPI_Allreduce(MPI_SUM); decode.
 */
double CellMeasureAccumulator::Integrate(const mfem::Vector& value) const {
    const int n = m_samples.Size();
    ThrowIfAnyRank(m_comm, value.Size() != n,
                   "CellMeasureAccumulator::Integrate: value size differs from the sample count "
                   "on at least one rank");

    // Step 1: terms and the non-finite counter
    mfem::Vector terms(n);
    terms.UseDevice(true);
    mfem::Array<Accumulator> bad_count(1);
    {
        Accumulator* bad_w = bad_count.Write();
        mfem::forall(1, [=] MFEM_HOST_DEVICE(int k) { bad_w[k] = 0ULL; });
    }
    if (n > 0) {
        const double* w = m_samples.weight.Read();
        const double* v = value.Read();
        double* t = terms.Write();
        Accumulator* bad = bad_count.ReadWrite();
        mfem::forall(n, [=] MFEM_HOST_DEVICE(int q) {
            const double term = w[q] * v[q];
            if (!(fabs(term) <= DBL_MAX)) {
                AtomicAdd(bad[0], 1ULL);
                t[q] = 0.0;
            } else {
                t[q] = term;
            }
        });
    }

    // Step 2: global largest term and error flag
    double reduce_buffer[2] = {0.0, 0.0};
    if (n > 0) {
        reduce_buffer[0] = terms.Normlinf();
    }
    if (bad_count.HostRead()[0] != 0ULL) {
        reduce_buffer[1] = 1.0;
    }
    MPI_Allreduce(MPI_IN_PLACE, reduce_buffer, 2, MPI_DOUBLE, MPI_MAX, m_comm);
    if (reduce_buffer[1] != 0.0) {
        throw std::invalid_argument("CellMeasureAccumulator::Integrate: a non-finite integrand "
                                    "value on at least one rank");
    }
    const double max_term = reduce_buffer[0];
    if (max_term == 0.0) {
        return 0.0; // every term is zero (or there are no samples), on every rank
    }

    // Step 3: reproducible sum of the terms
    RequireRepresentableScale(max_term, "the largest integrand term |W_q v_q|");
    const FixedPointScale scale = FixedPointScale::ForRange(0.0, max_term, m_total_samples);
    mfem::Array<Accumulator> acc(1);
    {
        Accumulator* acc_w = acc.Write();
        mfem::forall(1, [=] MFEM_HOST_DEVICE(int k) { acc_w[k] = 0ULL; });
    }
    if (n > 0) {
        const double* t = terms.Read();
        Accumulator* sum = acc.ReadWrite();
        mfem::forall(n, [=] MFEM_HOST_DEVICE(int q) {
            AtomicAdd(sum[0], static_cast<Accumulator>(scale.Encode(t[q])));
        });
    }
    const std::vector<Accumulator> total = AllreduceAccumulators(m_comm, acc);
    return scale.Decode(ToSigned(total[0]));
}

/*
 * ComputeCellMeasures: setup and one measurement, for callers that measure only once.
 */
CellMeasures ComputeCellMeasures(MPI_Comm comm, const WeightedSamples& samples,
                                 const mfem::Array<int>& sample_seed, int num_seeds) {
    const CellMeasureAccumulator accumulator(comm, samples, num_seeds);
    return accumulator.Measure(sample_seed);
}

} // namespace exaconstit::microstructure
