/*
 * Implementation of the Lloyd relaxation (lloyd.hpp).
 *
 * Terms used in this file:
 * - Collective: a function every rank must call, because the ranks exchange data inside it.
 * - Bitmask: an array of 64-bit integers used as yes/no flags, one bit per item: bit (i mod 64)
 *   of word i / 64 belongs to item i. Combining masks over ranks with a bitwise OR
 *   (MPI_Allreduce with MPI_BOR) answers "is this true on at least one rank?".
 * - User-defined MPI reduction (MPI_Op_create): a function MPI applies to combine the values of
 *   two ranks, here "keep the lexicographically smaller (distance, x, y, z) record". It is
 *   commutative and associative, so the result is the same for any rank count and reduction
 *   order.
 */
#include "microstructure/morphology/lloyd.hpp"
#include "microstructure/mesh/straight_sided.hpp"

#include "microstructure/morphology/sample_labels.hpp"
#include "microstructure/seeding/material_point_locator.hpp"

#include "mfem.hpp"

#include <cmath>
#include <cstdint>
#include <float.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/**
 * @brief CVT energy sum_q W_q |x_q - s_label(q)|^2 of the given labels. Collective (one
 * reproducible integral).
 *
 * A kernel evaluates |x_q - s|^2 for every sample with its seed's position (copied into
 * mfem::Vectors so the kernel can read them on the device), and the accumulator integrates it
 * with fixed-point sums.
 */
double CvtEnergy(const CellMeasureAccumulator& accumulator, const WeightedSamples& samples,
                 const SeedSet& seeds, const mfem::Array<int>& labels) {
    const int num_seeds = static_cast<int>(seeds.Size());
    mfem::Vector seed_x(num_seeds), seed_y(num_seeds), seed_z(num_seeds);
    for (int i = 0; i < num_seeds; ++i) {
        const auto s = static_cast<std::size_t>(i);
        seed_x[i] = seeds.x[s];
        seed_y[i] = seeds.y[s];
        seed_z[i] = seeds.z[s];
    }
    const int n = samples.Size();
    mfem::Vector squared_distance(n);
    squared_distance.UseDevice(true);
    if (n > 0) {
        const double* sx = seed_x.Read();
        const double* sy = seed_y.Read();
        const double* sz = seed_z.Read();
        const double* x = samples.x.Read();
        const double* y = samples.y.Read();
        const double* z = samples.z.Read();
        const int* label = labels.Read();
        double* out = squared_distance.Write();
        mfem::forall(n, [=] MFEM_HOST_DEVICE(int q) {
            // Labels were validated by the preceding Measure call (all in [0, N))
            const int s = label[q];
            const double dx = x[q] - sx[s];
            const double dy = y[q] - sy[s];
            const double dz = z[q] - sz[s];
            out[q] = dx * dx + dy * dy + dz * dz;
        });
    }
    return accumulator.Integrate(squared_distance);
}

/**
 * @brief MPI reduction function: element-wise lexicographic minimum of (d^2, x, y, z) records.
 * Signature required by MPI_Op_create; `inout` receives the result.
 */
void MinRecordOp(void* in, void* inout, int* len, MPI_Datatype* /*type*/) {
    const double* a = static_cast<const double*>(in);
    double* b = static_cast<double*>(inout);
    for (int r = 0; r < *len; ++r) {
        const double* ra = a + 4 * r;
        double* rb = b + 4 * r;
        bool a_smaller = false;
        for (int k = 0; k < 4; ++k) {
            if (ra[k] != rb[k]) {
                a_smaller = ra[k] < rb[k];
                break;
            }
        }
        if (a_smaller) {
            for (int k = 0; k < 4; ++k) {
                rb[k] = ra[k];
            }
        }
    }
}

/**
 * @brief Replaces the centroids that lie outside the material by the closest sample of their
 * cell. Collective. Returns the number of replaced centroids.
 *
 * Steps:
 *  1. Inside test of every non-empty cell's centroid on this rank's elements
 *     (MaterialPointLocator::MarkInside, host), then a bitwise-OR MPI_Allreduce: a centroid is
 *     inside the material if it is inside an element of any rank.
 *  2. For each outside centroid c_i, each rank scans its samples labeled i and keeps the record
 *     (|x_q - c_i|^2, x_q, y_q, z_q) that is smallest in lexicographic order (distance first,
 *     then the coordinates as a tie-break that does not depend on the partition).
 *  3. One MPI_Allreduce with the user-defined lexicographic-minimum reduction over the records of
 *     all outside cells; the winning sample becomes the new position. A non-empty cell has at
 *     least one sample on some rank, so a winner always exists.
 */
int ClampToMaterial(MPI_Comm comm, const MaterialPointLocator& locator,
                    const WeightedSamples& samples, const mfem::Array<int>& labels,
                    const CellMeasures& measures, std::vector<double>& new_x,
                    std::vector<double>& new_y, std::vector<double>& new_z) {
    const std::size_t num_seeds = measures.volume.size();

    // Step 1: inside test of the non-empty cells' centroids
    std::vector<int> tested; // seed index of each tested centroid
    std::vector<double> px, py, pz;
    for (std::size_t i = 0; i < num_seeds; ++i) {
        if (measures.volume[i] > 0.0) {
            tested.push_back(static_cast<int>(i));
            px.push_back(new_x[i]);
            py.push_back(new_y[i]);
            pz.push_back(new_z[i]);
        }
    }
    const std::size_t num_words = (tested.size() + 63) / 64;
    std::vector<std::uint64_t> inside(num_words, 0);
    if (!tested.empty()) {
        locator.MarkInside(static_cast<std::int64_t>(tested.size()), px.data(), py.data(),
                           pz.data(), inside.data());
    }
    MPI_Allreduce(MPI_IN_PLACE, inside.data(), static_cast<int>(num_words), MPI_UINT64_T, MPI_BOR,
                  comm);

    // outside_slot[i] = position of seed i in the list of outside centroids, or -1
    std::vector<int> outside_slot(num_seeds, -1);
    std::vector<int> outside_seed;
    for (std::size_t k = 0; k < tested.size(); ++k) {
        const bool is_inside = ((inside[k / 64] >> (k % 64)) & std::uint64_t(1)) != 0;
        if (!is_inside) {
            outside_slot[static_cast<std::size_t>(tested[k])] =
                static_cast<int>(outside_seed.size());
            outside_seed.push_back(tested[k]);
        }
    }
    if (outside_seed.empty()) {
        return 0; // same decision on every rank (the mask is global)
    }

    // Step 2: local closest sample of every outside cell (host scan of the samples)
    std::vector<double> record(4 * outside_seed.size(), DBL_MAX);
    const int n = samples.Size();
    if (n > 0) {
        const double* x = samples.x.HostRead();
        const double* y = samples.y.HostRead();
        const double* z = samples.z.HostRead();
        const int* label = labels.HostRead();
        for (int q = 0; q < n; ++q) {
            const int slot = outside_slot[static_cast<std::size_t>(label[q])];
            if (slot < 0) {
                continue;
            }
            const auto s = static_cast<std::size_t>(label[q]);
            const double dx = x[q] - new_x[s];
            const double dy = y[q] - new_y[s];
            const double dz = z[q] - new_z[s];
            double candidate[4] = {dx * dx + dy * dy + dz * dz, x[q], y[q], z[q]};
            int len = 1;
            MinRecordOp(candidate, &record[4 * static_cast<std::size_t>(slot)], &len, nullptr);
        }
    }

    // Step 3: global lexicographic minimum per outside cell
    MPI_Datatype record_type;
    MPI_Type_contiguous(4, MPI_DOUBLE, &record_type);
    MPI_Type_commit(&record_type);
    MPI_Op min_record;
    MPI_Op_create(&MinRecordOp, 1, &min_record);
    MPI_Allreduce(MPI_IN_PLACE, record.data(), static_cast<int>(outside_seed.size()), record_type,
                  min_record, comm);
    MPI_Op_free(&min_record);
    MPI_Type_free(&record_type);

    for (std::size_t k = 0; k < outside_seed.size(); ++k) {
        const auto s = static_cast<std::size_t>(outside_seed[k]);
        new_x[s] = record[4 * k + 1];
        new_y[s] = record[4 * k + 2];
        new_z[s] = record[4 * k + 3];
    }
    return static_cast<int>(outside_seed.size());
}

} // namespace

/*
 * ValidateLloydParameters: Lloyd needs a metric whose cells are "nearest seed" cells with planar
 * boundaries (Voronoi, or Laguerre for power cells).
 */
void ValidateLloydParameters(const LloydParameters& params) {
    if (params.iterations < 0) {
        throw std::invalid_argument("Lloyd: iterations must be >= 0");
    }
    if (params.num_bins < 1) {
        throw std::invalid_argument("Lloyd: num_bins must be >= 1");
    }
    if (!params.centroidal_power && params.metric != MetricType::VORONOI &&
        params.metric != MetricType::LAGUERRE) {
        throw std::invalid_argument("Lloyd: metric must be VORONOI or LAGUERRE");
    }
    if (params.centroidal_power) {
        WeightFitParameters fit = params.weight_fit;
        fit.metric = MetricType::LAGUERRE;
        ValidateWeightFitParameters(fit);
    }
}

/*
 * RunLloyd
 *
 * Algorithm:
 *  1. Checks; build the inside-test structure (MaterialPointLocator, local and host only) and the
 *     cell-measure accumulator (collective setup of the fixed-point scales).
 *  2. Centroidal power mode: fit the weights for the starting positions.
 *  3. For each iteration:
 *     a. label every sample with the current seeds (exact nearest-seed search);
 *     b. measure the cells (volumes and centroids, reproducible) and record the CVT energy of
 *        these cells around the current seeds;
 *     c. new position = cell centroid; empty cells keep their seed;
 *     d. clamp centroids outside the material (if enabled);
 *     e. move the seeds; in centroidal power mode, refit the weights for the new positions,
 *        warm-started from the previous weights.
 *  4. Label once more and record the final energy.
 * Every quantity that decides a position is identical on all ranks, so all ranks move every
 * seed identically.
 */
SeedSet RunLloyd(const mfem::Mesh& local_mesh, MPI_Comm comm, const WeightedSamples& samples,
                 const SeedSet& seeds, const LloydParameters& params,
                 const std::vector<double>& target_volume, LloydReport* report) {
    // Step 1: checks and setup
    ValidateLloydParameters(params);
    const std::size_t num_seeds = seeds.Size();
    if (num_seeds == 0) {
        throw std::invalid_argument("Lloyd: the seed set is empty");
    }
    if (params.centroidal_power && target_volume.size() != num_seeds) {
        throw std::invalid_argument("Lloyd: centroidal power mode needs one target volume per "
                                    "seed");
    }
    MetricType metric = params.metric;
    WeightFitParameters fit_params = params.weight_fit;
    if (params.centroidal_power) {
        metric = MetricType::LAGUERRE;
        fit_params.metric = MetricType::LAGUERRE;
    }
    std::unique_ptr<MaterialPointLocator> locator;
    if (params.clamp_to_material) {
        // params are identical on every rank, so every rank enters the collective decision
        locator = std::make_unique<MaterialPointLocator>(local_mesh,
                                                         IsStraightSided(local_mesh, comm));
    }
    const CellMeasureAccumulator accumulator(comm, samples, static_cast<int>(num_seeds));
    LloydReport local_report;
    SeedSet current = seeds;

    // Step 2: weights for the starting positions (centroidal power mode)
    if (params.centroidal_power) {
        WeightFitResult fit =
            FitWeights(comm, samples, current, target_volume, current.weight, fit_params);
        current.weight = std::move(fit.weight);
        local_report.fits.push_back(std::move(fit.report));
    }

    // Step 3: Lloyd iterations
    const MetricSetup setup{};
    mfem::Array<int> labels;
    for (int iteration = 0; iteration < params.iterations; ++iteration) {
        // 3a-3b: cells of the current seeds, their measures and energy
        LabelSamples(samples, current, metric, setup, params.num_bins, labels, nullptr);
        const CellMeasures measures = accumulator.Measure(labels);
        local_report.energy.push_back(CvtEnergy(accumulator, samples, current, labels));

        // 3c: centroids (empty cells stay put)
        std::vector<double> new_x = current.x;
        std::vector<double> new_y = current.y;
        std::vector<double> new_z = current.z;
        int num_empty = 0;
        for (std::size_t i = 0; i < num_seeds; ++i) {
            if (measures.volume[i] > 0.0) {
                new_x[i] = measures.cx[i];
                new_y[i] = measures.cy[i];
                new_z[i] = measures.cz[i];
            } else {
                ++num_empty;
            }
        }

        // 3d: keep seeds inside the material
        int num_clamped = 0;
        if (locator) {
            num_clamped = ClampToMaterial(comm, *locator, samples, labels, measures, new_x, new_y,
                                          new_z);
        }

        // 3e: move, record the largest displacement, refit in centroidal power mode
        double max_move = 0.0;
        for (std::size_t i = 0; i < num_seeds; ++i) {
            const double dx = new_x[i] - current.x[i];
            const double dy = new_y[i] - current.y[i];
            const double dz = new_z[i] - current.z[i];
            max_move = std::fmax(max_move, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        current.x = std::move(new_x);
        current.y = std::move(new_y);
        current.z = std::move(new_z);
        local_report.max_displacement.push_back(max_move);
        local_report.num_clamped.push_back(num_clamped);
        local_report.num_empty.push_back(num_empty);
        if (params.centroidal_power) {
            WeightFitResult fit =
                FitWeights(comm, samples, current, target_volume, current.weight, fit_params);
            current.weight = std::move(fit.weight);
            local_report.fits.push_back(std::move(fit.report));
        }
    }

    // Step 4: energy of the final seeds
    LabelSamples(samples, current, metric, setup, params.num_bins, labels, nullptr);
    accumulator.Measure(labels); // validates the labels (all ranks throw together if not)
    local_report.energy.push_back(CvtEnergy(accumulator, samples, current, labels));

    if (report != nullptr) {
        *report = std::move(local_report);
    }
    return current;
}

} // namespace exaconstit::microstructure
