#include "microstructure/seeding/roi_filler.hpp"

#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/util/collective_errors.hpp"
#include "microstructure/util/fixed_point.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** Elements evaluated per call of the quadrature-point sampler (bounds the temporary memory). */
constexpr int ELEMENT_CHUNK = 8192;

/**
 * Highest tetrahedron quadrature order whose MFEM v4.10 rule has only positive weights (the same
 * limit as labeling/cell_measures.cpp). Higher orders fall back to Grundmann-Moller rules with
 * negative weights, which would make "volume inside the box" ill-defined.
 */
constexpr int MAX_POSITIVE_TET_QUAD_ORDER = 20;

/** Candidate indices of the seed generator stay below 2^40 (seed_generators.cpp MAX_CANDIDATES). */
constexpr std::int64_t GENERATOR_MAX_CANDIDATES = std::int64_t(1) << 40;

/*
 * RequirePositiveRule: the same check as labeling/cell_measures.cpp (RequirePositiveSampleRule).
 * A quadrature point with a weight <= 0 would add zero or negative volume to one side of the ROI
 * box, so such rules are refused: tetrahedron orders above MAX_POSITIVE_TET_QUAD_ORDER, and any
 * rule with a non-positive weight. Called with the globally agreed geometry and order, so every
 * rank throws together. Host only.
 */
void RequirePositiveRule(mfem::Geometry::Type geometry, int order) {
    const std::string name = mfem::Geometry::Name[geometry];
    if (geometry == mfem::Geometry::TETRAHEDRON && order > MAX_POSITIVE_TET_QUAD_ORDER) {
        throw std::runtime_error("FillOutsideRoi: tetrahedron quadrature order " +
                                 std::to_string(order) + " > " +
                                 std::to_string(MAX_POSITIVE_TET_QUAD_ORDER) +
                                 " has no positive-weight rule in MFEM (mesh node order too high)");
    }
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(geometry, order);
    for (int k = 0; k < rule.GetNPoints(); ++k) {
        if (!(rule.IntPoint(k).weight > 0.0)) {
            throw std::runtime_error("FillOutsideRoi: MFEM's " + name +
                                     " quadrature rule of order " + std::to_string(order) +
                                     " has a non-positive weight; MFEM v4.10 is required");
        }
    }
}

/** Volumes of the material inside and outside the ROI box (identical on every rank). */
struct RoiVolumes {
    double inside = 0.0;
    double outside = 0.0;
};

/*
 * MeasureRoiVolumes (collective)
 *
 * Steps:
 *  1. Agree on the quadrature rule: every rank proposes DefaultSampleQuadOrder of its own elements
 *     and its element geometry (-1 without elements), and MPI_Allreduce(MAX) picks one rule for
 *     all ranks, so every element is integrated the same way whatever the partition. The rule
 *     must have only positive weights (RequirePositiveRule).
 *  2. Walk the local elements in chunks with a QuadraturePointSampler (mesh/mesh_sampler.hpp):
 *     each point q gives its weight W_q = w_q detJ_q (length^3) and whether it lies strictly inside
 *     the box. This is a one-time host pass over HostRead() data. Local setup errors (e.g. mixed
 *     element geometries) are combined over the ranks so all ranks throw together.
 *  3. MPI_Allreduce: total point count (SUM) and largest |W_q| (MAX); these set the fixed-point
 *     scale (FixedPointScale::ForRange), the same on every rank.
 *  4. Encode every W_q, add the integers into an "inside" and an "outside" sum, MPI_SUM them, and
 *     decode. Integer sums do not depend on the order of the terms, so the volumes are bitwise
 *     identical for any number of ranks.
 */
RoiVolumes MeasureRoiVolumes(mfem::Mesh& mesh, MPI_Comm comm, const std::array<double, 3>& lo,
                             const std::array<double, 3>& hi) {
    // Step 1: one rule for every rank
    const int ne = mesh.GetNE();
    int rule_info[2] = {-1, -1}; // [order, geometry]
    if (ne > 0) {
        rule_info[0] = DefaultSampleQuadOrder(mesh);
        rule_info[1] = static_cast<int>(mesh.GetElementBaseGeometry(0));
    }
    MPI_Allreduce(MPI_IN_PLACE, rule_info, 2, MPI_INT, MPI_MAX, comm);
    if (rule_info[1] < 0) {
        throw std::runtime_error("FillOutsideRoi: the mesh has no elements");
    }
    RequirePositiveRule(static_cast<mfem::Geometry::Type>(rule_info[1]), rule_info[0]);

    // Step 2: per-point weights and inside flags of the local elements
    std::vector<double> weight;
    std::vector<char> inside;
    std::string local_error;
    try {
        if (ne > 0) {
            const mfem::IntegrationRule& rule = mfem::IntRules.Get(
                static_cast<mfem::Geometry::Type>(rule_info[1]), rule_info[0]);
            QuadraturePointSampler sampler(mesh, rule);
            QuadraturePointSamples samples;
            for (int begin = 0; begin < ne; begin += ELEMENT_CHUNK) {
                const int end = std::min(ne, begin + ELEMENT_CHUNK);
                sampler.Sample(begin, end, samples);
                const int count = samples.weight.Size();
                const double* x = samples.x.HostRead();
                const double* y = samples.y.HostRead();
                const double* z = samples.z.HostRead();
                const double* w = samples.weight.HostRead();
                for (int q = 0; q < count; q++) {
                    // Strictly inside, the same test as the seed generator's exclusion box
                    const bool in_box = x[q] > lo[0] && x[q] < hi[0] && y[q] > lo[1] &&
                                        y[q] < hi[1] && z[q] > lo[2] && z[q] < hi[2];
                    weight.push_back(w[q]);
                    inside.push_back(in_box ? 1 : 0);
                }
            }
        }
    } catch (const std::exception& error) {
        local_error = std::string("FillOutsideRoi: quadrature sampling failed: ") + error.what();
    }
    ThrowIfAnyRankFailed(comm, local_error);

    // Step 3: global count and largest term, for the fixed-point scale
    std::int64_t num_points = static_cast<std::int64_t>(weight.size());
    double max_weight = 0.0;
    for (const double w : weight) {
        max_weight = std::max(max_weight, std::fabs(w));
    }
    MPI_Allreduce(MPI_IN_PLACE, &num_points, 1, MPI_INT64_T, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &max_weight, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (!(max_weight > 0.0) || !std::isfinite(max_weight)) {
        throw std::runtime_error("FillOutsideRoi: the mesh has zero or undefined volume");
    }
    const FixedPointScale scale = FixedPointScale::ForRange(0.0, max_weight, num_points);

    // Step 4: integer sums inside / outside, then decode
    std::int64_t sums[2] = {0, 0}; // [inside, outside]
    for (std::size_t q = 0; q < weight.size(); q++) {
        sums[inside[q] ? 0 : 1] += scale.Encode(weight[q]);
    }
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_INT64_T, MPI_SUM, comm);
    RoiVolumes volumes;
    volumes.inside = scale.DecodeSum(sums[0], num_points);
    volumes.outside = scale.DecodeSum(sums[1], num_points);
    return volumes;
}

} // namespace

/*
 * FillOutsideRoi (collective)
 *
 * Steps:
 *  1. Check the parameters (identical on every rank, so all ranks throw together).
 *  2. Measure V_roi and V_outside (MeasureRoiVolumes).
 *  3. Number of filler seeds: the given count, or round(num_roi_seeds * V_outside / V_roi); 0 when
 *     V_outside = 0. The automatic formula needs V_roi > 0 and num_roi_seeds > 0.
 *  4. GenerateSeeds (default rejection sampler) with the ROI box as exclusion box, the id offset,
 *     the outside fraction V_outside / (V_roi + V_outside) (sets the automatic stall cap) and,
 *     with min_distance_factor > 0, the absolute spacing
 *     d = min_distance_factor * (V_outside / num_filler)^(1/3), which is the usual factor applied
 *     to the mean filler grain volume instead of the whole mesh volume.
 */
RoiFillerResult FillOutsideRoi(mfem::Mesh& local_mesh,
                               MPI_Comm comm,
                               const RoiFillerParameters& params,
                               std::int64_t num_roi_seeds) {
    // Step 1
    for (int d = 0; d < 3; d++) {
        if (!(params.hi[d] > params.lo[d]) || !std::isfinite(params.lo[d]) ||
            !std::isfinite(params.hi[d])) {
            throw std::invalid_argument("FillOutsideRoi: the ROI box must be finite with hi > lo "
                                        "along every axis");
        }
    }
    if (params.num_filler < 0 || num_roi_seeds < 0) {
        throw std::invalid_argument("FillOutsideRoi: num_filler and num_roi_seeds must be >= 0");
    }
    // The generator adds candidate indices (< 2^40) to the offset, which must not overflow int64
    if (params.source_id_offset < 0 ||
        params.source_id_offset > std::numeric_limits<std::int64_t>::max() -
                                      GENERATOR_MAX_CANDIDATES) {
        throw std::invalid_argument(
            "FillOutsideRoi: source_id_offset = " + std::to_string(params.source_id_offset) +
            " leaves no room for filler ids: it must be in [0, 2^63 - 1 - 2^40], i.e. the seed "
            "table ids must stay below 2^63 - 1 - 2^40");
    }
    if (!(params.min_distance_factor >= 0.0)) {
        throw std::invalid_argument("FillOutsideRoi: min_distance_factor must be >= 0");
    }

    // Step 2
    RoiFillerResult result;
    const RoiVolumes volumes = MeasureRoiVolumes(local_mesh, comm, params.lo, params.hi);
    result.roi_material_volume = volumes.inside;
    result.outside_material_volume = volumes.outside;

    // Step 3: requested count
    std::int64_t requested = params.num_filler;
    if (requested == 0 && volumes.outside > 0.0) {
        if (!(volumes.inside > 0.0) || num_roi_seeds == 0) {
            throw std::invalid_argument(
                "FillOutsideRoi: the automatic filler count needs material inside the ROI box and "
                "at least one ROI seed (V_roi = " +
                std::to_string(volumes.inside) + ", ROI seeds = " +
                std::to_string(num_roi_seeds) + "); give num_filler explicitly");
        }
        const double count =
            std::round(static_cast<double>(num_roi_seeds) * volumes.outside / volumes.inside);
        if (count > static_cast<double>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("FillOutsideRoi: the automatic filler count exceeds "
                                        "2^31 - 1; the ROI is too small for the mesh");
        }
        requested = static_cast<std::int64_t>(count);
    } else if (requested > 0 && !(volumes.outside > 0.0)) {
        throw std::invalid_argument("FillOutsideRoi: " + std::to_string(requested) +
                                    " filler seeds requested but there is no material outside "
                                    "the ROI box");
    }
    result.num_requested = requested;
    if (requested == 0) {
        return result; // nothing outside the ROI (or a tiny outside volume that rounds to 0)
    }

    // Step 4: rejection sampling outside the box
    SeedingParameters seeding;
    seeding.placement = SeedPlacement::REJECTION;
    seeding.num_seeds = requested;
    seeding.rng_seed = params.rng_seed;
    seeding.use_exclude_box = true;
    seeding.exclude_box_lo = params.lo;
    seeding.exclude_box_hi = params.hi;
    seeding.source_id_offset = params.source_id_offset;
    // Only the outside material can accept candidates: tell the generator, so its automatic stall
    // cap and its stall message account for it (bitwise the same on every rank: both volumes are
    // fixed-point sums)
    seeding.exclude_box_outside_fraction =
        std::min(1.0, volumes.outside / (volumes.inside + volumes.outside));
    if (params.min_distance_factor > 0.0) {
        seeding.min_distance = params.min_distance_factor *
                               std::cbrt(volumes.outside / static_cast<double>(requested));
    }
    result.filler = GenerateSeeds(local_mesh, comm, seeding, &result.report);
    return result;
}

} // namespace exaconstit::microstructure
