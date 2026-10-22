/*
 * Johnson-Mehl nucleation and growth (johnson_mehl.hpp): space-time Poisson candidates, the
 * in-material test, rate thinning, the "already transformed" test, and the
 * transformation-completion bound that ends the process.
 *
 * Terms used in this file:
 * - Poisson process in space and time: events (candidate nuclei) that occur independently, at an
 *   average rate of I per unit volume and unit time. The waiting time between two consecutive
 *   events in a region of volume V is exponentially distributed with mean 1 / (I V), and each
 *   event's position is uniform in the region.
 * - Thinning: keeping each event of a rate-I_max process independently with probability
 *   I(t) / I_max gives a process with the time-dependent rate I(t).
 * - Hash grid: space is cut into equal cubic cells and every accepted nucleus is filed under the
 *   cell that contains it, so a search near a point only looks at the nearby cells.
 * - Collective: a function every rank of the communicator must call, in the same order, because
 *   it exchanges data between ranks.
 */
#include "microstructure/seeding/johnson_mehl.hpp"

#include "microstructure/mesh/straight_sided.hpp"
#include "microstructure/seeding/material_point_locator.hpp"
#include "microstructure/seeding/seed_generators.hpp"
#include "microstructure/util/collective_errors.hpp"
#include "microstructure/util/counter_rng.hpp"
#include "microstructure/util/random_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

namespace {

/** Default stall cap: candidates generated before giving up (2^26, about 67 million). */
constexpr std::int64_t DEFAULT_MAX_CANDIDATES = std::int64_t(1) << 26;
/** First batch size, and the largest one (batches double in size up to this). */
constexpr std::int64_t FIRST_BATCH = 256;
constexpr std::int64_t MAX_BATCH = std::int64_t(1) << 16;
/** Upper limit on the number of hash-grid cells (memory: one std::vector per cell). */
constexpr std::int64_t MAX_GRID_CELLS = std::int64_t(1) << 18;
/**
 * Relative enlargement of the element radius in the completion bound, so that points accepted by
 * the slightly tolerant inside test (up to about 1e-8 of an element size outside the element,
 * material_point_locator.hpp) are still covered by the bound.
 */
constexpr double RADIUS_SAFETY = 1.0e-6;

// ThrowIfAnyRankFailed (collective; util/collective_errors.hpp): local setup steps that can fail
// on some ranks only (element boxes, element sampler, surface triangles) pass their error message
// (empty = no error); the lowest failing rank broadcasts it and every rank throws it, instead of
// the healthy ranks waiting forever in the next collective call.

/** Human-readable stop reason for the summary. */
const char* StopName(JohnsonMehlStop stop) {
    switch (stop) {
    case JohnsonMehlStop::END_TIME:
        return "end_time reached";
    case JohnsonMehlStop::NUM_NUCLEI:
        return "num_nuclei reached";
    case JohnsonMehlStop::FULLY_TRANSFORMED:
        return "material fully transformed";
    case JohnsonMehlStop::STALLED:
        return "STALLED (max_candidates reached)";
    }
    return "unknown";
}

/**
 * Hash grid of the accepted nuclei, answering the two questions of the Johnson-Mehl loop:
 * "has some grain already reached point c at time t?" (FindCovering) and "when does the first
 * grain reach point x?" (ArrivalTime).
 *
 * The grid covers a fixed box with cubic cells of edge h; it is dense (one list of nucleus
 * indices per cell, at most MAX_GRID_CELLS cells). The nuclei themselves live in the SeedSet
 * being built; the grid only stores their indices. Host only; every rank builds the identical
 * grid from the identical accepted nuclei.
 */
class NucleusGrid {
public:
    /**
     * @param lo, hi box containing every nucleus and every query point (mesh units).
     * @param h      requested cell edge (> 0); enlarged if the cell count would exceed
     *               MAX_GRID_CELLS.
     * @param speed  growth speed v.
     * @param seeds  the nuclei (x, y, z, nucleation_time are read); must outlive the grid.
     */
    NucleusGrid(const std::array<double, 3>& lo, const std::array<double, 3>& hi, double h,
                double speed, const SeedSet& seeds)
        : m_lo(lo), m_speed(speed), m_seeds(seeds) {
        // Grow h until the number of cells is acceptable, then count cells per axis
        double edge = h;
        while (true) {
            std::int64_t total = 1;
            for (int d = 0; d < 3; d++) {
                const double cells = std::ceil((hi[d] - lo[d]) / edge);
                m_num_cells[d] = std::max<std::int64_t>(1, static_cast<std::int64_t>(cells));
                total *= m_num_cells[d];
            }
            if (total <= MAX_GRID_CELLS) {
                m_cells.resize(static_cast<std::size_t>(total));
                break;
            }
            edge *= 1.25;
        }
        m_h = edge;
    }

    /** Files nucleus i (already appended to the SeedSet) under its cell. */
    void Insert(int i) {
        const std::int64_t c[3] = {Cell(m_seeds.x[i], 0), Cell(m_seeds.y[i], 1),
                                   Cell(m_seeds.z[i], 2)};
        m_cells[static_cast<std::size_t>(Index(c[0], c[1], c[2]))].push_back(i);
        if (m_count == 0) {
            m_earliest = m_seeds.nucleation_time[i];
        }
        m_count++;
    }

    /**
     * Lowest index j of an accepted nucleus whose grain covers point c at time t, i.e.
     * |c - s_j| < v (t - t_j), or -1 if none does.
     *
     * Only nuclei within distance R = v (t - t_earliest) of c can cover it (t_earliest is the
     * birth time of the first nucleus, so no grain is larger than R at time t). The cells
     * overlapping the cube [c - R, c + R] are searched; when that cube spans more cells than
     * there are nuclei, a plain loop over all nuclei is cheaper and is used instead.
     */
    int FindCovering(const std::array<double, 3>& c, double t) const {
        if (m_count == 0) {
            return -1;
        }
        const double reach = m_speed * (t - m_earliest);
        if (!(reach > 0.0)) {
            return -1;
        }

        // Cell range of the cube [c - R, c + R], clamped to the grid
        std::int64_t c_lo[3];
        std::int64_t c_hi[3];
        double num_range_cells = 1.0;
        for (int d = 0; d < 3; d++) {
            c_lo[d] = Cell(c[d] - reach, d);
            c_hi[d] = Cell(c[d] + reach, d);
            num_range_cells *= static_cast<double>(c_hi[d] - c_lo[d] + 1);
        }

        int best = -1;
        auto test = [&](int j) {
            if (best >= 0 && j > best) {
                return; // a lower covering index is already known
            }
            const double radius = m_speed * (t - m_seeds.nucleation_time[j]);
            if (!(radius > 0.0)) {
                return; // born at the same time or later: cannot cover c yet
            }
            const double dx = c[0] - m_seeds.x[j];
            const double dy = c[1] - m_seeds.y[j];
            const double dz = c[2] - m_seeds.z[j];
            if (dx * dx + dy * dy + dz * dz < radius * radius) {
                best = j;
            }
        };

        if (num_range_cells > static_cast<double>(m_count)) {
            // Brute force is cheaper than scanning mostly empty cells
            for (int j = 0; j < m_count; j++) {
                test(j);
            }
            return best;
        }
        for (std::int64_t k = c_lo[2]; k <= c_hi[2]; k++) {
            for (std::int64_t jj = c_lo[1]; jj <= c_hi[1]; jj++) {
                for (std::int64_t i = c_lo[0]; i <= c_hi[0]; i++) {
                    for (const int j : m_cells[static_cast<std::size_t>(Index(i, jj, k))]) {
                        test(j);
                    }
                }
            }
        }
        return best;
    }

    /**
     * Arrival time of the first grain at point x: T(x) = min_j (t_j + |x - s_j| / v); +infinity
     * without nuclei.
     *
     * Ring search: cells are visited in shells ("rings") of growing Chebyshev distance r from the
     * cell of x (r = 0 is the cell itself, r = 1 its 26 neighbors, ...). Any point of a cell in
     * ring r >= 1 is at least (r - 1) h away from x, so no nucleus in ring r or beyond can arrive
     * before t_earliest + (r - 1) h / v; the search stops once that exceeds the best time found.
     * When the next ring's cube would hold more cells than there are nuclei, a plain loop over
     * all nuclei finishes the job instead (cost never above O(number of nuclei)).
     */
    double ArrivalTime(const std::array<double, 3>& x) const {
        double best = std::numeric_limits<double>::infinity();
        if (m_count == 0) {
            return best;
        }
        auto test = [&](int j) {
            const double dx = x[0] - m_seeds.x[j];
            const double dy = x[1] - m_seeds.y[j];
            const double dz = x[2] - m_seeds.z[j];
            const double arrival =
                m_seeds.nucleation_time[j] + std::sqrt(dx * dx + dy * dy + dz * dz) / m_speed;
            best = std::min(best, arrival);
        };

        const std::int64_t center[3] = {Cell(x[0], 0), Cell(x[1], 1), Cell(x[2], 2)};
        const std::int64_t max_ring =
            std::max({m_num_cells[0], m_num_cells[1], m_num_cells[2]});
        for (std::int64_t r = 0; r <= max_ring; r++) {
            // Stop: nothing in ring r or beyond can beat the best arrival time
            if (r >= 1 && m_earliest + static_cast<double>(r - 1) * m_h / m_speed >= best) {
                return best;
            }
            // Switch to brute force when the ring cube becomes larger than the nucleus count
            const double side = static_cast<double>(2 * r + 1);
            if (side * side * side > static_cast<double>(m_count)) {
                for (int j = 0; j < m_count; j++) {
                    test(j);
                }
                return best;
            }
            // Visit the cells of ring r (Chebyshev distance exactly r) that lie in the grid
            for (std::int64_t dk = -r; dk <= r; dk++) {
                for (std::int64_t dj = -r; dj <= r; dj++) {
                    for (std::int64_t di = -r; di <= r; di++) {
                        const std::int64_t ring =
                            std::max({std::abs(di), std::abs(dj), std::abs(dk)});
                        if (ring != r) {
                            continue;
                        }
                        const std::int64_t i = center[0] + di;
                        const std::int64_t j = center[1] + dj;
                        const std::int64_t k = center[2] + dk;
                        if (i < 0 || j < 0 || k < 0 || i >= m_num_cells[0] ||
                            j >= m_num_cells[1] || k >= m_num_cells[2]) {
                            continue;
                        }
                        for (const int n : m_cells[static_cast<std::size_t>(Index(i, j, k))]) {
                            test(n);
                        }
                    }
                }
            }
        }
        return best;
    }

private:
    /** Cell coordinate of p along axis d, clamped to [0, n_d - 1]. */
    std::int64_t Cell(double p, int d) const {
        const double u = std::floor((p - m_lo[d]) / m_h);
        if (!(u > 0.0)) {
            return 0;
        }
        const double last = static_cast<double>(m_num_cells[d] - 1);
        return static_cast<std::int64_t>(std::min(u, last));
    }

    /** Linear cell index, x fastest. */
    std::int64_t Index(std::int64_t i, std::int64_t j, std::int64_t k) const {
        return i + m_num_cells[0] * (j + m_num_cells[1] * k);
    }

    std::array<double, 3> m_lo;                ///< grid box lower corner
    std::array<std::int64_t, 3> m_num_cells{}; ///< cells per axis
    double m_h = 1.0;                          ///< cell edge
    double m_speed = 1.0;                      ///< growth speed v
    const SeedSet& m_seeds;                    ///< the nuclei (not owned)
    int m_count = 0;                           ///< nuclei filed so far
    double m_earliest = 0.0;                   ///< birth time of the first nucleus
    std::vector<std::vector<int>> m_cells;     ///< nucleus indices per cell
};

/*
 * CompletionTimeBound (collective)
 *
 * Proven upper bound of the time by which every material point has been reached by a grain,
 * given the nuclei accepted so far (proof in GenerateJohnsonMehlSeeds' header comment):
 *
 *     T_done = max over local elements e of [ T(c_e) + r_e / v ],  then MPI_MAX over ranks,
 *
 * with c_e the center of element e's bounding box, r_e half the box diagonal (times
 * 1 + RADIUS_SAFETY) and T the arrival time of NucleusGrid::ArrivalTime. The element boxes are
 * the locator's (exact vertex boxes for straight-sided elements, padded boxes for curved ones).
 * A rank without elements contributes -infinity. The maximum of doubles is exact and
 * order-independent, so all ranks get the same value for any partition.
 */
double CompletionTimeBound(const MaterialPointLocator& locator, const NucleusGrid& grid,
                           double speed, MPI_Comm comm) {
    double local_max = -std::numeric_limits<double>::infinity();
    for (int e = 0; e < locator.NumElements(); e++) {
        const auto& box = locator.ElementBox(e);
        std::array<double, 3> center;
        double half_diag_sq = 0.0;
        for (int d = 0; d < 3; d++) {
            center[d] = 0.5 * (box.getMin()[d] + box.getMax()[d]);
            const double half = 0.5 * (box.getMax()[d] - box.getMin()[d]);
            half_diag_sq += half * half;
        }
        const double radius = std::sqrt(half_diag_sq) * (1.0 + RADIUS_SAFETY);
        local_max = std::max(local_max, grid.ArrivalTime(center) + radius / speed);
    }
    double global_max = local_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global_max;
}

} // namespace

/*
 * ValidateJohnsonMehlParameters
 *
 * Plain field checks, in declaration order, so the first problem is named. Runs on identical
 * parameters on every rank, so every rank throws the same error.
 */
void ValidateJohnsonMehlParameters(const JohnsonMehlParameters& params) {
    auto fail = [](const std::string& what) {
        throw std::invalid_argument("Johnson-Mehl parameters: " + what);
    };
    if (!(std::isfinite(params.nucleation_rate) && params.nucleation_rate > 0.0)) {
        fail("nucleation_rate must be finite and > 0");
    }
    if (!(std::isfinite(params.growth_speed) && params.growth_speed > 0.0)) {
        fail("growth_speed must be finite and > 0");
    }
    if (!(std::isfinite(params.end_time) && params.end_time >= 0.0)) {
        fail("end_time must be finite and >= 0 (0 = no time limit)");
    }
    if (params.num_nuclei < 0) {
        fail("num_nuclei must be >= 0 (0 = no count limit)");
    }
    if (params.num_nuclei > std::numeric_limits<int>::max()) {
        fail("num_nuclei must be <= 2^31 - 1");
    }
    if (params.max_candidates < 0) {
        fail("max_candidates must be >= 0 (0 = default)");
    }
    if (params.use_sampling_box) {
        for (int d = 0; d < 3; d++) {
            if (!(std::isfinite(params.sampling_box_lo[d]) &&
                  std::isfinite(params.sampling_box_hi[d]) &&
                  params.sampling_box_hi[d] > params.sampling_box_lo[d])) {
                fail("sampling box must be finite with hi > lo on every axis");
            }
        }
    }
}

/*
 * JohnsonMehlReport::Summary: a few lines for the rank-0 log.
 */
std::string JohnsonMehlReport::Summary() const {
    std::ostringstream os;
    os << "Johnson-Mehl nuclei: " << accepted << " (" << StopName(stop) << ")\n";
    os << "  candidates generated " << candidates_generated << ", in material "
       << candidates_in_material << ", removed by rate " << rejected_by_rate
       << ", already transformed " << rejected_covered << "\n";
    os << "  last candidate time " << last_candidate_time << ", completion time bound "
       << completion_time_bound << "\n";
    os << "  sampling box [" << sampling_lo[0] << ", " << sampling_hi[0] << "] x ["
       << sampling_lo[1] << ", " << sampling_hi[1] << "] x [" << sampling_lo[2] << ", "
       << sampling_hi[2] << "]\n";
    return os.str();
}

/*
 * JohnsonMehlGrainDensity
 *
 * For a constant rate I, the untransformed fraction at time t in infinite space is
 * 1 - X(t) = exp(-(pi/3) I v^3 t^4) (Johnson-Mehl-Avrami-Kolmogorov), and nuclei are accepted at
 * the rate I (1 - X(t)) per unit volume. Integrating over all time:
 *     N_V = I * Gamma(5/4) * ((pi/3) I v^3)^(-1/4) = Gamma(5/4) (3/pi)^(1/4) (I/v)^(3/4)
 *         ~= 0.8960 (I/v)^(3/4).
 */
double JohnsonMehlGrainDensity(double nucleation_rate, double growth_speed) {
    const double pi = 3.14159265358979323846;
    return std::tgamma(1.25) * std::pow(3.0 / pi, 0.25) *
           std::pow(nucleation_rate / growth_speed, 0.75);
}

/*
 * GenerateJohnsonMehlSeeds (collective)
 *
 * Steps:
 *  1. Validate the parameters (identical everywhere, so all ranks throw together).
 *  2. Local setup: element boxes + BVH + exact inside test (MaterialPointLocator). A failure on
 *     any rank becomes the same exception on every rank (ThrowIfAnyRankFailed in
 *     util/collective_errors.hpp).
 *  3. Global facts: element count (MPI_SUM) and the union of the element boxes (MPI_MIN/MAX).
 *     The sampling box is that union rounded outward (RoundBoxOutward, exact power-of-two
 *     arithmetic, so it does not move with round-off), or the user box.
 *  4. Hash grid over the union of the sampling box and the mesh box, with a cell edge near the
 *     expected nucleus spacing: (V_grid / num_nuclei)^(1/3) if a count is given, otherwise the
 *     Johnson-Mehl length (v / I)^(1/4), whichever is smaller.
 *  5. Batches of candidates (sizes 256, 512, ... up to 2^16; the batch size never changes the
 *     result, because every decision is made in candidate order):
 *     a. birth times by summing exponential waiting times, positions uniform in the box; the
 *        batch ends early at the first candidate born after end_time or after the completion
 *        bound (no later candidate can be accepted then);
 *     b. in-material bitmask: MarkInside on the local elements, then MPI_Allreduce(MPI_BOR);
 *     c. in candidate order: rate thinning, coverage test, acceptance; stop at num_nuclei;
 *     d. if more than half of this batch's in-material candidates were already covered (the
 *        material is mostly transformed) and nuclei were added since the last evaluation,
 *        recompute the completion bound (collective, MPI_MAX).
 *     The decision in (d) depends only on replicated counts, so all ranks make the collective
 *     call together.
 */
SeedSet GenerateJohnsonMehlSeeds(const mfem::Mesh& local_mesh, MPI_Comm comm,
                                 const JohnsonMehlParameters& params, JohnsonMehlReport* report) {
    // ---- Step 1: parameters ----
    ValidateJohnsonMehlParameters(params);
    JohnsonMehlReport local_report;
    JohnsonMehlReport& rep = (report != nullptr) ? *report : local_report;
    rep = JohnsonMehlReport{};
    rep.completion_time_bound = std::numeric_limits<double>::infinity();
    const double speed = params.growth_speed;
    const double rate_max = params.nucleation_rate;
    const std::int64_t max_candidates =
        (params.max_candidates > 0) ? params.max_candidates : DEFAULT_MAX_CANDIDATES;

    // ---- Step 2: local setup (may fail on some ranks only) ----
    // Straight-sided or curved is decided over all ranks (collective), so every rank builds the
    // same kind of element boxes and the mesh box below does not depend on the partition
    const bool straight_sided = IsStraightSided(local_mesh, comm);
    std::unique_ptr<MaterialPointLocator> locator_ptr;
    std::string setup_error;
    try {
        locator_ptr = std::make_unique<MaterialPointLocator>(local_mesh, straight_sided);
    } catch (const std::exception& error) {
        setup_error = std::string("GenerateJohnsonMehlSeeds: mesh setup failed: ") + error.what();
    }
    ThrowIfAnyRankFailed(comm, setup_error, false);
    const MaterialPointLocator& locator = *locator_ptr;

    // ---- Step 3: global element count and mesh box (union of the element boxes) ----
    std::int64_t local_elements = locator.NumElements();
    std::int64_t global_elements = 0;
    MPI_Allreduce(&local_elements, &global_elements, 1, MPI_INT64_T, MPI_SUM, comm);
    if (global_elements == 0) {
        throw std::runtime_error("GenerateJohnsonMehlSeeds: the mesh has no elements");
    }
    std::array<double, 3> mesh_lo;
    std::array<double, 3> mesh_hi;
    {
        double lo[3];
        double hi[3];
        for (int d = 0; d < 3; d++) {
            lo[d] = std::numeric_limits<double>::infinity();
            hi[d] = -std::numeric_limits<double>::infinity();
            if (locator.NumElements() > 0) {
                lo[d] = locator.LocalBounds().getMin()[d];
                hi[d] = locator.LocalBounds().getMax()[d];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, lo, 3, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, hi, 3, MPI_DOUBLE, MPI_MAX, comm);
        for (int d = 0; d < 3; d++) {
            mesh_lo[d] = lo[d];
            mesh_hi[d] = hi[d];
        }
    }
    RoundBoxOutward(mesh_lo, mesh_hi);

    std::array<double, 3> box_lo = mesh_lo;
    std::array<double, 3> box_hi = mesh_hi;
    if (params.use_sampling_box) {
        box_lo = params.sampling_box_lo;
        box_hi = params.sampling_box_hi;
    }
    double box_volume = 1.0;
    for (int d = 0; d < 3; d++) {
        box_volume *= (box_hi[d] - box_lo[d]);
    }
    if (!(box_volume > 0.0)) {
        throw std::runtime_error("GenerateJohnsonMehlSeeds: the sampling box has zero volume");
    }
    rep.sampling_lo = box_lo;
    rep.sampling_hi = box_hi;

    // ---- Step 4: hash grid over sampling box + mesh box ----
    SeedSet seeds;
    std::array<double, 3> grid_lo;
    std::array<double, 3> grid_hi;
    double grid_volume = 1.0;
    for (int d = 0; d < 3; d++) {
        grid_lo[d] = std::min(box_lo[d], mesh_lo[d]);
        grid_hi[d] = std::max(box_hi[d], mesh_hi[d]);
        grid_volume *= (grid_hi[d] - grid_lo[d]);
    }
    double cell = std::pow(speed / rate_max, 0.25); // Johnson-Mehl length
    if (params.num_nuclei > 0) {
        cell = std::min(cell, std::cbrt(grid_volume / static_cast<double>(params.num_nuclei)));
    }
    NucleusGrid grid(grid_lo, grid_hi, cell, speed, seeds);

    // ---- Step 5: batches of candidates ----
    const double inv_box_rate = 1.0 / (rate_max * box_volume); // mean waiting time
    std::int64_t next_k = 0;         // index of the next candidate to generate
    double previous_time = 0.0;      // birth time of the last generated candidate
    std::int64_t batch_size = FIRST_BATCH;
    std::int64_t accepted_at_last_bound = 0;
    bool stop = false;

    std::vector<double> cx;
    std::vector<double> cy;
    std::vector<double> cz;
    std::vector<double> ct;
    std::vector<std::uint64_t> mask;
    while (!stop) {
        // Step 5a: generate up to batch_size candidates, in birth order
        const std::int64_t room = max_candidates - next_k;
        if (room <= 0) {
            rep.stop = JohnsonMehlStop::STALLED;
            if (!params.allow_stall) {
                throw std::runtime_error(
                    "GenerateJohnsonMehlSeeds: stall cap reached after " +
                    std::to_string(next_k) + " candidates with " +
                    std::to_string(seeds.Size()) +
                    " nuclei; increase max_candidates, set end_time, or check that the "
                    "sampling box overlaps the material");
            }
            break;
        }
        const std::int64_t wanted = std::min(batch_size, room);
        cx.clear();
        cy.clear();
        cz.clear();
        ct.clear();
        bool time_limit = false;
        for (std::int64_t i = 0; i < wanted; i++) {
            const CounterRng rng(params.rng_seed, RngPurpose::JOHNSON_MEHL,
                                 static_cast<std::uint64_t>(next_k + i));
            // Exponential waiting time: -ln(1 - u) with u in [0, 1), so the log is finite
            const double waiting = -std::log(1.0 - rng.UniformDouble(1)) * inv_box_rate;
            const double t = previous_time + waiting;
            if ((params.end_time > 0.0 && t > params.end_time) ||
                t > rep.completion_time_bound) {
                time_limit = true;
                rep.stop = (params.end_time > 0.0 && t > params.end_time)
                               ? JohnsonMehlStop::END_TIME
                               : JohnsonMehlStop::FULLY_TRANSFORMED;
                break;
            }
            double u[3];
            UniformUnitCube(rng, 0, u);
            cx.push_back(box_lo[0] + u[0] * (box_hi[0] - box_lo[0]));
            cy.push_back(box_lo[1] + u[1] * (box_hi[1] - box_lo[1]));
            cz.push_back(box_lo[2] + u[2] * (box_hi[2] - box_lo[2]));
            ct.push_back(t);
            previous_time = t;
        }
        const std::int64_t n = static_cast<std::int64_t>(ct.size());

        // Step 5b: in-material bitmask, combined over all ranks (same n on every rank)
        const std::int64_t words = (n + 63) / 64;
        mask.assign(static_cast<std::size_t>(words), 0);
        if (n > 0) {
            locator.MarkInside(n, cx.data(), cy.data(), cz.data(), mask.data());
            MPI_Allreduce(MPI_IN_PLACE, mask.data(), static_cast<int>(words), MPI_UINT64_T,
                          MPI_BOR, comm);
        }

        // Step 5c: filters in candidate order (identical on every rank)
        std::int64_t batch_in_material = 0;
        std::int64_t batch_covered = 0;
        std::int64_t processed = 0;
        for (std::int64_t i = 0; i < n && !stop; i++) {
            processed = i + 1;
            const std::int64_t k = next_k + i;
            rep.last_candidate_time = ct[i];
            const bool inside = (mask[i / 64] >> (i % 64)) & 1u;
            if (!inside) {
                continue;
            }
            batch_in_material++;

            // Optional time-dependent rate: keep with probability I(t) / I_max
            if (params.rate_function) {
                const double rate = params.rate_function(ct[i]);
                if (!(rate >= 0.0 && rate <= rate_max)) {
                    throw std::invalid_argument(
                        "GenerateJohnsonMehlSeeds: rate_function(" + std::to_string(ct[i]) +
                        ") = " + std::to_string(rate) +
                        " is outside [0, nucleation_rate]; nucleation_rate must bound it");
                }
                const CounterRng rng(params.rng_seed, RngPurpose::JOHNSON_MEHL,
                                     static_cast<std::uint64_t>(k));
                if (!(rng.UniformDouble(2) * rate_max < rate)) {
                    rep.rejected_by_rate++;
                    continue;
                }
            }

            // Already transformed? (some earlier grain has reached the candidate)
            const std::array<double, 3> c = {cx[i], cy[i], cz[i]};
            const int covering = grid.FindCovering(c, ct[i]);
            if (covering >= 0) {
                batch_covered++;
                rep.rejected_covered++;
                if (params.record_rejected) {
                    rep.rejected_xyz.insert(rep.rejected_xyz.end(), c.begin(), c.end());
                    rep.rejected_time.push_back(ct[i]);
                    rep.rejected_by.push_back(covering);
                }
                continue;
            }

            // Accept: a new nucleus with Additive weight a = -v t
            seeds.x.push_back(c[0]);
            seeds.y.push_back(c[1]);
            seeds.z.push_back(c[2]);
            seeds.nucleation_time.push_back(ct[i]);
            seeds.weight.push_back(-speed * ct[i]);
            seeds.source_id.push_back(k);
            grid.Insert(static_cast<int>(seeds.Size()) - 1);
            if (params.num_nuclei > 0 &&
                static_cast<std::int64_t>(seeds.Size()) >= params.num_nuclei) {
                rep.stop = JohnsonMehlStop::NUM_NUCLEI;
                stop = true;
            }
        }
        rep.candidates_generated += processed;
        rep.candidates_in_material += batch_in_material;
        next_k += processed;
        if (stop || time_limit) {
            break;
        }

        // Step 5d: mostly transformed -> tighten the completion bound (collective; the
        // condition uses replicated counts only)
        const std::int64_t accepted = static_cast<std::int64_t>(seeds.Size());
        if (2 * batch_covered > batch_in_material && accepted > accepted_at_last_bound) {
            rep.completion_time_bound = CompletionTimeBound(locator, grid, speed, comm);
            accepted_at_last_bound = accepted;
        }
        batch_size = std::min(2 * batch_size, MAX_BATCH);
    }

    rep.accepted = static_cast<std::int64_t>(seeds.Size());
    return seeds;
}

/*
 * ParMesh overload: the local part of a ParMesh is an mfem::Mesh; its communicator is the one the
 * mesh is distributed over.
 */
SeedSet GenerateJohnsonMehlSeeds(const mfem::ParMesh& pmesh, const JohnsonMehlParameters& params,
                                 JohnsonMehlReport* report) {
    return GenerateJohnsonMehlSeeds(static_cast<const mfem::Mesh&>(pmesh), pmesh.GetComm(),
                                    params, report);
}

} // namespace exaconstit::microstructure
