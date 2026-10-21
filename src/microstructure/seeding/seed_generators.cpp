#include "microstructure/seeding/seed_generators.hpp"

#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/seeding/boundary_clearance.hpp"
#include "microstructure/seeding/material_point_locator.hpp"
#include "microstructure/seeding/min_distance_grid.hpp"
#include "microstructure/util/counter_rng.hpp"
#include "microstructure/util/fixed_point.hpp"
#include "microstructure/util/random_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

/*
 * Implementation notes (see seed_generators.hpp for the user-level description).
 *
 * All three generators (rejection, lattice, element-weighted) produce a *candidate stream*: an
 * ordered list of points with a global candidate index, identical on every rank. The stream is
 * cut into batches; for each batch
 *   1. every rank sets bits in a two-part bitmask: "inside the material" (its own elements) and
 *      "too close to the surface" (its own surface triangles);
 *   2. one MPI_Allreduce(MPI_BOR) makes the mask global (a bit is set if any rank set it);
 *   3. every rank walks the batch in index order and accepts candidates that are inside, not too
 *      close to the surface and, with RSA, at least d away from the seeds accepted so far
 *      (SeedAcceptor).
 * Steps 2 and 3 use only replicated data, so every rank accepts exactly the same seeds, and every
 * decision that ends the run (done, stalled, switch to the fallback, errors) is taken identically
 * on all ranks, which keeps the collective calls matched.
 */

namespace exaconstit::microstructure {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

/** Largest volume fraction N (pi/6) d^3 / V accepted by the RSA feasibility guard. */
constexpr double FEASIBLE_PACKING_FRACTION = 0.3;

/** Rejection candidates generated before the measured acceptance may trigger the fallback. */
constexpr std::int64_t FALLBACK_PROBE_CANDIDATES = std::int64_t(1) << 20;

/** Lower bound of the automatic stall cap. */
constexpr std::int64_t MIN_AUTO_CANDIDATE_CAP = std::int64_t(1) << 22;

/** Upper bound of any candidate count (keeps all products far from int64 overflow). */
constexpr std::int64_t MAX_CANDIDATES = std::int64_t(1) << 40;

/**
 * Largest allowed batch (2^24 candidates): keeps the MPI counts of the bitmask reduction
 * (2 * 2^24 / 64 words) and of the element-weighted gather far below INT_MAX.
 */
constexpr std::int64_t MAX_BATCH_SIZE = std::int64_t(1) << 24;

/** Attempts of the in-box rejection that places a point uniformly inside a hex/curved element. */
constexpr int MAX_IN_ELEMENT_ATTEMPTS = 1000;

/** Human-readable placement name for messages. */
const char* PlacementName(SeedPlacement placement) {
    switch (placement) {
    case SeedPlacement::REJECTION:
        return "rejection";
    case SeedPlacement::ELEMENT_WEIGHTED:
        return "element_weighted";
    case SeedPlacement::LATTICE:
        return "lattice";
    case SeedPlacement::JOHNSON_MEHL:
        return "johnson_mehl";
    case SeedPlacement::FILE:
        return "file";
    }
    return "unknown";
}

/*
 * ThrowIfAnyRankFailed (collective)
 *
 * Local setup (element boxes, element sampler, surface triangles) can fail on some ranks only,
 * e.g. when only one partition holds an element of an unsupported geometry. If such a rank simply
 * threw, the other ranks would wait forever in the next collective call. Instead each rank passes
 * its own error message (empty = no error) and:
 *  1. MPI_Allreduce(MPI_MIN) of "my rank if I failed, else a huge value" finds the lowest failing
 *     rank (or none);
 *  2. that rank broadcasts the length and text of its message;
 *  3. every rank throws std::runtime_error with that message.
 */
void ThrowIfAnyRankFailed(MPI_Comm comm, const std::string& local_error) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    const int no_failure = std::numeric_limits<int>::max();

    // Step 1: lowest failing rank
    int candidate = local_error.empty() ? no_failure : rank;
    int failing_rank = no_failure;
    MPI_Allreduce(&candidate, &failing_rank, 1, MPI_INT, MPI_MIN, comm);
    if (failing_rank == no_failure) {
        return;
    }

    // Step 2: broadcast its message
    int length = (rank == failing_rank) ? static_cast<int>(local_error.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, failing_rank, comm);
    std::string message = (rank == failing_rank) ? local_error : std::string(length, ' ');
    MPI_Bcast(message.data(), length, MPI_CHAR, failing_rank, comm);

    // Step 3: the same exception everywhere
    throw std::runtime_error(message + " (reported by rank " + std::to_string(failing_rank) + ")");
}

/**
 * Candidates of one batch: positions plus their global candidate index (the future source_id).
 * Host, replicated on every rank.
 */
struct CandidateBatch {
    std::vector<double> x, y, z;
    std::vector<std::int64_t> index;

    std::int64_t Size() const {
        return static_cast<std::int64_t>(x.size());
    }
    void Clear() {
        x.clear();
        y.clear();
        z.clear();
        index.clear();
    }
    void Push(double px, double py, double pz, std::int64_t k) {
        x.push_back(px);
        y.push_back(py);
        z.push_back(pz);
        index.push_back(k);
    }
};

/**
 * Two-part candidate bitmask of one batch: bit i of part 0 = "candidate i is in the domain",
 * bit i of part 1 = "candidate i is closer than the clearance to the surface". Each part has
 * `words` 64-bit words; candidate i is bit i % 64 of word i / 64 (least significant bit first).
 * Both parts live in one array so a single MPI_Allreduce combines them.
 */
struct BatchMask {
    std::int64_t words = 0;
    std::vector<std::uint64_t> bits;

    void Reset(std::int64_t n) {
        words = (n + 63) / 64;
        bits.assign(static_cast<std::size_t>(2 * words), 0);
    }
    std::uint64_t* InDomain() {
        return bits.data();
    }
    std::uint64_t* NearSurface() {
        return bits.data() + words;
    }
    bool InDomain(std::int64_t i) const {
        return (bits[static_cast<std::size_t>(i / 64)] >> (i % 64)) & 1u;
    }
    bool NearSurface(std::int64_t i) const {
        return (bits[static_cast<std::size_t>(words + i / 64)] >> (i % 64)) & 1u;
    }
    void SetInDomain(std::int64_t i) {
        bits[static_cast<std::size_t>(i / 64)] |= std::uint64_t(1) << (i % 64);
    }
};

/** Global facts about the distributed mesh, identical on every rank. */
struct MeshSummary {
    std::array<double, 3> lo; ///< global bounding box of the vertices / sampled element points
    std::array<double, 3> hi;
    /**
     * Union of the (padded, for curved elements) element boxes: every point of the material lies
     * in it. Equal to [lo, hi] for straight-sided meshes.
     */
    std::array<double, 3> reach_lo;
    std::array<double, 3> reach_hi;
    double volume = 0.0;                  ///< material volume V (fixed-point sum)
    std::int64_t num_elements = 0;        ///< global element count
    std::int64_t num_nonpositive = 0;     ///< global count of elements with volume <= 0 (or NaN)
};

/*
 * SummarizeMesh (collective)
 *
 * Steps:
 *  1. Global bounding boxes: MPI_MIN of the local lower corners and MPI_MAX of the upper corners
 *     (min and max are exact and order independent, so the boxes are the same for any rank
 *     count), both for the evaluated element points (lo, hi) and for the padded element boxes
 *     (reach_lo, reach_hi). A rank without elements contributes +inf / -inf, which leaves the
 *     result unchanged.
 *  2. Global element count and count of elements with non-positive volume (MPI_SUM), and the
 *     largest element volume (MPI_MAX), which sets the scale of the fixed-point sum.
 *  3. Material volume as a fixed-point sum: each element volume is rounded to an integer number
 *     of tiny "quanta" (util/fixed_point.hpp) and the integers are added. Integer addition does not
 *     depend on the order, unlike floating-point addition, so V (and everything derived from it:
 *     the RSA distance, batch sizes, the fallback decision) is bitwise identical for any rank
 *     count.
 */
MeshSummary SummarizeMesh(const MaterialPointLocator& locator, const ElementSamples& samples,
                          MPI_Comm comm) {
    MeshSummary summary;
    const int ne = locator.NumElements();

    // Step 1: global bounding boxes; entries 0-2 are the point box, 3-5 the padded box
    double lo[6];
    double hi[6];
    for (int d = 0; d < 6; d++) {
        lo[d] = std::numeric_limits<double>::infinity();
        hi[d] = -std::numeric_limits<double>::infinity();
    }
    if (ne > 0) {
        const auto& points = locator.LocalPointBounds();
        const auto& padded = locator.LocalBounds();
        for (int d = 0; d < 3; d++) {
            lo[d] = points.getMin()[d];
            hi[d] = points.getMax()[d];
            lo[3 + d] = padded.getMin()[d];
            hi[3 + d] = padded.getMax()[d];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, lo, 6, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, hi, 6, MPI_DOUBLE, MPI_MAX, comm);
    for (int d = 0; d < 3; d++) {
        summary.lo[d] = lo[d];
        summary.hi[d] = hi[d];
        summary.reach_lo[d] = lo[3 + d];
        summary.reach_hi[d] = hi[3 + d];
    }

    // Step 2: element counts and largest element volume
    const double* volume = (ne > 0) ? samples.volume.HostRead() : nullptr;
    std::int64_t local_counts[2] = {ne, 0}; // [elements, elements with volume <= 0]
    double max_volume = 0.0;
    for (int e = 0; e < ne; e++) {
        if (!(volume[e] > 0.0)) {
            local_counts[1]++;
        }
        max_volume = std::max(max_volume, std::fabs(volume[e]));
    }
    std::int64_t global_counts[2] = {0, 0};
    MPI_Allreduce(local_counts, global_counts, 2, MPI_INT64_T, MPI_SUM, comm);
    summary.num_elements = global_counts[0];
    summary.num_nonpositive = global_counts[1];
    MPI_Allreduce(MPI_IN_PLACE, &max_volume, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (summary.num_elements == 0 || !(max_volume > 0.0)) {
        return summary; // the caller reports the empty mesh
    }

    // Step 3: order-independent volume sum
    const FixedPointScale scale = FixedPointScale::ForRange(0.0, max_volume, summary.num_elements);
    std::int64_t local_sum = 0;
    for (int e = 0; e < ne; e++) {
        local_sum += scale.Encode(volume[e]);
    }
    std::int64_t global_sum = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, comm);
    summary.volume = scale.DecodeSum(global_sum, summary.num_elements);
    return summary;
}

/**
 * Accepts candidates in index order: in the domain, not near the surface, and (RSA) at least d
 * away from every accepted seed. Identical state on every rank. Host only.
 */
class SeedAcceptor {
public:
    /**
     * @param target  stop after this many seeds; < 0 means no limit (lattices).
     * @param min_distance RSA distance d (0: no RSA).
     * @param grid_lo,grid_hi box of the RSA hash grid (the period along periodic axes).
     * @param periodic periodic axes.
     */
    SeedAcceptor(std::int64_t target, double min_distance, const std::array<double, 3>& grid_lo,
                 const std::array<double, 3>& grid_hi, const std::array<bool, 3>& periodic)
        : m_target(target) {
        if (min_distance > 0.0) {
            m_grid = std::make_unique<MinDistanceGrid>(grid_lo, grid_hi, min_distance, periodic);
        }
    }

    /** True once the target count is reached. */
    bool Full() const {
        return m_target >= 0 && static_cast<std::int64_t>(m_seeds.Size()) >= m_target;
    }

    std::int64_t NumAccepted() const {
        return static_cast<std::int64_t>(m_seeds.Size());
    }

    /*
     * Offer: walk the batch in order, stopping as soon as the target is reached or `cap`
     * candidates have been examined in total (report.candidates_generated). The remaining
     * candidates of the batch are not counted as examined. Stopping at exactly `cap` makes the
     * stall outcome independent of how the stream was cut into batches or windows (whose sizes
     * may depend on floating-point sums). Updates the report counters.
     */
    void Offer(const CandidateBatch& batch, const BatchMask& mask, std::int64_t cap,
               SeedingReport& report) {
        for (std::int64_t i = 0; i < batch.Size(); i++) {
            if (Full() || report.candidates_generated >= cap) {
                return;
            }
            report.candidates_generated++;
            if (!mask.InDomain(i)) {
                continue;
            }
            report.candidates_in_domain++;
            if (mask.NearSurface(i)) {
                report.rejected_by_clearance++;
                continue;
            }
            const std::array<double, 3> p = {batch.x[i], batch.y[i], batch.z[i]};
            if (m_grid) {
                if (m_grid->HasNeighborCloserThan(p)) {
                    report.rejected_by_distance++;
                    continue;
                }
                m_grid->Insert(p);
            }
            m_seeds.x.push_back(p[0]);
            m_seeds.y.push_back(p[1]);
            m_seeds.z.push_back(p[2]);
            m_seeds.source_id.push_back(batch.index[i]);
        }
    }

    SeedSet& Seeds() {
        return m_seeds;
    }

private:
    std::int64_t m_target;
    std::unique_ptr<MinDistanceGrid> m_grid;
    SeedSet m_seeds;
};

/** Shared state of one GenerateSeeds call. */
struct SeedingContext {
    MPI_Comm comm;
    const SeedingParameters* params;
    const MaterialPointLocator* locator;
    const BoundaryClearance* clearance; ///< null when boundary_clearance = 0
    bool material_mode;                 ///< sampling_domain == "material"
    std::array<double, 3> box_lo;       ///< sampling box
    std::array<double, 3> box_hi;
    double volume_ref;                  ///< V used for d, lattice spacing and the guard
    double min_distance;                ///< d
    std::int64_t candidate_cap;         ///< stall cap
};

/*
 * ClassifyBatch (collective)
 *
 * Fills the two-part mask for a batch:
 *  - in-domain part: with test_inside, each rank marks candidates inside its own elements
 *    (MaterialPointLocator::MarkInside); otherwise every candidate is in the domain, except that
 *    a user sampling box can be imposed with filter_box (element-weighted points come from the
 *    whole material);
 *  - surface part: each rank marks candidates closer than the clearance to its own surface
 *    triangles.
 * When any part is rank-local, one MPI_Allreduce(MPI_BOR) over both parts (2 * words 64-bit
 * words) gives every rank the global mask. Bits set identically on all ranks (filter_box) are
 * unchanged by the OR.
 */
void ClassifyBatch(const SeedingContext& ctx, const CandidateBatch& batch, bool test_inside,
                   bool filter_box, BatchMask& mask) {
    const std::int64_t n = batch.Size();
    mask.Reset(n);

    // In-domain bits
    if (test_inside) {
        ctx.locator->MarkInside(n, batch.x.data(), batch.y.data(), batch.z.data(),
                                mask.InDomain());
    } else {
        for (std::int64_t i = 0; i < n; i++) {
            bool in_box = true;
            if (filter_box) {
                const double p[3] = {batch.x[i], batch.y[i], batch.z[i]};
                for (int d = 0; d < 3; d++) {
                    in_box = in_box && p[d] >= ctx.box_lo[d] && p[d] <= ctx.box_hi[d];
                }
            }
            if (in_box) {
                mask.SetInDomain(i);
            }
        }
    }

    // Near-surface bits
    if (ctx.clearance) {
        ctx.clearance->MarkNear(n, batch.x.data(), batch.y.data(), batch.z.data(),
                                mask.NearSurface());
    }

    // Global OR over ranks when any bit is rank-local
    if ((test_inside || ctx.clearance) && mask.words > 0) {
        // 2 * words <= 2 * 2^24 / 64 = 2^19 (max_batch_size is capped at 2^24), so it fits an int
        MPI_Allreduce(MPI_IN_PLACE, mask.bits.data(), static_cast<int>(2 * mask.words),
                      MPI_UINT64_T, MPI_BOR, ctx.comm);
    }
}

/** How a generator run ended. */
enum class RunStatus { DONE, STALLED, FALLBACK };

/*
 * Batch size for the next batch: enough candidates to finish with ~20% margin at the acceptance
 * rate seen so far (or the a-priori estimate before the first batch), clamped to
 * [min(1024, max_batch), max_batch] and to the candidates left under the cap. It only affects
 * speed and memory, never which seeds are accepted, because candidates are always taken in index
 * order.
 */
std::int64_t NextBatchSize(const SeedingContext& ctx, std::int64_t remaining, double rate,
                           std::int64_t left_under_cap) {
    const double wanted = 1.2 * static_cast<double>(remaining) / std::max(rate, 1.0e-12) + 256.0;
    const std::int64_t max_batch = ctx.params->max_batch_size;
    const std::int64_t min_batch = std::min<std::int64_t>(1024, max_batch);
    std::int64_t size = static_cast<std::int64_t>(std::min(wanted, double(max_batch)));
    size = std::max(size, min_batch);
    return std::max<std::int64_t>(1, std::min(size, left_under_cap));
}

/*
 * RunRejection: the default generator.
 *
 * Candidate k is c_k = lo + U_k (hi - lo), with U_k = UniformUnitCube of the RNG stream
 * (rng_seed, SEED_POSITION, k), sample 0. Batches of consecutive k are classified (in material
 * mode: point-in-material test on every rank + OR reduction) and offered in order.
 *
 * Ends with DONE (N seeds), STALLED (cap reached), or FALLBACK: in material mode, once at least
 * 2^20 candidates were examined, if the measured fraction inside the material is below
 * fallback_acceptance (the caller then restarts with element-weighted sampling).
 */
RunStatus RunRejection(const SeedingContext& ctx, double prior_rate, SeedAcceptor& acceptor,
                       SeedingReport& report) {
    const SeedingParameters& params = *ctx.params;
    CandidateBatch batch;
    BatchMask mask;
    std::int64_t next_k = 0;

    while (!acceptor.Full()) {
        // Stop conditions, evaluated on replicated counters (same decision on every rank)
        if (next_k >= ctx.candidate_cap) {
            return RunStatus::STALLED;
        }
        if (ctx.material_mode && params.fallback_acceptance > 0.0 &&
            report.candidates_generated >= FALLBACK_PROBE_CANDIDATES &&
            static_cast<double>(report.candidates_in_domain) <
                params.fallback_acceptance * static_cast<double>(report.candidates_generated)) {
            return RunStatus::FALLBACK;
        }

        // Size of this batch from the acceptance rate so far
        const double rate =
            (report.candidates_generated > 0)
                ? double(acceptor.NumAccepted() + 1) / double(report.candidates_generated + 1)
                : prior_rate;
        const std::int64_t size = NextBatchSize(ctx, params.num_seeds - acceptor.NumAccepted(),
                                                rate, ctx.candidate_cap - next_k);

        // Candidate positions c_k = lo + U_k (hi - lo)
        batch.Clear();
        for (std::int64_t i = 0; i < size; i++) {
            const std::int64_t k = next_k + i;
            const CounterRng rng(params.rng_seed, RngPurpose::SEED_POSITION,
                                 static_cast<std::uint64_t>(k));
            double u[3];
            UniformUnitCube(rng, 0, u);
            batch.Push(ctx.box_lo[0] + u[0] * (ctx.box_hi[0] - ctx.box_lo[0]),
                       ctx.box_lo[1] + u[1] * (ctx.box_hi[1] - ctx.box_lo[1]),
                       ctx.box_lo[2] + u[2] * (ctx.box_hi[2] - ctx.box_lo[2]), k);
        }

        // Global classification, then in-order acceptance
        ClassifyBatch(ctx, batch, ctx.material_mode, false, mask);
        acceptor.Offer(batch, mask, ctx.candidate_cap, report);
        next_k += size;
    }
    return RunStatus::DONE;
}

/** One lattice: basis points in fractional cell coordinates and the cell shape. */
struct LatticeDefinition {
    std::vector<std::array<double, 3>> basis; ///< fractional positions in [0, 1)^3
    std::array<double, 3> cell_aspect;        ///< cell edge lengths divided by a
};

/*
 * Lattice definitions (conventional cells):
 *  - cubic: one point per cube of edge a;
 *  - bcc:   cube corners + body center (2 points per cube);
 *  - fcc:   cube corners + face centers (4 points per cube);
 *  - hcp:   orthohexagonal cell a x sqrt(3) a x c with the ideal c = sqrt(8/3) a and 4 points at
 *           (0,0,0), (1/2,1/2,0), (1/2,1/6,1/2), (0,2/3,1/2): close-packed layers A at z = 0 and B
 *           at z = c/2, nearest-neighbor distance a.
 */
LatticeDefinition MakeLattice(const std::string& type) {
    LatticeDefinition lattice;
    lattice.cell_aspect = {1.0, 1.0, 1.0};
    if (type == "cubic") {
        lattice.basis = {{0.0, 0.0, 0.0}};
    } else if (type == "bcc") {
        lattice.basis = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
    } else if (type == "fcc") {
        lattice.basis = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
    } else { // "hcp" (validated before)
        lattice.cell_aspect = {1.0, std::sqrt(3.0), std::sqrt(8.0 / 3.0)};
        lattice.basis = {{0.0, 0.0, 0.0},
                         {0.5, 0.5, 0.0},
                         {0.5, 1.0 / 6.0, 0.5},
                         {0.0, 2.0 / 3.0, 0.5}};
    }
    return lattice;
}

/*
 * RunLattice
 *
 * Steps:
 *  1. Lattice parameter a from the target count: the cell holds n_b points and has volume
 *     a^3 * aspect_x * aspect_y * aspect_z, so N points in volume V need
 *     a = (n_b V / (N * aspect_x aspect_y aspect_z))^(1/3).
 *  2. Lattice points lo + (i + f_b + 1/4) * cell, for cells i = 0 .. ceil(extent / cell) along each
 *     axis and basis points f_b. The 1/4-cell shift keeps all points off the faces of a box whose
 *     edges are whole multiples of the cell (so a [0,1]^3 box with a = 1/4 holds exactly
 *     4^3 n_b points, none of them on the boundary).
 *  3. Optional jitter: an offset (U - 1/2) * jitter * a per axis, U from the RNG stream
 *     (rng_seed, LATTICE_JITTER, lattice index).
 *  4. Points outside the sampling box are skipped; the rest are classified (inside the material
 *     in material mode, surface clearance) and offered like any candidate. All points are kept;
 *     num_seeds only sets the spacing.
 *
 * Lattice index = ((iz * n_y + iy) * n_x + ix) * n_b + b, which is the source_id of the seed.
 */
void RunLattice(const SeedingContext& ctx, SeedAcceptor& acceptor, SeedingReport& report) {
    const SeedingParameters& params = *ctx.params;
    const LatticeDefinition lattice = MakeLattice(params.lattice_type);
    const std::int64_t nb = static_cast<std::int64_t>(lattice.basis.size());

    // Step 1: lattice parameter and cell edge lengths
    const double aspect_volume =
        lattice.cell_aspect[0] * lattice.cell_aspect[1] * lattice.cell_aspect[2];
    const double a = std::cbrt(double(nb) * ctx.volume_ref /
                               (double(params.num_seeds) * aspect_volume));
    report.lattice_parameter = a;
    std::array<double, 3> cell;
    std::array<std::int64_t, 3> num_cells;
    double total = double(nb);
    for (int d = 0; d < 3; d++) {
        cell[d] = a * lattice.cell_aspect[d];
        const double extent = ctx.box_hi[d] - ctx.box_lo[d];
        num_cells[d] = static_cast<std::int64_t>(std::ceil(extent / cell[d])) + 1;
        total *= double(num_cells[d]);
    }
    if (total > double(MAX_CANDIDATES)) {
        throw std::runtime_error("GenerateSeeds: lattice spacing too small for the sampling box: "
                                 "reduce num_seeds or set a smaller sampling_box");
    }
    const std::int64_t num_points = static_cast<std::int64_t>(total);

    CandidateBatch batch;
    BatchMask mask;
    for (std::int64_t start = 0; start < num_points; start += params.max_batch_size) {
        const std::int64_t end = std::min(num_points, start + params.max_batch_size);
        batch.Clear();
        for (std::int64_t idx = start; idx < end; idx++) {
            // Step 2: decompose the lattice index into (cell ix, iy, iz; basis b)
            const std::int64_t b = idx % nb;
            std::int64_t rest = idx / nb;
            const std::int64_t ix = rest % num_cells[0];
            rest /= num_cells[0];
            const std::int64_t iy = rest % num_cells[1];
            const std::int64_t iz = rest / num_cells[1];
            const std::int64_t cell_index[3] = {ix, iy, iz};

            // Step 3: jitter offsets (zero without jitter)
            double jitter[3] = {0.0, 0.0, 0.0};
            if (params.lattice_jitter > 0.0) {
                const CounterRng rng(params.rng_seed, RngPurpose::LATTICE_JITTER,
                                     static_cast<std::uint64_t>(idx));
                double u[3];
                UniformUnitCube(rng, 0, u);
                for (int d = 0; d < 3; d++) {
                    jitter[d] = (u[d] - 0.5) * params.lattice_jitter * a;
                }
            }

            // Position, then step 4: drop points outside the sampling box
            double p[3];
            bool in_box = true;
            for (int d = 0; d < 3; d++) {
                p[d] = ctx.box_lo[d] +
                       (double(cell_index[d]) + lattice.basis[b][d] + 0.25) * cell[d] + jitter[d];
                in_box = in_box && p[d] >= ctx.box_lo[d] && p[d] <= ctx.box_hi[d];
            }
            if (in_box) {
                batch.Push(p[0], p[1], p[2], idx);
            }
        }
        ClassifyBatch(ctx, batch, ctx.material_mode, false, mask);
        acceptor.Offer(batch, mask, ctx.candidate_cap, report);
    }
}

/** splitmix64 finalizer: a fixed bijective mixing of 64-bit integers into well-spread hashes. */
std::uint64_t Mix64(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/** One point drawn by an element in element-weighted sampling. Plain data (sent as bytes). */
struct Arrival {
    double t;          ///< arrival time (the sort key)
    std::uint64_t key; ///< element key (tie-break)
    std::uint32_t j;   ///< arrival number within the element (tie-break)
    double x, y, z;    ///< position
};

/** Strict total order of arrivals: time, then element key, then arrival number, then position. */
bool ArrivalBefore(const Arrival& a, const Arrival& b) {
    if (a.t != b.t) {
        return a.t < b.t;
    }
    if (a.key != b.key) {
        return a.key < b.key;
    }
    if (a.j != b.j) {
        return a.j < b.j;
    }
    if (a.x != b.x) {
        return a.x < b.x;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    return a.z < b.z;
}

/*
 * SamplePointInElement: uniform point in element e for arrival j of its stream.
 *  - straight-sided tet: the "folding" method (Rocchini and Cignoni, "Generating random points in
 *    a tetrahedron", J. Graphics Tools 2000): three uniforms (s, t, u) in the unit cube are folded
 *    into the corner simplex s + t + u <= 1 by two reflections that preserve volume, giving
 *    uniform barycentric coordinates (1 - s - t - u, s, t, u);
 *  - any other element: uniform points in the element's bounding box until one is inside the
 *    element (exact test), at most MAX_IN_ELEMENT_ATTEMPTS times; if all fail (only for an
 *    extremely thin element) the element centroid is used.
 * RNG blocks: (j, 0) is the arrival time; attempt m uses blocks (j, 1 + 2m) and (j, 2 + 2m).
 */
void SamplePointInElement(const mfem::Mesh& mesh, const MaterialPointLocator& locator, int e,
                          const CounterRng& rng, std::uint32_t j, const double centroid[3],
                          double out[3]) {
    if (locator.StraightSided() && mesh.GetElementGeometry(e) == mfem::Geometry::TETRAHEDRON) {
        const PhiloxBlock b0 = rng.Block(j, 1);
        const PhiloxBlock b1 = rng.Block(j, 2);
        double s = BitsToUnitDouble(b0.v[0], b0.v[1]);
        double t = BitsToUnitDouble(b0.v[2], b0.v[3]);
        double u = BitsToUnitDouble(b1.v[0], b1.v[1]);
        // First fold: into the prism s + t <= 1
        if (s + t > 1.0) {
            s = 1.0 - s;
            t = 1.0 - t;
        }
        // Second fold: into the tetrahedron s + t + u <= 1
        if (t + u > 1.0) {
            const double tmp = u;
            u = 1.0 - s - t;
            t = 1.0 - tmp;
        } else if (s + t + u > 1.0) {
            const double tmp = u;
            u = s + t + u - 1.0;
            s = 1.0 - t - tmp;
        }
        const double w0 = 1.0 - s - t - u;
        const int* v = mesh.GetElement(e)->GetVertices();
        const double* v0 = mesh.GetVertex(v[0]);
        const double* v1 = mesh.GetVertex(v[1]);
        const double* v2 = mesh.GetVertex(v[2]);
        const double* v3 = mesh.GetVertex(v[3]);
        for (int d = 0; d < 3; d++) {
            out[d] = w0 * v0[d] + s * v1[d] + t * v2[d] + u * v3[d];
        }
        return;
    }

    // Rejection inside the element box
    const auto& box = locator.ElementBox(e);
    for (int m = 0; m < MAX_IN_ELEMENT_ATTEMPTS; m++) {
        const PhiloxBlock b0 = rng.Block(j, static_cast<std::uint32_t>(1 + 2 * m));
        const PhiloxBlock b1 = rng.Block(j, static_cast<std::uint32_t>(2 + 2 * m));
        const double u[3] = {BitsToUnitDouble(b0.v[0], b0.v[1]),
                             BitsToUnitDouble(b0.v[2], b0.v[3]),
                             BitsToUnitDouble(b1.v[0], b1.v[1])};
        std::array<double, 3> p;
        for (int d = 0; d < 3; d++) {
            p[d] = box.getMin()[d] + u[d] * (box.getMax()[d] - box.getMin()[d]);
        }
        if (locator.PointInElement(e, p)) {
            out[0] = p[0];
            out[1] = p[1];
            out[2] = p[2];
            return;
        }
    }
    out[0] = centroid[0];
    out[1] = centroid[1];
    out[2] = centroid[2];
}

/*
 * ElementKey: a 64-bit identity of element e that does not depend on the partition or on the
 * execution space.
 *
 * The key hashes exact input bits, not computed values: the vertex coordinates of the element
 * (copied bit for bit from the serial mesh into every partition), sorted lexicographically so the
 * local vertex order does not matter, and folded with the splitmix64 mixer over their raw IEEE bit
 * patterns. In a conforming mesh no two elements have the same vertex set, so distinct elements get
 * distinct keys up to a 64-bit hash collision (probability ~ n_elements^2 / 2^65). Using all
 * vertices (not only the smallest one) matters: the 6 tets of a split hex share their smallest
 * vertex. A computed centroid would not do: CPU and GPU round-off can differ in the last bit and
 * change a quantized value.
 */
std::uint64_t ElementKey(const mfem::Mesh& mesh, int e) {
    mfem::Array<int> vertices;
    mesh.GetElementVertices(e, vertices);
    std::vector<std::array<double, 3>> coords(static_cast<std::size_t>(vertices.Size()));
    for (int v = 0; v < vertices.Size(); v++) {
        const double* x = mesh.GetVertex(vertices[v]);
        coords[static_cast<std::size_t>(v)] = {x[0], x[1], x[2]};
    }
    std::sort(coords.begin(), coords.end()); // lexicographic (x, then y, then z)

    std::uint64_t h = 0x9E3779B97F4A7C15ULL; // arbitrary nonzero start value
    for (const auto& c : coords) {
        for (int d = 0; d < 3; d++) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &c[d], sizeof(bits)); // exact bit pattern of the coordinate
            h = Mix64(h ^ bits);
        }
    }
    return h;
}

/*
 * RunElementWeighted
 *
 * The idea in plain words: imagine every element firing points at random times, independently,
 * at a rate proportional to its volume (a Poisson process in space and time). Then, at any time,
 * the next point fired is in element e with probability V_e / V and is uniform inside it, so the
 * points listed in order of firing time are independent and uniform in the material, exactly
 * like accepted rejection-sampling candidates, but with no wasted candidates.
 *
 * Steps:
 *  1. Element key (ElementKey): a hash of the element's exact vertex coordinates, the same for
 *     any rank count and execution space. The element's RNG stream is
 *     (rng_seed, ELEMENT_WEIGHTED_POSITION, key). That purpose is reserved for these streams, so
 *     they never coincide with a rejection candidate's (SEED_POSITION, k) stream.
 *  2. Arrival times of element e: t_{j+1} = t_j + E_j / V_e with E_j = -ln(1 - U_j) an
 *     exponential random number (the waiting time between events of a Poisson process).
 *     Elements that cannot contribute (box outside a user sampling box, or volume <= 0, which
 *     GenerateSeeds already rejects) get t = +infinity and never emit.
 *  3. Time windows [T, T + W): each rank emits the arrivals of its elements in the window,
 *     MPI_Allgatherv collects them on every rank, and all ranks sort them by
 *     (time, key, j). The window width W = (expected points needed) / V_active, so a window holds
 *     about the points still needed (with a 20% margin, capped at max_batch_size). Windows are
 *     consecutive and disjoint in time, so the global time order does not depend on the window
 *     widths; together with Offer stopping at exactly the stall cap, the result (including a
 *     stall) does not depend on them either. That is why V_active may be a plain floating-point
 *     sum, whose last bits change with the rank count.
 *  4. The sorted arrivals get consecutive stream indices (their source_id), are classified
 *     (user sampling box, surface clearance) and offered in order.
 */
RunStatus RunElementWeighted(const SeedingContext& ctx, const mfem::Mesh& mesh,
                             const ElementSamples& samples, SeedAcceptor& acceptor,
                             SeedingReport& report) {
    const SeedingParameters& params = *ctx.params;
    const int ne = ctx.locator->NumElements();
    const double* volume = (ne > 0) ? samples.volume.HostRead() : nullptr;
    const double* cx = (ne > 0) ? samples.cx.HostRead() : nullptr;
    const double* cy = (ne > 0) ? samples.cy.HostRead() : nullptr;
    const double* cz = (ne > 0) ? samples.cz.HostRead() : nullptr;
    const double infinity = std::numeric_limits<double>::infinity();

    // Step 1: element keys and which elements contribute
    std::vector<std::uint64_t> key(static_cast<std::size_t>(ne));
    std::vector<char> active(static_cast<std::size_t>(ne), 1);
    for (int e = 0; e < ne; e++) {
        const auto ue = static_cast<std::size_t>(e);
        key[ue] = ElementKey(mesh, e);
        if (!(volume[e] > 0.0)) {
            active[ue] = 0;
        }
        if (params.use_sampling_box) {
            const auto& box = ctx.locator->ElementBox(e);
            for (int d = 0; d < 3; d++) {
                if (box.getMax()[d] < ctx.box_lo[d] || box.getMin()[d] > ctx.box_hi[d]) {
                    active[ue] = 0;
                }
            }
        }
    }

    // Volume of the contributing elements. It only sets the window width (see step 3), so a plain
    // floating-point sum is fine; zero means the user box misses the material and no point could
    // ever be drawn.
    double active_volume = 0.0;
    for (int e = 0; e < ne; e++) {
        if (active[static_cast<std::size_t>(e)]) {
            active_volume += volume[e];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &active_volume, 1, MPI_DOUBLE, MPI_SUM, ctx.comm);
    if (!(active_volume > 0.0)) {
        throw std::runtime_error("GenerateSeeds: the sampling box does not overlap the material");
    }

    // Step 2 setup: first arrival time of every contributing element, +inf for the others
    auto exponential = [&](int e, std::uint32_t j) {
        const CounterRng rng(params.rng_seed, RngPurpose::ELEMENT_WEIGHTED_POSITION,
                             key[static_cast<std::size_t>(e)]);
        const PhiloxBlock b = rng.Block(j, 0);
        return -std::log(1.0 - BitsToUnitDouble(b.v[0], b.v[1]));
    };
    std::vector<double> next_t(static_cast<std::size_t>(ne), infinity);
    std::vector<std::uint32_t> next_j(static_cast<std::size_t>(ne), 0);
    for (int e = 0; e < ne; e++) {
        if (active[static_cast<std::size_t>(e)]) {
            next_t[static_cast<std::size_t>(e)] = exponential(e, 0) / volume[e];
        }
    }

    // MPI datatype of one Arrival (its bytes, contiguous), so the gather counts and displacements
    // are in arrivals, not bytes, and stay far below INT_MAX
    MPI_Datatype arrival_type;
    MPI_Type_contiguous(static_cast<int>(sizeof(Arrival)), MPI_BYTE, &arrival_type);
    MPI_Type_commit(&arrival_type);

    int nranks = 1;
    MPI_Comm_size(ctx.comm, &nranks);
    std::vector<int> counts(static_cast<std::size_t>(nranks));
    std::vector<int> displs(static_cast<std::size_t>(nranks));
    std::vector<Arrival> local;
    std::vector<Arrival> all;
    CandidateBatch batch;
    BatchMask mask;
    double window_start = 0.0;
    std::int64_t stream_position = 0;
    RunStatus status = RunStatus::DONE;

    while (!acceptor.Full()) {
        if (report.candidates_generated >= ctx.candidate_cap) {
            status = RunStatus::STALLED;
            break;
        }

        // Step 3a: window width from the points still needed and the acceptance so far
        const double rate =
            (report.candidates_generated > 0)
                ? double(acceptor.NumAccepted() + 1) / double(report.candidates_generated + 1)
                : 1.0;
        const double wanted =
            std::min(1.2 * double(params.num_seeds - acceptor.NumAccepted()) / rate + 256.0,
                     double(params.max_batch_size));
        const double window_end = window_start + wanted / active_volume;

        // Step 3b: this rank's arrivals in [window_start, window_end) (inactive elements have
        // next_t = +inf and are skipped by the while condition)
        local.clear();
        for (int e = 0; e < ne; e++) {
            const auto ue = static_cast<std::size_t>(e);
            while (next_t[ue] < window_end) {
                const std::uint32_t j = next_j[ue];
                const CounterRng rng(params.rng_seed, RngPurpose::ELEMENT_WEIGHTED_POSITION,
                                     key[ue]);
                const double centroid[3] = {cx[e], cy[e], cz[e]};
                Arrival a;
                a.t = next_t[ue];
                a.key = key[ue];
                a.j = j;
                double p[3];
                SamplePointInElement(mesh, *ctx.locator, e, rng, j, centroid, p);
                a.x = p[0];
                a.y = p[1];
                a.z = p[2];
                local.push_back(a);
                next_j[ue] = j + 1;
                next_t[ue] += exponential(e, j + 1) / volume[e];
            }
        }

        // Step 3c: gather all arrivals on every rank and sort them. A window expects at most
        // max_batch_size (<= 2^24) arrivals, so the int counts cannot overflow in practice; the
        // check below turns a pathological case into the same error on every rank.
        const int my_count = static_cast<int>(local.size());
        MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, ctx.comm);
        std::int64_t total = 0;
        for (int r = 0; r < nranks; r++) {
            displs[static_cast<std::size_t>(r)] = static_cast<int>(total);
            total += counts[static_cast<std::size_t>(r)];
        }
        if (total > std::numeric_limits<int>::max()) {
            MPI_Type_free(&arrival_type);
            throw std::runtime_error("GenerateSeeds: too many element-weighted points in one "
                                     "window; reduce max_batch_size");
        }
        all.resize(static_cast<std::size_t>(total));
        MPI_Allgatherv(local.data(), my_count, arrival_type, all.data(), counts.data(),
                       displs.data(), arrival_type, ctx.comm);
        std::sort(all.begin(), all.end(), ArrivalBefore);

        // Step 4: stream indices, classification and in-order acceptance
        batch.Clear();
        for (const Arrival& a : all) {
            batch.Push(a.x, a.y, a.z, stream_position++);
        }
        ClassifyBatch(ctx, batch, false, params.use_sampling_box, mask);
        acceptor.Offer(batch, mask, ctx.candidate_cap, report);
        window_start = window_end;
    }
    MPI_Type_free(&arrival_type);
    return status;
}

} // namespace

/* MakeSeedingParameters: field-by-field copy of the [Seeds] options (including min_distance, the
 * user sampling box and allow_fewer_seeds) and Tessellation.periodic; the stall and batch
 * controls, which have no TOML key, keep their defaults. */
SeedingParameters MakeSeedingParameters(const MicrostructureOptions& options) {
    SeedingParameters params;
    params.placement = options.seeds.placement;
    params.num_seeds = options.seeds.num_seeds;
    params.rng_seed = options.seeds.rng_seed;
    params.min_distance_factor = options.seeds.min_distance_factor;
    params.min_distance = options.seeds.min_distance;
    params.boundary_clearance = options.seeds.boundary_clearance;
    params.sampling_domain = options.seeds.sampling_domain;
    params.use_sampling_box = options.seeds.use_sampling_box;
    params.sampling_box_lo = options.seeds.sampling_box_lo;
    params.sampling_box_hi = options.seeds.sampling_box_hi;
    params.allow_fewer_seeds = options.seeds.allow_fewer_seeds;
    params.lattice_type = options.seeds.lattice_type;
    params.lattice_jitter = options.seeds.lattice_jitter;
    params.periodic = options.tessellation.periodic;
    return params;
}

/* ValidateSeedingParameters: one check per field, first failure thrown. */
void ValidateSeedingParameters(const SeedingParameters& params) {
    auto fail = [](const std::string& what) {
        throw std::invalid_argument("GenerateSeeds: " + what);
    };
    if (params.num_seeds <= 0) {
        fail("num_seeds must be > 0");
    }
    if (params.num_seeds > std::numeric_limits<int>::max()) {
        fail("num_seeds must be <= 2^31 - 1");
    }
    if (params.placement != SeedPlacement::REJECTION &&
        params.placement != SeedPlacement::ELEMENT_WEIGHTED &&
        params.placement != SeedPlacement::LATTICE) {
        fail(std::string("placement \"") + PlacementName(params.placement) +
             "\" is not generated here (supported: rejection, element_weighted, lattice)");
    }
    if (params.sampling_domain != "material" && params.sampling_domain != "bbox") {
        fail("sampling_domain must be \"material\" or \"bbox\"");
    }
    if (params.placement == SeedPlacement::ELEMENT_WEIGHTED &&
        params.sampling_domain != "material") {
        fail("element_weighted placement requires sampling_domain = \"material\"");
    }
    if (!(params.min_distance >= 0.0) || !(params.min_distance_factor >= 0.0)) {
        fail("min_distance and min_distance_factor must be >= 0");
    }
    if (!(params.boundary_clearance >= 0.0)) {
        fail("boundary_clearance must be >= 0");
    }
    if (params.placement == SeedPlacement::LATTICE) {
        if (params.lattice_type != "cubic" && params.lattice_type != "bcc" &&
            params.lattice_type != "fcc" && params.lattice_type != "hcp") {
            fail("lattice_type must be \"cubic\", \"bcc\", \"fcc\" or \"hcp\"");
        }
        if (!(params.lattice_jitter >= 0.0)) {
            fail("lattice_jitter must be >= 0");
        }
    }
    if (params.use_sampling_box) {
        for (int d = 0; d < 3; d++) {
            if (!(params.sampling_box_hi[d] > params.sampling_box_lo[d])) {
                fail("the sampling box must have hi > lo along every axis");
            }
        }
    }
    if (params.max_candidates < 0) {
        fail("max_candidates must be >= 0 (0 = automatic)");
    }
    if (!(params.fallback_acceptance >= 0.0 && params.fallback_acceptance <= 1.0)) {
        fail("fallback_acceptance must be in [0, 1]");
    }
    if (params.max_batch_size < 64 || params.max_batch_size > MAX_BATCH_SIZE) {
        fail("max_batch_size must be in [64, 2^24]");
    }
}

/*
 * RoundBoxOutward
 *
 * q = 2^(floor(log2(largest edge)) - 6), i.e. between 1/128 and 1/64 of the largest edge. Each
 * lower corner goes down to floor(lo / q) q and each upper corner up to ceil(hi / q) q. Because q
 * is a power of two these products are exact in floating point, so the rounded box is an exact
 * function of the input box, and every input box inside the same q-grid cell gives the same
 * output.
 */
void RoundBoxOutward(std::array<double, 3>& lo, std::array<double, 3>& hi) {
    double extent = 0.0;
    for (int d = 0; d < 3; d++) {
        extent = std::max(extent, hi[d] - lo[d]);
    }
    if (!(extent > 0.0)) {
        // Degenerate box: use the coordinate magnitude (or 1) to pick a length scale
        extent = 1.0;
        for (int d = 0; d < 3; d++) {
            extent = std::max({extent, std::fabs(lo[d]), std::fabs(hi[d])});
        }
    }
    int exponent = 0;
    std::frexp(extent, &exponent); // extent = m 2^exponent, m in [0.5, 1)
    const double q = std::ldexp(1.0, exponent - 1 - 6);
    for (int d = 0; d < 3; d++) {
        lo[d] = std::floor(lo[d] / q) * q;
        hi[d] = std::ceil(hi[d] / q) * q;
        if (!(hi[d] > lo[d])) {
            hi[d] = lo[d] + q;
        }
    }
}

/* Summary: one line per fact, for the rank-0 log. */
std::string SeedingReport::Summary() const {
    std::ostringstream os;
    os << "Seeds: " << achieved << " of " << requested << " requested ("
       << PlacementName(placement_used) << (used_fallback ? ", element-weighted fallback" : "")
       << (stalled ? ", STALLED" : "") << ")\n";
    os << "  candidates examined " << candidates_generated << ", in domain "
       << candidates_in_domain << ", rejected by clearance " << rejected_by_clearance
       << ", rejected by min distance " << rejected_by_distance << "\n";
    os << "  material volume " << material_volume << ", min distance " << min_distance;
    if (lattice_parameter > 0.0) {
        os << ", lattice parameter " << lattice_parameter;
    }
    os << "\n  sampling box [" << sampling_lo[0] << ", " << sampling_hi[0] << "] x ["
       << sampling_lo[1] << ", " << sampling_hi[1] << "] x [" << sampling_lo[2] << ", "
       << sampling_hi[2] << "]\n";
    for (const std::string& note : notes) {
        os << "  note: " << note << "\n";
    }
    return os.str();
}

/*
 * GenerateSeeds (collective)
 *
 * Steps:
 *  1. Validate the parameters (identical on every rank, so all ranks throw together).
 *  2. Local setup: element boxes + BVH (MaterialPointLocator), element volumes and centroids
 *     (ComputeElementSamples). A failure on any rank (e.g. an unsupported geometry in one
 *     partition) is turned into the same exception on every rank by ThrowIfAnyRankFailed.
 *  3. Global mesh facts (SummarizeMesh): bounding boxes, element counts, material volume. Elements
 *     with volume <= 0 (inverted or degenerate) are an error on all ranks: they have no meaningful
 *     inside, and element-weighted sampling needs positive volumes.
 *  4. Sampling box: the user box, else the exact point bounding box (bbox mode, lattices) or the
 *     padded "reach" box rounded outward (rejection in material mode; for curved elements the
 *     padding makes bulges between sampled points reachable, for straight-sided meshes the two
 *     boxes coincide).
 *  5. RSA distance d and the feasibility guard N (pi/6) d^3 <= 0.3 V.
 *  6. Choose the generator: element-weighted if requested, or if rejection would accept fewer
 *     than fallback_acceptance of its candidates (estimated as V / V_box, an upper bound when a
 *     user box cuts the mesh).
 *  7. Run it; a measured-acceptance FALLBACK restarts with element-weighted sampling from
 *     scratch, so the result never mixes two streams.
 *  8. Stall handling: error, or (allow_fewer_seeds) a note in the report.
 */
SeedSet GenerateSeeds(mfem::Mesh& local_mesh, MPI_Comm comm, const SeedingParameters& params,
                      SeedingReport* report_out) {
    // Step 1
    ValidateSeedingParameters(params);
    SeedingReport report;
    report.requested = params.num_seeds;

    // Step 2: local setup, failures combined over ranks
    std::unique_ptr<MaterialPointLocator> locator_ptr;
    ElementSamples samples;
    std::string setup_error;
    try {
        locator_ptr = std::make_unique<MaterialPointLocator>(local_mesh);
        ComputeElementSamples(local_mesh, -1, samples);
    } catch (const std::exception& error) {
        setup_error = std::string("GenerateSeeds: mesh setup failed: ") + error.what();
    }
    ThrowIfAnyRankFailed(comm, setup_error);
    const MaterialPointLocator& locator = *locator_ptr;

    // Step 3: global facts
    const MeshSummary summary = SummarizeMesh(locator, samples, comm);
    if (summary.num_elements == 0) {
        throw std::runtime_error("GenerateSeeds: the mesh has no elements");
    }
    if (summary.num_nonpositive > 0) {
        throw std::runtime_error("GenerateSeeds: " + std::to_string(summary.num_nonpositive) +
                                 " element(s) have zero, negative or undefined volume (inverted "
                                 "or degenerate elements); check the mesh with CheckMesh");
    }
    if (!(summary.volume > 0.0)) {
        throw std::runtime_error("GenerateSeeds: the mesh has zero volume");
    }
    report.material_volume = summary.volume;
    report.bbox_lo = summary.lo;
    report.bbox_hi = summary.hi;

    // Step 4: sampling box
    SeedingContext ctx;
    ctx.comm = comm;
    ctx.params = &params;
    ctx.locator = &locator;
    ctx.clearance = nullptr;
    ctx.material_mode = (params.sampling_domain == "material");
    if (params.use_sampling_box) {
        ctx.box_lo = params.sampling_box_lo;
        ctx.box_hi = params.sampling_box_hi;
    } else {
        ctx.box_lo = summary.lo;
        ctx.box_hi = summary.hi;
        if (ctx.material_mode && params.placement != SeedPlacement::LATTICE) {
            ctx.box_lo = summary.reach_lo;
            ctx.box_hi = summary.reach_hi;
            RoundBoxOutward(ctx.box_lo, ctx.box_hi);
        }
    }
    double box_volume = 1.0;
    for (int d = 0; d < 3; d++) {
        box_volume *= (ctx.box_hi[d] - ctx.box_lo[d]);
    }
    if (!(box_volume > 0.0)) {
        throw std::runtime_error("GenerateSeeds: the sampling box has zero volume (flat mesh?)");
    }
    report.sampling_lo = ctx.box_lo;
    report.sampling_hi = ctx.box_hi;
    ctx.volume_ref = ctx.material_mode ? summary.volume : box_volume;

    // Step 5: RSA distance and feasibility guard
    ctx.min_distance = params.min_distance;
    if (ctx.min_distance == 0.0 && params.min_distance_factor > 0.0) {
        ctx.min_distance =
            params.min_distance_factor * std::cbrt(ctx.volume_ref / double(params.num_seeds));
    }
    report.min_distance = ctx.min_distance;
    if (ctx.min_distance > 0.0) {
        const double d = ctx.min_distance;
        const double n = double(params.num_seeds);
        const double fraction = n * (PI / 6.0) * d * d * d / ctx.volume_ref;
        if (fraction > FEASIBLE_PACKING_FRACTION) {
            const double d_max =
                std::cbrt(FEASIBLE_PACKING_FRACTION * ctx.volume_ref * 6.0 / (PI * n));
            const double n_max =
                FEASIBLE_PACKING_FRACTION * ctx.volume_ref * 6.0 / (PI * d * d * d);
            std::ostringstream os;
            os << "GenerateSeeds: minimum distance " << d << " is infeasible for "
               << params.num_seeds << " seeds in volume " << ctx.volume_ref
               << ": N (pi/6) d^3 / V = " << fraction
               << " exceeds " << FEASIBLE_PACKING_FRACTION
               << ". Use min_distance <= " << d_max << " (min_distance_factor <= "
               << d_max / std::cbrt(ctx.volume_ref / double(params.num_seeds))
               << ") or at most " << static_cast<std::int64_t>(n_max) << " seeds.";
            throw std::runtime_error(os.str());
        }
    }

    // Surface clearance (local surface triangles; built only when requested). Construction is
    // local and can fail on one rank only (unsupported face geometry), hence the combined check.
    std::unique_ptr<BoundaryClearance> clearance;
    if (params.boundary_clearance > 0.0) {
        std::string clearance_error;
        try {
            clearance =
                std::make_unique<BoundaryClearance>(local_mesh, params.boundary_clearance);
        } catch (const std::exception& error) {
            clearance_error =
                std::string("GenerateSeeds: surface clearance setup failed: ") + error.what();
        }
        ThrowIfAnyRankFailed(comm, clearance_error);
        ctx.clearance = clearance.get();
    }

    // RSA hash grid box: the period (exact mesh box) along periodic axes, the sampling box
    // elsewhere
    std::array<double, 3> grid_lo = ctx.box_lo;
    std::array<double, 3> grid_hi = ctx.box_hi;
    for (int d = 0; d < 3; d++) {
        if (params.periodic[d]) {
            grid_lo[d] = summary.lo[d];
            grid_hi[d] = summary.hi[d];
        }
    }

    // Step 6: choose the generator and the stall cap
    const double prior_rate =
        ctx.material_mode ? std::min(1.0, summary.volume / box_volume) : 1.0;
    SeedPlacement placement = params.placement;
    if (placement == SeedPlacement::REJECTION && ctx.material_mode &&
        prior_rate < params.fallback_acceptance) {
        placement = SeedPlacement::ELEMENT_WEIGHTED;
        report.used_fallback = true;
        std::ostringstream os;
        os << "the material fills only " << prior_rate
           << " of the sampling box; using element-weighted sampling";
        report.notes.push_back(os.str());
    }
    auto auto_cap = [&](double rate) {
        if (params.max_candidates > 0) {
            return params.max_candidates;
        }
        const double cap = std::max(double(MIN_AUTO_CANDIDATE_CAP),
                                    1000.0 * double(params.num_seeds) / std::max(rate, 1e-12));
        return static_cast<std::int64_t>(std::min(cap, double(MAX_CANDIDATES)));
    };

    // Step 7: run
    auto acceptor = std::make_unique<SeedAcceptor>(params.num_seeds, ctx.min_distance, grid_lo,
                                                   grid_hi, params.periodic);
    RunStatus status = RunStatus::DONE;
    if (placement == SeedPlacement::LATTICE) {
        ctx.candidate_cap = MAX_CANDIDATES;
        acceptor = std::make_unique<SeedAcceptor>(-1, ctx.min_distance, grid_lo, grid_hi,
                                                  params.periodic);
        RunLattice(ctx, *acceptor, report);
    } else {
        if (placement == SeedPlacement::REJECTION) {
            ctx.candidate_cap = auto_cap(prior_rate);
            status = RunRejection(ctx, prior_rate, *acceptor, report);
            if (status == RunStatus::FALLBACK) {
                std::ostringstream os;
                os << "only " << report.candidates_in_domain << " of "
                   << report.candidates_generated
                   << " rejection candidates were inside the material; restarted with "
                      "element-weighted sampling";
                report.notes.push_back(os.str());
                report.used_fallback = true;
                report.candidates_generated = 0;
                report.candidates_in_domain = 0;
                report.rejected_by_clearance = 0;
                report.rejected_by_distance = 0;
                acceptor = std::make_unique<SeedAcceptor>(params.num_seeds, ctx.min_distance,
                                                          grid_lo, grid_hi, params.periodic);
                placement = SeedPlacement::ELEMENT_WEIGHTED;
            }
        }
        if (placement == SeedPlacement::ELEMENT_WEIGHTED) {
            ctx.candidate_cap = auto_cap(1.0);
            status = RunElementWeighted(ctx, local_mesh, samples, *acceptor, report);
        }
    }
    report.placement_used = placement;
    report.achieved = acceptor->NumAccepted();

    // Step 8: stall handling (same counters on every rank, so all ranks agree)
    if (status == RunStatus::STALLED) {
        report.stalled = true;
        std::ostringstream os;
        os << "placed only " << report.achieved << " of " << params.num_seeds << " seeds after "
           << report.candidates_generated << " candidates (stall cap " << ctx.candidate_cap
           << "; in domain " << report.candidates_in_domain << ", rejected by min distance "
           << report.rejected_by_distance << ", by clearance " << report.rejected_by_clearance
           << ")";
        if (!params.allow_fewer_seeds) {
            throw std::runtime_error("GenerateSeeds: " + os.str() +
                                     ". Reduce min_distance/boundary_clearance or num_seeds.");
        }
        report.notes.push_back(os.str());
    }
    if (report_out) {
        *report_out = report;
    }
    return std::move(acceptor->Seeds());
}

/* ParMesh overload: the mesh's own communicator. */
SeedSet GenerateSeeds(mfem::ParMesh& pmesh, const SeedingParameters& params,
                      SeedingReport* report) {
    return GenerateSeeds(static_cast<mfem::Mesh&>(pmesh), pmesh.GetComm(), params, report);
}

} // namespace exaconstit::microstructure
