/*
 * Implementation of grain_statistics.hpp: per-grain element counts, volumes, centroids, neighbor
 * counts and shared face areas, all with fixed-point (order-independent) global sums, plus the
 * comparison of element-level seed volumes with target volumes.
 *
 * Terms used in this file:
 * - Fixed-point sum: every term is rounded once to a whole number of a tiny "quantum" and the
 *   integer counts are added; integer addition is exact, so the total does not depend on the
 *   order of the terms, i.e. on the mesh partition or the number of ranks (util/fixed_point.hpp).
 * - Ghost element: an element owned by another rank that shares a face with a local element
 *   (mesh/mesh_adjacency.hpp).
 * - Collective: a function every rank of the communicator must call, because ranks exchange data
 *   inside it.
 */
#include "microstructure/labeling/grain_statistics.hpp"

#include "microstructure/labeling/cell_measures.hpp"
#include "microstructure/util/fixed_point.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <map>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace exaconstit::microstructure {

namespace {

using Point3 = std::array<double, 3>;

/** Lexicographic "a < b": compare x, then y, then z. */
bool LexLess(const Point3& a, const Point3& b) {
    for (std::size_t d = 0; d < 3; ++d) {
        if (a[d] != b[d]) {
            return a[d] < b[d];
        }
    }
    return false;
}

/** b - a. */
Point3 Minus(const Point3& b, const Point3& a) {
    return {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
}

/** Euclidean norm of the cross product a x b. */
double CrossNorm(const Point3& a, const Point3& b) {
    const double cx = a[1] * b[2] - a[2] * b[1];
    const double cy = a[2] * b[0] - a[0] * b[2];
    const double cz = a[0] * b[1] - a[1] * b[0];
    return std::sqrt(cx * cx + cy * cy + cz * cz);
}

/*
 * MeshNodeOrder: polynomial order of the mesh geometry. 1 for a mesh without Nodes (vertices
 * only) and for a mesh whose Nodes are order 1 (e.g. read from a file with a nodes section);
 * the Nodes' order otherwise (FiniteElementCollection::GetOrder of the nodal space).
 */
int MeshNodeOrder(const mfem::Mesh& mesh) {
    const mfem::GridFunction* nodes = mesh.GetNodes();
    if (nodes == nullptr) {
        return 1;
    }
    return nodes->FESpace()->FEColl()->GetOrder();
}

/*
 * CanonicalFaceArea: area of local face f of a mesh with straight-sided (order <= 1) geometry,
 * computed from its corner points in a canonical order so the result is bitwise the same
 * whichever rank (and whichever stored orientation of the face) computes it.
 *
 * Steps:
 *  1. The face's corner points: the point matrix of Mesh::GetFaceTransformation (mesh.cpp:539-
 *     581). Without Nodes these are the face vertices in MFEM's cyclic order around the face;
 *     with order-1 Nodes they are the face's nodal points, which for an order-1 H1 space are
 *     the same corners in the same cyclic vertex order (vertex dofs first, in the reference
 *     element's vertex order). Using the point matrix rather than Mesh::GetVertex also follows
 *     Nodes that were moved after the mesh was read.
 *  2. Canonical order: start at the lexicographically smallest corner (compare x, then y, then
 *     z) and walk around the face toward the smaller of its two neighbors. Any rotation or
 *     reversal of the cyclic list gives the same canonical list, and a triangle or a bilinear
 *     quadrilateral is the same surface for every rotation/reversal of its corner list, so only
 *     the rounding is fixed by this choice.
 *  3. Triangle: |(q1 - q0) x (q2 - q0)| / 2.
 *     Quadrilateral: the bilinear map x(s, t) of the unit square with corners q0 (0,0), q1 (1,0),
 *     q2 (1,1), q3 (0,1); area = integral of |dx/ds x dx/dt| with mfem::IntRules' order-3 square
 *     rule (2 x 2 Gauss points), the order MeshAdjacency uses for linear meshes.
 */
double CanonicalFaceArea(const mfem::Mesh& mesh, int f) {
    // Step 1: corner points in cyclic order (column k of the point matrix is corner k)
    mfem::IsoparametricTransformation transformation;
    mesh.GetFaceTransformation(f, &transformation);
    const mfem::DenseMatrix& corners = transformation.GetPointMat();
    const int n = corners.Width();
    if ((n != 3 && n != 4) || corners.Height() != 3) {
        // Faces of a linear 3D mesh are triangles or quadrilaterals with 3 or 4 corner points,
        // so this cannot happen. A throw on one rank only would leave the others waiting in the
        // collective calls that follow, so abort instead (as MeshAdjacency does for impossible
        // face data).
        mfem::mfem_error("ComputeGrainStatistics: a face is neither a linear triangle nor a "
                         "linear quadrilateral");
    }
    std::vector<Point3> p(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        p[static_cast<std::size_t>(k)] = {corners(0, k), corners(1, k), corners(2, k)};
    }

    // Step 2: canonical start and direction
    int start = 0;
    for (int k = 1; k < n; ++k) {
        if (LexLess(p[static_cast<std::size_t>(k)], p[static_cast<std::size_t>(start)])) {
            start = k;
        }
    }
    const int next = (start + 1) % n;
    const int previous = (start + n - 1) % n;
    const int step = LexLess(p[static_cast<std::size_t>(next)],
                             p[static_cast<std::size_t>(previous)])
                         ? 1
                         : n - 1; // n - 1 steps forward = one step backward
    std::vector<Point3> q(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        q[static_cast<std::size_t>(k)] = p[static_cast<std::size_t>((start + k * step) % n)];
    }

    // Step 3: area of the triangle or the bilinear quadrilateral
    if (n == 3) {
        return 0.5 * CrossNorm(Minus(q[1], q[0]), Minus(q[2], q[0]));
    }
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(mfem::Geometry::SQUARE, 3);
    double area = 0.0;
    for (int i = 0; i < rule.GetNPoints(); ++i) {
        const mfem::IntegrationPoint& ip = rule.IntPoint(i);
        const double s = ip.x;
        const double t = ip.y;
        // dx/ds = (1 - t)(q1 - q0) + t (q2 - q3);  dx/dt = (1 - s)(q3 - q0) + s (q2 - q1)
        Point3 ds;
        Point3 dt;
        for (std::size_t d = 0; d < 3; ++d) {
            ds[d] = (1.0 - t) * (q[1][d] - q[0][d]) + t * (q[2][d] - q[3][d]);
            dt[d] = (1.0 - s) * (q[3][d] - q[0][d]) + s * (q[2][d] - q[1][d]);
        }
        area += ip.weight * CrossNorm(ds, dt);
    }
    return area;
}

/** One face between two grains, counted on this rank: smaller grain, larger grain, area. */
struct PairFace {
    int a;
    int b;
    double area;
};

} // namespace

/*
 * ComputeGrainStatistics (collective)
 *
 * The steps follow the header. Communication pattern: CellMeasureAccumulator (Allreduce of the
 * fixed-point sums), one ExchangeLabels (neighbor-to-neighbor messages, plus one Alltoallv for
 * periodic ghosts), two small Allreduces (largest face area, face count), one Gather of the
 * per-rank record counts and one Gatherv of the pair records to `root`, and one Bcast of the
 * neighbor counts from `root`.
 */
GrainStatistics ComputeGrainStatistics(const MeshAdjacency& adjacency,
                                       const mfem::Array<int>& element_grain,
                                       int num_grains,
                                       int root) {
    mfem::ParMesh& pmesh = adjacency.GetParMesh();
    MPI_Comm comm = pmesh.GetComm();
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    if (num_grains < 1 || root < 0 || root >= nranks) {
        // Same arguments on every rank, so every rank throws here before communicating
        throw std::invalid_argument("ComputeGrainStatistics: num_grains must be >= 1 and root a "
                                    "rank of the mesh communicator");
    }
    const int ne = pmesh.GetNE();

    // Step 1: element count, volume and centroid per grain (fixed-point sums). The accumulator
    // wants 0-based labels; it checks the size and range on every rank and throws on all ranks
    // together if any rank has a bad label.
    WeightedSamples samples;
    MakeElementWeightedSamples(pmesh, samples);
    mfem::Array<int> sample_label(element_grain.Size());
    {
        const int* grain = element_grain.HostRead();
        int* label = sample_label.HostWrite();
        for (int e = 0; e < element_grain.Size(); ++e) {
            label[e] = grain[e] - 1;
        }
    }
    const CellMeasureAccumulator accumulator(comm, samples, num_grains);
    const CellMeasures measures = accumulator.Measure(sample_label);

    GrainStatistics stats;
    stats.num_grains = num_grains;
    stats.num_elements = measures.sample_count;
    stats.volume = measures.volume;
    stats.total_volume = measures.total_volume;
    stats.centroid.resize(3 * static_cast<std::size_t>(num_grains));
    for (std::size_t g = 0; g < static_cast<std::size_t>(num_grains); ++g) {
        stats.centroid[3 * g + 0] = measures.cx[g];
        stats.centroid[3 * g + 1] = measures.cy[g];
        stats.centroid[3 * g + 2] = measures.cz[g];
    }

    // Step 2: grain ids of the ghost elements (the labels are valid: step 1 checked them)
    mfem::Array<int> ghost_grain;
    adjacency.ExchangeLabels(element_grain, ghost_grain);

    // Step 3: faces between different grains, each counted once globally, from the side of the
    // smaller grain id. Row e of the adjacency lists every face neighbor n of local element e
    // (n < NE local, n >= NE ghost n - NE). Linear geometry (no Nodes or order-1 Nodes) uses the
    // canonical corner formula; higher-order Nodes use the adjacency's curved-face quadrature.
    const bool curved = (MeshNodeOrder(pmesh) > 1);
    const int* grain = element_grain.HostRead();
    const int* offsets = adjacency.Offsets().HostRead();
    const int* neighbors = adjacency.Neighbors().HostRead();
    const int* face_index = adjacency.FaceIndices().HostRead();
    const double* face_area = adjacency.FaceAreas().HostRead();
    std::vector<PairFace> faces;
    double local_max_area = 0.0;
    for (int e = 0; e < ne; ++e) {
        for (int k = offsets[e]; k < offsets[e + 1]; ++k) {
            const int n = neighbors[k];
            const int other = (n < ne) ? grain[n] : ghost_grain[n - ne];
            if (grain[e] >= other) {
                continue; // same grain, or counted from the other side
            }
            const double area = curved ? face_area[k] : CanonicalFaceArea(pmesh, face_index[k]);
            faces.push_back({grain[e], other, area});
            local_max_area = std::max(local_max_area, area);
        }
    }

    // Step 4a: one fixed-point scale for all face areas, from global bounds (exact reductions)
    double max_area = 0.0;
    MPI_Allreduce(&local_max_area, &max_area, 1, MPI_DOUBLE, MPI_MAX, comm);
    long long local_count = static_cast<long long>(faces.size());
    long long total_count = 0;
    MPI_Allreduce(&local_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, comm);
    // A single grain has no pairs; a zero largest area (degenerate faces) still needs a valid
    // scale, any positive bound will do then
    const FixedPointScale scale =
        FixedPointScale::ForRange(0.0, (max_area > 0.0) ? max_area : 1.0,
                                  std::max<long long>(total_count, 1));

    // Step 4b: this rank's integer sums per pair, flattened into records (a, b, sum, count)
    std::map<std::pair<int, int>, std::pair<std::int64_t, std::int64_t>> local_pairs;
    for (const PairFace& face : faces) {
        auto& entry = local_pairs[{face.a, face.b}];
        entry.first += scale.Encode(face.area);
        entry.second += 1;
    }
    std::vector<std::int64_t> send;
    send.reserve(4 * local_pairs.size());
    for (const auto& [key, value] : local_pairs) {
        send.push_back(key.first);
        send.push_back(key.second);
        send.push_back(value.first);
        send.push_back(value.second);
    }

    // Step 4c: gather every rank's records on `root` only (counts first, then the data). Root
    // receives one record per (rank, pair) it touches, i.e. about the number of grain pairs
    // (~7 per grain in 3D) plus duplicates for pairs whose faces span ranks. The receive buffer
    // is indexed with int, which caps the total at INT_MAX / 4 (~5.4e8) records; root checks
    // that and broadcasts the verdict so every rank throws together.
    const long long local_values = static_cast<long long>(send.size());
    std::vector<long long> all_values(static_cast<std::size_t>(nranks), 0);
    MPI_Gather(&local_values, 1, MPI_LONG_LONG, all_values.data(), 1, MPI_LONG_LONG, root, comm);
    long long total_values = 0;
    for (const long long v : all_values) {
        total_values += v; // zeros except on root
    }
    int too_many = (rank == root && total_values > INT_MAX) ? 1 : 0;
    MPI_Bcast(&too_many, 1, MPI_INT, root, comm);
    if (too_many != 0) {
        throw std::runtime_error("ComputeGrainStatistics: too many grain pair records to gather "
                                 "on one rank (more than INT_MAX / 4)");
    }
    std::vector<int> counts(static_cast<std::size_t>(nranks), 0);
    std::vector<int> displacements(static_cast<std::size_t>(nranks), 0);
    int running = 0;
    for (std::size_t r = 0; r < counts.size(); ++r) {
        counts[r] = static_cast<int>(all_values[r]);
        displacements[r] = running;
        running += counts[r];
    }
    std::vector<std::int64_t> received(static_cast<std::size_t>(total_values));
    MPI_Gatherv(send.data(), static_cast<int>(send.size()), MPI_INT64_T, received.data(),
                counts.data(), displacements.data(), MPI_INT64_T, root, comm);

    // Step 4d (root): merge by exact integer addition (a pair whose faces lie on several ranks
    // arrives once per rank); std::map keeps the pairs sorted by (a, b). Step 5 (root): pair
    // table and neighbor counts.
    stats.num_neighbors.assign(static_cast<std::size_t>(num_grains), 0);
    if (rank == root) {
        std::map<std::pair<int, int>, std::pair<std::int64_t, std::int64_t>> pairs;
        for (std::size_t i = 0; i + 3 < received.size(); i += 4) {
            auto& entry =
                pairs[{static_cast<int>(received[i]), static_cast<int>(received[i + 1])}];
            entry.first += received[i + 2];
            entry.second += received[i + 3];
        }
        for (const auto& [key, value] : pairs) {
            stats.pair_a.push_back(key.first);
            stats.pair_b.push_back(key.second);
            stats.pair_area.push_back(scale.DecodeSum(value.first, value.second));
            stats.pair_faces.push_back(value.second);
            stats.num_neighbors[static_cast<std::size_t>(key.first - 1)] += 1;
            stats.num_neighbors[static_cast<std::size_t>(key.second - 1)] += 1;
        }
    }

    // Step 5b: the neighbor counts (one int per grain) to every rank
    MPI_Bcast(stats.num_neighbors.data(), num_grains, MPI_INT, root, comm);
    return stats;
}

/*
 * CompareWithTargetVolumes: add each grain's volume to its seed (grain order), then the relative
 * error |V - V_target| / V_target seed by seed; the maximum (first seed on ties) and the mean.
 */
TargetVolumeComparison CompareWithTargetVolumes(const std::vector<double>& grain_volume,
                                                const std::vector<int>& grain_seed,
                                                const std::vector<double>& seed_target_volume) {
    if (grain_volume.size() != grain_seed.size()) {
        throw std::invalid_argument("CompareWithTargetVolumes: grain_volume and grain_seed differ "
                                    "in size");
    }
    TargetVolumeComparison out;
    out.num_seeds = static_cast<int>(seed_target_volume.size());
    out.seed_volume.assign(seed_target_volume.size(), 0.0);
    for (const double target : seed_target_volume) {
        if (!(target > 0.0) || !std::isfinite(target)) {
            throw std::invalid_argument("CompareWithTargetVolumes: target volumes must be positive "
                                        "and finite");
        }
    }

    // Element-level volume of every seed: the sum over the grains that descend from it
    for (std::size_t g = 0; g < grain_volume.size(); ++g) {
        const int s = grain_seed[g];
        if (s < 0 || s >= out.num_seeds) {
            throw std::invalid_argument("CompareWithTargetVolumes: grain " + std::to_string(g + 1) +
                                        " has seed index " + std::to_string(s) +
                                        " outside the target list");
        }
        out.seed_volume[static_cast<std::size_t>(s)] += grain_volume[g];
    }

    // Relative errors, seed by seed
    double sum = 0.0;
    for (std::size_t s = 0; s < seed_target_volume.size(); ++s) {
        const double error =
            std::fabs(out.seed_volume[s] - seed_target_volume[s]) / seed_target_volume[s];
        sum += error;
        if (out.seed_volume[s] == 0.0) {
            out.num_empty_seeds += 1;
        }
        if (out.worst_seed < 0 || error > out.max_relative_error) {
            out.max_relative_error = error;
            out.worst_seed = static_cast<int>(s);
        }
    }
    if (out.num_seeds > 0) {
        out.mean_relative_error = sum / static_cast<double>(out.num_seeds);
    }
    return out;
}

} // namespace exaconstit::microstructure
