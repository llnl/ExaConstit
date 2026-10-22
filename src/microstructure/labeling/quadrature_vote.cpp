/*
 * Implementation of the quadrature-point vote (quadrature_vote.hpp): subcell-center voting points
 * of each element -> exact nearest seed per point -> volume-weighted majority per element.
 *
 * Terms used in this file:
 * - BVH (bounding volume hierarchy): a tree of nested bounding boxes that lets us skip seeds that
 *   are provably too far away; SeedIndex (geometry/seed_index.hpp) builds one and searches it.
 * - Execution space: the Axom tag type (axom::SEQ_EXEC, axom::CUDA_EXEC<256>, ...) that says
 *   where the BVH kernels run. DispatchAxomExec (exec_dispatch.hpp) picks the one that matches
 *   the configured mfem::Device.
 * - mfem::forall: MFEM's parallel loop; a GPU kernel when the mfem::Device has a CUDA or HIP
 *   backend, a plain host loop otherwise. MFEM_HOST_DEVICE marks code callable inside it.
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it.
 * - Chunk: a contiguous range of local elements whose voting points are processed together,
 *   to bound memory.
 */
#include "microstructure/labeling/quadrature_vote.hpp"

#include "microstructure/exec_dispatch.hpp"
#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief A point of the reference element (x, y, z). */
using RefPoint = std::array<double, 3>;

/** @brief Midpoint of two reference points. */
RefPoint Midpoint(const RefPoint& a, const RefPoint& b) {
    return {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])};
}

/**
 * @brief Appends the centers of the 8^levels children of tet (v0, v1, v2, v3) to `centers`.
 *
 * One split (the uniform-refinement pattern): with m_ij the midpoint of edge ij, the 4 corner
 * tets (v0, m01, m02, m03), (m01, v1, m12, m13), (m02, m12, v2, m23), (m03, m13, m23, v3), and
 * the middle octahedron m01 m02 m03 m12 m13 m23 cut along its diagonal m02-m13 into the 4 tets
 * around that diagonal. Each child has 1/8 of the parent volume: the corner tets are the parent
 * scaled by 1/2, and the octahedron (1/2 of the parent) is split into 4 tets of equal volume (for
 * the regular tet they are congruent by symmetry, and an affine map keeps volume ratios). The
 * center of a tet is the average of its vertices. Host only, recursive (depth = levels <= 3).
 */
void AppendTetCenters(const RefPoint& v0, const RefPoint& v1, const RefPoint& v2,
                      const RefPoint& v3, int levels, std::vector<RefPoint>& centers) {
    if (levels == 0) {
        centers.push_back({0.25 * (v0[0] + v1[0] + v2[0] + v3[0]),
                           0.25 * (v0[1] + v1[1] + v2[1] + v3[1]),
                           0.25 * (v0[2] + v1[2] + v2[2] + v3[2])});
        return;
    }
    const RefPoint m01 = Midpoint(v0, v1);
    const RefPoint m02 = Midpoint(v0, v2);
    const RefPoint m03 = Midpoint(v0, v3);
    const RefPoint m12 = Midpoint(v1, v2);
    const RefPoint m13 = Midpoint(v1, v3);
    const RefPoint m23 = Midpoint(v2, v3);
    // Corner tets
    AppendTetCenters(v0, m01, m02, m03, levels - 1, centers);
    AppendTetCenters(m01, v1, m12, m13, levels - 1, centers);
    AppendTetCenters(m02, m12, v2, m23, levels - 1, centers);
    AppendTetCenters(m03, m13, m23, v3, levels - 1, centers);
    // Octahedron: 4 tets around the diagonal m02-m13; the equator m01, m12, m23, m03 in cycle
    AppendTetCenters(m02, m13, m01, m12, levels - 1, centers);
    AppendTetCenters(m02, m13, m12, m23, levels - 1, centers);
    AppendTetCenters(m02, m13, m23, m03, levels - 1, centers);
    AppendTetCenters(m02, m13, m03, m01, levels - 1, centers);
}

/**
 * @brief Builds the voting rule for one geometry and level count (see VoteRule in the header).
 * Hex: tensor product of the 1D composite midpoint rule with 2^levels points of weight 2^-levels
 * (IntegrationRule's 3-rule constructor stores x fastest, intrules.cpp:56-84). Tet: the centers
 * from AppendTetCenters, each with weight (1/6) / 8^levels. Host only.
 */
std::unique_ptr<mfem::IntegrationRule> BuildVoteRule(mfem::Geometry::Type geometry, int levels) {
    if (geometry == mfem::Geometry::CUBE) {
        const int n = 1 << levels; // subcells per direction
        mfem::IntegrationRule line(n);
        for (int i = 0; i < n; ++i) {
            line.IntPoint(i).x = (i + 0.5) / n;
            line.IntPoint(i).weight = 1.0 / n;
        }
        return std::make_unique<mfem::IntegrationRule>(line, line, line);
    }
    // Tetrahedron
    std::vector<RefPoint> centers;
    AppendTetCenters({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, levels,
                     centers);
    const int num_points = static_cast<int>(centers.size());
    auto rule = std::make_unique<mfem::IntegrationRule>(num_points);
    for (int i = 0; i < num_points; ++i) {
        mfem::IntegrationPoint& ip = rule->IntPoint(i);
        ip.Set3(centers[static_cast<std::size_t>(i)][0], centers[static_cast<std::size_t>(i)][1],
                centers[static_cast<std::size_t>(i)][2]);
        ip.weight = (1.0 / 6.0) / num_points;
    }
    return rule;
}

/**
 * @brief Throws std::runtime_error unless the mesh is a 3D volume mesh with one element geometry
 * (checked before the rule lookup, so a 2D or mixed mesh gets this message rather than an
 * "unsupported geometry" error for whatever element 0 happens to be). Host only.
 */
void RequireVotableMesh(const mfem::Mesh& mesh) {
    if (mesh.SpaceDimension() != 3 || mesh.Dimension() != 3) {
        throw std::runtime_error("Quadrature vote: only 3D volume meshes are supported");
    }
    if (mesh.GetNumGeometries(mesh.Dimension()) > 1) {
        throw std::runtime_error("Quadrature vote: mixed element geometries are not supported");
    }
}

/**
 * @brief Throws std::invalid_argument unless 0 <= levels <= 3.
 *
 * The point work per element grows as 8^levels: level 3 (512 points) resolves a boundary band of
 * 1/8 of the element size, and level 4 (4096 points) would cost 8 times more per element to halve
 * it again, so it is not offered. (The points are evaluated
 * by the shared sampler's own kernel, so MFEM's QuadratureInterpolator size limits do not apply.)
 */
void RequireValidLevels(int levels) {
    if (levels < 0 || levels > 3) {
        throw std::invalid_argument("Quadrature vote: levels must be in 0..3 (8^levels points per "
                                    "element)");
    }
}

/**
 * @brief The vote kernel for one chunk of elements (one mfem::forall iteration per element).
 *
 * For element e of the chunk, its voting points are entries base .. base + nq - 1 of the
 * chunk arrays, base = (e - chunk_begin) * nq (the QuadraturePointSamples layout). The label of
 * point q is owner[best[base + q].id1]; a point with id1 = -1 (non-finite position) has no label
 * and does not vote.
 *
 * The vote needs no scratch memory, so it works the same on a GPU: for each point q whose label
 * has not appeared at an earlier point of the element, the total weight of that label is summed
 * over the points q..nq-1 (in increasing order, so the sum is the same on every rank and device),
 * and compared with the best total so far. Larger total wins; an exactly equal total goes to the
 * lower label. The cost per element is about nq x (number of different labels among its points):
 * the "seen before" scan stops at the first earlier point with the same label, which for an
 * element inside one grain is point 0. Cheap for the default 64 points.
 *
 * It is a named function, not a lambda inside the dispatch lambdas of
 * LabelElementsByQuadratureVote: CUDA does not allow a device lambda to be defined inside a
 * generic (auto-parameter) lambda, but calling a function that contains one is fine.
 *
 * Host function launching one kernel; not collective.
 *
 * @param chunk_begin   first local element of the chunk.
 * @param chunk_elems   number of elements in the chunk (> 0).
 * @param nq            voting points per element.
 * @param best          search result per chunk point (size chunk_elems * nq).
 * @param weight        w_q detJ_q per chunk point (size chunk_elems * nq).
 * @param owner_table   seed -> owner seed (size N).
 * @param element_seed  output, written at chunk_begin .. chunk_begin + chunk_elems - 1.
 * @param element_split output, 1 where the element's points voted for more than one label.
 */
void VoteChunk(int chunk_begin,
               int chunk_elems,
               int nq,
               const mfem::Array<BestPair>& best,
               const mfem::Vector& weight,
               const mfem::Array<int>& owner_table,
               mfem::Array<int>& element_seed,
               mfem::Array<int>& element_split) {
    const BestPair* point_best = best.Read();
    const double* point_weight = weight.Read();
    const int* owner = owner_table.Read();
    // ReadWrite (not Write): the arrays cover all local elements and only this chunk's range is
    // written here, so the other entries must be kept.
    int* seed_out = element_seed.ReadWrite();
    int* split_out = element_split.ReadWrite();

    mfem::forall(chunk_elems, [=] MFEM_HOST_DEVICE(int local) {
        const int base = local * nq; // first voting point of this element in the chunk
        int winner = -1;             // label with the largest total weight so far
        double winner_weight = 0.0;
        int first_label = -1;        // label of the first voting point, to detect a split
        int split = 0;

        for (int q = 0; q < nq; ++q) {
            // Label of point q (skip points without a seed: non-finite position)
            const int id_q = point_best[base + q].id1;
            if (id_q < 0) {
                continue;
            }
            const int label = owner[id_q];
            if (first_label < 0) {
                first_label = label;
            } else if (label != first_label) {
                split = 1;
            }

            // Count each label once: skip it if an earlier point already carried it
            bool seen_before = false;
            for (int p = 0; p < q; ++p) {
                const int id_p = point_best[base + p].id1;
                if (id_p >= 0 && owner[id_p] == label) {
                    seen_before = true;
                    break;
                }
            }
            if (seen_before) {
                continue;
            }

            // Total weight (volume share) of this label, summed in increasing point order
            double total = 0.0;
            for (int p = q; p < nq; ++p) {
                const int id_p = point_best[base + p].id1;
                if (id_p >= 0 && owner[id_p] == label) {
                    total += point_weight[base + p];
                }
            }

            // Larger total wins; exact tie -> lower label
            if (winner < 0 || total > winner_weight ||
                (total == winner_weight && label < winner)) {
                winner = label;
                winner_weight = total;
            }
        }
        seed_out[chunk_begin + local] = winner;
        split_out[chunk_begin + local] = split;
    });
}

} // namespace

/*
 * VoteRule
 *
 * Validates the arguments, then returns the rule from a static map keyed by (geometry, levels),
 * building it on first use (BuildVoteRule). The map owns the rules until program exit, so their
 * addresses stay valid for MFEM's per-rule caches (see the header).
 */
const mfem::IntegrationRule& VoteRule(mfem::Geometry::Type geometry, int levels) {
    RequireValidLevels(levels);
    if (geometry != mfem::Geometry::CUBE && geometry != mfem::Geometry::TETRAHEDRON) {
        throw std::invalid_argument("Quadrature vote: only hexahedral and tetrahedral elements "
                                    "are supported");
    }
    static std::map<std::pair<int, int>, std::unique_ptr<mfem::IntegrationRule>> rules;
    std::unique_ptr<mfem::IntegrationRule>& rule = rules[{static_cast<int>(geometry), levels}];
    if (!rule) {
        rule = BuildVoteRule(geometry, levels);
    }
    return *rule;
}

/*
 * ComputeVotePoints: checks, then the shared sampler (ComputeQuadraturePointSamples with the
 * VoteRule, mesh/mesh_sampler.hpp) for the requested range. A rank without elements returns
 * empty vectors.
 */
void ComputeVotePoints(mfem::Mesh& mesh,
                       int levels,
                       int elem_begin,
                       int elem_end,
                       QuadraturePointSamples& out) {
    RequireValidLevels(levels);
    const int ne = mesh.GetNE();
    if (elem_begin < 0 || elem_end < elem_begin || elem_end > ne) {
        throw std::runtime_error("ComputeVotePoints: invalid element range [" +
                                 std::to_string(elem_begin) + ", " + std::to_string(elem_end) +
                                 ") for " + std::to_string(ne) + " local elements");
    }
    if (ne == 0) {
        out = QuadraturePointSamples{};
        return;
    }
    RequireVotableMesh(mesh);
    ComputeQuadraturePointSamples(mesh, VoteRule(mesh.GetElementGeometry(0), levels), elem_begin,
                                  elem_end, out);
}

/*
 * LabelElementsByQuadratureVote
 *
 * Algorithm:
 *  1. Check the parameters and build the seed -> owner table (identity when seed_owner is empty)
 *     as an mfem::Array so the vote kernel can read it on the device.
 *  2. A QuadraturePointSampler for the VoteRule (mesh/mesh_sampler.hpp: it gathers the element
 *     node coordinates once and evaluates any element range on demand, so memory is bounded by
 *     the chunk), the points per element nq = 8^levels, and the chunk size in whole elements:
 *     max(1, max_points_per_chunk / nq).
 *  3. Choose the metric functor type (DispatchMetric) and the Axom execution space
 *     (DispatchAxomExec), build the SeedIndex once, then for each chunk:
 *       a. coordinates and weights w_q detJ_q of the chunk's voting points
 *          (QuadraturePointSampler::Sample);
 *       b. exact nearest seed of every point (SeedIndex::FindBest; no second best needed);
 *       c. the vote (VoteChunk).
 *     The index is built once and reused across chunks, so only the points are chunked.
 *  4. Report the statistics (split elements are counted on the host).
 *
 * Why the result is partition independent: an element's voting points depend only on its own
 * nodes, the search result per point only on the point and the replicated seeds, and the vote
 * sums the weights in a fixed order inside the element.
 */
void LabelElementsByQuadratureVote(mfem::Mesh& local_mesh,
                                   const SeedSet& seeds,
                                   const QuadratureVoteParameters& params,
                                   mfem::Array<int>& element_seed,
                                   const std::vector<int>& seed_owner,
                                   QuadratureVoteReport* report) {
    // Step 1: parameter checks and the owner table
    const std::size_t num_seeds = seeds.Size();
    if (num_seeds == 0) {
        throw std::invalid_argument("Quadrature vote: the seed set is empty");
    }
    if (num_seeds > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Quadrature vote: more than 2^31 - 1 seeds");
    }
    if (params.num_bins < 1) {
        throw std::invalid_argument("Quadrature vote: num_bins must be >= 1");
    }
    if (params.max_points_per_chunk < 1) {
        throw std::invalid_argument("Quadrature vote: max_points_per_chunk must be >= 1");
    }
    RequireValidLevels(params.levels);
    if (!seed_owner.empty() && seed_owner.size() != num_seeds) {
        throw std::invalid_argument("Quadrature vote: seed_owner must be empty or have one "
                                    "entry per seed");
    }
    const int n = static_cast<int>(num_seeds);
    mfem::Array<int> owner_table(n);
    for (int i = 0; i < n; ++i) {
        if (seed_owner.empty()) {
            owner_table[i] = i;
        } else {
            owner_table[i] = seed_owner[static_cast<std::size_t>(i)];
            if (owner_table[i] < 0) {
                throw std::invalid_argument("Quadrature vote: negative seed_owner entry");
            }
        }
    }

    // Step 2: point sampler, points per element and chunk size (nothing to sample on a rank
    // without elements)
    const int num_elements = local_mesh.GetNE();
    element_seed.SetSize(num_elements);
    mfem::Array<int> element_split(num_elements);
    std::unique_ptr<QuadraturePointSampler> sampler;
    int nq = 0;
    int chunk_elems = 0;
    if (num_elements > 0) {
        RequireVotableMesh(local_mesh);
        sampler = std::make_unique<QuadraturePointSampler>(
            local_mesh, VoteRule(local_mesh.GetElementGeometry(0), params.levels));
        nq = sampler->NumPointsPerElement();
        chunk_elems = std::max(1, params.max_points_per_chunk / nq);
        chunk_elems = std::min(chunk_elems, num_elements);
    }

    // Quadric parameters: global tensor, or per-seed tensors when the seed set carries them
    MetricSetup setup;
    setup.global_tensor = params.global_tensor;
    setup.per_seed_shape = !seeds.shape.empty();

    int coincident = 0;
    int num_chunks = 0;

    // Step 3: metric type and execution space, one index, then the chunks
    DispatchMetric(params.metric, [&](auto metric_tag) {
        using Metric = decltype(metric_tag);
        DispatchAxomExec([&](auto exec_tag) {
            using Exec = decltype(exec_tag);

            // The index is built on every rank, even without elements, so the coincident-seed
            // count is the same everywhere.
            SeedIndex<Exec, Metric> index;
            index.Build(seeds, setup, params.num_bins, AxomAllocatorIdForCurrentDevice());
            coincident = index.GetReport().num_coincident;

            QuadraturePointSamples points;
            mfem::Array<BestPair> best;
            for (int begin = 0; begin < num_elements; begin += chunk_elems) {
                const int end = std::min(num_elements, begin + chunk_elems);

                // Step 3a: voting points of elements [begin, end)
                sampler->Sample(begin, end, points);

                // Step 3b: nearest seed of every point
                index.FindBest(points.x, points.y, points.z, best, false);

                // Step 3c: volume-weighted vote per element
                VoteChunk(begin, end - begin, points.num_qp, best, points.weight, owner_table,
                          element_seed, element_split);
                ++num_chunks;
            }
        });
    });

    // Step 4: statistics
    if (report != nullptr) {
        report->num_points_per_element = nq;
        report->num_chunks = num_chunks;
        report->num_coincident_seeds = coincident;
        report->num_split_elements = 0;
        if (num_elements > 0) {
            const int* split = element_split.HostRead();
            for (int e = 0; e < num_elements; ++e) {
                report->num_split_elements += split[e];
            }
        }
    }
}

/*
 * LabelTessellationByQuadratureVote: the vote, the collective check for unlabeled elements, then
 * CompactSeedLabels (tessellation_labeler.hpp) over the owner seeds, so grain ids are 1..N in
 * owner-seed order exactly as for centroid labeling.
 */
TessellationLabels LabelTessellationByQuadratureVote(mfem::Mesh& local_mesh,
                                                     MPI_Comm comm,
                                                     const SeedSet& seeds,
                                                     const QuadratureVoteParameters& params,
                                                     const std::vector<int>& seed_owner,
                                                     int num_owner_seeds) {
    // Size of the owner range (the seed_grain table)
    int owners = num_owner_seeds;
    if (owners < 0) {
        if (seed_owner.empty()) {
            owners = static_cast<int>(std::min<std::size_t>(
                seeds.Size(), static_cast<std::size_t>(std::numeric_limits<int>::max())));
        } else {
            owners = 1 + *std::max_element(seed_owner.begin(), seed_owner.end());
        }
    }

    // Step 1: vote (no communication)
    mfem::Array<int> element_seed;
    QuadratureVoteReport report;
    LabelElementsByQuadratureVote(local_mesh, seeds, params, element_seed, seed_owner, &report);

    // Step 2: all ranks agree that every element has a seed
    ThrowIfAnyElementUnlabeled(comm, element_seed,
                               "Quadrature vote: no voting point of the element has a finite "
                               "position; the element has non-finite node coordinates");

    // Step 3: compact grain ids (collective; also range-checks the owner indices on all ranks)
    TessellationLabels labels;
    CompactSeedLabels(comm, element_seed, owners, labels);
    labels.num_coincident_seeds = report.num_coincident_seeds;
    return labels;
}

} // namespace exaconstit::microstructure
