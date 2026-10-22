/**
 * @file test_sampling_modes.cpp
 * @brief Tests of quadrature-point voting (labeling/quadrature_vote.hpp) and periodic seed
 * images (geometry/periodic_images.hpp).
 *
 * Every test runs on all ranks. The labelers themselves are not collective, but compaction,
 * the periodic box and the reductions used for checks are (every rank of the communicator must
 * call them, because the ranks exchange data inside). Rank-count independence is checked inside
 * each run where it matters: the distributed result over MPI_COMM_WORLD is gathered into serial
 * element order and compared with a reference computed on the whole serial mesh with
 * MPI_COMM_SELF, which is exactly the np = 1 answer. Running the executable at np = 1..4
 * (test_sampling_modes.cmake) therefore compares np = 2, 3, 4 against np = 1.
 *
 * Meshes: test/data/cube_tet.mesh (750 tets in the unit cube), n x n x n hex boxes of the unit
 * cube (mfem::Mesh::MakeCartesian3D, lexicographic element order), and a single hand-built tet.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever.
 */
#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/geometry/periodic_images.hpp"
#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/labeling/quadrature_vote.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/util/counter_rng.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

const fs::path data_dir = fs::path(MS_SAMPLING_DATA_DIR);

/** test/data/cube_tet.mesh, read the way ExaConstit reads a [Mesh] type = "file" mesh. */
mfem::Mesh CubeTet() {
    return mfem::Mesh((data_dir / "cube_tet.mesh").string().c_str(), 1, 1, true);
}

/** n x n x n hexes filling the unit cube, lexicographic element order. */
mfem::Mesh HexBox(int n) {
    return mfem::Mesh::MakeCartesian3D(n, n, n, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
}

/**
 * n random seeds in the unit cube from the counter-based RNG (random numbers computed from
 * (key, purpose, id), so every rank builds the identical set). With `quantize` the coordinates
 * (and weights) are rounded down to multiples of 2^-20: shifting such a seed by a multiple of
 * 1/8 and wrapping it back into the box is then exact in floating point, which the periodic
 * shift test needs. With `max_weight` > 0 each seed gets a Laguerre weight in [0, max_weight).
 */
ms::SeedSet RandomSeeds(int n, std::uint64_t key, double max_weight = 0.0, bool quantize = false) {
    auto q = [quantize](double v) {
        if (!quantize) {
            return v;
        }
        return std::floor(v * 1048576.0) / 1048576.0;
    };
    ms::SeedSet seeds;
    for (int i = 0; i < n; ++i) {
        const ms::CounterRng rng(key, ms::RngPurpose::GENERIC, static_cast<std::uint64_t>(i));
        seeds.x.push_back(q(rng.UniformDouble(0u)));
        seeds.y.push_back(q(rng.UniformDouble(1u)));
        seeds.z.push_back(q(rng.UniformDouble(2u)));
        if (max_weight > 0.0) {
            seeds.weight.push_back(q(max_weight * rng.UniformDouble(3u)));
        }
        seeds.source_id.push_back(static_cast<std::int64_t>(i));
    }
    return seeds;
}

/** The distributed mesh with the serial-order bookkeeping needed to gather results. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/** Partitions `serial` over `comm` (collective), keeping local -> serial element ids. */
Distributed Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    Distributed d;
    d.order = ms::BuildSerialOrder(serial, comm);
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/** Host copy of an mfem::Array<int> (works for device-resident data too). */
std::vector<int> ToHost(const mfem::Array<int>& a) {
    std::vector<int> out(static_cast<std::size_t>(a.Size()));
    if (a.Size() > 0) {
        const int* data = a.HostRead();
        for (int i = 0; i < a.Size(); ++i) {
            out[static_cast<std::size_t>(i)] = data[i];
        }
    }
    return out;
}

/** Gathers per-element ints to rank 0 in serial element order (collective). */
std::vector<int> Gather(const Distributed& d, const mfem::Array<int>& local) {
    local.HostRead(); // GatherToSerial reads the host copy
    return ms::GatherToSerial(MPI_COMM_WORLD, d.order, local, 0);
}

/** Sum over all ranks (collective). */
double GlobalSum(double v) {
    double out = 0.0;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

/**
 * Brute-force vote on the local elements: every voting point (ComputeVotePoints) is labeled by a
 * loop over all seeds with the same metric functor the search uses (strict `<` keeps the lowest
 * index on ties), mapped through `owner`, and the element goes to the label with the largest total
 * w detJ (exact ties: lowest label). The totals are summed in increasing point order, as the
 * kernel does, so the comparison can be exact.
 */
template <typename Metric>
std::vector<int> BruteForceVote(mfem::Mesh& mesh, const ms::SeedSet& seeds,
                                const ms::MetricSetup& setup, const std::vector<int>& owner,
                                int levels) {
    ms::SeedIndex<axom::SEQ_EXEC, Metric> index;
    index.Build(seeds, setup, 1, axom::execution_space<axom::SEQ_EXEC>::allocatorID());
    const Metric& metric = index.GetMetric();

    const int ne = mesh.GetNE();
    std::vector<int> labels(static_cast<std::size_t>(ne), -1);
    if (ne == 0) {
        return labels;
    }
    ms::QuadraturePointSamples points;
    ms::ComputeVotePoints(mesh, levels, 0, ne, points);
    const int nq = points.num_qp;
    const double* x = points.x.HostRead();
    const double* y = points.y.HostRead();
    const double* z = points.z.HostRead();
    const double* w = points.weight.HostRead();
    const int n = static_cast<int>(seeds.Size());

    for (int e = 0; e < ne; ++e) {
        // Label of every point of the element
        std::vector<int> point_label(static_cast<std::size_t>(nq));
        for (int q = 0; q < nq; ++q) {
            const int i_pt = e * nq + q;
            int best = -1;
            double best_f = 0.0;
            for (int i = 0; i < n; ++i) {
                const double f = metric.Eval(i, x[i_pt], y[i_pt], z[i_pt]);
                if (best < 0 || f < best_f) {
                    best = i;
                    best_f = f;
                }
            }
            point_label[static_cast<std::size_t>(q)] =
                owner.empty() ? best : owner[static_cast<std::size_t>(best)];
        }
        // Totals per label in first-appearance order, each summed over increasing q
        int winner = -1;
        double winner_w = 0.0;
        for (int q = 0; q < nq; ++q) {
            const int label = point_label[static_cast<std::size_t>(q)];
            bool seen = false;
            for (int p = 0; p < q; ++p) {
                seen = seen || (point_label[static_cast<std::size_t>(p)] == label);
            }
            if (seen) {
                continue;
            }
            double total = 0.0;
            for (int p = q; p < nq; ++p) {
                if (point_label[static_cast<std::size_t>(p)] == label) {
                    total += w[e * nq + p];
                }
            }
            if (winner < 0 || total > winner_w || (total == winner_w && label < winner)) {
                winner = label;
                winner_w = total;
            }
        }
        labels[static_cast<std::size_t>(e)] = winner;
    }
    return labels;
}

/** Brute force for a runtime metric type (DispatchMetric picks the functor). */
std::vector<int> BruteForceVote(ms::MetricType type, mfem::Mesh& mesh, const ms::SeedSet& seeds,
                                const ms::MetricSetup& setup, const std::vector<int>& owner,
                                int levels) {
    std::vector<int> out;
    ms::DispatchMetric(type, [&](auto metric_tag) {
        out = BruteForceVote<decltype(metric_tag)>(mesh, seeds, setup, owner, levels);
    });
    return out;
}

/** Element volumes of the local mesh (host vector). */
std::vector<double> ElementVolumes(mfem::Mesh& mesh) {
    ms::ElementSamples samples;
    ms::ComputeElementSamples(mesh, -1, samples);
    std::vector<double> v(static_cast<std::size_t>(mesh.GetNE()));
    if (mesh.GetNE() > 0) {
        const double* vol = samples.volume.HostRead();
        for (int e = 0; e < mesh.GetNE(); ++e) {
            v[static_cast<std::size_t>(e)] = vol[e];
        }
    }
    return v;
}

} // namespace

//--------------------------------------------------------------------------------------------
// Quadrature-point voting
//--------------------------------------------------------------------------------------------

/**
 * The vote equals a brute-force vote (every voting point compared with every seed, same
 * weights, same tie rule) on the local elements of every rank, for weighted and unweighted
 * metrics, tets and hexes, non-default subdivision levels, tiny chunks (several elements per
 * chunk, then one element per chunk: the chunking must not change anything), and an owner map
 * that groups seeds (as periodic images do). Exact equality is expected because both sides use
 * the same metric functor, the same voting points and the same summation order.
 */
TEST(QuadratureVote, MatchesBruteForce) {
    struct Case {
        const char* name;
        bool tets;
        ms::MetricType metric;
        double max_weight;
        int levels;
    };
    const Case cases[] = {
        {"cube_tet voronoi", true, ms::MetricType::VORONOI, 0.0, 2},
        {"cube_tet laguerre levels 1", true, ms::MetricType::LAGUERRE, 0.02, 1},
        {"hex6 additive", false, ms::MetricType::ADDITIVE, 0.1, 2},
        {"hex6 voronoi levels 3", false, ms::MetricType::VORONOI, 0.0, 3},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        mfem::Mesh serial = c.tets ? CubeTet() : HexBox(6);
        Distributed dist = Distribute(serial, MPI_COMM_WORLD);
        const ms::SeedSet seeds = RandomSeeds(25, 11u, c.max_weight);

        ms::QuadratureVoteParameters params;
        params.metric = c.metric;
        params.levels = c.levels;
        params.num_bins = 4;
        mfem::Array<int> voted;
        ms::QuadratureVoteReport report;
        ms::LabelElementsByQuadratureVote(*dist.pmesh, seeds, params, voted, {}, &report);

        const ms::MetricSetup setup;
        const std::vector<int> brute =
            BruteForceVote(c.metric, *dist.pmesh, seeds, setup, {}, c.levels);
        EXPECT_EQ(ToHost(voted), brute);
        if (dist.pmesh->GetNE() > 0) {
            EXPECT_EQ(report.num_chunks, 1); // default chunk holds all local points
            EXPECT_GT(report.num_points_per_element, 1);
        }

        // Chunking: 3 elements per chunk, then 1 element per chunk
        for (int elems : {3, 1}) {
            params.max_points_per_chunk = elems * std::max(1, report.num_points_per_element);
            mfem::Array<int> chunked;
            ms::QuadratureVoteReport chunk_report;
            ms::LabelElementsByQuadratureVote(*dist.pmesh, seeds, params, chunked, {},
                                              &chunk_report);
            EXPECT_EQ(ToHost(chunked), brute);
            const int ne = dist.pmesh->GetNE();
            EXPECT_EQ(chunk_report.num_chunks, (ne + elems - 1) / elems);
        }
        params.max_points_per_chunk = 1 << 21;

        // Owner map: seed i votes for owner i % 7 (several seeds per grain, like images)
        std::vector<int> owner(seeds.Size());
        for (std::size_t i = 0; i < owner.size(); ++i) {
            owner[i] = static_cast<int>(i % 7);
        }
        mfem::Array<int> grouped;
        ms::LabelElementsByQuadratureVote(*dist.pmesh, seeds, params, grouped, owner);
        EXPECT_EQ(ToHost(grouped),
                  BruteForceVote(c.metric, *dist.pmesh, seeds, setup, owner, c.levels));
    }
}

/**
 * Hand-checked case where centroid labeling and the vote disagree. One tet with vertices
 * (0,0,0), (1,0,0), (0,1,0), (0,0,1) is cut by the Voronoi bisector plane x + y + z = 0.78 of two
 * seeds placed symmetrically about it on the line x = y = z. The corner part x + y + z < 0.78 is
 * the tet scaled by 0.78, so it holds 0.78^3 = 47.5% of the volume; the centroid has
 * x + y + z = 0.75 < 0.78, so it lies in that minority part. Centroid labeling must therefore
 * pick the corner seed (0) and the vote, which measures the volume shares, the other seed (1).
 * The vote uses 3 subdivision levels here (512 subcell centers). How well 512 points resolve the
 * 47.5% / 52.5% split is measured, not assumed: the share of seed 1 estimated from the same
 * voting points (weights of the points with x + y + z >= 0.78) is 0.570 against the exact
 * 1 - 0.78^3 = 0.525. The error (0.045) is larger than the 2.5% margin to 1/2, because this plane
 * is parallel to a tet face, which is also the orientation of the layers of subcell centers, so
 * whole layers switch sides together (the worst case for this rule); the estimate still lands on
 * the majority side. The test checks share > 1/2 and that the error stays below 0.05.
 */
TEST(QuadratureVote, CentroidInMinorityPart) {
    mfem::Mesh tet(3, 4, 1, 0, 3);
    const double v[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (const auto& p : v) {
        tet.AddVertex(p[0], p[1], p[2]);
    }
    const int ids[4] = {0, 1, 2, 3};
    tet.AddTet(ids, 1);
    tet.FinalizeTetMesh(1, 1, true);

    // Seeds at p -/+ t n with p = (c/3)(1,1,1) on the plane and n the unit normal
    const double c = 0.78;
    const double t = 0.2 / std::sqrt(3.0);
    ms::SeedSet seeds;
    seeds.x = {c / 3 - t, c / 3 + t};
    seeds.y = seeds.x;
    seeds.z = seeds.x;
    seeds.source_id = {0, 1};

    mfem::Array<int> by_centroid;
    ms::LabelElementsWithSeeds(tet, seeds, ms::TessellationOptions{}, by_centroid);
    ms::QuadratureVoteParameters params;
    params.levels = 3;
    mfem::Array<int> by_vote;
    ms::QuadratureVoteReport report;
    ms::LabelElementsByQuadratureVote(tet, seeds, params, by_vote, {}, &report);

    EXPECT_EQ(ToHost(by_centroid), std::vector<int>{0});
    EXPECT_EQ(ToHost(by_vote), std::vector<int>{1});
    EXPECT_EQ(report.num_split_elements, 1);
    EXPECT_EQ(report.num_points_per_element, 512);

    // The measured share behind the vote (same points as the vote uses)
    ms::QuadraturePointSamples points;
    ms::ComputeVotePoints(tet, 3, 0, 1, points);
    double share1 = 0.0;
    double total = 0.0;
    for (int q = 0; q < points.num_qp; ++q) {
        const double w = points.weight.HostRead()[q];
        total += w;
        if (points.x.HostRead()[q] + points.y.HostRead()[q] + points.z.HostRead()[q] >= c) {
            share1 += w;
        }
    }
    share1 /= total;
    EXPECT_GT(share1, 0.5);
    EXPECT_NEAR(share1, 1.0 - c * c * c, 0.05);
    std::cout << "[ single tet ] measured share of seed 1 at level 3: " << share1
              << " (exact " << 1.0 - c * c * c << ")" << std::endl;
}

/**
 * The vote assigns grain volumes closer to the analytic tessellation than centroid labeling on a
 * coarse mesh (cube_tet: 5 x 5 x 5 cubes of 6 tets each, element size 0.2). Two families of
 * two-seed tessellations with known grain-0 volume:
 *  - Planes: Voronoi seeds at p -/+ 0.1 n with p = c n have the bisector plane n . x = c, and
 *    for a normal n with positive components the volume of {n . x < c} inside the unit cube is
 *    V = sum over the 8 cube corners v of (-1)^(number of ones in v) max(0, c - n . v)^3 /
 *    (6 n_x n_y n_z) (inclusion-exclusion over the corners). Oblique normals cut the tets
 *    generically; axis-aligned planes would cut this structured mesh symmetrically, where the
 *    centroid side and the majority side always coincide and nothing could be learned.
 *  - Balls: two coincident Quadric seeds at (0.5, 0.5, 0.5) with tensors I and I/4 and weights 0
 *    and -0.75 R^2 make grain 0 the ball |x - c| < R exactly (f0 = |d|^2 < f1 = |d|^2/4 +
 *    0.75 R^2 when |d| < R), whose volume is 4/3 pi R^3.
 * Checked quantities (labels gathered to rank 0 in serial order; checks on rank 0):
 *  - the volume error |V_0 - V|, summed over all cases: smaller for the vote;
 *  - the misassigned volume (parts of elements lying in the other grain than their label says),
 *    summed over all cases: smaller for the vote. It is measured on a reference copy of the mesh
 *    refined twice (each tet split into 64), with the 14-point positive-weight rule on every fine
 *    tet, and each fine tet inherits the label of its coarse parent;
 *  - the same misassigned volume measured with the vote's own quadrature points, per case: the
 *    vote picks the label with the larger share of exactly these weights, so it can never be
 *    worse than the centroid label there (a mathematical guarantee, hence checked per case).
 */
TEST(QuadratureVote, VolumesCloserToAnalyticThanCentroid) {
    mfem::Mesh serial = CubeTet();
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    const bool root = mfem::Mpi::Root();

    // Reference mesh: refined twice, with the coarse parent of every fine element. Uniform
    // refinement records each child's parent (Mesh::GetRefinementTransforms().embeddings),
    // for the last refinement only, so the two levels are composed.
    mfem::Mesh fine(serial);
    fine.UniformRefinement();
    std::vector<int> parent_of_mid;
    for (int c = 0; c < fine.GetNE(); ++c) {
        parent_of_mid.push_back(fine.GetRefinementTransforms().embeddings[c].parent);
    }
    fine.UniformRefinement();
    std::vector<int> parent(static_cast<std::size_t>(fine.GetNE()));
    for (int c = 0; c < fine.GetNE(); ++c) {
        const int mid = fine.GetRefinementTransforms().embeddings[c].parent;
        parent[static_cast<std::size_t>(c)] = parent_of_mid[static_cast<std::size_t>(mid)];
    }
    EXPECT_EQ(fine.GetNE(), 64 * serial.GetNE());
    ms::QuadraturePointSamples reference;
    ms::ComputeQuadraturePointSamples(fine, 5, 0, fine.GetNE(), reference);

    // Sanity of the parent map: the fine tets of every coarse tet add up to its volume
    const std::vector<double> volume = ElementVolumes(serial);
    {
        std::vector<double> child_sum(volume.size(), 0.0);
        const double* w = reference.weight.HostRead();
        for (int c = 0; c < fine.GetNE(); ++c) {
            for (int q = 0; q < reference.num_qp; ++q) {
                child_sum[static_cast<std::size_t>(parent[static_cast<std::size_t>(c)])] +=
                    w[c * reference.num_qp + q];
            }
        }
        for (std::size_t e = 0; e < volume.size(); ++e) {
            EXPECT_NEAR(child_sum[e], volume[e], 1e-14);
        }
    }

    // The vote's own points on the coarse elements (serial order)
    ms::QuadraturePointSamples own;
    ms::ComputeVotePoints(serial, ms::QuadratureVoteParameters{}.levels, 0, serial.GetNE(), own);

    // One test case: seeds, metric, analytic grain-0 volume and the exact grain-0 indicator
    struct Case {
        ms::SeedSet seeds;
        ms::MetricType metric;
        double analytic_volume;
        std::function<bool(double, double, double)> in_grain0;
    };
    std::vector<Case> cases;
    const double pi = 3.14159265358979323846;
    for (const std::array<double, 3>& direction :
         {std::array<double, 3>{1.0, 0.37, 0.61}, std::array<double, 3>{0.3, 1.0, 0.8},
          std::array<double, 3>{0.55, 0.45, 1.0}}) {
        const double norm = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] +
                                      direction[2] * direction[2]);
        const std::array<double, 3> nrm = {direction[0] / norm, direction[1] / norm,
                                           direction[2] / norm};
        for (double fraction : {0.3, 0.45, 0.6, 0.72}) {
            const double c = fraction * (nrm[0] + nrm[1] + nrm[2]);
            Case k;
            k.metric = ms::MetricType::VORONOI;
            for (double side : {-0.1, 0.1}) {
                k.seeds.x.push_back((c + side) * nrm[0]);
                k.seeds.y.push_back((c + side) * nrm[1]);
                k.seeds.z.push_back((c + side) * nrm[2]);
            }
            // Inclusion-exclusion over the 8 corners of the unit cube
            double v = 0.0;
            for (int corner = 0; corner < 8; ++corner) {
                const int bx = corner & 1;
                const int by = (corner >> 1) & 1;
                const int bz = (corner >> 2) & 1;
                const double gap = c - (bx * nrm[0] + by * nrm[1] + bz * nrm[2]);
                double sign = -1.0;
                if ((bx + by + bz) % 2 == 0) {
                    sign = 1.0;
                }
                if (gap > 0.0) {
                    v += sign * gap * gap * gap;
                }
            }
            k.analytic_volume = v / (6.0 * nrm[0] * nrm[1] * nrm[2]);
            k.in_grain0 = [nrm, c](double x, double y, double z) {
                return nrm[0] * x + nrm[1] * y + nrm[2] * z < c;
            };
            cases.push_back(k);
        }
    }
    for (double radius : {0.15, 0.22, 0.3, 0.37}) {
        Case k;
        k.metric = ms::MetricType::QUADRIC;
        k.seeds.x = {0.5, 0.5};
        k.seeds.y = {0.5, 0.5};
        k.seeds.z = {0.5, 0.5};
        k.seeds.weight = {0.0, -0.75 * radius * radius};
        k.seeds.shape = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0};
        k.analytic_volume = 4.0 / 3.0 * pi * radius * radius * radius;
        k.in_grain0 = [radius](double x, double y, double z) {
            const double dx = x - 0.5;
            const double dy = y - 0.5;
            const double dz = z - 0.5;
            return dx * dx + dy * dy + dz * dz < radius * radius;
        };
        cases.push_back(k);
    }

    // Misassigned volume of coarse labels (grain 0 = seed 0) measured with the points `pts`,
    // which belong to the elements of `mesh_of_pts`; element_label maps those elements to the
    // coarse label (identity for the coarse mesh, the parent for the fine mesh)
    auto misassigned = [](const ms::QuadraturePointSamples& pts, int num_elements,
                          const std::function<int(int)>& element_label, const Case& k) {
        const double* x = pts.x.HostRead();
        const double* y = pts.y.HostRead();
        const double* z = pts.z.HostRead();
        const double* w = pts.weight.HostRead();
        double sum = 0.0;
        for (int e = 0; e < num_elements; ++e) {
            const int label = element_label(e);
            for (int q = 0; q < pts.num_qp; ++q) {
                const int i = e * pts.num_qp + q;
                int true_label = 1;
                if (k.in_grain0(x[i], y[i], z[i])) {
                    true_label = 0;
                }
                if (true_label != label) {
                    sum += w[i];
                }
            }
        }
        return sum;
    };

    double centroid_volume_error = 0.0;
    double vote_volume_error = 0.0;
    double centroid_mis = 0.0;
    double vote_mis = 0.0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const Case& k = cases[i];
        ms::TessellationOptions options;
        options.metric = k.metric;
        ms::QuadratureVoteParameters params;
        params.metric = k.metric;
        mfem::Array<int> by_centroid;
        ms::LabelElementsWithSeeds(*dist.pmesh, k.seeds, options, by_centroid);
        mfem::Array<int> by_vote;
        ms::LabelElementsByQuadratureVote(*dist.pmesh, k.seeds, params, by_vote);
        const std::vector<int> lc = Gather(dist, by_centroid); // serial order, rank 0 only
        const std::vector<int> lv = Gather(dist, by_vote);
        if (!root) {
            continue;
        }

        double vc = 0.0;
        double vv = 0.0;
        for (std::size_t e = 0; e < volume.size(); ++e) {
            if (lc[e] == 0) {
                vc += volume[e];
            }
            if (lv[e] == 0) {
                vv += volume[e];
            }
        }
        centroid_volume_error += std::fabs(vc - k.analytic_volume);
        vote_volume_error += std::fabs(vv - k.analytic_volume);

        auto via_parent = [&parent](const std::vector<int>& labels) {
            return [&parent, &labels](int c) {
                return labels[static_cast<std::size_t>(parent[static_cast<std::size_t>(c)])];
            };
        };
        auto direct = [](const std::vector<int>& labels) {
            return [&labels](int e) { return labels[static_cast<std::size_t>(e)]; };
        };
        centroid_mis += misassigned(reference, fine.GetNE(), via_parent(lc), k);
        vote_mis += misassigned(reference, fine.GetNE(), via_parent(lv), k);
        EXPECT_LE(misassigned(own, serial.GetNE(), direct(lv), k),
                  misassigned(own, serial.GetNE(), direct(lc), k) + 1e-14)
            << "case " << i;
    }
    if (root) {
        EXPECT_LT(vote_volume_error, centroid_volume_error);
        EXPECT_LT(vote_mis, centroid_mis);
        std::cout << "[ cube_tet, " << cases.size() << " cases ] volume error sum: centroid "
                  << centroid_volume_error << ", vote " << vote_volume_error
                  << "; misassigned volume sum: centroid " << centroid_mis << ", vote "
                  << vote_mis << std::endl;
    }
}

/**
 * The voting rule and points. On the reference elements the rule has 8^levels points with equal
 * positive weights (a negative weight would make a point vote against its grain; MFEM's tet
 * rules had such weights before v4.10) summing to the reference volume (1 for the cube, 1/6 for
 * the tet), all inside the element, and their mean is the reference centroid (the subcells have
 * equal volume and tile the element, so the mean of their centers is the element centroid). In
 * physical space the weights of an element sum to its volume and their weighted mean is its
 * centroid, exactly (to round-off) for the affine elements of cube_tet and a hex box (compared
 * with ComputeElementSamples), and the rule pointer is stable across calls (MFEM caches by
 * address).
 */
TEST(QuadratureVote, VotingPoints) {
    for (mfem::Geometry::Type geometry : {mfem::Geometry::CUBE, mfem::Geometry::TETRAHEDRON}) {
        const bool cube = (geometry == mfem::Geometry::CUBE);
        const double ref_volume = cube ? 1.0 : 1.0 / 6.0;
        const double ref_center = cube ? 0.5 : 0.25;
        for (int levels = 0; levels <= 3; ++levels) {
            const mfem::IntegrationRule& rule = ms::VoteRule(geometry, levels);
            EXPECT_EQ(&rule, &ms::VoteRule(geometry, levels));
            EXPECT_EQ(rule.GetNPoints(), 1 << (3 * levels));
            double sum_w = 0.0;
            double mean[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < rule.GetNPoints(); ++i) {
                const mfem::IntegrationPoint& ip = rule.IntPoint(i);
                EXPECT_DOUBLE_EQ(ip.weight, rule.IntPoint(0).weight);
                EXPECT_GT(ip.weight, 0.0);
                EXPECT_GT(ip.x, 0.0);
                EXPECT_GT(ip.y, 0.0);
                EXPECT_GT(ip.z, 0.0);
                if (cube) {
                    EXPECT_LT(std::max({ip.x, ip.y, ip.z}), 1.0);
                } else {
                    EXPECT_LT(ip.x + ip.y + ip.z, 1.0);
                }
                sum_w += ip.weight;
                mean[0] += ip.x / rule.GetNPoints();
                mean[1] += ip.y / rule.GetNPoints();
                mean[2] += ip.z / rule.GetNPoints();
            }
            EXPECT_NEAR(sum_w, ref_volume, 1e-15);
            for (double m : mean) {
                EXPECT_NEAR(m, ref_center, 1e-15);
            }
        }
    }
    EXPECT_THROW(ms::VoteRule(mfem::Geometry::CUBE, 4), std::invalid_argument);
    EXPECT_THROW(ms::VoteRule(mfem::Geometry::TETRAHEDRON, -1), std::invalid_argument);
    EXPECT_THROW(ms::VoteRule(mfem::Geometry::PRISM, 1), std::invalid_argument);

    // Physical points: volumes and centroids per element (each rank its own elements)
    for (bool tets : {true, false}) {
        mfem::Mesh serial = tets ? CubeTet() : HexBox(3);
        Distributed dist = Distribute(serial, MPI_COMM_WORLD);
        const int ne = dist.pmesh->GetNE();
        ms::QuadraturePointSamples points;
        ms::ComputeVotePoints(*dist.pmesh, 2, 0, ne, points);
        ms::ElementSamples samples;
        ms::ComputeElementSamples(*dist.pmesh, -1, samples);
        if (ne == 0) {
            continue;
        }
        EXPECT_EQ(points.num_qp, 64);
        const double* x = points.x.HostRead();
        const double* y = points.y.HostRead();
        const double* z = points.z.HostRead();
        const double* w = points.weight.HostRead();
        const double* vol = samples.volume.HostRead();
        const double* cx = samples.cx.HostRead();
        const double* cy = samples.cy.HostRead();
        const double* cz = samples.cz.HostRead();
        for (int e = 0; e < ne; ++e) {
            double v = 0.0;
            double m[3] = {0.0, 0.0, 0.0};
            for (int q = 0; q < points.num_qp; ++q) {
                const int i = e * points.num_qp + q;
                // Every vote must count for its grain: no negative weight
                EXPECT_GT(w[i], 0.0);
                v += w[i];
                m[0] += w[i] * x[i];
                m[1] += w[i] * y[i];
                m[2] += w[i] * z[i];
            }
            EXPECT_NEAR(v, vol[e], 1e-15);
            EXPECT_NEAR(m[0] / v, cx[e], 1e-14);
            EXPECT_NEAR(m[1] / v, cy[e], 1e-14);
            EXPECT_NEAR(m[2] / v, cz[e], 1e-14);
        }
        // A sub-range copies the same entries
        ms::QuadraturePointSamples part;
        ms::ComputeVotePoints(*dist.pmesh, 2, ne / 2, ne, part);
        EXPECT_EQ(part.x.Size(), 64 * (ne - ne / 2));
        if (part.x.Size() > 0) {
            EXPECT_EQ(part.x.HostRead()[0], x[64 * (ne / 2)]);
        }
    }
}

/**
 * The chunk-by-chunk point evaluation of the shared sampler (QuadraturePointSampler, used by
 * ComputeVotePoints) reproduces what mfem::GeometricFactors computes for all elements at once:
 * bitwise for tets, where GeometricFactors uses the same non-tensor kernel (same sums in the same
 * order), and to round-off (1e-14 relative) for hexes, where GeometricFactors uses a
 * sum-factorized tensor kernel that adds the same terms in another order. Checked on linear
 * meshes (cube_tet, a 3^3 hex box) and curved order-2 meshes with perturbed interior nodes, at
 * levels 1 and 3, and chunk by chunk (2 elements per Sample call) against one call for all.
 */
TEST(QuadratureVote, ChunkedPointsMatchGeometricFactors) {
    struct Case {
        const char* name;
        bool tets;
        bool curved;
    };
    const Case cases[] = {{"cube_tet", true, false},
                          {"hex3", false, false},
                          {"cube_tet order 2", true, true},
                          {"hex3 order 2", false, true}};
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        mfem::Mesh mesh = c.tets ? CubeTet() : HexBox(3);
        if (c.curved) {
            // Order-2 Nodes, every node moved by a smooth displacement that vanishes on the
            // boundary of the unit cube (elements stay valid, faces curve)
            mesh.SetCurvature(2);
            mfem::GridFunction& nodes = *mesh.GetNodes();
            const mfem::FiniteElementSpace& space = *nodes.FESpace();
            for (int i = 0; i < space.GetNDofs(); ++i) {
                const double x = nodes(space.DofToVDof(i, 0));
                const double y = nodes(space.DofToVDof(i, 1));
                const double z = nodes(space.DofToVDof(i, 2));
                const double bump = 0.03 * std::sin(3.14159265358979 * x) *
                                    std::sin(3.14159265358979 * y) *
                                    std::sin(3.14159265358979 * z);
                nodes(space.DofToVDof(i, 0)) += bump;
            }
        }
        const int ne = mesh.GetNE();
        for (int levels : {1, 3}) {
            const mfem::IntegrationRule& rule = ms::VoteRule(mesh.GetElementGeometry(0), levels);
            const int nq = rule.GetNPoints();

            // Reference: GeometricFactors on all elements (temporary order-1 nodes if needed)
            std::unique_ptr<mfem::H1_FECollection> fec;
            std::unique_ptr<mfem::FiniteElementSpace> fes;
            std::unique_ptr<mfem::GridFunction> linear_nodes;
            const mfem::GridFunction* nodes = mesh.GetNodes();
            if (nodes == nullptr) {
                fec = std::make_unique<mfem::H1_FECollection>(1, 3);
                fes = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get(), 3,
                                                                 mfem::Ordering::byNODES);
                linear_nodes = std::make_unique<mfem::GridFunction>(fes.get());
                mesh.GetNodes(*linear_nodes);
                nodes = linear_nodes.get();
            }
            const mfem::GeometricFactors factors(*nodes, rule,
                                                 mfem::GeometricFactors::COORDINATES |
                                                     mfem::GeometricFactors::DETERMINANTS);
            const double* fx = factors.X.HostRead();
            const double* fdet = factors.detJ.HostRead();

            // Sampler, chunk by chunk
            ms::QuadraturePointSampler sampler(mesh, rule);
            int mismatches = 0;
            double max_rel = 0.0;
            for (int begin = 0; begin < ne; begin += 2) {
                const int end = std::min(ne, begin + 2);
                ms::QuadraturePointSamples chunk;
                sampler.Sample(begin, end, chunk);
                const double* coord[3] = {chunk.x.HostRead(), chunk.y.HostRead(),
                                          chunk.z.HostRead()};
                const double* w = chunk.weight.HostRead();
                for (int e = begin; e < end; ++e) {
                    for (int q = 0; q < nq; ++q) {
                        const int i = (e - begin) * nq + q;
                        double values[4];
                        double expected[4];
                        for (int d = 0; d < 3; ++d) {
                            values[d] = coord[d][i];
                            expected[d] = fx[q + nq * (d + 3 * e)];
                        }
                        values[3] = w[i];
                        expected[3] = rule.IntPoint(q).weight * fdet[q + nq * e];
                        for (int k = 0; k < 4; ++k) {
                            if (values[k] != expected[k]) {
                                ++mismatches;
                            }
                            const double scale = std::max(1.0, std::fabs(expected[k]));
                            max_rel = std::max(max_rel,
                                               std::fabs(values[k] - expected[k]) / scale);
                        }
                    }
                }
            }
            if (c.tets) {
                EXPECT_EQ(mismatches, 0) << "levels " << levels;
            }
            EXPECT_LT(max_rel, 1e-14) << "levels " << levels;
            if (mfem::Mpi::Root()) {
                std::cout << "[ " << c.name << ", levels " << levels << " ] values differing from "
                          << "GeometricFactors: " << mismatches << ", max relative difference "
                          << max_rel << std::endl;
            }
        }
    }
}

/**
 * Rank-count independence of the full vote + compaction, with Laguerre weights and periodic
 * images on the tet cube: grain ids gathered in serial order, num_grains and the grain -> seed
 * table must equal the single-rank (MPI_COMM_SELF) reference exactly, because each element's
 * label depends only on its own quadrature points and the replicated seeds.
 */
TEST(QuadratureVote, RankCountIndependent) {
    mfem::Mesh serial = CubeTet();
    const ms::SeedSet seeds = RandomSeeds(30, 5u, 0.01);
    ms::QuadratureVoteParameters params;
    params.metric = ms::MetricType::LAGUERRE;

    auto run = [&](MPI_Comm comm, Distributed& d) {
        const ms::PeriodicBox box = ms::PeriodicBoxFromMesh(*d.pmesh, comm, {true, true, true});
        const ms::PeriodicImages images =
            ms::BuildPeriodicImages(seeds, box, ms::PeriodicImageParameters{});
        return ms::LabelTessellationByQuadratureVote(*d.pmesh, comm, images.seeds, params,
                                                     images.original, images.num_original);
    };

    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    const ms::TessellationLabels labels = run(MPI_COMM_WORLD, dist);
    const std::vector<int> gathered = Gather(dist, labels.element_grain);

    Distributed alone = Distribute(serial, MPI_COMM_SELF);
    const ms::TessellationLabels reference = run(MPI_COMM_SELF, alone);

    EXPECT_EQ(labels.num_grains, reference.num_grains);
    EXPECT_EQ(labels.grain_seed, reference.grain_seed);
    EXPECT_EQ(labels.seed_grain.size(), seeds.Size());
    if (mfem::Mpi::Root()) {
        EXPECT_EQ(gathered, ToHost(reference.element_grain));
    }
}

/** Invalid parameters throw std::invalid_argument (not collective: each rank checks alone). */
TEST(QuadratureVote, InvalidParameters) {
    mfem::Mesh mesh = HexBox(2);
    const ms::SeedSet seeds = RandomSeeds(3, 1u);
    mfem::Array<int> out;
    ms::QuadratureVoteParameters bad_chunk;
    bad_chunk.max_points_per_chunk = 0;
    EXPECT_THROW(ms::LabelElementsByQuadratureVote(mesh, seeds, bad_chunk, out),
                 std::invalid_argument);
    ms::QuadratureVoteParameters bad_levels;
    bad_levels.levels = 4;
    EXPECT_THROW(ms::LabelElementsByQuadratureVote(mesh, seeds, bad_levels, out),
                 std::invalid_argument);
    ms::QuadratureVoteParameters bad_bins;
    bad_bins.num_bins = 0;
    EXPECT_THROW(ms::LabelElementsByQuadratureVote(mesh, seeds, bad_bins, out),
                 std::invalid_argument);
    EXPECT_THROW(ms::LabelElementsByQuadratureVote(mesh, ms::SeedSet{}, {}, out),
                 std::invalid_argument);
    EXPECT_THROW(ms::LabelElementsByQuadratureVote(mesh, seeds, {}, out, {0, 1}),
                 std::invalid_argument);
    EXPECT_THROW(ms::LabelElementsByQuadratureVote(mesh, seeds, {}, out, {0, -1, 2}),
                 std::invalid_argument);
}

//--------------------------------------------------------------------------------------------
// Periodic images
//--------------------------------------------------------------------------------------------

/**
 * Image counts, order, wrapping and copied data. With p periodic axes and the default (exact)
 * margin every seed gets (2k + 1)^p copies; originals come first with their input order and
 * indices; seeds outside the box are moved in by whole box lengths (1.25 -> 0.25, -0.25 -> 0.75
 * on a unit box), seeds inside are bitwise unchanged; every per-seed array is copied to the
 * images; and a margin keeps exactly the images within that distance of the box.
 */
TEST(PeriodicImages, CountsWrappingAndData) {
    ms::SeedSet seeds = RandomSeeds(5, 3u, 0.5);
    seeds.x[1] = 1.25;  // outside on +x
    seeds.y[2] = -0.25; // outside on -y
    for (std::size_t i = 0; i < seeds.Size(); ++i) {
        for (int k = 0; k < 4; ++k) {
            seeds.quat.push_back(static_cast<double>(10 * i + k));
        }
        seeds.phase.push_back(static_cast<int>(i) + 1);
    }
    ms::PeriodicBox box; // unit cube
    const std::size_t n = seeds.Size();

    // No periodic axis: a plain copy
    {
        const ms::PeriodicImages images = ms::BuildPeriodicImages(seeds, box, {});
        EXPECT_EQ(images.seeds.Size(), n);
        EXPECT_EQ(images.num_images, 0);
        EXPECT_EQ(images.seeds.x, seeds.x); // no wrapping on non-periodic axes
    }
    // x only: 3 copies; x, y, z: 27 copies; two layers on all axes: 125 copies
    struct Count {
        std::array<bool, 3> periodic;
        int layers;
        std::size_t copies;
    };
    for (const Count& c : {Count{{true, false, false}, 1, 3}, Count{{true, true, true}, 1, 27},
                           Count{{true, true, true}, 2, 125}}) {
        box.periodic = c.periodic;
        ms::PeriodicImageParameters params;
        params.layers = c.layers;
        const ms::PeriodicImages images = ms::BuildPeriodicImages(seeds, box, params);
        EXPECT_EQ(images.seeds.Size(), c.copies * n);
        EXPECT_EQ(images.num_original, static_cast<int>(n));
        EXPECT_EQ(images.num_images, static_cast<int>((c.copies - 1) * n));
        ASSERT_EQ(images.original.size(), images.seeds.Size());
        ASSERT_EQ(images.seeds.weight.size(), images.seeds.Size());
        ASSERT_EQ(images.seeds.quat.size(), 4 * images.seeds.Size());
        ASSERT_EQ(images.seeds.source_id.size(), images.seeds.Size());
        ASSERT_EQ(images.seeds.phase.size(), images.seeds.Size());
        for (std::size_t k = 0; k < images.seeds.Size(); ++k) {
            const auto i = static_cast<std::size_t>(images.original[k]);
            if (k < n) {
                EXPECT_EQ(i, k); // originals first, same order
            }
            EXPECT_EQ(images.seeds.weight[k], seeds.weight[i]);
            EXPECT_EQ(images.seeds.source_id[k], seeds.source_id[i]);
            EXPECT_EQ(images.seeds.phase[k], seeds.phase[i]);
            EXPECT_EQ(images.seeds.quat[4 * k + 3], seeds.quat[4 * i + 3]);
            // An image differs from its wrapped original by whole box lengths
            const double shift_x = images.seeds.x[k] - images.seeds.x[i];
            EXPECT_DOUBLE_EQ(shift_x, std::round(shift_x));
        }
        // Wrapping on the periodic axes, bitwise identity for in-box coordinates
        EXPECT_DOUBLE_EQ(images.seeds.x[1], 0.25);
        EXPECT_EQ(images.seeds.x[0], seeds.x[0]);
        if (c.periodic[1]) {
            EXPECT_DOUBLE_EQ(images.seeds.y[2], 0.75);
        } else {
            EXPECT_EQ(images.seeds.y[2], -0.25);
        }
    }

    // Margin: keep images within 0.2 of the box on every periodic axis (count by hand)
    box.periodic = {true, true, false};
    ms::PeriodicImageParameters margin_params;
    margin_params.margin = 0.2;
    const ms::PeriodicImages near = ms::BuildPeriodicImages(seeds, box, margin_params);
    const ms::PeriodicImages wrapped = ms::BuildPeriodicImages(seeds, box, {});
    std::size_t expected = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (int nx = -1; nx <= 1; ++nx) {
            for (int ny = -1; ny <= 1; ++ny) {
                const double x = wrapped.seeds.x[i] + nx;
                const double y = wrapped.seeds.y[i] + ny;
                if (x >= -0.2 && x <= 1.2 && y >= -0.2 && y <= 1.2) {
                    ++expected;
                }
            }
        }
    }
    EXPECT_EQ(near.seeds.Size(), expected);
    EXPECT_LT(near.seeds.Size(), wrapped.seeds.Size());

    // Invalid input
    ms::PeriodicImageParameters zero_layers;
    zero_layers.layers = 0;
    EXPECT_THROW(ms::BuildPeriodicImages(seeds, box, zero_layers), std::invalid_argument);
    EXPECT_THROW(ms::BuildPeriodicImages(ms::SeedSet{}, box, {}), std::invalid_argument);
    ms::PeriodicBox flat = box;
    flat.hi[0] = flat.lo[0];
    EXPECT_THROW(ms::BuildPeriodicImages(seeds, flat, {}), std::invalid_argument);
}

/**
 * The periodic box from the mesh is the bounding box of all nodes of all ranks, the same on
 * every rank (here the unit cube for cube_tet). For a topologically periodic MFEM mesh
 * (Mesh::MakePeriodic of a 4^3 unit hex box, periodic in x) the vertices at x = 1 are merged
 * into those at x = 0, so the vertices alone end at x = 0.75; the box must still be [0, 1],
 * taken from the discontinuous Nodes that keep every element's corners. Every rank checks that
 * serial mesh alone (MPI_COMM_SELF).
 */
TEST(PeriodicImages, BoxFromMesh) {
    mfem::Mesh serial = CubeTet();
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    const ms::PeriodicBox box =
        ms::PeriodicBoxFromMesh(*dist.pmesh, MPI_COMM_WORLD, {true, false, true});
    for (int d = 0; d < 3; ++d) {
        EXPECT_DOUBLE_EQ(box.lo[d], 0.0);
        EXPECT_DOUBLE_EQ(box.hi[d], 1.0);
    }
    EXPECT_TRUE(box.periodic[0]);
    EXPECT_FALSE(box.periodic[1]);

    // Topologically periodic mesh: merged vertices, discontinuous Nodes
    mfem::Mesh plain = HexBox(4);
    mfem::Vector shift_x({1.0, 0.0, 0.0});
    mfem::Mesh periodic_mesh = mfem::Mesh::MakePeriodic(
        plain, plain.CreatePeriodicVertexMapping({shift_x}));
    double max_vertex_x = 0.0;
    for (int v = 0; v < periodic_mesh.GetNV(); ++v) {
        max_vertex_x = std::max(max_vertex_x, periodic_mesh.GetVertex(v)[0]);
    }
    EXPECT_DOUBLE_EQ(max_vertex_x, 0.75); // what a vertex-based box would have used
    const ms::PeriodicBox periodic_box =
        ms::PeriodicBoxFromMesh(periodic_mesh, MPI_COMM_SELF, {true, false, false});
    for (int d = 0; d < 3; ++d) {
        EXPECT_DOUBLE_EQ(periodic_box.lo[d], 0.0);
        EXPECT_DOUBLE_EQ(periodic_box.hi[d], 1.0);
    }
}

/**
 * Image layers: 1 for Voronoi/Laguerre/Additive and for axis-aligned Quadric tensors; for a
 * sheared tensor the documented bound k = ceil(sqrt(mu_max / mu_min) R / L_min) (R = sqrt(3)/2
 * for a fully periodic unit cube); a semidefinite sheared tensor has no finite exact bound and
 * throws. The bound is then checked in use on a fully periodic 6^3 hex box with a nearly
 * columnar tensor A = I - 0.99 u u^T (eigenvalues 0.01, 1, 1; u along (1, 0.618, 0.382), not
 * aligned with any lattice direction of the box) and 2 seeds, which gives k = 9: labels computed
 * with k layers equal labels computed with 2k + 2 layers (no copy beyond k wins, even far
 * beyond), and, as a control that makes the test able to fail, labels with a single layer differ
 * (along the nearly free direction u, a seed copy several box lengths away can be "closer" than
 * any of the 27 neighbors; measured: 10 of the 216 elements change).
 */
TEST(PeriodicImages, RequiredLayers) {
    ms::PeriodicBox box;
    box.periodic = {true, true, true};
    const ms::SeedSet seeds = RandomSeeds(8, 21u);
    const std::array<double, 6> identity = {1, 1, 1, 0, 0, 0};
    EXPECT_EQ(ms::RequiredImageLayers(box, ms::MetricType::VORONOI, identity, seeds), 1);
    EXPECT_EQ(ms::RequiredImageLayers(box, ms::MetricType::ADDITIVE, identity, seeds), 1);
    EXPECT_EQ(ms::RequiredImageLayers(box, ms::MetricType::QUADRIC, {4, 1, 0.01, 0, 0, 0},
                                      seeds),
              1);
    ms::PeriodicBox none;
    EXPECT_EQ(ms::RequiredImageLayers(none, ms::MetricType::QUADRIC, {1, 1, 1, 0.9, 0, 0},
                                      seeds),
              1);

    // Sheared tensor: eigenvalues of [[1,0,0],[0,1,0.9],[0,0.9,1]] are 0.1, 1, 1.9
    const std::array<double, 6> sheared = {1.0, 1.0, 1.0, 0.9, 0.0, 0.0};
    const int k = ms::RequiredImageLayers(box, ms::MetricType::QUADRIC, sheared, seeds);
    EXPECT_EQ(k, static_cast<int>(std::ceil(std::sqrt(1.9 / 0.1) * std::sqrt(3.0) / 2.0)));
    EXPECT_THROW(ms::RequiredImageLayers(box, ms::MetricType::QUADRIC, {1, 1, 0, 1, 0, 0},
                                         seeds),
                 std::invalid_argument);

    // In use: 1, k and 2k + 2 layers for a nearly columnar tensor, A = I - (1 - eps) u u^T
    const double eps = 0.01;
    const double norm_u = std::sqrt(1.0 + 0.618 * 0.618 + 0.382 * 0.382);
    const double u[3] = {1.0 / norm_u, 0.618 / norm_u, 0.382 / norm_u};
    const std::array<double, 6> columnar = {
        1.0 - (1.0 - eps) * u[0] * u[0], 1.0 - (1.0 - eps) * u[1] * u[1],
        1.0 - (1.0 - eps) * u[2] * u[2], -(1.0 - eps) * u[1] * u[2],
        -(1.0 - eps) * u[0] * u[2],      -(1.0 - eps) * u[0] * u[1]};
    const ms::SeedSet two_seeds = RandomSeeds(2, 21u);
    const int k_columnar =
        ms::RequiredImageLayers(box, ms::MetricType::QUADRIC, columnar, two_seeds);
    EXPECT_EQ(k_columnar, static_cast<int>(std::ceil(std::sqrt(1.0 / eps) * std::sqrt(3.0) / 2.0)));
    mfem::Mesh mesh = HexBox(6);
    ms::TessellationOptions options;
    options.metric = ms::MetricType::QUADRIC;
    options.global_tensor = columnar;
    auto label_with_layers = [&](int layers) {
        ms::PeriodicImageParameters params;
        params.layers = layers;
        const ms::PeriodicImages images = ms::BuildPeriodicImages(two_seeds, box, params);
        mfem::Array<int> element_seed;
        ms::LabelElementsWithSeeds(mesh, images.seeds, options, element_seed);
        ms::MapImageLabelsToOriginal(images, element_seed);
        return ToHost(element_seed);
    };
    const std::vector<int> with_one = label_with_layers(1);
    const std::vector<int> with_k = label_with_layers(k_columnar);
    const std::vector<int> with_many = label_with_layers(2 * k_columnar + 2);
    EXPECT_EQ(with_k, with_many);
    int differing = 0;
    for (std::size_t e = 0; e < with_k.size(); ++e) {
        if (with_one[e] != with_k[e]) {
            ++differing;
        }
    }
    EXPECT_GT(differing, 0) << "one layer should not be enough for this tensor";
    std::cout << "[ columnar quadric ] k = " << k_columnar
              << "; elements labeled differently with 1 layer: "
              << differing << " of " << with_k.size() << std::endl;
}

/**
 * Periodic shift invariance. On a fully periodic 8^3 hex box, translating every seed by
 * t = (2, 1, 3) element widths and wrapping it back into the box must translate the labels:
 * the element at centroid c + t (mod 1) gets the seed that the element at c had. This holds
 * exactly only if the images make the tessellation truly periodic, because the translation
 * moves seeds across the box faces. Seeds and weights are on a 2^-20 grid so the shift and wrap
 * are exact in floating point. Checked for centroid labeling and for the vote, Voronoi and
 * Laguerre, on the distributed mesh (gathered to rank 0). As a control, the same comparison
 * without images must fail for at least one element (otherwise the test would prove nothing).
 */
TEST(PeriodicImages, ShiftInvariance) {
    const int n = 8;
    mfem::Mesh serial = HexBox(n);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    const ms::PeriodicBox box =
        ms::PeriodicBoxFromMesh(*dist.pmesh, MPI_COMM_WORLD, {true, true, true});
    const int shift[3] = {2, 1, 3};

    // Serial element id from integer centroid coordinates (centroid = (i + 1/2) / n)
    ms::ElementSamples centroids;
    ms::ComputeElementSamples(serial, -1, centroids);
    const double* cx = centroids.cx.HostRead();
    const double* cy = centroids.cy.HostRead();
    const double* cz = centroids.cz.HostRead();
    std::map<std::array<int, 3>, int> element_at;
    std::vector<std::array<int, 3>> cell_of(static_cast<std::size_t>(serial.GetNE()));
    for (int e = 0; e < serial.GetNE(); ++e) {
        const std::array<int, 3> cell = {static_cast<int>(std::floor(cx[e] * n)),
                                         static_cast<int>(std::floor(cy[e] * n)),
                                         static_cast<int>(std::floor(cz[e] * n))};
        element_at[cell] = e;
        cell_of[static_cast<std::size_t>(e)] = cell;
    }

    // Labels (original seed index per serial element) for one configuration, on rank 0
    auto label = [&](const ms::SeedSet& seeds, ms::MetricType metric, bool vote, bool periodic) {
        ms::PeriodicBox use_box = box;
        if (!periodic) {
            use_box.periodic = {false, false, false};
        }
        const ms::PeriodicImages images = ms::BuildPeriodicImages(seeds, use_box, {});
        mfem::Array<int> element_seed;
        if (vote) {
            ms::QuadratureVoteParameters params;
            params.metric = metric;
            ms::LabelElementsByQuadratureVote(*dist.pmesh, images.seeds, params, element_seed,
                                              images.original);
        } else {
            ms::TessellationOptions options;
            options.metric = metric;
            ms::LabelElementsWithSeeds(*dist.pmesh, images.seeds, options, element_seed);
            ms::MapImageLabelsToOriginal(images, element_seed);
        }
        return Gather(dist, element_seed);
    };

    // Number of elements whose label does not follow the shift
    auto mismatches = [&](const std::vector<int>& before, const std::vector<int>& after) {
        int count = 0;
        for (std::size_t e = 0; e < before.size(); ++e) {
            std::array<int, 3> moved;
            for (int d = 0; d < 3; ++d) {
                moved[static_cast<std::size_t>(d)] =
                    (cell_of[e][static_cast<std::size_t>(d)] + shift[d]) % n;
            }
            const auto target = static_cast<std::size_t>(element_at.at(moved));
            count += (after[target] != before[e]) ? 1 : 0;
        }
        return count;
    };

    for (ms::MetricType metric : {ms::MetricType::VORONOI, ms::MetricType::LAGUERRE}) {
        double max_weight = 0.0;
        if (metric == ms::MetricType::LAGUERRE) {
            max_weight = 0.01;
        }
        const ms::SeedSet seeds = RandomSeeds(12, 77u, max_weight, true);
        ms::SeedSet shifted = seeds;
        for (std::size_t i = 0; i < seeds.Size(); ++i) {
            shifted.x[i] += shift[0] / static_cast<double>(n); // wrapped by BuildPeriodicImages
            shifted.y[i] += shift[1] / static_cast<double>(n);
            shifted.z[i] += shift[2] / static_cast<double>(n);
        }
        for (bool vote : {false, true}) {
            SCOPED_TRACE(std::string(vote ? "vote" : "centroid") +
                         (metric == ms::MetricType::LAGUERRE ? " laguerre" : " voronoi"));
            const std::vector<int> before = label(seeds, metric, vote, true);
            const std::vector<int> after = label(shifted, metric, vote, true);
            if (mfem::Mpi::Root()) {
                EXPECT_EQ(mismatches(before, after), 0);
            }
        }
    }

    // Control: without images the shifted labels do not follow the shift
    const ms::SeedSet seeds = RandomSeeds(12, 77u, 0.0, true);
    ms::SeedSet shifted = seeds;
    for (std::size_t i = 0; i < seeds.Size(); ++i) {
        shifted.x[i] = std::fmod(shifted.x[i] + shift[0] / static_cast<double>(n), 1.0);
        shifted.y[i] = std::fmod(shifted.y[i] + shift[1] / static_cast<double>(n), 1.0);
        shifted.z[i] = std::fmod(shifted.z[i] + shift[2] / static_cast<double>(n), 1.0);
    }
    const std::vector<int> before = label(seeds, ms::MetricType::VORONOI, false, false);
    const std::vector<int> after = label(shifted, ms::MetricType::VORONOI, false, false);
    if (mfem::Mpi::Root()) {
        EXPECT_GT(mismatches(before, after), 0);
    }
}

/**
 * A grain that crosses a periodic face is one grain. On an 8^3 hex box periodic in x, seed 0 at
 * x = 0.0625 and seed 1 at x = 0.46875 (same y, z). Seed 0's image at x = 1.0625 makes the cell
 * of seed 0 wrap around: its boundaries are the bisectors x = 0.265625 (with seed 1) and
 * x = 0.765625 (between seed 1 and seed 0's image). So the element columns with centroid
 * x = 0.0625, 0.1875 (left face) and 0.8125, 0.9375 (right face) form grain 1, the rest grain 2;
 * there are exactly 2 grains and grain 1 holds half the volume. Checked for the centroid labeler
 * and the vote, after compaction over all ranks. The bisectors are placed 1/64 inside the element
 * columns 0.25-0.375 and 0.75-0.875, before the first of the two Gauss points per direction
 * (0.25 + 0.125 (1/2 -/+ 1/sqrt(12)) = 0.279 and 0.346): every voting point of a cut element is
 * then on the majority side, so both modes must give the same answer. (A bisector between the two
 * Gauss points would give an exact 50/50 vote, decided by the lowest seed index.)
 */
TEST(PeriodicImages, GrainCrossingPeriodicFaceIsOneGrain) {
    mfem::Mesh serial = HexBox(8);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    mfem::ParMesh& mesh = *dist.pmesh;
    const ms::PeriodicBox box = ms::PeriodicBoxFromMesh(mesh, MPI_COMM_WORLD, {true, false, false});
    ms::SeedSet seeds;
    seeds.x = {0.0625, 0.46875};
    seeds.y = {0.5, 0.5};
    seeds.z = {0.5, 0.5};
    seeds.source_id = {0, 1};
    const ms::PeriodicImages images = ms::BuildPeriodicImages(seeds, box, {});
    EXPECT_EQ(images.seeds.Size(), 6u);

    ms::ElementSamples samples;
    ms::ComputeElementSamples(mesh, -1, samples);
    const int ne = mesh.GetNE();

    for (bool vote : {false, true}) {
        SCOPED_TRACE(vote ? "vote" : "centroid");
        ms::TessellationLabels labels;
        if (vote) {
            labels = ms::LabelTessellationByQuadratureVote(mesh, MPI_COMM_WORLD, images.seeds,
                                                           ms::QuadratureVoteParameters{},
                                                           images.original, images.num_original);
        } else {
            mfem::Array<int> element_seed;
            ms::LabelElementsWithSeeds(mesh, images.seeds, ms::TessellationOptions{},
                                       element_seed);
            ms::MapImageLabelsToOriginal(images, element_seed);
            ms::CompactSeedLabels(MPI_COMM_WORLD, element_seed, images.num_original, labels);
        }
        EXPECT_EQ(labels.num_grains, 2);
        EXPECT_EQ(labels.grain_seed, (std::vector<int>{0, 1}));

        const std::vector<int> grain = ToHost(labels.element_grain);
        double grain1_volume = 0.0;
        int wrong = 0;
        if (ne > 0) {
            const double* cx = samples.cx.HostRead();
            const double* vol = samples.volume.HostRead();
            for (int e = 0; e < ne; ++e) {
                const bool in_grain1 = cx[e] < 0.265625 || cx[e] > 0.765625;
                wrong += (grain[static_cast<std::size_t>(e)] != (in_grain1 ? 1 : 2)) ? 1 : 0;
                if (grain[static_cast<std::size_t>(e)] == 1) {
                    grain1_volume += vol[e];
                }
            }
        }
        EXPECT_EQ(wrong, 0);
        EXPECT_NEAR(GlobalSum(grain1_volume), 0.5, 1e-12);
    }
}

/** MapImageLabelsToOriginal maps images to originals, keeps -1, and rejects bad indices. */
TEST(PeriodicImages, MapToOriginal) {
    ms::SeedSet seeds = RandomSeeds(2, 9u);
    ms::PeriodicBox box;
    box.periodic = {true, false, false};
    const ms::PeriodicImages images = ms::BuildPeriodicImages(seeds, box, {});
    ASSERT_EQ(images.seeds.Size(), 6u); // 0,1 originals; 2,3 shift -1; 4,5 shift +1
    mfem::Array<int> labels({0, 1, 2, 3, 4, 5, -1});
    ms::MapImageLabelsToOriginal(images, labels);
    EXPECT_EQ(ToHost(labels), (std::vector<int>{0, 1, 0, 1, 0, 1, -1}));
    mfem::Array<int> bad({6});
    EXPECT_THROW(ms::MapImageLabelsToOriginal(images, bad), std::invalid_argument);
}

/** Initializes MPI (mfem::Mpi finalizes it at exit) and runs every test on every rank. */
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
