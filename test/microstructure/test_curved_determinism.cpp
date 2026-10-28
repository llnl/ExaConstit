/*
 * test_curved_determinism: vertex positions, element keys, seeding, fragment cleanup and periodic
 * face pairing on genuinely curved meshes and on meshes refined in parallel (WP-CV1).
 *
 * Background. For a mesh with high-order Nodes, MFEM's vertex array is a partition-dependent
 * average (ParMesh) or stale after node motion (Mesh::Transform), and after a parallel
 * UniformRefinement the copies of a shared vertex in the Nodes differ in the last bits between
 * ranks. The generator therefore reads vertex positions from the Nodes, broadcasts the owner's
 * value to every copy (VertexCoordinates), rounds coordinates to a grid before hashing
 * element keys, and tests periodic plane faces with every nodal point.
 *
 * "Genuinely curved" below means the Nodes are moved off the straight map (Warp), not only
 * projected with SetCurvature. Every test compares the MPI_COMM_WORLD run (np = 1-4) with a
 * single-rank reference computed in the same executable on MPI_COMM_SELF.
 */
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <vector>

#include "microstructure/geometry/periodic_images.hpp"
#include "microstructure/labeling/fragment_cleanup.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/mesh/straight_sided.hpp"
#include "microstructure/mesh/vertex_coordinates.hpp"
#include "microstructure/seeding/seed_element_locator.hpp"
#include "microstructure/seeding/seed_generators.hpp"
#include <gtest/gtest.h>

namespace ms = exaconstit::microstructure;

namespace {

constexpr double PI = 3.141592653589793;

/** Smooth warp that moves every interior and boundary node (genuinely curved elements). */
void Warp(const mfem::Vector& x, mfem::Vector& y) {
    y = x;
    y(0) += 0.05 * std::sin(3.0 * x(1)) * std::cos(2.0 * x(2));
    y(1) += 0.04 * std::sin(2.0 * x(0) + x(2));
    y(2) += 0.03 * std::cos(3.0 * x(0) * x(1));
}

/** Straight affine map giving non-dyadic coordinates (so last-bit differences can appear). */
void Skew(const mfem::Vector& x, mfem::Vector& y) {
    y = x;
    y(0) += 0.3 * x(1) + 0.1 * x(2);
    y(1) += 0.2 * x(2);
    y(2) *= 1.7;
}

/**
 * Periodic warp of the unit cube: u_d vanishes on the planes x_d = 0, 1 (sin(2 pi x_d)), and the
 * tangential components are periodic, so the curved face meshes on opposite planes still match
 * after a shift by one box length. Faces stay on their planes; edges inside them curve.
 */
void PeriodicWarp(const mfem::Vector& x, mfem::Vector& y) {
    y = x;
    const double a = 0.03;
    const double c[3] = {std::cos(2 * PI * x(0)), std::cos(2 * PI * x(1)), std::cos(2 * PI * x(2))};
    y(0) += a * std::sin(2 * PI * x(0)) * (1.0 + 0.5 * c[1] * c[2]);
    y(1) += a * std::sin(2 * PI * x(1)) * (1.0 + 0.5 * c[0] * c[2]);
    y(2) += a * std::sin(2 * PI * x(2)) * (1.0 + 0.5 * c[0] * c[1]);
}

/** Scale and shift, applied to Nodes only by Mesh::Transform: the vertex array goes stale. */
void ScaleShift(const mfem::Vector& x, mfem::Vector& y) {
    y = x;
    for (int d = 0; d < 3; d++) {
        y(d) = 2.0 * x(d) + 0.5;
    }
}

/** Bulges the x = 1 face of the unit cube between its corners (see the failure test). */
void BulgeRightFace(const mfem::Vector& x, mfem::Vector& y) {
    y = x;
    y(0) += 0.05 * std::sin(2 * PI * x(1)) * std::sin(2 * PI * x(2)) * std::pow(x(0), 4);
}

/** A box mesh of hexes or tets with order-p Nodes moved by Warp. */
mfem::Mesh WarpedBox(int order, mfem::Element::Type type) {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(4, 3, 3, type, 1.0, 1.0, 1.0, false);
    mesh.SetCurvature(order);
    mesh.Transform(Warp);
    return mesh;
}

/** The distributed mesh plus the serial-order bookkeeping. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

Distributed Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    Distributed d;
    d.order = ms::BuildSerialOrder(serial, comm);
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/** All local keys of every rank, sorted (a multiset), identical on every rank. Collective. */
std::vector<std::uint64_t> GlobalSortedKeys(const std::vector<std::uint64_t>& local,
                                            MPI_Comm comm) {
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);
    const int count = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(nranks));
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
    int total = 0;
    for (int r = 0; r < nranks; r++) {
        displs[static_cast<std::size_t>(r)] = total;
        total += counts[static_cast<std::size_t>(r)];
    }
    std::vector<std::uint64_t> all(static_cast<std::size_t>(total));
    MPI_Allgatherv(local.data(),
                   count,
                   MPI_UINT64_T,
                   all.data(),
                   counts.data(),
                   displs.data(),
                   MPI_UINT64_T,
                   comm);
    std::sort(all.begin(), all.end());
    return all;
}

/** Bitwise equality of seed positions and source ids. */
void ExpectSameSeeds(const ms::SeedSet& a, const ms::SeedSet& b) {
    ASSERT_EQ(a.Size(), b.Size());
    for (std::size_t i = 0; i < a.Size(); i++) {
        EXPECT_EQ(a.x[i], b.x[i]) << "seed " << i;
        EXPECT_EQ(a.y[i], b.y[i]) << "seed " << i;
        EXPECT_EQ(a.z[i], b.z[i]) << "seed " << i;
        EXPECT_EQ(a.source_id[i], b.source_id[i]) << "seed " << i;
    }
}

/**
 * Label of each local element: the seed nearest to the element centroid shifted by a
 * pseudo-random offset of up to 0.6 element sizes, drawn from the element key (so it does not
 * depend on the partition). The noise breaks grains into fragments, which exercises the
 * smallest-key tie-breaks of fragment cleanup. Labels are 1-based.
 */
mfem::Array<int> NoisyVoronoiLabels(mfem::ParMesh& pmesh, const ms::SeedSet& seeds, double h) {
    const int ne = pmesh.GetNE();
    mfem::Array<int> label(ne);
    if (ne == 0) {
        return label;
    }
    ms::ElementSamples samples;
    ms::ComputeElementSamples(pmesh, -1, samples);
    const double* cx = samples.cx.HostRead();
    const double* cy = samples.cy.HostRead();
    const double* cz = samples.cz.HostRead();
    const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(pmesh);
    for (int e = 0; e < ne; e++) {
        const std::uint64_t k = keys[static_cast<std::size_t>(e)];
        double p[3] = {cx[e], cy[e], cz[e]};
        for (int d = 0; d < 3; d++) {
            const double u = static_cast<double>((k >> (16 * d)) & 0xFFFF) / 65535.0; // [0, 1]
            p[d] += 0.6 * h * (2.0 * u - 1.0);
        }
        int best = 0;
        double best_d2 = 1e300;
        for (std::size_t s = 0; s < seeds.Size(); s++) {
            const double dx = p[0] - seeds.x[s];
            const double dy = p[1] - seeds.y[s];
            const double dz = p[2] - seeds.z[s];
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best_d2) {
                best_d2 = d2;
                best = static_cast<int>(s);
            }
        }
        label[e] = best + 1;
    }
    return label;
}

/** Element-weighted seeding, then noisy labels and fragment cleanup, gathered to serial order. */
struct PipelineResult {
    ms::SeedSet seeds;
    std::vector<int> labels; // after cleanup, serial order
    int num_grains = 0;
    ms::FragmentCleanupReport report;
};

PipelineResult RunPipeline(mfem::Mesh& serial, MPI_Comm comm) {
    Distributed d = Distribute(serial, comm);
    ms::SeedingParameters params;
    params.placement = ms::SeedPlacement::ELEMENT_WEIGHTED;
    params.num_seeds = 12;
    params.rng_seed = 20261022;
    PipelineResult out;
    out.seeds = ms::GenerateSeeds(*d.pmesh, params);
    const mfem::Array<int> label = NoisyVoronoiLabels(*d.pmesh, out.seeds, 0.25);

    const ms::MeshAdjacency adjacency(*d.pmesh);
    ms::FragmentCleanupParameters cleanup;
    cleanup.policy = ms::FragmentPolicy::SPLIT_LARGE_MERGE_TINY;
    cleanup.min_fragment_elements = 3;
    cleanup.min_grain_elements = 3;
    std::vector<std::int64_t> source(out.seeds.Size());
    for (std::size_t i = 0; i < source.size(); i++) {
        source[i] = static_cast<std::int64_t>(i);
    }
    const ms::FragmentCleanupResult result = ms::CleanupFragments(
        adjacency, label, static_cast<int>(out.seeds.Size()), source, cleanup);
    out.labels = ms::GatherToSerial(comm, d.order, result.element_grain, 0);
    out.labels.resize(static_cast<std::size_t>(serial.GetNE()));
    MPI_Bcast(out.labels.data(), serial.GetNE(), MPI_INT, 0, comm);
    out.num_grains = result.num_grains;
    out.report = result.report;
    return out;
}

} // namespace

// VertexCoordinates reads the Nodes vertex values (not the averaged vertex array), and after two
// parallel refinements of a warped order-3 mesh every copy of a shared vertex holds the same
// bits. The check broadcasts nothing itself: it reduces each shared vertex's coordinates with MIN
// and with MAX over its group (through the Nodes space's own GroupComm, the independent MFEM
// path) and requires both to equal the local value. The raw Nodes copies are also counted, to
// show the broadcast had something to fix (reported, not required).
TEST(VertexCoordinates, NodesSourceAndIdenticalCopiesAfterParallelRefinement) {
    mfem::Mesh serial = WarpedBox(3, mfem::Element::HEXAHEDRON);
    // Serial: exactly the Nodes vertex dofs
    {
        const ms::VertexCoordinates coords(serial);
        EXPECT_TRUE(coords.FromNodes());
        EXPECT_FALSE(coords.Broadcast());
        const mfem::GridFunction& nodes = *serial.GetNodes();
        mfem::Array<int> dofs;
        for (int v = 0; v < serial.GetNV(); v++) {
            nodes.FESpace()->GetVertexDofs(v, dofs);
            for (int d = 0; d < 3; d++) {
                EXPECT_EQ(coords.Vertex(v)[d], nodes(nodes.FESpace()->DofToVDof(dofs[0], d)));
            }
        }
    }

    mfem::ParMesh pmesh(MPI_COMM_WORLD, serial);
    pmesh.UniformRefinement();
    pmesh.UniformRefinement();
    const ms::VertexCoordinates coords(pmesh);
    EXPECT_TRUE(coords.Broadcast());

    auto* nodes = dynamic_cast<mfem::ParGridFunction*>(pmesh.GetNodes());
    ASSERT_NE(nodes, nullptr);
    mfem::ParFiniteElementSpace* fes = nodes->ParFESpace();
    const int nvdofs = fes->GetVSize();
    long long raw_differ = 0;
    for (const int use_coords : {0, 1}) {
        // Vertex values placed at their vdofs (other dofs 0), reduced with MIN and with MAX
        mfem::Vector lo(nvdofs), hi(nvdofs), own(nvdofs);
        own = 0.0;
        mfem::Array<int> dofs;
        for (int v = 0; v < pmesh.GetNV(); v++) {
            fes->GetVertexDofs(v, dofs);
            for (int d = 0; d < 3; d++) {
                const int vdof = fes->DofToVDof(dofs[0], d);
                own(vdof) = use_coords ? coords.Vertex(v)[d] : (*nodes)(vdof);
            }
        }
        lo = own;
        hi = own;
        fes->GroupComm().Reduce<double>(lo.HostReadWrite(), mfem::GroupCommunicator::Min<double>);
        fes->GroupComm().Bcast(lo.HostReadWrite());
        fes->GroupComm().Reduce<double>(hi.HostReadWrite(), mfem::GroupCommunicator::Max<double>);
        fes->GroupComm().Bcast(hi.HostReadWrite());
        long long differ = 0;
        for (int i = 0; i < nvdofs; i++) {
            differ += (lo(i) != own(i) || hi(i) != own(i)) ? 1 : 0;
        }
        MPI_Allreduce(MPI_IN_PLACE, &differ, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (use_coords) {
            EXPECT_EQ(differ, 0) << "copies of a shared vertex differ after the owner broadcast";
        } else {
            raw_differ = differ;
        }
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "[ info ] raw Nodes vertex-dof entries whose copies differ before the "
                     "broadcast: "
                  << raw_differ << std::endl;
    }
}

// Element keys round coordinates to a grid of about 2^-20 of the element's length scale: moving
// every Nodes value by a few units in the last place leaves every key unchanged, while moving one
// corner by 1e-3 of the element size (far above the grid spacing) changes that element's key. The
// key of a straight mesh with projected Nodes equals the key of the same mesh without Nodes (both
// read the same corners).
TEST(ElementKey, QuantizedAgainstLastBitNoise) {
    for (const auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        mfem::Mesh plain = mfem::Mesh::MakeCartesian3D(3, 3, 2, type, 1.0, 1.0, 1.0, false);
        plain.Transform(Skew);
        mfem::Mesh curved(plain);
        curved.SetCurvature(2);
        EXPECT_EQ(ms::ComputeElementKeys(plain), ms::ComputeElementKeys(curved));

        const std::vector<std::uint64_t> before = ms::ComputeElementKeys(curved);
        mfem::GridFunction& nodes = *curved.GetNodes();
        for (int i = 0; i < nodes.Size(); i++) {
            const double up = std::nextafter(nodes(i), 1e300);
            nodes(i) = std::nextafter(up, 1e300); // two ulps up
        }
        EXPECT_EQ(ms::ComputeElementKeys(curved), before);

        // A real move of vertex 0 (1e-3 of the element size) changes the keys of its elements
        mfem::Array<int> dofs;
        nodes.FESpace()->GetVertexDofs(0, dofs);
        nodes(nodes.FESpace()->DofToVDof(dofs[0], 0)) += 1.0e-3 / 3.0;
        const std::vector<std::uint64_t> moved = ms::ComputeElementKeys(curved);
        int changed = 0;
        for (std::size_t e = 0; e < moved.size(); e++) {
            changed += (moved[e] != before[e]) ? 1 : 0;
        }
        EXPECT_GT(changed, 0);
        // Keys remain unique
        std::vector<std::uint64_t> sorted = moved;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_TRUE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
    }
}

// Element-weighted seeding and fragment cleanup on genuinely warped order-2 and order-3 hex and
// tet meshes: the distributed run gives exactly the seeds, labels, grain count and cleanup counts
// of the single-rank run. Element keys name the seeding streams and break the cleanup ties, so
// this checks that keys read from the Nodes (not the partition-dependent vertex average) are the
// same on every partition.
TEST(CurvedDeterminism, WarpedElementWeightedSeedingAndCleanup) {
    for (const auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        for (int order = 2; order <= 3; order++) {
            mfem::Mesh serial = WarpedBox(order, type);
            ASSERT_FALSE(ms::StraightSidedLocal(serial)); // genuinely curved
            const PipelineResult world = RunPipeline(serial, MPI_COMM_WORLD);
            const PipelineResult self = RunPipeline(serial, MPI_COMM_SELF);
            ExpectSameSeeds(world.seeds, self.seeds);
            EXPECT_EQ(world.labels, self.labels) << "order " << order;
            EXPECT_EQ(world.num_grains, self.num_grains);
            EXPECT_EQ(world.report.fragments_found, self.report.fragments_found);
            EXPECT_EQ(world.report.fragments_split, self.report.fragments_split);
            EXPECT_EQ(world.report.fragments_merged, self.report.fragments_merged);
            EXPECT_EQ(world.report.small_grains_merged, self.report.small_grains_merged);
        }
    }
}

namespace {

/** Keys of `m` that are not in `reference` plus keys of `reference` not in `m` (both sorted). */
std::size_t KeyMismatches(const std::vector<std::uint64_t>& m,
                          const std::vector<std::uint64_t>& reference) {
    std::vector<std::uint64_t> diff;
    std::set_symmetric_difference(
        m.begin(), m.end(), reference.begin(), reference.end(), std::back_inserter(diff));
    return diff.size();
}

/** Seeds whose position or source id differs (bitwise) between two seed sets of equal size. */
std::size_t SeedMismatches(const ms::SeedSet& a, const ms::SeedSet& b) {
    std::size_t differ = 0;
    for (std::size_t i = 0; i < std::min(a.Size(), b.Size()); i++) {
        const bool same = a.x[i] == b.x[i] && a.y[i] == b.y[i] && a.z[i] == b.z[i] &&
                          a.source_id[i] == b.source_id[i];
        differ += same ? 0 : 1;
    }
    return differ + (std::max(a.Size(), b.Size()) - std::min(a.Size(), b.Size()));
}

/**
 * Element label = nearest seed to the element centroid, as a map key -> label over all ranks.
 * Collective over comm.
 */
std::map<std::uint64_t, std::uint64_t>
LabelsByKey(mfem::ParMesh& m, const ms::SeedSet& seeds, MPI_Comm comm) {
    const int ne = m.GetNE();
    const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(m);
    std::vector<std::uint64_t> pairs; // key, label, key, label, ...
    if (ne > 0) {
        ms::ElementSamples s;
        ms::ComputeElementSamples(m, -1, s);
        for (int e = 0; e < ne; e++) {
            int best = 0;
            double best_d2 = 1e300;
            for (std::size_t j = 0; j < seeds.Size(); j++) {
                const double dx = s.cx(e) - seeds.x[j];
                const double dy = s.cy(e) - seeds.y[j];
                const double dz = s.cz(e) - seeds.z[j];
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best = static_cast<int>(j);
                }
            }
            pairs.push_back(keys[static_cast<std::size_t>(e)]);
            pairs.push_back(static_cast<std::uint64_t>(best));
        }
    }
    const std::vector<std::uint64_t> all = [&] {
        int nranks = 1;
        MPI_Comm_size(comm, &nranks);
        const int count = static_cast<int>(pairs.size());
        std::vector<int> counts(static_cast<std::size_t>(nranks));
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
        std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
        int total = 0;
        for (int r = 0; r < nranks; r++) {
            displs[static_cast<std::size_t>(r)] = total;
            total += counts[static_cast<std::size_t>(r)];
        }
        std::vector<std::uint64_t> gathered(static_cast<std::size_t>(total));
        MPI_Allgatherv(pairs.data(),
                       count,
                       MPI_UINT64_T,
                       gathered.data(),
                       counts.data(),
                       displs.data(),
                       MPI_UINT64_T,
                       comm);
        return gathered;
    }();
    std::map<std::uint64_t, std::uint64_t> out;
    for (std::size_t i = 0; i + 1 < all.size(); i += 2) {
        out[all[i]] = all[i + 1];
    }
    return out;
}

/**
 * The hook-style comparison: the serial mesh is distributed over MPI_COMM_WORLD and over
 * MPI_COMM_SELF, both are refined `refinements` times in parallel, and the two runs must give
 * bitwise the same multiset of element keys and the same seeds for rejection sampling and for
 * `num_seeds` element-weighted seeds. Mismatch counts are printed before the checks, so a failure
 * says how far off it is. Nearest-seed labels are compared by key and must agree on at least
 * 99.9% of the elements (a centroid within round-off of a bisector may go either way).
 */
void CompareRefinedRuns(const mfem::Mesh& serial,
                        int refinements,
                        int num_seeds,
                        const char* name) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    mfem::Mesh copy(serial);
    mfem::ParMesh world(MPI_COMM_WORLD, copy);
    mfem::ParMesh self(MPI_COMM_SELF, copy);
    for (int k = 0; k < refinements; k++) {
        world.UniformRefinement();
        self.UniformRefinement();
    }

    // Keys as a multiset
    const std::vector<std::uint64_t> world_keys = GlobalSortedKeys(ms::ComputeElementKeys(world),
                                                                   MPI_COMM_WORLD);
    std::vector<std::uint64_t> self_keys = ms::ComputeElementKeys(self);
    std::sort(self_keys.begin(), self_keys.end());
    const std::size_t key_mismatch = KeyMismatches(world_keys, self_keys);
    if (rank == 0) {
        std::cout << "[ info ] " << name << ": " << self_keys.size() << " elements, "
                  << key_mismatch << " keys differ from np 1" << std::endl;
    }
    EXPECT_EQ(key_mismatch, 0u) << name;

    for (const auto placement :
         {ms::SeedPlacement::REJECTION, ms::SeedPlacement::ELEMENT_WEIGHTED}) {
        ms::SeedingParameters params;
        params.placement = placement;
        params.num_seeds = num_seeds;
        params.rng_seed = 77;
        const ms::SeedSet a = ms::GenerateSeeds(world, params);
        const ms::SeedSet b = ms::GenerateSeeds(self, params);
        const std::size_t seed_mismatch = SeedMismatches(a, b);
        const char* kind = (placement == ms::SeedPlacement::REJECTION) ? "rejection"
                                                                       : "element-weighted";
        if (rank == 0) {
            std::cout << "[ info ] " << name << ": " << seed_mismatch << " of " << b.Size() << " "
                      << kind << " seeds differ from np 1" << std::endl;
        }
        EXPECT_EQ(a.Size(), static_cast<std::size_t>(num_seeds)) << name << " " << kind;
        EXPECT_EQ(seed_mismatch, 0u) << name << " " << kind;

        const auto lw = LabelsByKey(world, b, MPI_COMM_WORLD);
        const auto ls = LabelsByKey(self, b, MPI_COMM_SELF);
        std::size_t agree = 0;
        for (const auto& entry : lw) {
            const auto it = ls.find(entry.first);
            agree += (it != ls.end() && it->second == entry.second) ? 1 : 0;
        }
        const double fraction = ls.empty()
                                    ? 1.0
                                    : static_cast<double>(agree) / static_cast<double>(ls.size());
        EXPECT_GE(fraction, 0.999) << name << " " << kind;
        if (rank == 0) {
            std::cout << "[ info ] " << name << ": " << kind << " label agreement " << agree
                      << " of " << ls.size() << std::endl;
        }
    }
}

/** Skew, then shift every coordinate by `offset` (a domain far from the origin). */
mfem::Mesh SkewedBox(int nx, int ny, int nz, double offset, int order) {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        nx, ny, nz, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
    mesh.Transform(Skew);
    if (offset != 0.0) {
        mesh.Transform([offset](const mfem::Vector& x, mfem::Vector& y) {
            y = x;
            for (int d = 0; d < 3; d++) {
                y(d) += offset;
            }
        });
    }
    mesh.SetCurvature(order);
    return mesh;
}

} // namespace

// The ExaConstit hook path: a straight (skewed, non-dyadic) mesh given order-3 Nodes with
// SetCurvature, distributed, then refined three times in parallel (9216 elements). Copies of
// shared Nodes differ in the last bits between ranks and from the single-rank refinement; keys and
// element-weighted sampling boxes are rounded to the key grid (about 2^-20 of the element's length
// scale, mesh/element_key.hpp), so keys and all 1000 element-weighted seeds must be bitwise those
// of np 1. Rejection seeds use the rounded mesh box and never read keys.
TEST(CurvedDeterminism, HookStyleParallelRefinement) {
    CompareRefinedRuns(SkewedBox(3, 3, 2, 0.0, 3), 3, 1000, "hook order 3, 3 refinements");
}

// The same far from the origin, where the round-off between copies of a coordinate grows with
// |x| rather than with the element size: the unit-cube mesh shifted to x ~ 100 (|x| / h about
// 1e3 after two refinements) and to x ~ 1e4 (|x| / h about 1e5). The quantum grows with |x| (up to
// 1024 element sizes), so keys and seeds are still bitwise those of np 1.
TEST(CurvedDeterminism, OffsetDomainsParallelRefinement) {
    CompareRefinedRuns(SkewedBox(3, 3, 3, 100.0, 2), 2, 1000, "offset 100, order 2");
    CompareRefinedRuns(SkewedBox(3, 3, 3, 1.0e4, 2), 2, 1000, "offset 1e4, order 2");
}

// Genuinely curved meshes refined in parallel: the warped order-2 box (hexes refined twice, tets
// once). Curved element boxes (sampled lattice plus bulge margin) are computed from Nodes whose
// shared copies differ in the last bits after the refinement; the element-weighted sampler rounds
// them to the key grid, so keys and seeds must again be bitwise those of np 1.
TEST(CurvedDeterminism, WarpedParallelRefinement) {
    CompareRefinedRuns(WarpedBox(2, mfem::Element::HEXAHEDRON), 2, 1000, "warped hex order 2");
    CompareRefinedRuns(WarpedBox(2, mfem::Element::TETRAHEDRON), 1, 1000, "warped tet order 2");
}

// Periodic pairing on a curved periodic box: the unit cube of order-2 hexes moved by
// PeriodicWarp keeps its boundary faces on the box planes (their nodal points move only within
// the plane) and the curved face meshes match after a shift, so all 6 n^2 boundary faces are
// plane faces and form 3 n^2 pairs.
TEST(PeriodicPairing, CurvedPeriodicBox) {
    const int n = 3;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n, n, n, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
    serial.SetCurvature(2);
    serial.Transform(PeriodicWarp);
    ASSERT_FALSE(ms::StraightSidedLocal(serial));
    Distributed d = Distribute(serial, MPI_COMM_WORLD);
    const ms::PeriodicBox box = ms::PeriodicBoxFromMesh(
        *d.pmesh, MPI_COMM_WORLD, {true, true, true});
    ms::PeriodicFaceParameters pairing;
    pairing.periodic = box.periodic;
    pairing.lo = box.lo;
    pairing.hi = box.hi;
    const ms::MeshAdjacency adjacency(*d.pmesh, pairing);
    EXPECT_EQ(adjacency.PeriodicReport().plane_faces, 6 * n * n);
    EXPECT_EQ(adjacency.PeriodicReport().matched_pairs, 3 * n * n);
}

// Periodic pairing on a mesh whose Nodes were moved after distribution (ParMesh::Transform moves
// only the Nodes, mesh.cpp:14072-14078), so the vertex array still describes the unit cube while
// the geometry is [0.5, 2.5]^3. The box comes from the Nodes; the faces must be found from the
// Nodes too (with the vertex array no face lies on the planes, which used to skip the pairing
// silently).
TEST(PeriodicPairing, StaleVertexArray) {
    const int n = 3;
    for (const auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(n, n, n, type, 1.0, 1.0, 1.0, false);
        serial.SetCurvature(2);
        Distributed d = Distribute(serial, MPI_COMM_WORLD);
        d.pmesh->Transform(ScaleShift);
        // The vertex array is stale (still the unit cube) on every rank that has vertices
        if (d.pmesh->GetNV() > 0) {
            const ms::VertexCoordinates coords(*d.pmesh);
            bool stale = false;
            for (int v = 0; v < d.pmesh->GetNV(); v++) {
                stale = stale || coords.Vertex(v)[0] != d.pmesh->GetVertex(v)[0];
            }
            EXPECT_TRUE(stale);
        } else {
            const ms::VertexCoordinates coords(*d.pmesh); // collective: every rank builds it
        }
        const ms::PeriodicBox box = ms::PeriodicBoxFromMesh(
            *d.pmesh, MPI_COMM_WORLD, {true, true, false});
        EXPECT_DOUBLE_EQ(box.lo[0], 0.5);
        EXPECT_DOUBLE_EQ(box.hi[0], 2.5);
        ms::PeriodicFaceParameters pairing;
        pairing.periodic = box.periodic;
        pairing.lo = box.lo;
        pairing.hi = box.hi;
        const ms::MeshAdjacency adjacency(*d.pmesh, pairing);
        const int faces_per_plane = (type == mfem::Element::HEXAHEDRON) ? n * n : 2 * n * n;
        EXPECT_EQ(adjacency.PeriodicReport().plane_faces, 4 * faces_per_plane);
        EXPECT_EQ(adjacency.PeriodicReport().matched_pairs, 2 * faces_per_plane);
    }
}

// Failure cases, thrown on every rank: (1) a periodic box larger than the mesh puts no face on
// the planes of an axis (zero plane faces is an error, no longer a silent skip); (2) a face whose
// corners lie on the plane x = 1 but whose middle bulges off it is not a plane face, so its
// partner on x = 0 is unmatched.
TEST(PeriodicPairing, ZeroPlaneFacesAndBulgedFaceThrow) {
    const int n = 2;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        n, n, n, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
    serial.SetCurvature(2);
    {
        Distributed d = Distribute(serial, MPI_COMM_WORLD);
        ms::PeriodicFaceParameters pairing;
        pairing.periodic = {false, false, true};
        pairing.lo = {0.0, 0.0, -1.0};
        pairing.hi = {1.0, 1.0, 2.0};
        EXPECT_THROW(ms::MeshAdjacency(*d.pmesh, pairing), std::runtime_error);
    }
    {
        // Bulge the x = 1 face outward: x += 0.05 sin(2 pi y) sin(2 pi z) x^4 moves the nodes
        // inside the face's edges and its middle (y or z = 0.25, 0.75) but not its corners
        // (y, z in {0, 0.5, 1}, where the sines vanish up to round-off)
        mfem::Mesh bulged(serial);
        bulged.Transform(BulgeRightFace);
        Distributed d = Distribute(bulged, MPI_COMM_WORLD);
        ms::PeriodicFaceParameters pairing;
        pairing.periodic = {true, false, false};
        pairing.lo = {0.0, 0.0, 0.0};
        pairing.hi = {1.0, 1.0, 1.0};
        EXPECT_THROW(ms::MeshAdjacency(*d.pmesh, pairing), std::runtime_error);
    }
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
