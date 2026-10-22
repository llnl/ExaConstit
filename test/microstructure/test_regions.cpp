/**
 * @file test_regions.cpp
 * @brief Tests of phases -> regions and synthetic phase fractions (regions/region_assigner) and
 * of attribute domains (mesh/attribute_domain: grains only in the active part of a mesh, the
 * passthrough parts kept as one region each).
 *
 * The region and phase-fraction tests are host-only and need no MPI; every rank runs them.
 *
 * The attribute-domain tests are collective: AttributeDomain, seeding, labeling, adjacency and
 * cleanup all communicate, so every rank calls them. Rank-count independence is checked inside
 * each run: the parent labels computed over MPI_COMM_WORLD are gathered into serial element
 * order and compared with the labels every rank computes alone on the whole serial mesh
 * (MPI_COMM_SELF), which is exactly the np = 1 answer. Running the executable at np = 1..4
 * (test_regions.cmake) therefore compares np = 2, 3, 4 against np = 1.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever.
 */
#include "microstructure/labeling/fragment_cleanup.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/attribute_domain.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_options.hpp"
#include "microstructure/regions/region_assigner.hpp"
#include "microstructure/seeding/seed_generators.hpp"
#include "microstructure/util/counter_rng.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

//------------------------------------------------------------------------------------------------
// Phases -> regions
//------------------------------------------------------------------------------------------------

/*
 * Phases 0, 1, 2 already compact: region = phase + 1 (ascending phase order), one region per
 * phase, all crystalline by default.
 */
TEST(Regions, CompactPhases) {
    const std::vector<int> phase = {2, 0, 1, 1, 0, 2, 2};
    const ms::RegionAssignment r = ms::AssignRegionsFromPhases(phase, {});
    EXPECT_EQ(r.num_regions, 3);
    EXPECT_EQ(r.grain_region, (std::vector<int>{3, 1, 2, 2, 1, 3, 3}));
    EXPECT_EQ(r.region_phases, (std::vector<std::vector<int>>{{0}, {1}, {2}}));
    EXPECT_EQ(r.region_crystalline, (std::vector<char>{1, 1, 1}));
}

/*
 * Sparse phases {0, 3, 7}: the regions are still 1..3 in ascending phase order (0 -> 1, 3 -> 2,
 * 7 -> 3), so they line up with three [[Materials]] blocks.
 */
TEST(Regions, SparsePhases) {
    const std::vector<int> phase = {7, 3, 0, 7, 3};
    const ms::RegionAssignment r = ms::AssignRegionsFromPhases(phase, {});
    EXPECT_EQ(r.num_regions, 3);
    EXPECT_EQ(r.grain_region, (std::vector<int>{3, 2, 1, 3, 2}));
    EXPECT_EQ(r.region_phases, (std::vector<std::vector<int>>{{0}, {3}, {7}}));
}

/*
 * Explicit map 0 -> 2, 3 -> 3, 7 -> 1 is used as given; a key for a phase no grain has (9) is
 * ignored.
 */
TEST(Regions, ExplicitMap) {
    ms::PhaseRegionParameters params;
    params.phase_to_region = {{0, 2}, {3, 3}, {7, 1}, {9, 17}};
    const std::vector<int> phase = {7, 3, 0, 7};
    const ms::RegionAssignment r = ms::AssignRegionsFromPhases(phase, params);
    EXPECT_EQ(r.num_regions, 3);
    EXPECT_EQ(r.grain_region, (std::vector<int>{1, 3, 2, 1}));
    EXPECT_EQ(r.region_phases, (std::vector<std::vector<int>>{{7}, {0}, {3}}));
}

/*
 * Many-to-one explicit map: phases 0 and 3 share region 1, phase 7 is region 2. The region ids
 * used are {1, 2} = 1..2, so there are 2 regions, region 1 holding phases {0, 3} (ascending).
 */
TEST(Regions, ManyToOneMap) {
    ms::PhaseRegionParameters params;
    params.phase_to_region = {{0, 1}, {3, 1}, {7, 2}};
    const ms::RegionAssignment r = ms::AssignRegionsFromPhases({0, 3, 7}, params);
    EXPECT_EQ(r.num_regions, 2);
    EXPECT_EQ(r.grain_region, (std::vector<int>{1, 1, 2}));
    EXPECT_EQ(r.region_phases, (std::vector<std::vector<int>>{{0, 3}, {7}}));
    EXPECT_EQ(r.region_crystalline, (std::vector<char>{1, 1}));
}

/*
 * Invalid explicit maps: a present phase missing (3); region ids {1, 3} leave region 2 unused
 * (not 1..M); a region id 4 with only 3 phases; a region id < 1 or a negative key, even for a
 * phase no grain has; negative phase ids; no grains.
 */
TEST(Regions, InvalidMapsThrow) {
    const std::vector<int> phase = {0, 3, 7};
    ms::PhaseRegionParameters missing;
    missing.phase_to_region = {{0, 1}, {7, 2}};
    EXPECT_THROW(ms::AssignRegionsFromPhases(phase, missing), std::invalid_argument);

    ms::PhaseRegionParameters gap;
    gap.phase_to_region = {{0, 1}, {3, 1}, {7, 3}};
    EXPECT_THROW(ms::AssignRegionsFromPhases(phase, gap), std::invalid_argument);

    ms::PhaseRegionParameters out_of_range;
    out_of_range.phase_to_region = {{0, 1}, {3, 2}, {7, 4}};
    EXPECT_THROW(ms::AssignRegionsFromPhases(phase, out_of_range), std::invalid_argument);

    ms::PhaseRegionParameters bad_absent_entry;
    bad_absent_entry.phase_to_region = {{0, 1}, {3, 2}, {7, 3}, {9, 0}};
    EXPECT_THROW(ms::AssignRegionsFromPhases(phase, bad_absent_entry), std::invalid_argument);

    ms::PhaseRegionParameters negative_key;
    negative_key.phase_to_region = {{-2, 1}, {0, 1}, {3, 2}, {7, 3}};
    EXPECT_THROW(ms::AssignRegionsFromPhases(phase, negative_key), std::invalid_argument);

    EXPECT_THROW(ms::AssignRegionsFromPhases({0, -1}, {}), std::invalid_argument);
    EXPECT_THROW(ms::AssignRegionsFromPhases({}, {}), std::invalid_argument);
}

/*
 * Non-crystalline phases: phase 3 is listed, so its region (2) is flagged 0; phase 5 is listed
 * but absent and changes nothing.
 */
TEST(Regions, NonCrystallineFlags) {
    ms::PhaseRegionParameters params;
    params.non_crystalline_phases = {5, 3};
    const ms::RegionAssignment r = ms::AssignRegionsFromPhases({0, 3, 7, 3}, params);
    EXPECT_EQ(r.region_crystalline, (std::vector<char>{1, 0, 1}));
}

/*
 * One region is one material model: mapping crystalline phase 0 and non-crystalline phase 3 to
 * the same region throws; mapping two non-crystalline phases together is fine.
 */
TEST(Regions, MixedCrystallinityThrows) {
    ms::PhaseRegionParameters params;
    params.non_crystalline_phases = {3};
    params.phase_to_region = {{0, 1}, {3, 1}, {7, 2}};
    EXPECT_THROW(ms::AssignRegionsFromPhases({0, 3, 7}, params), std::invalid_argument);

    params.non_crystalline_phases = {3, 7};
    params.phase_to_region = {{0, 1}, {3, 2}, {7, 2}};
    const ms::RegionAssignment r = ms::AssignRegionsFromPhases({0, 3, 7}, params);
    EXPECT_EQ(r.region_crystalline, (std::vector<char>{1, 0}));
}

//------------------------------------------------------------------------------------------------
// Synthetic phase fractions
//------------------------------------------------------------------------------------------------

/** N grains with log-uniform volumes in [0.1, 10) and non-contiguous source ids 3 g + 11. */
void MakeGrains(int n, std::vector<double>& volume, std::vector<std::int64_t>& source) {
    volume.resize(static_cast<std::size_t>(n));
    source.resize(static_cast<std::size_t>(n));
    for (int g = 0; g < n; g++) {
        const ms::CounterRng rng(99, ms::RngPurpose::GENERIC, static_cast<std::uint64_t>(g));
        volume[static_cast<std::size_t>(g)] = 0.1 * std::pow(100.0, rng.UniformDouble(0));
        source[static_cast<std::size_t>(g)] = 3 * g + 11;
    }
}

/*
 * The greedy-by-deficit assignment satisfies |achieved_k - f_k| <= v_max / V (proof in
 * region_assigner.hpp), the achieved fractions match the volumes actually assigned, and every
 * grain gets one of the requested phases.
 */
TEST(PhaseFractions, WithinBound) {
    std::vector<double> volume;
    std::vector<std::int64_t> source;
    for (const int n : {5, 37, 400}) {
        MakeGrains(n, volume, source);
        ms::PhaseFractionParameters params;
        params.phase_id = {4, 1, 9};
        params.fraction = {0.5, 0.3, 0.2};
        params.rng_seed = 2024;
        const ms::PhaseFractionResult res = ms::AssignPhaseFractions(volume, source, params);

        double total = 0.0;
        double v_max = 0.0;
        for (const double v : volume) {
            total += v;
            v_max = std::max(v_max, v);
        }
        std::map<int, double> phase_volume;
        for (std::size_t g = 0; g < volume.size(); g++) {
            phase_volume[res.grain_phase[g]] += volume[g];
        }
        double sum = 0.0;
        for (std::size_t k = 0; k < 3; k++) {
            const double achieved = res.achieved_fraction[k];
            sum += achieved;
            EXPECT_LE(std::fabs(achieved - params.fraction[k]), v_max / total + 1.0e-12)
                << "n = " << n << ", phase " << params.phase_id[k];
            EXPECT_NEAR(achieved, phase_volume[params.phase_id[k]] / total, 1.0e-12);
        }
        EXPECT_NEAR(sum, 1.0, 1.0e-12);
        EXPECT_EQ(phase_volume.size() <= 3, true);
    }
}

/*
 * Determinism and input-order independence: the same call twice gives the same phases, and
 * listing the grains in a different order gives every grain (identified by its source id) the
 * same phase, because the visit order depends only on (rng_seed, source id). A different rng_seed
 * changes the assignment.
 */
TEST(PhaseFractions, DeterministicAndOrderIndependent) {
    std::vector<double> volume;
    std::vector<std::int64_t> source;
    MakeGrains(300, volume, source);
    ms::PhaseFractionParameters params;
    params.phase_id = {0, 1};
    params.fraction = {0.7, 0.3};
    params.rng_seed = 7;
    const ms::PhaseFractionResult a = ms::AssignPhaseFractions(volume, source, params);
    const ms::PhaseFractionResult b = ms::AssignPhaseFractions(volume, source, params);
    EXPECT_EQ(a.grain_phase, b.grain_phase);

    // Permute the grains: g -> (g * 7 + 3) mod 300 (7 and 300 are coprime, so it is a bijection)
    const std::size_t n = volume.size();
    std::vector<double> pvolume(n);
    std::vector<std::int64_t> psource(n);
    std::vector<std::size_t> where(n);
    for (std::size_t g = 0; g < n; g++) {
        const std::size_t p = (g * 7 + 3) % n;
        pvolume[p] = volume[g];
        psource[p] = source[g];
        where[g] = p;
    }
    const ms::PhaseFractionResult c = ms::AssignPhaseFractions(pvolume, psource, params);
    for (std::size_t g = 0; g < n; g++) {
        EXPECT_EQ(a.grain_phase[g], c.grain_phase[where[g]]);
    }
    for (std::size_t k = 0; k < 2; k++) {
        EXPECT_EQ(a.achieved_fraction[k], c.achieved_fraction[k]);
    }

    params.rng_seed = 8;
    const ms::PhaseFractionResult d = ms::AssignPhaseFractions(volume, source, params);
    EXPECT_NE(a.grain_phase, d.grain_phase);
}

/* Invalid inputs: fractions not summing to 1, a zero fraction, repeated source ids or phases. */
TEST(PhaseFractions, InvalidInputsThrow) {
    const std::vector<double> volume = {1.0, 2.0, 3.0};
    const std::vector<std::int64_t> source = {0, 1, 2};
    ms::PhaseFractionParameters params;
    params.phase_id = {0, 1};
    params.fraction = {0.5, 0.4};
    EXPECT_THROW(ms::AssignPhaseFractions(volume, source, params), std::invalid_argument);
    params.fraction = {0.5, 0.5000005}; // within 1e-6 of 1: accepted (and normalized)
    EXPECT_NO_THROW(ms::AssignPhaseFractions(volume, source, params));
    params.fraction = {1.0, 0.0};
    EXPECT_THROW(ms::AssignPhaseFractions(volume, source, params), std::invalid_argument);
    params.fraction = {0.5, 0.5};
    EXPECT_THROW(ms::AssignPhaseFractions(volume, {0, 1, 1}, params), std::invalid_argument);
    params.phase_id = {1, 1};
    EXPECT_THROW(ms::AssignPhaseFractions(volume, source, params), std::invalid_argument);
}

//------------------------------------------------------------------------------------------------
// Attribute domains: a 6 x 6 x 8 hex box whose two bottom layers (z < 2) are a build plate
//------------------------------------------------------------------------------------------------

constexpr int PLATE_ATTRIBUTE = 2;
constexpr int PART_ATTRIBUTE = 1;
constexpr int GRIP_ATTRIBUTE = 3;

/**
 * The box [0,6] x [0,6] x [0,8] with unit hexes; elements with centroid z < 2 get attribute 2
 * (the plate, 72 elements), the others attribute 1 (the part). With `grip`, the top layer
 * (centroid z > 7, 36 elements) gets attribute 3 instead, leaving 180 part elements.
 */
mfem::Mesh PlateBox(bool grip = false) {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(6, 6, 8, mfem::Element::HEXAHEDRON, 6.0, 6.0,
                                                  8.0);
    mfem::Vector center;
    for (int e = 0; e < mesh.GetNE(); e++) {
        mesh.GetElementCenter(e, center);
        int attribute = PART_ATTRIBUTE;
        if (center(2) < 2.0) {
            attribute = PLATE_ATTRIBUTE;
        } else if (grip && center(2) > 7.0) {
            attribute = GRIP_ATTRIBUTE;
        }
        mesh.SetAttribute(e, attribute);
    }
    mesh.SetAttributes();
    return mesh;
}

/** The distributed mesh with the serial-order bookkeeping needed to gather results. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/**
 * Distributes `serial` over comm. METIS partition (plate_on_last_rank = false), or: every plate
 * element on the last rank and the part elements dealt in contiguous blocks to the other ranks,
 * so the last rank owns no active element at all (with >= 2 ranks). Collective.
 */
Distributed Distribute(mfem::Mesh& serial, MPI_Comm comm, bool plate_on_last_rank) {
    Distributed d;
    if (!plate_on_last_rank) {
        d.order = ms::BuildSerialOrder(serial, comm);
    } else {
        int rank = 0;
        int nranks = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &nranks);
        const int owners = (nranks > 1) ? nranks - 1 : 1;
        const int ne = serial.GetNE();
        d.order.partitioning.SetSize(ne);
        int part_index = 0;
        const int num_part = 6 * 6 * 6;
        for (int e = 0; e < ne; e++) {
            if (serial.GetAttribute(e) == PLATE_ATTRIBUTE) {
                d.order.partitioning[e] = nranks - 1;
            } else {
                d.order.partitioning[e] = (part_index * owners) / num_part;
                part_index++;
            }
        }
        std::vector<int> local;
        for (int e = 0; e < ne; e++) {
            if (d.order.partitioning[e] == rank) {
                local.push_back(e);
            }
        }
        d.order.local_to_serial.SetSize(static_cast<int>(local.size()));
        for (std::size_t i = 0; i < local.size(); i++) {
            d.order.local_to_serial[static_cast<int>(i)] = local[i];
        }
    }
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/** Labels of the whole plate box in serial element order, identical on every rank. */
struct PlateResult {
    std::vector<int> parent_grain;
    int num_grains = 0;
    bool all_seeds_above_plate = true;
};

/**
 * The full pipeline on the active part: AttributeDomain (active {1}, passthrough {2}, or
 * {3, 2} for a mesh with a grip, listed unsorted on purpose), GenerateSeeds and
 * LabelTessellation on ActiveMesh(), MeshAdjacency and CleanupFragments on the sub-mesh,
 * ToParent, gathered into serial order on every rank. Collective over comm.
 */
PlateResult RunPlate(mfem::Mesh& serial, MPI_Comm comm, bool plate_on_last_rank,
                     bool grip = false) {
    Distributed d = Distribute(serial, comm, plate_on_last_rank);
    const std::vector<int> passthrough = grip
                                             ? std::vector<int>{GRIP_ATTRIBUTE, PLATE_ATTRIBUTE}
                                             : std::vector<int>{PLATE_ATTRIBUTE};
    ms::AttributeDomain domain(*d.pmesh, {PART_ATTRIBUTE}, passthrough);
    EXPECT_TRUE(domain.UsesSubMesh());
    // Passthrough attributes sorted ascending: plate (2) first, then grip (3)
    EXPECT_EQ(domain.PassthroughElementCounts(),
              grip ? (std::vector<std::int64_t>{72, 36}) : (std::vector<std::int64_t>{72}));
    EXPECT_EQ(domain.GlobalActiveElements(), grip ? 180 : 216);
    mfem::ParMesh& active = domain.ActiveMesh();

    // In the skewed partition the last rank (with >= 2 ranks) owns plate elements only, so its
    // part of the sub-mesh is empty: the rest of the pipeline must still work there
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    if (plate_on_last_rank && nranks > 1 && rank == nranks - 1) {
        EXPECT_EQ(active.GetNE(), 0);
    }

    // The sub-mesh keeps the parent attributes: every active element still has attribute 1
    for (int e = 0; e < active.GetNE(); e++) {
        EXPECT_EQ(active.GetAttribute(e), PART_ATTRIBUTE);
    }

    // Seeds inside the active material only: z >= 2 (the top of the plate), and z <= 7 (the
    // bottom of the grip) when there is a grip
    ms::SeedingParameters seeding;
    seeding.num_seeds = 20;
    seeding.rng_seed = 31415;
    const ms::SeedSet seeds = ms::GenerateSeeds(active, seeding);
    PlateResult out;
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        const double z_max = grip ? 7.0 : 8.0;
        out.all_seeds_above_plate =
            out.all_seeds_above_plate && (seeds.z[i] >= 2.0) && (seeds.z[i] <= z_max);
    }

    // Labeling, then adjacency and fragment cleanup on the sub-mesh
    const ms::TessellationOptions tessellation;
    const ms::TessellationLabels labels =
        ms::LabelTessellation(active, active.GetComm(), seeds, tessellation);
    const ms::MeshAdjacency adjacency(active);
    ms::FragmentCleanupParameters cleanup;
    cleanup.min_grain_elements = 2;
    const ms::FragmentCleanupResult cleaned =
        ms::CleanupFragments(adjacency, labels.element_grain, labels.num_grains, {}, cleanup);

    // Back to the parent and into serial order
    const mfem::Array<int> parent = domain.ToParent(cleaned.element_grain, cleaned.num_grains);
    out.parent_grain = ms::GatherToSerial(comm, d.order, parent, 0);
    out.parent_grain.resize(static_cast<std::size_t>(serial.GetNE()));
    MPI_Bcast(out.parent_grain.data(), serial.GetNE(), MPI_INT, 0, comm);
    out.num_grains = cleaned.num_grains;
    return out;
}

/*
 * Build plate: every seed lies above the plate, plate elements get grain id N + 1, part elements
 * get 1..N, and the gathered parent labels are identical for the METIS partition, the partition
 * whose last rank owns only plate elements (zero active elements on that rank), and the np = 1
 * reference.
 */
TEST(AttributeDomain, BuildPlate) {
    mfem::Mesh serial = PlateBox();
    const PlateResult world = RunPlate(serial, MPI_COMM_WORLD, false);
    const PlateResult skewed = RunPlate(serial, MPI_COMM_WORLD, true);
    const PlateResult self = RunPlate(serial, MPI_COMM_SELF, false);

    EXPECT_TRUE(world.all_seeds_above_plate);
    EXPECT_GE(world.num_grains, 2);
    for (int e = 0; e < serial.GetNE(); e++) {
        const int g = world.parent_grain[static_cast<std::size_t>(e)];
        if (serial.GetAttribute(e) == PLATE_ATTRIBUTE) {
            EXPECT_EQ(g, world.num_grains + 1);
        } else {
            EXPECT_GE(g, 1);
            EXPECT_LE(g, world.num_grains);
        }
    }
    for (const PlateResult* other : {&skewed, &self}) {
        EXPECT_EQ(world.num_grains, other->num_grains);
        EXPECT_EQ(world.parent_grain, other->parent_grain);
    }
}

/*
 * Two passthrough attributes, listed as {3, 2}: they are sorted, so plate elements (attribute 2)
 * get grain id N + 1 and grip elements (attribute 3) get N + 2; the part (180 elements) gets
 * 1..N, every seed lies in 2 <= z <= 7, and the labels agree with the np = 1 reference and the
 * skewed partition.
 */
TEST(AttributeDomain, PlateAndGrip) {
    mfem::Mesh serial = PlateBox(true);
    const PlateResult world = RunPlate(serial, MPI_COMM_WORLD, false, true);
    const PlateResult skewed = RunPlate(serial, MPI_COMM_WORLD, true, true);
    const PlateResult self = RunPlate(serial, MPI_COMM_SELF, false, true);

    EXPECT_TRUE(world.all_seeds_above_plate);
    for (int e = 0; e < serial.GetNE(); e++) {
        const int g = world.parent_grain[static_cast<std::size_t>(e)];
        if (serial.GetAttribute(e) == PLATE_ATTRIBUTE) {
            EXPECT_EQ(g, world.num_grains + 1);
        } else if (serial.GetAttribute(e) == GRIP_ATTRIBUTE) {
            EXPECT_EQ(g, world.num_grains + 2);
        } else {
            EXPECT_GE(g, 1);
            EXPECT_LE(g, world.num_grains);
        }
    }
    for (const PlateResult* other : {&skewed, &self}) {
        EXPECT_EQ(world.num_grains, other->num_grains);
        EXPECT_EQ(world.parent_grain, other->parent_grain);
    }
}

/*
 * Shared faces and face-neighbor data on the sub-mesh: the active 6 x 6 x 6 block has
 * 3 * 6 * 6 * 5 = 540 interior faces of area 1. MeshAdjacency lists every interior face twice
 * over all ranks (once from each side, whether local or shared), so the global entry count is
 * 1080 and the global face-area sum 1080, for every partition (including a rank without active
 * elements).
 */
TEST(AttributeDomain, SubMeshAdjacency) {
    mfem::Mesh serial = PlateBox();
    for (const bool skewed : {false, true}) {
        Distributed d = Distribute(serial, MPI_COMM_WORLD, skewed);
        ms::AttributeDomain domain(*d.pmesh, {PART_ATTRIBUTE}, {PLATE_ATTRIBUTE});
        const ms::MeshAdjacency adjacency(domain.ActiveMesh());
        long long entries = adjacency.Neighbors().Size();
        double area = 0.0;
        const double* a = adjacency.FaceAreas().HostRead();
        for (int k = 0; k < adjacency.FaceAreas().Size(); k++) {
            area += a[k];
        }
        long long global_entries = 0;
        double global_area = 0.0;
        MPI_Allreduce(&entries, &global_entries, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&area, &global_area, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(global_entries, 1080);
        EXPECT_NEAR(global_area, 1080.0, 1.0e-9);
    }
}

/*
 * Curved meshes: after SetCurvature(2) on the parent and a smooth bending of its nodes
 * (x += 0.05 sin(z), so the geometry is no longer given by the vertices), the sub-mesh has
 * second-order Nodes too, and every sub-mesh element maps reference points to exactly the same
 * physical points as its parent element (ActiveToParentElement): the curved geometry was
 * transferred, not rebuilt from the vertices.
 */
TEST(AttributeDomain, KeepsCurvedNodes) {
    mfem::Mesh serial = PlateBox();
    Distributed d = Distribute(serial, MPI_COMM_WORLD, false);
    d.pmesh->SetCurvature(2);
    mfem::GridFunction* parent_nodes = d.pmesh->GetNodes();
    {
        // Bend the parent: node (x, y, z) -> (x + 0.05 sin(z), y, z). Nodes are stored by the
        // FE space's DofToVDof(i, component) map.
        const mfem::FiniteElementSpace* fes = parent_nodes->FESpace();
        for (int i = 0; i < fes->GetNDofs(); i++) {
            const double z = (*parent_nodes)(fes->DofToVDof(i, 2));
            (*parent_nodes)(fes->DofToVDof(i, 0)) += 0.05 * std::sin(z);
        }
    }
    ms::AttributeDomain domain(*d.pmesh, {PART_ATTRIBUTE}, {PLATE_ATTRIBUTE});
    mfem::ParMesh& active = domain.ActiveMesh();
    const mfem::GridFunction* nodes = active.GetNodes();
    EXPECT_NE(nodes, nullptr);
    if (nodes != nullptr) {
        EXPECT_EQ(nodes->FESpace()->GetMaxElementOrder(), 2);
        // Compare physical points of a few reference points in every element pair
        const double ref_points[3][3] = {{0.5, 0.5, 0.5}, {0.1, 0.7, 0.3}, {0.9, 0.2, 0.95}};
        const mfem::Array<int>& to_parent = domain.ActiveToParentElement();
        double max_diff = 0.0;
        for (int i = 0; i < active.GetNE(); i++) {
            mfem::Vector x_sub(3);
            mfem::Vector x_par(3);
            for (const auto& p : ref_points) {
                mfem::IntegrationPoint ip;
                ip.Set3(p[0], p[1], p[2]);
                active.GetElementTransformation(i)->Transform(ip, x_sub);
                d.pmesh->GetElementTransformation(to_parent[i])->Transform(ip, x_par);
                for (int k = 0; k < 3; k++) {
                    max_diff = std::max(max_diff, std::fabs(x_sub(k) - x_par(k)));
                }
                EXPECT_GE(x_sub(2), 2.0 - 1.0e-12);
            }
        }
        EXPECT_LE(max_diff, 1.0e-13);
    }
}

/*
 * Validation (collective errors, thrown on every rank): attribute 2 classified as neither,
 * attribute 1 classified as both, a passthrough attribute without elements, and an active set
 * without elements.
 */
TEST(AttributeDomain, InvalidClassificationThrows) {
    mfem::Mesh serial = PlateBox();
    Distributed d = Distribute(serial, MPI_COMM_WORLD, false);
    EXPECT_THROW(ms::AttributeDomain(*d.pmesh, {1}, {}), std::invalid_argument);
    EXPECT_THROW(ms::AttributeDomain(*d.pmesh, {1}, {1, 2}), std::invalid_argument);
    EXPECT_THROW(ms::AttributeDomain(*d.pmesh, {1}, {2, 5}), std::invalid_argument);
    EXPECT_THROW(ms::AttributeDomain(*d.pmesh, {3}, {1, 2}), std::invalid_argument);
    EXPECT_THROW(ms::AttributeDomain(*d.pmesh, {0, 1}, {2}), std::invalid_argument);
}

/*
 * All active (empty lists, or both attributes listed active): no sub-mesh, ActiveMesh is the
 * parent, and ToParent returns the labels unchanged. An empty active list with passthrough {2}
 * means "every other attribute", i.e. {1}.
 */
TEST(AttributeDomain, AllActiveUsesParent) {
    mfem::Mesh serial = PlateBox();
    Distributed d = Distribute(serial, MPI_COMM_WORLD, false);
    for (const std::vector<int>& active : {std::vector<int>{}, std::vector<int>{1, 2}}) {
        ms::AttributeDomain domain(*d.pmesh, active, {});
        EXPECT_FALSE(domain.UsesSubMesh());
        EXPECT_EQ(&domain.ActiveMesh(), d.pmesh.get());
        mfem::Array<int> grain(d.pmesh->GetNE());
        for (int e = 0; e < grain.Size(); e++) {
            grain[e] = 1 + (e % 3);
        }
        const mfem::Array<int> parent = domain.ToParent(grain, 3);
        EXPECT_EQ(parent.Size(), grain.Size());
        for (int e = 0; e < grain.Size(); e++) {
            EXPECT_EQ(parent[e], grain[e]);
        }
    }
    ms::AttributeDomain implicit(*d.pmesh, {}, {PLATE_ATTRIBUTE});
    EXPECT_TRUE(implicit.UsesSubMesh());
    EXPECT_EQ(implicit.ActiveAttributes(), (std::vector<int>{PART_ATTRIBUTE}));
}

/* ToParent rejects a grain id outside [1, num_grains] on every rank. */
TEST(AttributeDomain, ToParentRejectsBadGrains) {
    mfem::Mesh serial = PlateBox();
    Distributed d = Distribute(serial, MPI_COMM_WORLD, false);
    ms::AttributeDomain domain(*d.pmesh, {PART_ATTRIBUTE}, {PLATE_ATTRIBUTE});
    mfem::Array<int> grain(domain.ActiveMesh().GetNE());
    grain = 1;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0 && grain.Size() > 0) {
        grain[0] = 5;
    }
    int any = grain.Size() > 0 && rank == 0 ? 1 : 0;
    int global_any = 0;
    MPI_Allreduce(&any, &global_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (global_any) {
        EXPECT_THROW(domain.ToParent(grain, 4), std::invalid_argument);
    }
}

} // namespace

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
