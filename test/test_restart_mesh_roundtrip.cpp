#include "sim_state/exa_sidre_parmesh.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "axom/sidre.hpp"
#include <gtest/gtest.h>

// Mesh round-trip test for the restart checkpoint mesh I/O (WriteParMeshBlueprint and
// ExaSidreParMesh, src/sim_state/exa_sidre_parmesh.*).
//
// Each case partitions a small perturbed serial mesh, writes every rank's piece with
// WriteParMeshBlueprint(), saves it to HDF5, loads it back into a fresh DataStore, rebuilds the
// parallel mesh with ExaSidreParMesh, and then checks that the rebuilt mesh is identical to the
// original in every way the finite element spaces depend on:
// - local topology counts and the global element count;
// - every communication group: its master and, entity by entity in group order, its shared
//   vertices, edges and faces, including each shared edge's and face's orientation;
// - for H1 spaces of order 1 to 4: identical local, true and global true sizes, bitwise
//   identical local and true-DOF vectors of a projected nonlinear field, identical
//   element-local DOF hashes, and a field that survives restriction to true DOFs and
//   prolongation back unchanged (shared DOFs matched consistently across ranks).
//
// Order 4 matters: a shared face then carries several interior DOFs, so a wrong shared-face
// orientation changes which DOFs are matched across ranks (see the background notes in
// exa_sidre_parmesh.cpp). The `empty_ranks` case partitions the mesh so that half of the ranks
// own no elements, covering zero-length arrays in the checkpoint and empty rebuilt meshes.
//
// Every case also runs a negative control that rewrites the shared edges and faces in each
// rank's local vertex order before saving (what axom::sidre::MFEMSidreDataCollection writes)
// and reports how far the shared-DOF round trip is then off. It is printed for information
// rather than checked, since whether a given partition happens to have mismatched
// orientations is not guaranteed; on 2 or more ranks it is typically 0.1 or larger for hexes at
// order >= 3, against ~1e-15 for the real writer.
//
// Run with any number of ranks, for example `mpirun -np 3 ./test_restart_mesh_roundtrip`.
// Scratch files go to restart_roundtrip_np<ranks>_<case>/ in the working directory.

using axom::sidre::DataStore;
using axom::sidre::Group;

namespace {

/**
 * @brief 64-bit FNV-1a hash, the same as the one used by restart_checkpoint.cpp
 */
class Fnv1a64 {
public:
    /**
     * @brief Add raw bytes to the hash
     *
     * @param data Bytes to add
     * @param byte_count Number of bytes
     */
    void AddBytes(const void* data, const std::size_t byte_count) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < byte_count; ++i) {
            m_state ^= bytes[i];
            m_state *= 1099511628211ULL;
        }
    }

    /**
     * @brief Current hash value
     *
     * @return The hash of every byte added so far
     */
    std::uint64_t Value() const {
        return m_state;
    }

private:
    /** @brief Running hash state, starting from the FNV-1a 64-bit offset basis */
    std::uint64_t m_state = 1469598103934665603ULL;
};

/**
 * @brief Hash of a grid function visited element by element in local DOF order
 *
 * @param field Grid function to hash
 * @return FNV-1a hash of the values in GetElementVDofs() order (sign-flipped for reversed DOFs)
 *
 * @details Same definition as ElementLocalFieldHash() in restart_checkpoint.cpp. Equal hashes
 * on the original and rebuilt meshes mean both number their local DOFs the same way.
 */
std::uint64_t ElementLocalFieldHash(const mfem::ParGridFunction& field) {
    const mfem::ParFiniteElementSpace* space = field.ParFESpace();
    const double* values = field.HostRead();
    mfem::Array<int> element_vdofs;
    Fnv1a64 hasher;
    for (int element_index = 0; element_index < space->GetNE(); ++element_index) {
        space->GetElementVDofs(element_index, element_vdofs);
        for (const int signed_vdof : element_vdofs) {
            const int vdof = signed_vdof >= 0 ? signed_vdof : -1 - signed_vdof;
            const double value = signed_vdof >= 0 ? values[vdof] : -values[vdof];
            hasher.AddBytes(&value, sizeof(value));
        }
    }
    return hasher.Value();
}

/**
 * @brief Largest change of a grid function under restriction to true DOFs and prolongation
 *
 * @param field Grid function to test
 * @return max |field - P(R(field))| over this rank's local DOFs
 *
 * @details Collective. Zero (up to rounding) exactly when every shared DOF is matched with
 * the same true DOF on every rank; see SharedDofInconsistency() in restart_checkpoint.cpp.
 */
double SharedDofInconsistency(const mfem::ParGridFunction& field) {
    mfem::Vector true_values;
    field.GetTrueDofs(true_values);
    mfem::ParGridFunction round_trip(field.ParFESpace());
    round_trip.SetFromTrueDofs(true_values);
    double max_difference = 0.0;
    for (int i = 0; i < field.Size(); ++i) {
        max_difference = std::max(max_difference, std::abs(field(i) - round_trip(i)));
    }
    return max_difference;
}

/**
 * @brief Smooth nonlinear vector field projected onto the H1 spaces
 *
 * @param x Point
 * @param y Field value at x
 *
 * @details Nonlinear in every coordinate, so every DOF of a high-order space gets a distinct
 * value and any DOF permutation changes the projected vector.
 */
void TestField(const mfem::Vector& x, mfem::Vector& y) {
    y(0) = std::sin(2.0 * x(0) + x(1)) + 0.3 * x(2) * x(2);
    y(1) = std::cos(x(0) - 3.0 * x(2)) + x(1) * x(1) * x(0);
    y(2) = x(0) * x(1) * x(2) + std::sin(1.3 * x(2) + 0.7 * x(1));
}

/**
 * @brief Move every vertex by a small smooth offset
 *
 * @param mesh Serial mesh to perturb
 * @param amplitude Size of the offset
 *
 * @details Makes the coordinates distinct and non-axis-aligned, so a mismatched vertex or DOF
 * always changes a value instead of possibly landing on an equal one.
 */
void PerturbVertices(mfem::Mesh& mesh, const double amplitude) {
    for (int vertex = 0; vertex < mesh.GetNV(); ++vertex) {
        double* x = mesh.GetVertex(vertex);
        const double x0 = x[0];
        const double x1 = x[1];
        const double x2 = x[2];
        x[0] += amplitude * std::sin(3.1 * x1 + 1.7 * x2 + 0.3);
        x[1] += amplitude * std::sin(2.3 * x2 + 0.9 * x0 + 1.1);
        x[2] += amplitude * std::sin(1.9 * x0 + 2.9 * x1 + 0.7);
    }
}

/**
 * @brief Largest value of a double over all ranks
 *
 * @param value This rank's value
 * @return The maximum over MPI_COMM_WORLD
 */
double GlobalMax(const double value) {
    double result = 0.0;
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
}

/**
 * @brief Sorted MPI ranks of a communication group
 *
 * @param mesh Parallel mesh
 * @param group Local group index
 * @return The member ranks in increasing order
 *
 * @details Groups are matched between the original and rebuilt meshes by their member ranks,
 * so the comparison does not assume anything about local group indices.
 */
std::vector<int> GroupRanks(const mfem::ParMesh& mesh, const int group) {
    std::vector<int> ranks;
    for (int member = 0; member < mesh.gtopo.GetGroupSize(group); ++member) {
        ranks.push_back(mesh.gtopo.GetNeighborRank(mesh.gtopo.GetGroup(group)[member]));
    }
    std::sort(ranks.begin(), ranks.end());
    return ranks;
}

/**
 * @brief Check that two meshes have identical communication groups and shared entities
 *
 * @param original Mesh that was written
 * @param rebuilt Mesh rebuilt from the checkpoint
 *
 * @details For every group of the original mesh, finds the rebuilt group with the same member
 * ranks and checks its master and, entity by entity in group order, its shared vertices, the
 * local vertices and orientation of its shared edges, and the local vertices and orientation
 * of its shared triangles and quadrilaterals. Both the order and the orientation matter (see
 * the background notes in exa_sidre_parmesh.cpp).
 */
void ExpectSameSharedStructures(const mfem::ParMesh& original, const mfem::ParMesh& rebuilt) {
    EXPECT_EQ(original.GetNGroups(), rebuilt.GetNGroups()) << "number of communication groups";
    std::map<std::vector<int>, int> rebuilt_group_index;
    for (int group = 1; group < rebuilt.GetNGroups(); ++group) {
        rebuilt_group_index[GroupRanks(rebuilt, group)] = group;
    }

    mfem::Array<int> original_vertices;
    mfem::Array<int> rebuilt_vertices;
    for (int ga = 1; ga < original.GetNGroups(); ++ga) {
        const auto it = rebuilt_group_index.find(GroupRanks(original, ga));
        if (it == rebuilt_group_index.end()) {
            ADD_FAILURE() << "original group " << ga << " has no rebuilt group with the same ranks";
            continue;
        }
        const int gb = it->second;
        EXPECT_EQ(original.gtopo.GetGroupMasterRank(ga), rebuilt.gtopo.GetGroupMasterRank(gb))
            << "group master of group " << ga;

        // Shared vertices, in group order.
        EXPECT_EQ(original.GroupNVertices(ga), rebuilt.GroupNVertices(gb));
        for (int i = 0; i < std::min(original.GroupNVertices(ga), rebuilt.GroupNVertices(gb));
             ++i) {
            EXPECT_EQ(original.GroupVertex(ga, i), rebuilt.GroupVertex(gb, i))
                << "shared vertex " << i << " of group " << ga;
        }

        // Shared edges: same local edge (by its vertices) and same orientation.
        EXPECT_EQ(original.GroupNEdges(ga), rebuilt.GroupNEdges(gb));
        for (int i = 0; i < std::min(original.GroupNEdges(ga), rebuilt.GroupNEdges(gb)); ++i) {
            int edge_a = -1;
            int orientation_a = 0;
            int edge_b = -1;
            int orientation_b = 0;
            original.GroupEdge(ga, i, edge_a, orientation_a);
            rebuilt.GroupEdge(gb, i, edge_b, orientation_b);
            original.GetEdgeVertices(edge_a, original_vertices);
            rebuilt.GetEdgeVertices(edge_b, rebuilt_vertices);
            EXPECT_TRUE(original_vertices[0] == rebuilt_vertices[0] &&
                        original_vertices[1] == rebuilt_vertices[1] &&
                        orientation_a == orientation_b)
                << "shared edge " << i << " of group " << ga;
        }

        // Shared faces: same local face (by its vertices) and same orientation.
        const auto expect_same_faces =
            [&](const int count_a, const int count_b, const bool triangles, const char* label) {
                EXPECT_EQ(count_a, count_b) << "number of shared " << label;
                for (int i = 0; i < std::min(count_a, count_b); ++i) {
                    int face_a = -1;
                    int orientation_a = 0;
                    int face_b = -1;
                    int orientation_b = 0;
                    if (triangles) {
                        original.GroupTriangle(ga, i, face_a, orientation_a);
                        rebuilt.GroupTriangle(gb, i, face_b, orientation_b);
                    } else {
                        original.GroupQuadrilateral(ga, i, face_a, orientation_a);
                        rebuilt.GroupQuadrilateral(gb, i, face_b, orientation_b);
                    }
                    original.GetFaceVertices(face_a, original_vertices);
                    rebuilt.GetFaceVertices(face_b, rebuilt_vertices);
                    bool same = original_vertices.Size() == rebuilt_vertices.Size() &&
                                orientation_a == orientation_b;
                    for (int k = 0; same && k < original_vertices.Size(); ++k) {
                        same = original_vertices[k] == rebuilt_vertices[k];
                    }
                    EXPECT_TRUE(same) << "shared " << label << " " << i << " of group " << ga;
                }
            };
        expect_same_faces(
            original.GroupNTriangles(ga), rebuilt.GroupNTriangles(gb), true, "triangle");
        expect_same_faces(original.GroupNQuadrilaterals(ga),
                          rebuilt.GroupNQuadrilaterals(gb),
                          false,
                          "quadrilateral");
    }
}

/**
 * @brief Overwrite the stored shared edges and faces with each rank's local vertex order
 *
 * @param mesh Mesh that was written
 * @param mesh_group Mesh group written by WriteParMeshBlueprint()
 *
 * @details Reproduces what axom::sidre::MFEMSidreDataCollection writes (local order, no
 * orientation), for the negative control. The arrays are edited in place, so everything else
 * the writer stored, including each group's local_group_index, is unchanged.
 */
void RewriteSharedEntitiesInLocalOrder(const mfem::ParMesh& mesh, Group* mesh_group) {
    Group* groups = mesh_group->getGroup("adjsets/mesh/groups");
    mfem::Array<int> vertices;
    for (int group = 1; group < mesh.GetNGroups(); ++group) {
        const std::string name = "g" + std::to_string(mesh.gtopo.GetGroupMasterRank(group)) + "_" +
                                 std::to_string(mesh.gtopo.GetGroupMasterGroup(group));
        Group* shared_group = groups->getGroup(name);
        if (mesh.GroupNEdges(group) > 0) {
            int* edges = shared_group->getView("edges")->getData<int*>();
            for (int i = 0; i < mesh.GroupNEdges(group); ++i) {
                int edge = -1;
                int orientation = 0;
                mesh.GroupEdge(group, i, edge, orientation);
                mesh.GetEdgeVertices(edge, vertices);
                edges[2 * i] = vertices[0];
                edges[2 * i + 1] = vertices[1];
            }
        }
        if (mesh.GroupNTriangles(group) > 0) {
            int* triangles = shared_group->getView("triangles")->getData<int*>();
            for (int i = 0; i < mesh.GroupNTriangles(group); ++i) {
                int face = -1;
                int orientation = 0;
                mesh.GroupTriangle(group, i, face, orientation);
                mesh.GetFaceVertices(face, vertices);
                for (int k = 0; k < 3; ++k) {
                    triangles[3 * i + k] = vertices[k];
                }
            }
        }
        if (mesh.GroupNQuadrilaterals(group) > 0) {
            int* quads = shared_group->getView("quadrilaterals")->getData<int*>();
            for (int i = 0; i < mesh.GroupNQuadrilaterals(group); ++i) {
                int face = -1;
                int orientation = 0;
                mesh.GroupQuadrilateral(group, i, face, orientation);
                mesh.GetFaceVertices(face, vertices);
                for (int k = 0; k < 4; ++k) {
                    quads[4 * i + k] = vertices[k];
                }
            }
        }
    }
}

/**
 * @brief Write a mesh through HDF5 and rebuild it, exactly as a checkpoint and restart do
 *
 * @param mesh Mesh to write
 * @param directory Directory for the rank files (created by rank 0)
 * @param local_order_control Whether to rewrite the shared entities in local order first
 * @return The rebuilt mesh
 *
 * @details The DataStore used for loading goes out of scope before the rebuilt mesh is used,
 * which also checks that the rebuilt mesh does not point into Sidre memory.
 */
std::shared_ptr<exaconstit::ExaSidreParMesh> RoundTrip(const mfem::ParMesh& mesh,
                                                       const std::filesystem::path& directory,
                                                       const bool local_order_control) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::filesystem::create_directories(directory);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const std::string file = (directory / ("rank_" + std::to_string(rank) + ".hdf5")).string();
    {
        DataStore store;
        Group* mesh_group = store.getRoot()->createGroup("mesh");
        exaconstit::WriteParMeshBlueprint(mesh, mesh_group);
        if (local_order_control) {
            RewriteSharedEntitiesInLocalOrder(mesh, mesh_group);
        }
        MFEM_VERIFY(store.getRoot()->save(file, "sidre_hdf5"), "Unable to save " << file);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    DataStore store;
    MFEM_VERIFY(store.getRoot()->load(file, "sidre_hdf5"), "Unable to load " << file);
    return exaconstit::ExaSidreParMesh::Create(store.getRoot()->getGroup("mesh"), MPI_COMM_WORLD);
}

/**
 * @brief Run one round-trip case
 *
 * @param label Case name, used for the scratch directory and the summary line
 * @param type Element type of the serial mesh
 * @param n Elements per cube edge of the serial mesh
 * @param ref_par Number of uniform parallel refinements after partitioning
 * @param leave_ranks_empty Whether to partition onto only half of the ranks (at least one), so
 *                          the remaining ranks own no elements
 *
 * @details Collective. See the comment at the top of this file for what is checked.
 */
void RunRoundTripCase(const std::string& label,
                      const mfem::Element::Type type,
                      const int n,
                      const int ref_par,
                      const bool leave_ranks_empty) {
    int rank = 0;
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(n, n, n, type, 1.0, 1.0, 1.0, false);
    PerturbVertices(serial, 0.04 / n);

    // With leave_ranks_empty, elements are dealt out round-robin over the first half of the
    // ranks only, so the other half own nothing. MFEM accepts a partitioning that leaves ranks
    // empty; this is the situation the recent zero-element fixes in the solver guard against.
    std::vector<int> partitioning;
    if (leave_ranks_empty) {
        const int active_ranks = std::max(1, num_ranks / 2);
        partitioning.resize(static_cast<std::size_t>(serial.GetNE()));
        for (int element = 0; element < serial.GetNE(); ++element) {
            partitioning[static_cast<std::size_t>(element)] = element % active_ranks;
        }
    }
    mfem::ParMesh original(
        MPI_COMM_WORLD, serial, leave_ranks_empty ? partitioning.data() : nullptr);
    for (int refinement = 0; refinement < ref_par; ++refinement) {
        original.UniformRefinement();
    }

    // The rank count is part of the directory name so tests with different rank counts can
    // run at the same time.
    const std::filesystem::path directory = std::filesystem::current_path() /
                                            ("restart_roundtrip_np" + std::to_string(num_ranks) +
                                             "_" + label);
    const auto rebuilt = RoundTrip(original, directory, false);

    // Topology and parallel structure.
    EXPECT_TRUE(
        exaconstit::CountMeshTopology(original).Matches(exaconstit::CountMeshTopology(*rebuilt)))
        << "local topology counts";
    EXPECT_EQ(original.GetGlobalNE(), rebuilt->GetGlobalNE()) << "global element count";
    EXPECT_EQ(original.bdr_attributes.Size(), rebuilt->bdr_attributes.Size())
        << "boundary attribute list";
    ExpectSameSharedStructures(original, *rebuilt);

    // H1 spaces of order 1 to 4 on both meshes must be indistinguishable.
    mfem::VectorFunctionCoefficient coefficient(3, TestField);
    double worst_inconsistency = 0.0;
    for (int order = 1; order <= 4; ++order) {
        const std::string context = label + ", order " + std::to_string(order);
        mfem::H1_FECollection fec(order, 3);
        mfem::ParFiniteElementSpace space_a(&original, &fec, 3);
        mfem::ParFiniteElementSpace space_b(rebuilt.get(), &fec, 3);
        EXPECT_EQ(space_a.GetVSize(), space_b.GetVSize()) << context;
        EXPECT_EQ(space_a.TrueVSize(), space_b.TrueVSize()) << context;
        EXPECT_EQ(space_a.GlobalTrueVSize(), space_b.GlobalTrueVSize()) << context;

        mfem::ParGridFunction field_a(&space_a);
        mfem::ParGridFunction field_b(&space_b);
        field_a.ProjectCoefficient(coefficient);
        field_b.ProjectCoefficient(coefficient);
        double local_difference = 0.0;
        for (int i = 0; i < std::min(field_a.Size(), field_b.Size()); ++i) {
            local_difference = std::max(local_difference, std::abs(field_a(i) - field_b(i)));
        }
        EXPECT_EQ(local_difference, 0.0) << "local DOF vectors differ, " << context;
        EXPECT_EQ(ElementLocalFieldHash(field_a), ElementLocalFieldHash(field_b))
            << "element-local DOF hash differs, " << context;

        mfem::Vector true_a;
        mfem::Vector true_b;
        field_a.GetTrueDofs(true_a);
        field_b.GetTrueDofs(true_b);
        double true_difference = 0.0;
        for (int i = 0; i < std::min(true_a.Size(), true_b.Size()); ++i) {
            true_difference = std::max(true_difference, std::abs(true_a(i) - true_b(i)));
        }
        EXPECT_EQ(true_difference, 0.0) << "true DOF vectors differ, " << context;

        const double inconsistency = GlobalMax(SharedDofInconsistency(field_b));
        worst_inconsistency = std::max(worst_inconsistency, inconsistency);
        EXPECT_LT(inconsistency, 1e-12) << "shared DOFs matched inconsistently, " << context;
    }

    // Negative control (reported, not checked; see the comment at the top of this file).
    const auto control = RoundTrip(original, directory.string() + "_control", true);
    std::array<double, 5> control_inconsistency{};
    for (int order = 1; order <= 4; ++order) {
        mfem::H1_FECollection fec(order, 3);
        mfem::ParFiniteElementSpace space_c(control.get(), &fec, 3);
        mfem::ParGridFunction field_c(&space_c);
        field_c.ProjectCoefficient(coefficient);
        control_inconsistency[static_cast<std::size_t>(order)] = GlobalMax(
            SharedDofInconsistency(field_c));
    }

    // GetGlobalNE() is collective, so it is called on every rank before printing on rank 0.
    const long long global_elements = original.GetGlobalNE();
    int local_empty = original.GetNE() == 0 ? 1 : 0;
    int empty_ranks = 0;
    MPI_Allreduce(&local_empty, &empty_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << label << ": " << num_ranks << " ranks (" << empty_ranks << " empty), "
                  << global_elements << " elements, max |x - P R x| = " << worst_inconsistency
                  << "; negative control at orders 1-4: " << control_inconsistency[1] << ", "
                  << control_inconsistency[2] << ", " << control_inconsistency[3] << ", "
                  << control_inconsistency[4] << std::endl;
    }
}

} // namespace

TEST(restart_mesh_roundtrip, hex) {
    RunRoundTripCase("hex", mfem::Element::HEXAHEDRON, 4, 0, false);
}

TEST(restart_mesh_roundtrip, hex_parallel_refinement) {
    RunRoundTripCase("hex_refpar", mfem::Element::HEXAHEDRON, 3, 1, false);
}

TEST(restart_mesh_roundtrip, tet) {
    RunRoundTripCase("tet", mfem::Element::TETRAHEDRON, 3, 0, false);
}

TEST(restart_mesh_roundtrip, tet_parallel_refinement) {
    RunRoundTripCase("tet_refpar", mfem::Element::TETRAHEDRON, 2, 1, false);
}

TEST(restart_mesh_roundtrip, empty_ranks) {
    RunRoundTripCase("empty_ranks", mfem::Element::HEXAHEDRON, 3, 0, true);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    mfem::Hypre::Init();
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
