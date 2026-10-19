#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// End-to-end restart test at the SimulationState level (no SystemDriver), for the checkpoint
// and restart code in src/sim_state/restart_checkpoint.* and the restart branches of the
// SimulationState constructor.
//
// Run A builds a real SimulationState from input files: a perturbed order-3 hex mesh whose
// element attributes are grain IDs, two UMAT regions assigned through a region-mapping file,
// and orientations for the first region. It is then given a synthetic evolved state (deformed
// coordinates, a velocity, distinct values in every quadrature function, three completed time
// steps) and writes a checkpoint. The mesh, region-mapping and orientation files are then
// deleted, and run B restarts from the checkpoint alone. The test checks that:
// - phase 1 reproduces the original reference coordinates bit for bit inside the constructor
//   (the vertex-geometry projection a rebuilt mesh would otherwise use differs in the last
//   bits; the summary line reports by how much);
// - after phase 2 the mesh attributes, grain IDs, region flags, every nodal field, every
//   checkpointed quadrature function and the full time state are bitwise identical, and the
//   time-step line the driver logs reads the same as in run A;
// - the tangent stiffness is not restored (it keeps its constructor value until the first
//   model setup recomputes it);
// - the restarted run can write its next checkpoint.
// One begin/end state-variable pair is deliberately made unequal, so both the `copy_of` and the
// stored-values paths of the quadrature-function checkpoint are exercised.
//
// The first region also covers orientation placement: its grain table has no
// ori_state_var_loc, so the quaternions go at the end of each state vector.
//
// Run with 2 or 3 ranks, for example `mpirun -np 2 ./test_restart_e2e`. Scratch files go to
// restart_e2e_np<ranks>/ in the working directory.

namespace fs = std::filesystem;

namespace {

/**
 * @brief Deterministic unit quaternion for a grain
 *
 * @param grain Grain ID
 * @return Normalized quaternion (w, x, y, z), different for every grain
 */
std::array<double, 4> GrainQuaternion(const int grain) {
    std::array<double, 4> quaternion = {1.0, 0.1 * grain, 0.2 * grain, 0.05 * grain};
    const double norm = std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                                  quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
    for (double& component : quaternion) {
        component /= norm;
    }
    return quaternion;
}

/**
 * @brief Check two vectors for bitwise equality
 *
 * @param a First vector
 * @param b Second vector
 * @return True if both have the same size and every entry compares equal
 */
bool BitwiseEqual(const mfem::Vector& a, const mfem::Vector& b) {
    if (a.Size() != b.Size()) {
        return false;
    }
    if (a.Size() == 0) {
        return true;
    }
    const double* x = a.HostRead();
    const double* y = b.HostRead();
    for (int i = 0; i < a.Size(); ++i) {
        if (x[i] != y[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Capture the time-step line the driver logs at the start of a cycle
 *
 * @param sim_state Simulation state whose time manager is printed
 * @return The output of SimulationState::PrintTimeStats(), printed with 17 significant digits
 *         so that any difference in the time steps shows up in the text
 */
std::string CaptureTimeStats(const SimulationState& sim_state) {
    std::ostringstream captured;
    std::streambuf* original = std::cout.rdbuf(captured.rdbuf());
    const std::streamsize original_precision = std::cout.precision(17);
    sim_state.PrintTimeStats();
    std::cout.precision(original_precision);
    std::cout.rdbuf(original);
    return captured.str();
}

/**
 * @brief Options for the test runs
 *
 * @param directory Scratch directory holding the input files and the checkpoints
 * @param read Whether this run restarts from a checkpoint
 * @return Options with two UMAT regions, AUTO time stepping and checkpoint writing enabled
 *
 * @details The first region ("soft") has 3 state variables plus quaternion orientations with
 * no ori_state_var_loc, so the orientations go at the end of its state vector (7 values per
 * point). The second region ("hard") has 5 state variables and no orientations.
 */
ExaOptions MakeOptions(const fs::path& directory, const bool read) {
    ExaOptions options;
    options.basename = "e2e";
    options.mesh.mesh_type = MeshType::FILE;
    options.mesh.mesh_file = directory / "mesh.mesh";
    options.mesh.order = 3;
    // AUTO stepping with dt_max well above the steps taken, so the Newton iteration counts
    // FillEvolvedState() reports make every step a different size and the logged "dt old was"
    // value differs from the next dt.
    options.time.time_type = TimeStepType::AUTO;
    options.time.auto_time = TimeOptions::AutoTimeOptions{};
    options.time.auto_time->dt_start = 0.1;
    options.time.auto_time->dt_min = 1e-3;
    options.time.auto_time->dt_max = 10.0;
    options.time.auto_time->t_final = 100.0;
    options.solvers.rtmodel = RTModel::CPU;
    options.region_mapping_file = directory / "regions.txt";
    for (int region = 0; region < 2; ++region) {
        MaterialOptions material;
        material.material_name = (region == 0) ? "soft" : "hard";
        material.region_id = region + 1;
        material.mech_type = MechType::UMAT;
        material.properties.num_props = 2;
        material.properties.properties = {10.0 + region, 20.0 + region};
        material.state_vars.num_vars = 3 + 2 * region;
        for (int k = 0; k < material.state_vars.num_vars; ++k) {
            material.state_vars.initial_values.push_back(1.0 + k + 10 * region);
        }
        if (region == 0) {
            GrainInfo grains;
            grains.orientation_file = directory / "orientations.txt";
            grains.ori_type = OriType::QUAT;
            grains.ori_stride = 4;
            grains.num_grains = 8;
            grains.ori_state_var_loc = -1;
            material.grain_info = grains;
        }
        options.materials.push_back(material);
    }
    options.restart.write = true;
    options.restart.interval = 1;
    options.restart.keep_last = 2;
    options.restart.read = read;
    options.restart.directory = directory / "restart";
    return options;
}

/**
 * @brief A quadrature function registry entry: base name and region (-1 for global)
 */
struct QuadratureFunctionName {
    /** @brief Base name, as passed to SimulationState::GetQuadratureFunction() */
    std::string base;
    /** @brief Region index, or -1 for a global function */
    int region;
};

/**
 * @brief The quadrature functions the test fills and compares
 *
 * @return Global and per-region names; functions not created on a rank are skipped there
 */
std::vector<QuadratureFunctionName> QuadratureFunctionNames() {
    std::vector<QuadratureFunctionName> names = {{"cauchy_stress_beg", -1},
                                                 {"cauchy_stress_end", -1},
                                                 {"cauchy_stress_avg", -1},
                                                 {"kinetic_grads", -1},
                                                 {"tangent_stiffness", -1}};
    for (int region = 0; region < 2; ++region) {
        for (const char* base : {"state_var_beg",
                                 "state_var_end",
                                 "state_var_avg",
                                 "cauchy_stress_beg",
                                 "cauchy_stress_end",
                                 "cauchy_stress_avg",
                                 "tangent_stiffness",
                                 "def_grad_beg"}) {
            names.push_back({base, region});
        }
    }
    return names;
}

/**
 * @brief Give a freshly constructed SimulationState a synthetic evolved state
 *
 * @param sim_state Simulation state to modify
 *
 * @details Mirrors what the driver leaves behind after three converged steps, in the same
 * order (solve, FinishCycle(), model update, post-processing, checkpoint):
 * - a velocity with distinct values per true DOF, and current coordinates displaced by it
 *   (built from true DOFs, so the field is continuous across ranks);
 * - distinct deterministic values in every quadrature function, different per name and rank;
 * - three completed AUTO time steps with different Newton iteration counts, so each step has
 *   a different size;
 * - SetupModelVariables(), which makes every end-of-step function equal to its
 *   beginning-of-step partner, as SystemDriver::UpdateModel() does.
 * Finally one state-variable pair is made unequal on purpose, so its end-of-step values are
 * written in full rather than as `copy_of`.
 */
void FillEvolvedState(SimulationState& sim_state) {
    auto space = sim_state.GetMeshParFiniteElementSpace();
    mfem::Vector& primal = *sim_state.GetPrimalField();
    const HYPRE_BigInt offset = space->GetMyTDofOffset();
    for (int i = 0; i < primal.Size(); ++i) {
        primal[i] = std::sin(0.37 * static_cast<double>(offset + i));
    }
    mfem::Vector reference_true;
    sim_state.GetRefCoords()->GetTrueDofs(reference_true);
    mfem::Vector current_true = reference_true;
    current_true.Add(0.01, primal);
    sim_state.GetCurrentCoords()->SetFromTrueDofs(current_true);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int salt = 1;
    for (const auto& name : QuadratureFunctionNames()) {
        auto function = sim_state.GetQuadratureFunction(name.base, name.region);
        ++salt;
        if (!function || function->Size() == 0) {
            continue;
        }
        double* data = function->HostReadWrite();
        for (int i = 0; i < function->Size(); ++i) {
            data[i] = std::cos(0.013 * i + 0.7 * salt + 0.11 * rank);
        }
    }

    for (const int newton_iterations : {3, 7, 12}) {
        sim_state.UpdateDeltaTime(newton_iterations, true);
    }
    sim_state.FinishCycle();
    sim_state.SetupModelVariables();

    auto end_state = sim_state.GetQuadratureFunction("state_var_end", 0);
    if (end_state && end_state->Size() > 0) {
        end_state->HostReadWrite()[0] += 1.0;
    }
}

/**
 * @brief Write the input files of run A (rank 0 only)
 *
 * @param directory Scratch directory
 *
 * @details A perturbed 4x4x4 hex mesh whose element attributes are grain IDs 1..8, an
 * orientation file with one quaternion per grain, and a region-mapping file sending grains
 * 1-4 to region 1 and grains 5-8 to region 2. The mesh is written with 17 significant digits so
 * the vertex positions survive the text round trip exactly.
 */
void WriteInputFiles(const fs::path& directory) {
    fs::remove_all(directory);
    fs::create_directories(directory);

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        4, 4, 4, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
    for (int vertex = 0; vertex < mesh.GetNV(); ++vertex) {
        double* x = mesh.GetVertex(vertex);
        const double x0 = x[0];
        const double x1 = x[1];
        const double x2 = x[2];
        x[0] += 0.01 * std::sin(3.1 * x1 + 1.7 * x2 + 0.3);
        x[1] += 0.01 * std::sin(2.3 * x2 + 0.9 * x0 + 1.1);
        x[2] += 0.01 * std::sin(1.9 * x0 + 2.9 * x1 + 0.7);
    }
    for (int element = 0; element < mesh.GetNE(); ++element) {
        mesh.SetAttribute(element, 1 + (element * 5) % 8);
    }
    mesh.SetAttributes();
    std::ofstream mesh_file(directory / "mesh.mesh");
    mesh_file.precision(17);
    mesh.Print(mesh_file);

    std::ofstream orientations(directory / "orientations.txt");
    orientations.precision(17);
    for (int grain = 1; grain <= 8; ++grain) {
        const auto quaternion = GrainQuaternion(grain);
        orientations << quaternion[0] << " " << quaternion[1] << " " << quaternion[2] << " "
                     << quaternion[3] << "\n";
    }

    std::ofstream regions(directory / "regions.txt");
    for (int grain = 1; grain <= 8; ++grain) {
        regions << grain << " " << (grain <= 4 ? 1 : 2) << "\n";
    }
}

/**
 * @brief Count the quadrature points of region 0 whose state does not match the inputs
 *
 * @param sim_state Freshly constructed simulation state of run A
 * @return Number of points whose state vector is not {1, 2, 3, quaternion of the grain}
 */
int CountMisplacedOrientations(SimulationState& sim_state) {
    auto state = sim_state.GetQuadratureFunction("state_var_beg", 0);
    if (!state || state->GetVDim() != 7) {
        return state ? 1 : 0;
    }
    auto space = state->GetPartialSpaceShared();
    const auto& local_to_global = space->GetLocal2Global();
    const int num_elements = space->GetNumLocalElements();
    if (num_elements == 0) {
        return 0;
    }
    const int points_per_element = state->Size() / 7 / num_elements;
    const double* data = state->HostRead();
    const double* grains = sim_state.GetGrains()->HostRead();
    int wrong = 0;
    for (int element = 0; element < num_elements; ++element) {
        const auto quaternion = GrainQuaternion(static_cast<int>(grains[local_to_global[element]]));
        for (int point = 0; point < points_per_element; ++point) {
            const double* s = data +
                              static_cast<std::size_t>(element * points_per_element + point) * 7;
            const bool ok = s[0] == 1.0 && s[1] == 2.0 && s[2] == 3.0 && s[3] == quaternion[0] &&
                            s[4] == quaternion[1] && s[5] == quaternion[2] && s[6] == quaternion[3];
            wrong += ok ? 0 : 1;
        }
    }
    return wrong;
}

/**
 * @brief Compare the original reference coordinates with a vertex-geometry projection
 *
 * @param directory Scratch directory holding the mesh file
 * @param reference Reference coordinates of run A
 * @return Number of differing entries on this rank and the largest difference
 *
 * @details A mesh rebuilt from a checkpoint has no Nodes, so GetNodes() projects from the
 * vertex positions. This measures how far that is from the original reference coordinates,
 * which is the gap the phase-1 coordinate overlay closes.
 */
std::pair<int, double> VertexProjectionDifference(const fs::path& directory,
                                                  const mfem::Vector& reference) {
    mfem::Mesh serial((directory / "mesh.mesh").string().c_str(), 1, 1, true);
    mfem::ParMesh vertex_mesh(MPI_COMM_WORLD, serial);
    mfem::H1_FECollection fec(3, 3);
    mfem::ParFiniteElementSpace space(&vertex_mesh, &fec, 3);
    mfem::ParGridFunction projected(&space);
    vertex_mesh.GetNodes(projected);
    int mismatches = 0;
    double max_difference = 0.0;
    if (projected.Size() == reference.Size()) {
        for (int i = 0; i < projected.Size(); ++i) {
            const double difference = std::abs(projected(i) - reference(i));
            mismatches += difference != 0.0 ? 1 : 0;
            max_difference = std::max(max_difference, difference);
        }
    }
    return {mismatches, max_difference};
}

} // namespace

TEST(restart_simulation_state, checkpoint_and_restart) {
    int rank = 0;
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    const fs::path directory = fs::current_path() / ("restart_e2e_np" + std::to_string(num_ranks));
    if (rank == 0) {
        WriteInputFiles(directory);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ---------------------------------------------------------------------
    // Run A: build from the input files, evolve, checkpoint
    // ---------------------------------------------------------------------
    ExaOptions options_a = MakeOptions(directory, false);
    SimulationState run_a(options_a);
    EXPECT_EQ(CountMisplacedOrientations(run_a), 0)
        << "region 0 state vectors should be {1, 2, 3} followed by the grain's quaternion";
    const auto [projection_mismatches, projection_max_difference] = VertexProjectionDifference(
        directory, *run_a.GetRefCoords());

    FillEvolvedState(run_a);
    run_a.WriteCheckpointIfDue(3);
    EXPECT_TRUE(fs::exists(directory / "restart" / "cycle_0000000003" / "commit.json"))
        << "run A should have committed a checkpoint for cycle 3";

    // The restart must not need the original input files.
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        fs::remove(directory / "mesh.mesh");
        fs::remove(directory / "regions.txt");
        fs::remove(directory / "orientations.txt");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ---------------------------------------------------------------------
    // Run B: phase 1 (constructor) must reproduce the original construction-time geometry
    // ---------------------------------------------------------------------
    ExaOptions options_b = MakeOptions(directory, true);
    SimulationState run_b(options_b);
    EXPECT_TRUE(BitwiseEqual(*run_a.GetRefCoords(), *run_b.GetRefCoords())) << "phase 1: mesh_ref";
    EXPECT_TRUE(BitwiseEqual(*run_a.GetRefCoords(), *run_b.GetCurrentCoords()))
        << "phase 1: mesh_current should equal the reference coordinates";
    EXPECT_TRUE(BitwiseEqual(*run_a.GetRefCoords(), *run_b.GetTimeStartCoords()))
        << "phase 1: mesh_t_beg should equal the reference coordinates";

    // ---------------------------------------------------------------------
    // Run B: phase 2 must restore everything bitwise
    // ---------------------------------------------------------------------
    run_b.RestoreFromCheckpoint();

    EXPECT_EQ(run_a.GetMesh()->GetNE(), run_b.GetMesh()->GetNE()) << "local element count";
    for (int element = 0; element < std::min(run_a.GetMesh()->GetNE(), run_b.GetMesh()->GetNE());
         ++element) {
        if (run_a.GetMesh()->GetAttribute(element) != run_b.GetMesh()->GetAttribute(element)) {
            ADD_FAILURE() << "region attribute of element " << element;
            break;
        }
    }
    EXPECT_TRUE(BitwiseEqual(*run_a.GetGrains(), *run_b.GetGrains())) << "grain IDs";
    for (int region = 0; region < 2; ++region) {
        EXPECT_EQ(run_a.IsRegionActive(region), run_b.IsRegionActive(region))
            << "region " << region << " active flag";
    }

    EXPECT_TRUE(BitwiseEqual(*run_a.GetRefCoords(), *run_b.GetRefCoords())) << "mesh_ref";
    EXPECT_TRUE(BitwiseEqual(*run_a.GetCurrentCoords(), *run_b.GetCurrentCoords()))
        << "mesh_current";
    EXPECT_TRUE(BitwiseEqual(*run_a.GetTimeStartCoords(), *run_b.GetTimeStartCoords()))
        << "mesh_t_beg";
    EXPECT_TRUE(BitwiseEqual(*run_a.GetPrimalField(), *run_b.GetPrimalField())) << "velocity";
    EXPECT_TRUE(BitwiseEqual(*run_a.GetPrimalFieldPrev(), *run_b.GetPrimalFieldPrev()))
        << "previous velocity";
    EXPECT_TRUE(BitwiseEqual(*run_a.GetVelocity(), *run_b.GetVelocity()))
        << "velocity grid function";
    EXPECT_TRUE(BitwiseEqual(*run_a.GetDisplacement(), *run_b.GetDisplacement())) << "displacement";

    int compared = 0;
    for (const auto& name : QuadratureFunctionNames()) {
        auto function_a = run_a.GetQuadratureFunction(name.base, name.region);
        auto function_b = run_b.GetQuadratureFunction(name.base, name.region);
        const std::string label = name.base + " (region " + std::to_string(name.region) + ")";
        if (!function_a || !function_b) {
            EXPECT_TRUE(!function_a && !function_b) << "present in only one run: " << label;
            continue;
        }
        if (name.base == "tangent_stiffness") {
            // Not checkpointed: it keeps the constructor's zeros until the first model setup.
            bool zero = true;
            if (function_b->Size() > 0) {
                const double* data = function_b->HostRead();
                for (int i = 0; i < function_b->Size(); ++i) {
                    zero = zero && data[i] == 0.0;
                }
            }
            EXPECT_TRUE(zero) << "the tangent should not be restored: " << label;
        } else {
            EXPECT_TRUE(BitwiseEqual(*function_a, *function_b)) << "restored values: " << label;
            ++compared;
        }
    }

    EXPECT_EQ(run_a.GetTime(), run_b.GetTime()) << "time";
    EXPECT_EQ(run_a.GetDeltaTime(), run_b.GetDeltaTime()) << "dt";
    EXPECT_EQ(run_a.GetSimulationCycle(), run_b.GetSimulationCycle()) << "cycle";
    EXPECT_EQ(run_a.GetTrueCycleTime(), run_b.GetTrueCycleTime()) << "completed time";
    EXPECT_EQ(run_a.IsLastStep(), run_b.IsLastStep()) << "time step tracker (last step)";
    EXPECT_EQ(run_a.IsFinished(), run_b.IsFinished()) << "time step tracker (finished)";
    EXPECT_EQ(run_b.GetRestartDriverStep(), 3u) << "driver step";
    // The line the driver logs at the start of the next cycle, including the logging-only
    // previous time step, must read exactly as in the uninterrupted run.
    EXPECT_EQ(CaptureTimeStats(run_a), CaptureTimeStats(run_b)) << "logged time-step change";

    // ---------------------------------------------------------------------
    // Both runs take one more step; run B writes its next checkpoint
    // ---------------------------------------------------------------------
    run_a.UpdateDeltaTime(5, true);
    run_a.FinishCycle();
    run_b.UpdateDeltaTime(5, true);
    run_b.FinishCycle();
    EXPECT_EQ(run_a.GetTime(), run_b.GetTime()) << "time after one more step";
    EXPECT_EQ(run_a.GetDeltaTime(), run_b.GetDeltaTime()) << "dt after one more step";
    EXPECT_EQ(CaptureTimeStats(run_a), CaptureTimeStats(run_b))
        << "logged time-step change after one more step";
    run_b.WriteCheckpointIfDue(4);
    EXPECT_TRUE(fs::exists(directory / "restart" / "cycle_0000000004" / "commit.json"))
        << "the restarted run should have committed a checkpoint for cycle 4";

    int global_mismatches = 0;
    double global_max_difference = 0.0;
    MPI_Allreduce(&projection_mismatches, &global_mismatches, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(
        &projection_max_difference, &global_max_difference, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Restart end-to-end: " << num_ranks << " ranks, " << compared
                  << " quadrature functions compared on rank 0; a vertex-geometry projection "
                     "would have differed from the original reference coordinates in "
                  << global_mismatches << " entries (max " << global_max_difference << ")"
                  << std::endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    mfem::Hypre::Init();
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
