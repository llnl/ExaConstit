/**
 * @file test_user_functions.cpp
 * @brief User functions for the tests of ExaConstit's user function interface
 *
 * This file is built into a small shared library (libtest_user_functions) that the tests list
 * in their input decks, exactly as a user would list their own library. It is NOT linked with
 * ExaConstit: like a real user library, it uses ExaConstit and MFEM through their headers only,
 * and the functions it calls (SimulationState::AddQuadratureFunctionStatePair, MFEM's memory
 * manager, ...) are found in the program that loads it (the mechanics executable, or the test
 * executable) when it is loaded.
 *
 * Contents, grouped by what they test:
 * - TestUmat: a UMAT that does nothing, so the UMAT region of the test decks has a material
 *   model to load (the tests never solve a step with it).
 * - State variable functions ([Materials.State_Vars] register_function / initialize_function),
 *   signature void f(std::shared_ptr<SimulationState> sim_state, int region):
 *   - correct functions, used by the success test (test_state_variable_functions.cpp);
 *   - deliberately wrong ones, used by the failure tests that run mechanics and check the
 *     error message (wrong argument types, extern "C", overlapping or too long slices).
 * - Post-processing user functions ([[PostProcessing.user_functions]]): a constant 1, the mean
 *   and von Mises stress, and the stress triaxiality, used by the post-processing test.
 *
 * The test material layouts these functions assume are in user_functions.toml.in:
 * - region 0: ExaCMech "evptn_FCC_A" (24 state variables + 4 quaternion components);
 * - region 1: UMAT with 10 state variables.
 *
 * The user functions are normal C++ functions at global scope (not extern "C", not in a
 * namespace), as ExaConstit requires, except the one that tests the extern "C" mistake. The UMAT
 * is extern "C", as UMATs are.
 */

#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <cmath>
#include <memory>

// =================================================================================================
// UMAT
// =================================================================================================

/**
 * @brief UMAT with the standard Abaqus arguments that does nothing
 *
 * The test decks load it with [Materials.Model.UMAT] library_path / function_name = "TestUmat".
 * The tests only build the material models and stop before the first step, so it is never
 * called; it exists so the UMAT region can be created.
 */
extern "C" void TestUmat(double* /* stress */,
                         double* /* statev */,
                         double* /* ddsdde */,
                         double* /* sse */,
                         double* /* spd */,
                         double* /* scd */,
                         double* /* rpl */,
                         double* /* ddsdt */,
                         double* /* drplde */,
                         double* /* drpldt */,
                         double* /* stran */,
                         double* /* dstran */,
                         double* /* time */,
                         double* /* deltaTime */,
                         double* /* tempk */,
                         double* /* dtemp */,
                         double* /* predef */,
                         double* /* dpred */,
                         char* /* cmname */,
                         int* /* ndi */,
                         int* /* nshr */,
                         int* /* ntens */,
                         int* /* nstatv */,
                         double* /* props */,
                         int* /* nprops */,
                         double* /* coords */,
                         double* /* drot */,
                         double* /* pnewdt */,
                         double* /* celent */,
                         double* /* dfgrd0 */,
                         double* /* dfgrd1 */,
                         int* /* noel */,
                         int* /* npt */,
                         int* /* layer */,
                         int* /* kspt */,
                         int* /* kstep */,
                         int* /* kinc */) {}

// =================================================================================================
// State variable functions: correct ones
// =================================================================================================

/**
 * @brief Names slices of the UMAT region's 10 state variables
 *
 * Layout (offset, length): damage (0, 1), back_stress (1, 6), accumulated_slip (7, 1),
 * num_iterations (8, 1) registered with plot = false (a bookkeeping slot). Slot 9 is left
 * unnamed. A second "damage" with a different slice is ignored (the first registration of a
 * name wins), which the success test checks.
 */
void RegisterUmatStateVariables(std::shared_ptr<SimulationState> sim_state, int region) {
    sim_state->AddQuadratureFunctionStatePair("damage", {0, 1}, region);
    sim_state->AddQuadratureFunctionStatePair("back_stress", {1, 6}, region);
    sim_state->AddQuadratureFunctionStatePair("accumulated_slip", {7, 1}, region);
    sim_state->AddQuadratureFunctionStatePair("num_iterations", {8, 1}, region, false);
    // Same name again: ignored, so "damage" stays at (0, 1)
    sim_state->AddQuadratureFunctionStatePair("damage", {9, 1}, region);
}

/**
 * @brief Adds one name to the ExaCMech region, after ExaCMech's own names
 *
 * ExaCMech names every slot of "evptn_FCC_A" except slot 3 (its count of function
 * evaluations), so that slot is free for a user name. A second "quats" pointing elsewhere is
 * ignored: the ExaCMech name was registered first and cannot be replaced.
 */
void RegisterExaCMechStateVariables(std::shared_ptr<SimulationState> sim_state, int region) {
    sim_state->AddQuadratureFunctionStatePair("num_func_evals", {3, 1}, region);
    sim_state->AddQuadratureFunctionStatePair("quats", {3, 1}, region);
}

/**
 * @brief Sets the UMAT region's initial state variables: slot k gets 0.5 + k at every point
 *
 * Writes the beginning-of-step values ("state_var_beg"); ExaConstit copies them to
 * "state_var_end" afterwards. Called only on ranks that own elements of the region. The
 * success test checks for exactly these values.
 */
void InitializeUmatStateVariables(std::shared_ptr<SimulationState> sim_state, int region) {
    auto state_var_beg = sim_state->GetQuadratureFunction("state_var_beg", region);
    const int num_vars = state_var_beg->GetVDim();
    const int num_points = state_var_beg->Size() / num_vars;
    // values(k, point): state variable k at quadrature point "point"
    auto values = mfem::Reshape(state_var_beg->Write(), num_vars, num_points);
    mfem::forall(num_points, [=] MFEM_HOST_DEVICE(int point) {
        for (int k = 0; k < num_vars; k++) {
            values(k, point) = 0.5 + k;
        }
    });
}

// =================================================================================================
// State variable functions: wrong ones (each must stop the run with a clear message)
// =================================================================================================

/**
 * @brief Wrong signature: the region is a double instead of an int
 *
 * Its symbol name differs from the required one, so ExaConstit reports it as not found and
 * prints the required signature.
 */
void RegisterWrongArguments(std::shared_ptr<SimulationState> /* sim_state */, double /* region */) {
}

/**
 * @brief Right arguments but declared extern "C", which drops the argument types from the name
 *
 * ExaConstit cannot check the arguments of such a function and refuses it, pointing out the
 * extern "C".
 */
extern "C" void RegisterExternC(std::shared_ptr<SimulationState> /* sim_state */,
                                int /* region */) {}

/**
 * @brief Two slices of the UMAT region that share slot 1 (must stop the run)
 */
void RegisterOverlappingStateVariables(std::shared_ptr<SimulationState> sim_state, int region) {
    sim_state->AddQuadratureFunctionStatePair("damage", {0, 2}, region);
    sim_state->AddQuadratureFunctionStatePair("porosity", {1, 1}, region);
}

/**
 * @brief A slice that runs past the UMAT region's 10 state variables (must stop the run)
 */
void RegisterOutOfRangeStateVariables(std::shared_ptr<SimulationState> sim_state, int region) {
    sim_state->AddQuadratureFunctionStatePair("damage", {9, 2}, region);
}

// =================================================================================================
// Post-processing user functions ([[PostProcessing.user_functions]])
// =================================================================================================
// Signature void f(std::shared_ptr<SimulationState> sim_state, int region,
//                  std::shared_ptr<mfem::expt::PartialQuadratureFunction> values).
// Used by the post-processing test (postprocessing.toml.in, check_postprocessing.py), which
// compares their outputs with quantities that have exact answers.

/**
 * @brief 1 at every point: its volume integral must equal the region volume
 */
void UnitValue(std::shared_ptr<SimulationState> /* sim_state */,
               int /* region */,
               std::shared_ptr<mfem::expt::PartialQuadratureFunction> values) {
    *values = 1.0;
}

/**
 * @brief Mean stress and von Mises stress at every point (2 components)
 *
 * The volume average of the mean stress must equal one third of the trace of the built-in
 * volume-averaged Cauchy stress, since both are linear in the stress.
 */
void StressInvariants(std::shared_ptr<SimulationState> sim_state,
                      int region,
                      std::shared_ptr<mfem::expt::PartialQuadratureFunction> values) {
    auto stress_qf = sim_state->GetQuadratureFunction("cauchy_stress_end", region);
    const int num_points = stress_qf->Size() / 6;
    const auto stress = mfem::Reshape(stress_qf->Read(), 6, num_points);
    auto invariants = mfem::Reshape(values->Write(), values->GetVDim(), num_points);
    mfem::forall(num_points, [=] MFEM_HOST_DEVICE(int ip) {
        const double mean = (stress(0, ip) + stress(1, ip) + stress(2, ip)) / 3.0;
        const double d11 = stress(0, ip) - stress(1, ip);
        const double d22 = stress(1, ip) - stress(2, ip);
        const double d33 = stress(2, ip) - stress(0, ip);
        const double shear = stress(3, ip) * stress(3, ip) + stress(4, ip) * stress(4, ip) +
                             stress(5, ip) * stress(5, ip);
        invariants(0, ip) = mean;
        invariants(1, ip) = sqrt(0.5 * (d11 * d11 + d22 * d22 + d33 * d33) + 3.0 * shear);
    });
}

/**
 * @brief Stress triaxiality (mean stress / von Mises stress) at every point, 0 where unloaded
 */
void StressTriaxiality(std::shared_ptr<SimulationState> sim_state,
                       int region,
                       std::shared_ptr<mfem::expt::PartialQuadratureFunction> values) {
    auto stress_qf = sim_state->GetQuadratureFunction("cauchy_stress_end", region);
    const int num_points = stress_qf->Size() / 6;
    const auto stress = mfem::Reshape(stress_qf->Read(), 6, num_points);
    auto triaxiality = mfem::Reshape(values->Write(), values->GetVDim(), num_points);
    mfem::forall(num_points, [=] MFEM_HOST_DEVICE(int ip) {
        const double mean = (stress(0, ip) + stress(1, ip) + stress(2, ip)) / 3.0;
        const double d11 = stress(0, ip) - stress(1, ip);
        const double d22 = stress(1, ip) - stress(2, ip);
        const double d33 = stress(2, ip) - stress(0, ip);
        const double shear = stress(3, ip) * stress(3, ip) + stress(4, ip) * stress(4, ip) +
                             stress(5, ip) * stress(5, ip);
        const double von_mises = sqrt(0.5 * (d11 * d11 + d22 * d22 + d33 * d33) + 3.0 * shear);
        triaxiality(0, ip) = (von_mises > 0.0) ? mean / von_mises : 0.0;
    });
}
