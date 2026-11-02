/**
 * @file my_exaconstit_functions.cpp
 * @brief Example ExaConstit user functions
 *
 * Three kinds of user functions, each optional and independent of the others:
 *
 * - register_function ([Materials.State_Vars]): gives names to slices of a material's state
 *   variables, so they can be visualized by name and used by other functions.
 *       void f(std::shared_ptr<SimulationState> sim_state, int region);
 * - initialize_function ([Materials.State_Vars]): sets the initial history variables.
 *       void f(std::shared_ptr<SimulationState> sim_state, int region);
 * - post-processing user function ([[PostProcessing.user_functions]]): fills one value (or
 *   several components) per quadrature point; ExaConstit visualizes it or volume-averages /
 *   integrates it, per region and globally, as the options entry says.
 *       void f(std::shared_ptr<SimulationState> sim_state, int region,
 *              std::shared_ptr<mfem::expt::PartialQuadratureFunction> values);
 *
 * Rules:
 * - Write them as normal C++ functions at global scope (not extern "C", not in a namespace)
 *   with exactly these argument types. ExaConstit checks the signature when it starts and stops
 *   with the required signature if it differs.
 * - region is the 0-based material index (material number - 1).
 * - Data lives where ExaConstit runs (CPU or GPU): read it with Read(), write with Write() or
 *   ReadWrite(), and loop with mfem::forall so the same code runs on both.
 *
 * The state variable layout used here is that of a UMAT with 9 state variables:
 *   [0] damage, [1-6] back stress (Voigt), [7] accumulated plastic strain,
 *   [8] number of local Newton iterations (bookkeeping, not worth plotting).
 */

#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <cmath>
#include <memory>

/**
 * @brief Name this material's state variables
 *
 * Each call registers (offset, length) inside the material's state variable vector. Slices must
 * fit in the state variables and must not overlap; ExaConstit stops with a message otherwise.
 * The optional last argument (default true) says whether the variable is written when all of
 * the material's state variables are asked for (visualize_state_vars = "all" or region_<N>).
 */
void RegisterMyStateVariables(std::shared_ptr<SimulationState> sim_state, int region) {
    sim_state->AddQuadratureFunctionStatePair("damage", {0, 1}, region);
    sim_state->AddQuadratureFunctionStatePair("back_stress", {1, 6}, region);
    sim_state->AddQuadratureFunctionStatePair("accumulated_slip", {7, 1}, region);
    sim_state->AddQuadratureFunctionStatePair("num_iterations", {8, 1}, region, false);
}

/**
 * @brief Set the initial history variables of this material
 *
 * Writes the beginning-of-step state variables ("state_var_beg"); ExaConstit copies them to the
 * end-of-step ones afterwards. Here: a small initial damage everywhere, everything else zero.
 */
void InitializeMyStateVariables(std::shared_ptr<SimulationState> sim_state, int region) {
    auto state_vars = sim_state->GetQuadratureFunction("state_var_beg", region);
    const int num_vars = state_vars->GetVDim();
    const int num_points = state_vars->Size() / num_vars;
    const int damage = sim_state->GetQuadratureFunctionStatePair("damage", region).first;

    auto state = mfem::Reshape(state_vars->Write(), num_vars, num_points);
    mfem::forall(num_points, [=] MFEM_HOST_DEVICE(int ip) {
        for (int k = 0; k < num_vars; k++) {
            state(k, ip) = 0.0;
        }
        state(damage, ip) = 1.0e-3;
    });
}

/**
 * @brief Stress triaxiality: mean stress over von Mises stress, at each quadrature point
 *
 * Reads the end-of-step Cauchy stress (Voigt: 11, 22, 33, 23, 13, 12) of the region and writes
 * one value per point. Zero where the von Mises stress is zero.
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
