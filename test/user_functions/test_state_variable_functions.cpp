/**
 * @file test_state_variable_functions.cpp
 * @brief Checks that the [Materials.State_Vars] user functions take effect
 *
 * The test reads the "success" input deck written from user_functions.toml.in (its path is the
 * first command line argument), builds the SimulationState and the material model
 * (MultiExaModel) exactly as the mechanics driver does, and checks the result of the user
 * functions in libtest_user_functions (test_user_functions.cpp):
 * - register_function: the named state variables of each region, in registration order.
 *   Region 0 (ExaCMech) lists ExaCMech's own nine names first and the user name
 *   "num_func_evals" last; the user's attempt to move "quats" is ignored. Region 1 (UMAT)
 *   lists the user names; the second "damage" is ignored. The lists must be the same on every
 *   rank, also on ranks without elements of a region.
 * - initialize_function: on ranks that own elements of region 1, "state_var_beg" and
 *   "state_var_end" both hold 0.5 + k in state variable slot k at every point.
 *
 * The user library is loaded into this test executable, so the executable exports its symbols
 * (ENABLE_EXPORTS in test/CMakeLists.txt), as the mechanics executable does.
 *
 * Part of the extensive suite (EXACONSTIT_ENABLE_EXTENSIVE_TESTS). Run with 1 or 2 ranks:
 *     mpirun -np 2 ./test_state_variable_functions <build>/test/user_functions/success.toml
 */

#include "models/mechanics_multi_model.hpp"
#include "options/option_parser_v2.hpp"
#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

/** @brief Input deck path, from the command line */
std::string g_deck_path;

/** @brief Named state variables of a region: (name, (offset, length)) in registration order */
using StatePairs = std::vector<std::pair<std::string, std::pair<int, int>>>;

/**
 * @brief Check that every value of a region's state variable function is 0.5 + slot
 *
 * @param values State variables of the region, vdim values per point
 * @param label Name used in failure messages
 */
void ExpectInitialValues(const mfem::expt::PartialQuadratureFunction& values,
                         const std::string& label) {
    const int num_vars = values.GetVDim();
    const int num_points = values.Size() / num_vars;
    ASSERT_GT(num_points, 0) << label;
    // data(k, point): state variable k at quadrature point "point", read on the host
    const auto data = mfem::Reshape(values.HostRead(), num_vars, num_points);
    int mismatches = 0;
    for (int point = 0; point < num_points; point++) {
        for (int k = 0; k < num_vars; k++) {
            mismatches += (data(k, point) != 0.5 + k) ? 1 : 0;
        }
    }
    EXPECT_EQ(mismatches, 0) << label << ": values differ from 0.5 + slot";
}

TEST(StateVariableFunctions, RegisterAndInitialize) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ExaOptions options;
    options.parse_options(g_deck_path, rank);
    auto sim_state = std::make_shared<SimulationState>(options);
    // The constructor runs the built-in model setup and then the user functions
    auto model = std::make_shared<MultiExaModel>(sim_state, options);

    // Region 0 (ExaCMech): ExaCMech's names in its own order, then the user name
    const StatePairs crystal = sim_state->GetQuadratureFunctionStatePairs(0);
    const std::vector<std::string> crystal_names = {"eq_pl_strain_rate",
                                                    "eq_pl_strain",
                                                    "plastic_work",
                                                    "quats",
                                                    "shear_rate",
                                                    "hardness",
                                                    "internal_energy",
                                                    "relative_volume",
                                                    "elastic_strain",
                                                    "num_func_evals"};
    ASSERT_EQ(crystal.size(), crystal_names.size());
    for (size_t i = 0; i < crystal.size(); i++) {
        EXPECT_EQ(crystal[i].first, crystal_names[i]) << "region 0, entry " << i;
    }
    // "quats" keeps ExaCMech's slice (ori_state_var_loc = 9, 4 components)
    EXPECT_EQ(sim_state->GetQuadratureFunctionStatePair("quats", 0), std::make_pair(9, 4));
    EXPECT_EQ(crystal.back().second, std::make_pair(3, 1));

    // Region 1 (UMAT): only the user names; the second "damage" did not replace the first
    const StatePairs umat = sim_state->GetQuadratureFunctionStatePairs(1);
    const StatePairs expected_umat = {{"damage", {0, 1}},
                                      {"back_stress", {1, 6}},
                                      {"accumulated_slip", {7, 1}},
                                      {"num_iterations", {8, 1}}};
    EXPECT_EQ(umat, expected_umat);
    // Plot flags: true by default, false where the register function said so
    EXPECT_TRUE(sim_state->GetQuadratureFunctionStatePlot("damage", 1));
    EXPECT_FALSE(sim_state->GetQuadratureFunctionStatePlot("num_iterations", 1));
    EXPECT_TRUE(sim_state->GetQuadratureFunctionStatePlot("eq_pl_strain", 0));

    // Initial values, on the ranks that own elements of the UMAT region
    if (sim_state->IsRegionActive(1)) {
        ExpectInitialValues(*sim_state->GetQuadratureFunction("state_var_beg", 1), "state_var_beg");
        ExpectInitialValues(*sim_state->GetQuadratureFunction("state_var_end", 1), "state_var_end");
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    mfem::Hypre::Init();
    ::testing::InitGoogleTest(&argc, argv);
    int result = 1;
    if (argc < 2) {
        std::cerr << "usage: test_state_variable_functions <input deck>" << std::endl;
    } else {
        g_deck_path = argv[1];
        result = RUN_ALL_TESTS();
    }
    MPI_Finalize();
    return result;
}
