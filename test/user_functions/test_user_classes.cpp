/**
 * @file test_user_classes.cpp
 * @brief User classes for the tests of the factory form of ExaConstit's user functions
 *
 * Built into libtest_user_classes exactly as a user on a shared machine would build theirs: one
 * plain shared library that includes only exaconstit_user.hpp (no other ExaConstit, MFEM or MPI
 * header). The decks name its factory instead of a function:
 *     key = { factory = "MakeTestClasses", variant = "<variant>" }
 *
 * The classes do what the functions of test_user_functions.cpp do, so the same tests and checks
 * run on both forms:
 * - register_function: "umat_names", "crystal_names";
 * - initialize_function: "umat_start" (slot k = 0.5 + k, after checking its inputs);
 * - post-processing: "unit_value", "stress_invariants", "stress_triaxiality", plus
 *   "check_inputs", which stops the run if an input does not hold what it should.
 * The others must stop the run with a clear message: "duplicate_names", "ask_unknown_input",
 * "throwing_initialize", and any variant the factory does not know.
 *
 * Test materials (user_functions.toml.in, postprocessing.toml.in): see test_user_functions.cpp.
 */

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exaconstit_user.hpp"

using namespace exaconstit::user;

namespace {

// =================================================================================================
// State variables
// =================================================================================================

/// The UMAT material's names: as RegisterUmatStateVariables, without the repeated name
class UmatNames : public RegisterFunction {
public:
    std::vector<StateVariableName> names(const Context&) override {
        return {{"damage", 0, 1},
                {"back_stress", 1, 6},
                {"accumulated_slip", 7, 1},
                {"num_iterations", 8, 1, false}};
    }
};

/// One name after ExaCMech's own: slot 3 of evptn_FCC_A is free
class CrystalNames : public RegisterFunction {
public:
    std::vector<StateVariableName> names(const Context&) override {
        return {{"num_func_evals", 3, 1}};
    }
};

/// The same name twice (must stop the run)
class DuplicateNames : public RegisterFunction {
public:
    std::vector<StateVariableName> names(const Context&) override {
        return {{"damage", 0, 1}, {"damage", 9, 1}};
    }
};

void Expect(bool condition, const std::string& what) {
    if (!condition) {
        throw std::runtime_error("check failed: " + what);
    }
}

/**
 * @brief Slot k = 0.5 + k at every point, after checking what initialize() is given
 *
 * Before the first step: zero stress, identity deformation gradient, coordinates inside the unit
 * cube, grain IDs from 1, and inputs that are state variables viewing the output's memory.
 */
class UmatStart : public InitializeFunction {
public:
    std::vector<std::string> inputs(const Context&) override {
        return {"cauchy_stress", "def_grad", "reference_coords", "grains", "state_vars", "damage"};
    }
    void initialize(const Context& context, const Inputs& in, OutputField& state_vars) override {
        const Field& stress = in["cauchy_stress"];
        const Field& def_grad = in["def_grad"];
        const Field& position = in["reference_coords"];
        const Field& grains = in["grains"];
        const Field& current = in["state_vars"];
        const Field& damage = in["damage"];
        def_grad.require({3, 3});
        Expect(context.num_points() == state_vars.num_points(), "point counts agree");
        Expect(context.num_state_vars == 10 && state_vars.vdim == 10, "10 state variables");
        Expect(context.state_variable("back_stress").offset == 1, "back_stress offset");
        Expect(current.data == state_vars.data, "state_vars input views the output");
        Expect(damage.data == state_vars.data + context.state_variable("damage").offset &&
                   damage.stride == 10,
               "damage is a window into the state variables");
        for (int p = 0; p < context.num_points(); p++) {
            for (int i = 0; i < 3; i++) {
                Expect(position(p, i) >= 0.0 && position(p, i) <= 1.0, "inside the unit cube");
                for (int j = 0; j < 3; j++) {
                    Expect(stress.tensor(p, i, j) == 0.0, "zero stress");
                    Expect(def_grad.tensor(p, i, j) == (i == j ? 1.0 : 0.0), "identity F");
                }
            }
            Expect(grains(p) >= 1.0, "grain IDs from 1");
        }
        for (int p = 0; p < state_vars.num_points(); p++) {
            for (int k = 0; k < state_vars.vdim; k++) {
                state_vars(p, k) = 0.5 + k;
            }
        }
    }
};

/// Asks for an input no material has (must stop the run with the list of inputs)
class AskUnknownInput : public InitializeFunction {
public:
    std::vector<std::string> inputs(const Context&) override {
        return {"no_such_input"};
    }
    void initialize(const Context&, const Inputs&, OutputField&) override {}
};

/// Throws (must stop the run with the message)
class ThrowingInitialize : public InitializeFunction {
public:
    void initialize(const Context&, const Inputs&, OutputField&) override {
        throw std::runtime_error("deliberate error from ThrowingInitialize");
    }
};

// =================================================================================================
// Post-processing
// =================================================================================================

/// 1 at every point: its volume integral must equal the material volume
class UnitValue : public PostProcessingFunction {
public:
    std::vector<std::string> inputs(const Context&) override {
        return {};
    }
    void compute(const Context&, const Inputs&, OutputField& output) override {
        for (int p = 0; p < output.num_points(); p++) {
            output(p) = 1.0;
        }
    }
};

double Mean(const Field& stress, int p) {
    return (stress.tensor(p, 0, 0) + stress.tensor(p, 1, 1) + stress.tensor(p, 2, 2)) / 3.0;
}

double VonMises(const Field& stress, int p) {
    const double d11 = stress.tensor(p, 0, 0) - stress.tensor(p, 1, 1);
    const double d22 = stress.tensor(p, 1, 1) - stress.tensor(p, 2, 2);
    const double d33 = stress.tensor(p, 2, 2) - stress.tensor(p, 0, 0);
    const double s23 = stress.tensor(p, 1, 2);
    const double s13 = stress.tensor(p, 0, 2);
    const double s12 = stress.tensor(p, 0, 1);
    return std::sqrt(0.5 * (d11 * d11 + d22 * d22 + d33 * d33) +
                     3.0 * (s23 * s23 + s13 * s13 + s12 * s12));
}

/// Mean stress and von Mises stress (2 components)
class StressInvariants : public PostProcessingFunction {
public:
    std::vector<std::string> inputs(const Context&) override {
        return {"cauchy_stress"};
    }
    void compute(const Context&, const Inputs& in, OutputField& output) override {
        const Field& stress = in["cauchy_stress"];
        for (int p = 0; p < output.num_points(); p++) {
            output(p, 0) = Mean(stress, p);
            output(p, 1) = VonMises(stress, p);
        }
    }
};

/// Mean stress / von Mises stress, 0 where unloaded
class StressTriaxiality : public PostProcessingFunction {
public:
    std::vector<std::string> inputs(const Context&) override {
        return {"cauchy_stress"};
    }
    void compute(const Context&, const Inputs& in, OutputField& output) override {
        const Field& stress = in["cauchy_stress"];
        for (int p = 0; p < output.num_points(); p++) {
            const double von_mises = VonMises(stress, p);
            output(p) = (von_mises > 0.0) ? Mean(stress, p) / von_mises : 0.0;
        }
    }
};

/**
 * @brief Checks the post-processing inputs, then outputs F33 - 1
 *
 * Stops the run if: coords - reference_coords differs from displacement; def_grad is not
 * stretched along z somewhere (the test pulls along z); a quaternion is not of unit length;
 * the layouts or shapes differ from the header's list; the output is not zero on entry.
 */
class CheckInputs : public PostProcessingFunction {
public:
    std::vector<std::string> inputs(const Context&) override {
        return {"coords",
                "reference_coords",
                "displacement",
                "def_grad",
                "quats",
                "eq_pl_strain",
                "velocity"};
    }
    void compute(const Context& context, const Inputs& in, OutputField& output) override {
        const Field& coords = in["coords"];
        const Field& reference = in["reference_coords"];
        const Field& displacement = in["displacement"];
        const Field& def_grad = in["def_grad"];
        const Field& quats = in["quats"];
        const Field& eps = in["eq_pl_strain"];
        Expect(coords.layout == Layout::vector && def_grad.layout == Layout::matrix &&
                   quats.layout == Layout::quaternion && eps.layout == Layout::scalar,
               "layouts");
        quats.require({4});
        Expect(eps.stride == context.num_state_vars, "eq_pl_strain is a window");
        Expect(context.time > 0.0 && context.delta_time > 0.0, "time of a converged step");
        bool stretched = false;
        for (int p = 0; p < output.num_points(); p++) {
            Expect(output(p) == 0.0, "output is zero on entry");
            for (int i = 0; i < 3; i++) {
                const double moved = coords(p, i) - reference(p, i);
                Expect(std::abs(moved - displacement(p, i)) <= 1e-12, "coords - ref = disp");
            }
            double norm = 0.0;
            for (int c = 0; c < 4; c++) {
                norm += quats(p, c) * quats(p, c);
            }
            Expect(std::abs(norm - 1.0) < 1e-8, "unit quaternions");
            Expect(eps(p) >= 0.0, "eq_pl_strain >= 0");
            stretched = stretched || def_grad.tensor(p, 2, 2) > 1.0;
            output(p) = def_grad.tensor(p, 2, 2) - 1.0;
        }
        Expect(output.num_points() == 0 || stretched, "F33 > 1 somewhere");
    }
};

} // namespace

/// The one factory of this library
std::unique_ptr<Function> MakeTestClasses(const FactoryArgs& args) {
    const std::string& v = args.variant;
    if (v == "umat_names")
        return std::make_unique<UmatNames>();
    if (v == "crystal_names")
        return std::make_unique<CrystalNames>();
    if (v == "duplicate_names")
        return std::make_unique<DuplicateNames>();
    if (v == "umat_start")
        return std::make_unique<UmatStart>();
    if (v == "ask_unknown_input")
        return std::make_unique<AskUnknownInput>();
    if (v == "throwing_initialize")
        return std::make_unique<ThrowingInitialize>();
    if (v == "unit_value")
        return std::make_unique<UnitValue>();
    if (v == "stress_invariants")
        return std::make_unique<StressInvariants>();
    if (v == "stress_triaxiality")
        return std::make_unique<StressTriaxiality>();
    if (v == "check_inputs")
        return std::make_unique<CheckInputs>();
    return nullptr;
}
