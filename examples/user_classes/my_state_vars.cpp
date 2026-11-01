// ============================================================================================
// Example 2: name a UMAT's state variables and give them starting values
//
// What it does, for a UMAT with 9 state variables:
//   - gives them names, so you can plot them by name in ParaView and use them in your outputs:
//       slot 0 = "damage", slots 1-6 = "back_stress", slot 7 = "accumulated_slip",
//       slot 8 = "num_iterations" (bookkeeping, so it is left out of "plot everything")
//   - starts with damage 0.01 inside a ball of radius 0.25 at the center of a 1 x 1 x 1 sample,
//     0.0001 elsewhere (change the center and radius below to fit your sample).
//
// How to use it, step by step:
//   1. Build it (use the compiler written at the top of exaconstit_user.hpp):
//        g++ -std=c++17 -O2 -fPIC -shared -I<exaconstit>/include my_state_vars.cpp -o libmy_vars.so
//      (<exaconstit> is the ExaConstit install folder; ask your system admins where it is)
//   2. In your options file, in the UMAT material's [Materials.State_Vars] table:
//        num_vars = 9
//        library_path = "./libmy_vars.so"
//        register_function   = { factory = "MakeMyStateVariables", variant = "names" }
//        initialize_function = { factory = "MakeMyStateVariables", variant = "starting_damage" }
//      and, to see damage in ParaView, in the same [[Materials]] entry (above its tables):
//        visualize_state_vars = ["damage"]
//   3. Run ExaConstit as usual.
// ============================================================================================
#include "exaconstit_user.hpp"

#include <cmath>

using namespace exaconstit::user;

// Part 1: the names. Each entry is {name, first slot, number of slots}; slots count from 0.
class MyNames : public RegisterFunction {
public:
    std::vector<StateVariableName> names(const Context&) override {
        return {
            {"damage", 0, 1},
            {"back_stress", 1, 6},
            {"accumulated_slip", 7, 1},
            {"num_iterations", 8, 1, false}, // false = not plotted with "plot everything"
        };
    }
};

// Part 2: the starting values
class MyStartingDamage : public InitializeFunction {
public:
    // The data needed: where each point is in the undeformed sample
    std::vector<std::string> inputs(const Context&) override { return {"reference_coords"}; }

    void initialize(const Context& context, const Inputs& inputs,
                    OutputField& state_vars) override {
        const Field& position = inputs["reference_coords"];
        // Which slot holds "damage" (the name given in Part 1)
        const int damage_slot = context.state_variable("damage").offset;

        // The ball: its center (x, y, z) and radius, in the units of your mesh
        const double center_x = 0.5, center_y = 0.5, center_z = 0.5;
        const double radius = 0.25;

        for (int point = 0; point < context.num_points(); point++) {
            const double dx = position(point, 0) - center_x;
            const double dy = position(point, 1) - center_y;
            const double dz = position(point, 2) - center_z;
            const double distance_from_center = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (distance_from_center < radius) {
                state_vars(point, damage_slot) = 0.01;
            } else {
                state_vars(point, damage_slot) = 0.0001;
            }
        }
    }
};

// Part 3: the factory named in the options file
std::unique_ptr<Function> MakeMyStateVariables(const FactoryArgs& args) {
    if (args.variant == "names") {
        return std::make_unique<MyNames>();
    }
    if (args.variant == "starting_damage") {
        return std::make_unique<MyStartingDamage>();
    }
    return nullptr; // unknown variant: ExaConstit stops and tells you
}
