// ============================================================================================
// Example 1: your own output quantity
//
// What it does: at every integration point it computes the pressure (minus the mean stress),
// but only where the material has started to yield (equivalent plastic strain > 0) and only in
// the left half of the sample (x < 0.5). Everywhere else the value is 0.
//
// How to use it, step by step:
//   1. Build it (one line, on the machine where you run ExaConstit; use the compiler written at
//      the top of exaconstit_user.hpp):
//        g++ -std=c++17 -O2 -fPIC -shared -I<exaconstit>/include my_output.cpp -o libmy_output.so
//      (<exaconstit> is the ExaConstit install folder; ask your system admins where it is)
//   2. Add this to your options file:
//        [[PostProcessing.user_functions]]
//            library_path = "./libmy_output.so"
//            function_name = { factory = "MakeMyOutput", variant = "pressure_where_yielding" }
//            type = "visualization"   # or "volume_average" / "volume_integral"
//   3. Run ExaConstit as usual. The field "pressure_where_yielding" appears in ParaView.
// ============================================================================================
#include "exaconstit_user.hpp"

using namespace exaconstit::user;

class PressureWhereYielding : public PostProcessingFunction {
public:
    // Step A: list the data you need, by name (the names are listed in exaconstit_user.hpp)
    std::vector<std::string> inputs(const Context&) override {
        return {"cauchy_stress", "eq_pl_strain", "coords"};
    }

    // Step B: compute your value at every point and write it into "output"
    void compute(const Context& context, const Inputs& inputs, OutputField& output) override {
        const Field& stress = inputs["cauchy_stress"];       // 6 values per point
        const Field& plastic_strain = inputs["eq_pl_strain"]; // 1 value per point
        const Field& position = inputs["coords"];            // x, y, z per point

        for (int point = 0; point < context.num_points(); point++) {
            const double x = position(point, 0);  // component 0 = x
            const double eps = plastic_strain(point);

            if (x < 0.5 && eps > 0.0) {
                // tensor(point, i, j) gives the stress component s_ij (0 = x, 1 = y, 2 = z)
                const double mean_stress = (stress.tensor(point, 0, 0) +
                                            stress.tensor(point, 1, 1) +
                                            stress.tensor(point, 2, 2)) / 3.0;
                output(point) = -mean_stress;
            }
            // Points you do not write stay 0
        }
    }
};

// Step C: the "factory" named in the options file. It hands ExaConstit the class whose
// variant name matches the options file. One library can hold many classes this way.
std::unique_ptr<Function> MakeMyOutput(const FactoryArgs& args) {
    if (args.variant == "pressure_where_yielding") {
        return std::make_unique<PressureWhereYielding>();
    }
    return nullptr; // unknown variant: ExaConstit stops and tells you
}
