#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <sstream>

BCTimeInfo BCTimeInfo::from_toml(const toml::value& toml_input) {
    BCTimeInfo info;

    if (toml_input.contains("time_dependent")) {
        info.time_dependent = toml::find<bool>(toml_input, "time_dependent");
    }

    if (toml_input.contains("cycle_dependent")) {
        info.cycle_dependent = toml::find<bool>(toml_input, "cycle_dependent");
    }

    if (toml_input.contains("times")) {
        info.times = toml::find<std::vector<double>>(toml_input, "times");
    }

    if (toml_input.contains("cycles")) {
        info.cycles = toml::find<std::vector<int>>(toml_input, "cycles");
    }

    return info;
}

VelocityBC VelocityBC::from_toml(const toml::value& toml_input) {
    VelocityBC bc;

    if (toml_input.contains("essential_ids")) {
        bc.essential_ids = toml::find<std::vector<int>>(toml_input, "essential_ids");
    }

    if (toml_input.contains("essential_comps")) {
        bc.essential_comps = toml::find<std::vector<int>>(toml_input, "essential_comps");
    }

    if (toml_input.contains("essential_vals")) {
        bc.essential_vals = toml::find<std::vector<double>>(toml_input, "essential_vals");
    }

    return bc;
}

VelocityGradientBC VelocityGradientBC::from_toml(const toml::value& toml_input) {
    VelocityGradientBC bc;

    if (toml_input.contains("velocity_gradient")) {
        auto temp = toml::find<std::vector<std::vector<double>>>(toml_input, "velocity_gradient");
        for (const auto& items : temp) {
            for (const auto& item : items) {
                bc.velocity_gradient.push_back(item);
            }
        }
    }

    if (toml_input.contains("essential_ids")) {
        bc.essential_ids = toml::find<std::vector<int>>(toml_input, "essential_ids");
    }

    if (toml_input.contains("essential_comps")) {
        bc.essential_comps = toml::find<std::vector<int>>(toml_input, "essential_comps");
    }

    if (toml_input.contains("origin")) {
        auto origin = toml::find<std::vector<double>>(toml_input, "origin");
        if (origin.size() >= 3) {
            bc.origin = std::array<double, 3>{origin[0], origin[1], origin[2]};
        }
    }

    return bc;
}

bool BoundaryOptions::validate() {
    // For simplicity, use the legacy format if velocity_bcs is empty
    auto is_empty = [](auto&& arg) -> bool {
        return std::visit(
            [](auto&& arg) -> bool {
                return arg.empty();
            },
            arg);
    };

    if (velocity_bcs.empty() && !is_empty(legacy_bcs.essential_ids)) {
        transform_legacy_format();
    }

    // Populate BCManager-compatible maps
    populate_bc_manager_maps();

    for (const auto& vel_bc : velocity_bcs) {
        // Add this BC's data to the maps
        for (size_t i = 0; i < vel_bc.essential_ids.size() && i < vel_bc.essential_comps.size();
             ++i) {
            // Add to velocity-specific maps
            if (vel_bc.essential_ids[i] <= 0) {
                WARNING_0_OPT("`BCs.velocity_bcs` has an `essential_ids` that <= 0. We've "
                              "fixed any negative values");
            }
            if (vel_bc.essential_comps[i] < 0) {
                WARNING_0_OPT("`BCs.velocity_bcs` has an `essential_comps` that < 0. "
                              "We've fixed any negative values");
            }
        }
        if (vel_bc.essential_ids.size() != vel_bc.essential_comps.size()) {
            WARNING_0_OPT("Error: `BCs.velocity_bcs` has unequal sizes of `essential_ids` and "
                          "`essential_comps`");
            return false;
        }
        // Add the values if available
        if (vel_bc.essential_vals.size() != (3 * vel_bc.essential_ids.size())) {
            WARNING_0_OPT("Error: `BCs.velocity_bcs` needs to have `essential_vals` that have 3 * "
                          "the size of `essential_ids` or `essential_comps` ");
            return false;
        }
    }

    for (const auto& vgrad_bc : vgrad_bcs) {
        // Add this BC's data to the maps
        for (size_t i = 0; i < vgrad_bc.essential_ids.size() && i < vgrad_bc.essential_comps.size();
             ++i) {
            // Add to velocity-specific maps
            if (vgrad_bc.essential_ids[i] <= 0) {
                WARNING_0_OPT("`BCs.velocity_gradient_bcs` has an `essential_ids` that <= "
                              "0. We've fixed any negative values");
            }
            if (vgrad_bc.essential_comps[i] < 0) {
                WARNING_0_OPT("`BCs.velocity_gradient_bcs` has an `essential_comps` that "
                              "< 0. We've fixed any negative values");
            }
        }

        if (vgrad_bc.essential_ids.size() != vgrad_bc.essential_comps.size()) {
            WARNING_0_OPT("Error: `BCs.velocity_gradient_bcs` has unequal sizes of `essential_ids` "
                          "and `essential_comps`");
            return false;
        }
        // Add the values if available
        if (vgrad_bc.velocity_gradient.size() != 9) {
            WARNING_0_OPT("Error: `BCs.velocity_gradient_bcs` needs to have `velocity_gradient` "
                          "needs to be a have 3 x 3 matrix");
            return false;
        }
    }

    // Make sure the velocity and velocity gradient BCs aren't constraining the same DOFs
    // on the same boundary attribute. The essential_comps values map to the constrained
    // {x, y, z} directions as follows (same convention as BCData::GetComponents):
    // 0 = none, 1 = x, 2 = y, 3 = z, 4 = xy, 5 = yz, 6 = xz, 7 = xyz
    auto constrained_directions = [](int comp) -> std::array<bool, 3> {
        switch (std::abs(comp)) {
        case 1:
            return {true, false, false};
        case 2:
            return {false, true, false};
        case 3:
            return {false, false, true};
        case 4:
            return {true, true, false};
        case 5:
            return {false, true, true};
        case 6:
            return {true, false, true};
        case 7:
            return {true, true, true};
        default:
            return {false, false, false};
        }
    };
    const std::array<std::string, 3> direction_names = {"x", "y", "z"};

    bool overlap_found = false;
    for (const auto& [step, vel_ids] : map_ess_id["ess_vel"]) {
        const auto& vel_comps = map_ess_comp["ess_vel"][step];
        const auto& vgrad_ids = map_ess_id["ess_vgrad"][step];
        const auto& vgrad_comps = map_ess_comp["ess_vgrad"][step];
        for (size_t i = 0; i < vel_ids.size() && i < vel_comps.size(); ++i) {
            for (size_t j = 0; j < vgrad_ids.size() && j < vgrad_comps.size(); ++j) {
                // Only BCs applied to the same boundary attribute can conflict
                if (vel_ids[i] != vgrad_ids[j]) {
                    continue;
                }
                const auto vel_dirs = constrained_directions(vel_comps[i]);
                const auto vgrad_dirs = constrained_directions(vgrad_comps[j]);
                std::string shared_dirs;
                for (size_t d = 0; d < 3; ++d) {
                    if (vel_dirs[d] && vgrad_dirs[d]) {
                        shared_dirs += direction_names[d] + " ";
                    }
                }
                if (!shared_dirs.empty()) {
                    std::ostringstream err;
                    err << "Error: `BCs` for update step " << step << " has boundary attribute "
                        << vel_ids[i] << " with the " << shared_dirs
                        << "component(s) constrained by both `velocity_bcs` (essential_comps = "
                        << vel_comps[i]
                        << ") and `velocity_gradient_bcs` (essential_comps = " << vgrad_comps[j]
                        << "). A DOF may only be constrained by one BC type.";
                    WARNING_0_OPT(err.str());
                    overlap_found = true;
                }
            }
        }
    }
    if (overlap_found) {
        return false;
    }

    if (time_info.cycles[0] != 1) {
        WARNING_0_OPT("Error: `BCs.time_info` needs to have the first value be 1");
        return false;
    }

    return true;
}

void BoundaryOptions::transform_legacy_format() {
    // Skip if we don't have legacy data
    auto is_empty = [](auto&& arg) -> bool {
        return std::visit(
            [](auto&& arg) -> bool {
                return arg.empty();
            },
            arg);
    };

    if (is_empty(legacy_bcs.essential_ids) || is_empty(legacy_bcs.essential_comps)) {
        return;
    }

    // First, ensure update_steps includes 1 (required for initialization)
    if (legacy_bcs.update_steps.empty() ||
        std::find(legacy_bcs.update_steps.begin(), legacy_bcs.update_steps.end(), 1) ==
            legacy_bcs.update_steps.end()) {
        legacy_bcs.update_steps.insert(legacy_bcs.update_steps.begin(), 1);
    }

    // Transfer update_steps to the object field
    update_steps = legacy_bcs.update_steps;

    // Handle time-dependent BCs case
    if (legacy_bcs.changing_ess_bcs) {
        // We need to match nested structures:
        // For each update step, we need corresponding essential_ids, essential_comps, etc.

        // Validate that array sizes match number of update steps
        const size_t num_steps = legacy_bcs.update_steps.size();
        // We expect nested arrays for time-dependent BCs
        if (std::holds_alternative<std::vector<std::vector<int>>>(legacy_bcs.essential_ids)) {
            auto& nested_ess_ids = std::get<std::vector<std::vector<int>>>(
                legacy_bcs.essential_ids);
            auto& nested_ess_comps = std::get<std::vector<std::vector<int>>>(
                legacy_bcs.essential_comps);

            if (is_empty(legacy_bcs.essential_vals)) {
                std::vector<std::vector<double>> tmp = {};
                legacy_bcs.essential_vals.emplace<1>(tmp);
            }
            if (is_empty(legacy_bcs.essential_vel_grad)) {
                std::vector<std::vector<std::vector<double>>> tmp = {};
                legacy_bcs.essential_vel_grad.emplace<1>(tmp);
            }

            auto& nested_ess_vals = std::get<std::vector<std::vector<double>>>(
                legacy_bcs.essential_vals);
            auto& nested_ess_vgrads = std::get<std::vector<std::vector<std::vector<double>>>>(
                legacy_bcs.essential_vel_grad);

            // Ensure sizes match
            if (nested_ess_ids.size() != num_steps || nested_ess_comps.size() != num_steps) {
                throw std::runtime_error("Mismatch in sizes of BC arrays vs. update_steps");
            }

            const auto empty_v1 = std::vector<double>();
            const auto empty_v2 = std::vector<std::vector<double>>();
            // Process each time step
            for (size_t i = 0; i < num_steps; ++i) {
                const int step = legacy_bcs.update_steps[i];
                const auto& ess_ids = nested_ess_ids[i];
                const auto& ess_comps = nested_ess_comps[i];

                const auto& ess_vals = (!is_empty(legacy_bcs.essential_vals)) ? nested_ess_vals[i]
                                                                              : empty_v1;
                const auto& ess_vgrads = (!is_empty(legacy_bcs.essential_vel_grad))
                                             ? nested_ess_vgrads[i]
                                             : empty_v2;

                // Create BCs for this time step
                create_boundary_conditions(step, ess_ids, ess_comps, ess_vals, ess_vgrads);
            }
        }
    }
    // Simple case: constant BCs
    else {
        // For non-changing BCs, we just have one set of values for all time steps
        create_boundary_conditions(
            1,
            std::get<std::vector<int>>(legacy_bcs.essential_ids),
            std::get<std::vector<int>>(legacy_bcs.essential_comps),
            std::get<std::vector<double>>(legacy_bcs.essential_vals),
            std::get<std::vector<std::vector<double>>>(legacy_bcs.essential_vel_grad));
    }
}

// Helper method to create BC objects from legacy arrays
void BoundaryOptions::create_boundary_conditions(
    int step,
    const std::vector<int>& ess_ids,
    const std::vector<int>& ess_comps,
    const std::vector<double>& essential_vals,
    const std::vector<std::vector<double>>& essential_vel_grad) {
    // Separate velocity and velocity gradient BCs
    std::vector<int> vel_ids, vel_comps, vgrad_ids, vgrad_comps;

    // Configure time dependency
    time_info.cycle_dependent = true;
    time_info.cycles.push_back(step);

    // Identify which BCs are velocity vs. velocity gradient
    for (size_t i = 0; i < ess_ids.size() && i < ess_comps.size(); ++i) {
        if (ess_comps[i] >= 0) {
            vel_ids.push_back(ess_ids[i]);
            vel_comps.push_back(ess_comps[i]);
        } else {
            vgrad_ids.push_back(ess_ids[i]);
            vgrad_comps.push_back(std::abs(ess_comps[i]));
        }
    }

    // Create velocity BC if needed
    if (!vel_ids.empty()) {
        VelocityBC vel_bc;
        vel_bc.essential_ids = vel_ids;
        vel_bc.essential_comps = vel_comps;

        // Find velocity values for this step
        if (essential_vals.size() >= vel_ids.size() * 3) {
            vel_bc.essential_vals = essential_vals;
        }
        velocity_bcs.push_back(vel_bc);
    }

    // Create velocity gradient BC if needed
    if (!vgrad_ids.empty()) {
        VelocityGradientBC vgrad_bc;
        vgrad_bc.essential_ids = vgrad_ids;
        vgrad_bc.essential_comps = vgrad_comps;

        // Find velocity gradient values for this step
        if (!essential_vel_grad.empty()) {
            // Flatten the 2D array to 1D
            for (const auto& row : essential_vel_grad) {
                vgrad_bc.velocity_gradient.insert(
                    vgrad_bc.velocity_gradient.end(), row.begin(), row.end());
            }
        }

        // Set origin if needed
        if (!legacy_bcs.vgrad_origin.empty() && legacy_bcs.vgrad_origin.size() >= 3) {
            vgrad_bc.origin = std::array<double, 3>{
                legacy_bcs.vgrad_origin[0], legacy_bcs.vgrad_origin[1], legacy_bcs.vgrad_origin[2]};
        }
        vgrad_bcs.push_back(vgrad_bc);
    }
}

void BoundaryOptions::populate_bc_manager_maps() {
    // Initialize the map structures
    map_ess_comp["total"] = std::unordered_map<int, std::vector<int>>();
    map_ess_comp["ess_vel"] = std::unordered_map<int, std::vector<int>>();
    map_ess_comp["ess_vgrad"] = std::unordered_map<int, std::vector<int>>();

    map_ess_id["total"] = std::unordered_map<int, std::vector<int>>();
    map_ess_id["ess_vel"] = std::unordered_map<int, std::vector<int>>();
    map_ess_id["ess_vgrad"] = std::unordered_map<int, std::vector<int>>();

    // Default entry for step 0 (used for initialization)
    map_ess_comp["total"][0] = std::vector<int>();
    map_ess_comp["ess_vel"][0] = std::vector<int>();
    map_ess_comp["ess_vgrad"][0] = std::vector<int>();

    map_ess_id["total"][0] = std::vector<int>();
    map_ess_id["ess_vel"][0] = std::vector<int>();
    map_ess_id["ess_vgrad"][0] = std::vector<int>();

    map_ess_vel[0] = std::vector<double>();
    map_ess_vgrad[0] = std::vector<double>(9, 0.0);

    // Determine which step(s) this BC applies to
    std::vector<int> steps;
    if (time_info.cycle_dependent && !time_info.cycles.empty()) {
        update_steps = time_info.cycles;
    } else if (update_steps.empty()) {
        // Default to step 1
        update_steps = {1};
    }

    for (int step : update_steps) {
        // Initialize maps for this step if needed
        if (map_ess_comp["total"].find(step) == map_ess_comp["total"].end()) {
            map_ess_comp["total"][step] = std::vector<int>();
            map_ess_comp["ess_vel"][step] = std::vector<int>();
            map_ess_comp["ess_vgrad"][step] = std::vector<int>();

            map_ess_id["total"][step] = std::vector<int>();
            map_ess_id["ess_vel"][step] = std::vector<int>();
            map_ess_id["ess_vgrad"][step] = std::vector<int>();

            map_ess_vel[step] = std::vector<double>();
            map_ess_vgrad[step] = std::vector<double>(9, 0.0);
        }
    }

    // Process velocity BCs
    size_t index = 0;
    for (const auto& vel_bc : velocity_bcs) {
        const int step = update_steps[index];
        // Add this BC's data to the maps
        for (size_t i = 0; i < vel_bc.essential_ids.size() && i < vel_bc.essential_comps.size();
             ++i) {
            // Add to total maps
            map_ess_id["total"][step].push_back(std::abs(vel_bc.essential_ids[i]));
            map_ess_comp["total"][step].push_back(std::abs(vel_bc.essential_comps[i]));

            // Add to velocity-specific maps
            map_ess_id["ess_vel"][step].push_back(std::abs(vel_bc.essential_ids[i]));
            map_ess_comp["ess_vel"][step].push_back(std::abs(vel_bc.essential_comps[i]));
        }
        // Add the values if available
        if (!vel_bc.essential_vals.empty()) {
            // Add the values to the map
            // Note: the original code expected values organized as triplets
            // of x, y, z values for each BC
            map_ess_vel[step] = vel_bc.essential_vals;
        }
        index++;
    }

    index = 0;
    // Process velocity gradient BCs
    for (const auto& vgrad_bc : vgrad_bcs) {
        const int step = update_steps[index];
        // Add this BC's data to the maps
        for (size_t i = 0; i < vgrad_bc.essential_ids.size(); ++i) {
            // Add to total maps with negative component to indicate vgrad BC
            map_ess_id["total"][step].push_back(std::abs(vgrad_bc.essential_ids[i]));
            map_ess_comp["total"][step].push_back(std::abs(vgrad_bc.essential_comps[i]));

            // Add to vgrad-specific maps
            map_ess_id["ess_vgrad"][step].push_back(std::abs(vgrad_bc.essential_ids[i]));
            map_ess_comp["ess_vgrad"][step].push_back(std::abs(vgrad_bc.essential_comps[i]));
        }
        // Add the gradient values if available
        if (!vgrad_bc.velocity_gradient.empty()) {
            map_ess_vgrad[step] = vgrad_bc.velocity_gradient;
        }
        index++;
    }
}

BoundaryOptions BoundaryOptions::from_toml(const toml::value& toml_input) {
    BoundaryOptions options;

    if (toml_input.contains("expt_mono_def_flag")) {
        options.legacy_bcs.mono_def_bcs = toml::find<bool>(toml_input, "expt_mono_def_flag");
        options.mono_def_bcs = options.legacy_bcs.mono_def_bcs;
    }

    // Parse legacy format flags
    if (toml_input.contains("changing_ess_bcs")) {
        options.legacy_bcs.changing_ess_bcs = toml::find<bool>(toml_input, "changing_ess_bcs");
    }

    if (toml_input.contains("update_steps")) {
        options.legacy_bcs.update_steps = toml::find<std::vector<int>>(toml_input, "update_steps");
    }

    if (toml_input.contains("time_info")) {
        options.time_info = BCTimeInfo::from_toml(toml::find(toml_input, "time_info"));
    }

    // Parse essential IDs based on format
    if (toml_input.contains("essential_ids")) {
        const auto& ids = toml_input.at("essential_ids");
        if (ids.is_array()) {
            // Check if first element is also an array (nested arrays)
            if (!ids.as_array().empty() && ids.as_array()[0].is_array()) {
                // Nested arrays for time-dependent BCs
                options.legacy_bcs.essential_ids = toml::find<std::vector<std::vector<int>>>(
                    toml_input, "essential_ids");
            } else {
                // Flat array for constant BCs
                options.legacy_bcs.essential_ids = toml::find<std::vector<int>>(toml_input,
                                                                                "essential_ids");
            }
        }
    }

    // Parse essential components based on format
    if (toml_input.contains("essential_comps")) {
        const auto& comps = toml_input.at("essential_comps");
        if (comps.is_array()) {
            // Check if first element is also an array (nested arrays)
            if (!comps.as_array().empty() && comps.as_array()[0].is_array()) {
                // Nested arrays for time-dependent BCs
                options.legacy_bcs.essential_comps = toml::find<std::vector<std::vector<int>>>(
                    toml_input, "essential_comps");
            } else {
                // Flat array for constant BCs
                options.legacy_bcs.essential_comps = toml::find<std::vector<int>>(
                    toml_input, "essential_comps");
            }
        }
    }

    // Parse essential values based on format
    if (toml_input.contains("essential_vals")) {
        const auto& vals = toml_input.at("essential_vals");
        if (vals.is_array()) {
            // Check if first element is also an array (nested arrays)
            if (!vals.as_array().empty() && vals.as_array()[0].is_array()) {
                // Nested arrays for time-dependent BCs
                options.legacy_bcs.essential_vals = toml::find<std::vector<std::vector<double>>>(
                    toml_input, "essential_vals");
            } else {
                // Flat array for constant BCs
                options.legacy_bcs.essential_vals = toml::find<std::vector<double>>(
                    toml_input, "essential_vals");
            }
        }
    }

    // Parse velocity gradient based on format
    if (toml_input.contains("essential_vel_grad")) {
        const auto& vgrad = toml_input.at("essential_vel_grad");
        if (vgrad.is_array()) {
            // Check if we have a triple-nested array structure
            if (!vgrad.as_array().empty() && vgrad.as_array()[0].is_array() &&
                !vgrad.as_array()[0].as_array().empty() &&
                vgrad.as_array()[0].as_array()[0].is_array()) {
                // Triple-nested arrays for time-dependent BCs with 2D gradient matrices
                options.legacy_bcs.essential_vel_grad =
                    toml::find<std::vector<std::vector<std::vector<double>>>>(toml_input,
                                                                              "essential_vel_grad");
            } else {
                // Double-nested arrays for constant BCs with 2D gradient matrix
                options.legacy_bcs.essential_vel_grad =
                    toml::find<std::vector<std::vector<double>>>(toml_input, "essential_vel_grad");
            }
        }
    }

    if (toml_input.contains("vgrad_origin")) {
        options.legacy_bcs.vgrad_origin = toml::find<std::vector<double>>(toml_input,
                                                                          "vgrad_origin");
    }

    // Parse modern structured format
    if (toml_input.contains("velocity_bcs")) {
        const auto vel_bcs = toml::find(toml_input, "velocity_bcs");
        if (vel_bcs.is_array()) {
            for (const auto& bc : vel_bcs.as_array()) {
                options.velocity_bcs.push_back(VelocityBC::from_toml(bc));
            }
        } else {
            options.velocity_bcs.push_back(VelocityBC::from_toml(vel_bcs));
        }
    }

    if (toml_input.contains("velocity_gradient_bcs")) {
        const auto vgrad_bcs = toml::find(toml_input, "velocity_gradient_bcs");
        if (vgrad_bcs.is_array()) {
            for (const auto& bc : vgrad_bcs.as_array()) {
                options.vgrad_bcs.push_back(VelocityGradientBC::from_toml(bc));
            }
        } else {
            options.vgrad_bcs.push_back(VelocityGradientBC::from_toml(vgrad_bcs));
        }
    }

    return options;
}

bool BCTimeInfo::validate() const {
    // Implement validation logic
    return true;
}

bool VelocityBC::validate() const {
    // Implement validation logic
    return !essential_ids.empty() && !essential_comps.empty() && !essential_vals.empty();
}

bool VelocityGradientBC::validate() const {
    // Implement validation logic
    return !velocity_gradient.empty();
}
