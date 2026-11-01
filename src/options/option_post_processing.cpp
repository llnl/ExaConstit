#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"

#include <algorithm>
#include <charconv>
#include <iostream>
#include <set>
#include <sstream>
#include <string_view>

/**
 * @brief Check if the TOML input contains legacy volume averaging options in [Visualizations]
 */
bool has_legacy_volume_averaging(const toml::value& toml_input) {
    if (!toml_input.contains("Visualizations")) {
        return false;
    }

    const auto viz_table = toml::find(toml_input, "Visualizations");

    // Check for legacy volume averaging indicators
    return viz_table.contains("avg_stress_fname") || viz_table.contains("additional_avgs") ||
           viz_table.contains("avg_def_grad_fname") || viz_table.contains("avg_pl_work_fname") ||
           viz_table.contains("avg_euler_strain_fname");
}

/**
 * @brief Check if the TOML input contains legacy light-up options in [Visualizations]
 */
bool has_legacy_light_up(const toml::value& toml_input) {
    if (!toml_input.contains("Visualizations")) {
        return false;
    }

    const auto viz_table = toml::find(toml_input, "Visualizations");

    // Check for legacy light-up indicators
    return viz_table.contains("light_up") || viz_table.contains("light_up_hkl") ||
           viz_table.contains("light_dist_tol") || viz_table.contains("light_s_dir") ||
           viz_table.contains("lattice_params") || viz_table.contains("lattice_basename");
}

/**
 * @brief Parse legacy light-up options from [Visualizations] table
 */
LightUpOptions parse_legacy_light_up(const toml::value& toml_input) {
    LightUpOptions options;

    if (!toml_input.contains("Visualizations")) {
        return options;
    }

    const auto viz_table = toml::find(toml_input, "Visualizations");

    // Check if light-up is enabled
    if (viz_table.contains("light_up")) {
        options.enabled = toml::find<bool>(viz_table, "light_up");
    }

    if (!options.enabled) {
        return options; // Return early if not enabled
    }

    // Parse HKL directions (light_up_hkl -> hkl_directions)
    if (viz_table.contains("light_up_hkl")) {
        const auto hkl_array = toml::find(viz_table, "light_up_hkl");
        if (hkl_array.is_array()) {
            options.hkl_directions.clear();
            for (const auto& direction : hkl_array.as_array()) {
                if (direction.is_array() && direction.as_array().size() >= 3) {
                    std::array<double, 3> hkl_dir;
                    if (direction.at(0).is(toml::value_t::integer)) {
                        auto dir_vec = toml::get<std::vector<int>>(direction);
                        std::copy_n(dir_vec.begin(), 3, hkl_dir.begin());
                    } else {
                        auto dir_vec = toml::get<std::vector<double>>(direction);
                        std::copy_n(dir_vec.begin(), 3, hkl_dir.begin());
                    }
                    options.hkl_directions.push_back(hkl_dir);
                }
            }
        }
    }

    // Parse distance tolerance (light_dist_tol -> distance_tolerance)
    if (viz_table.contains("light_dist_tol")) {
        options.distance_tolerance = toml::find<double>(viz_table, "light_dist_tol");
    }

    // Parse sample direction (light_s_dir -> sample_direction)
    if (viz_table.contains("light_s_dir")) {
        const auto dir = toml::find(toml_input, "light_s_dir");
        if (dir.at(0).is(toml::value_t::integer)) {
            auto dir_vec = toml::get<std::vector<int>>(dir);
            if (dir_vec.size() >= 3) {
                std::copy_n(dir_vec.begin(), 3, options.sample_direction.begin());
            }
        } else {
            auto dir_vec = toml::get<std::vector<double>>(dir);
            if (dir_vec.size() >= 3) {
                std::copy_n(dir_vec.begin(), 3, options.sample_direction.begin());
            }
        }
    }

    // Parse lattice parameters (lattice_params -> lattice_parameters)
    if (viz_table.contains("lattice_params")) {
        auto params = toml::find<std::vector<double>>(viz_table, "lattice_params");
        if (params.size() >= 3) {
            options.lattice_parameters = {params[0]};
        }
    }

    options.lattice_type = LatticeType::CUBIC;

    // Parse lattice basename
    if (viz_table.contains("lattice_basename")) {
        options.lattice_basename = toml::find<std::string>(viz_table, "lattice_basename");
    }

    return options;
}

LightUpOptions LightUpOptions::from_toml(const toml::value& toml_input) {
    LightUpOptions options;

    if (toml_input.contains("light_up")) {
        options.enabled = toml::find<bool>(toml_input, "light_up");
    } else if (toml_input.contains("enabled")) {
        options.enabled = toml::find<bool>(toml_input, "enabled");
    }

    // Parse material association - REQUIRED for multi-material support
    if (toml_input.contains("material_name")) {
        options.material_name = toml::find<std::string>(toml_input, "material_name");
    } else {
        WARNING_0_OPT("LightUp configuration missing 'material_name' field");
    }

    if (toml_input.contains("light_up_hkl")) {
        const auto hkl = toml::find(toml_input, "light_up_hkl");
        if (hkl.is_array()) {
            for (const auto& dir : hkl.as_array()) {
                if (dir.is_array() && dir.as_array().size() >= 3) {
                    std::array<double, 3> direction;
                    if (dir.at(0).is(toml::value_t::integer)) {
                        auto dir_vec = toml::get<std::vector<int>>(dir);
                        std::copy_n(dir_vec.begin(), 3, direction.begin());
                    } else {
                        auto dir_vec = toml::get<std::vector<double>>(dir);
                        std::copy_n(dir_vec.begin(), 3, direction.begin());
                    }
                    options.hkl_directions.push_back(direction);
                }
            }
        }
    } else if (toml_input.contains("hkl_directions")) {
        const auto hkl = toml::find(toml_input, "hkl_directions");
        if (hkl.is_array()) {
            for (const auto& dir : hkl.as_array()) {
                if (dir.is_array() && dir.as_array().size() >= 3) {
                    std::array<double, 3> direction;
                    if (dir.at(0).is(toml::value_t::integer)) {
                        auto dir_vec = toml::get<std::vector<int>>(dir);
                        std::copy_n(dir_vec.begin(), 3, direction.begin());
                    } else {
                        auto dir_vec = toml::get<std::vector<double>>(dir);
                        std::copy_n(dir_vec.begin(), 3, direction.begin());
                    }
                    options.hkl_directions.push_back(direction);
                }
            }
        }
    }

    if (toml_input.contains("light_dist_tol")) {
        options.distance_tolerance = toml::find<double>(toml_input, "light_dist_tol");
    } else if (toml_input.contains("distance_tolerance")) {
        options.distance_tolerance = toml::find<double>(toml_input, "distance_tolerance");
    }

    if (toml_input.contains("light_s_dir")) {
        const auto dir = toml::find(toml_input, "light_s_dir");
        if (dir.at(0).is(toml::value_t::integer)) {
            auto dir_vec = toml::get<std::vector<int>>(dir);
            if (dir_vec.size() >= 3) {
                std::copy_n(dir_vec.begin(), 3, options.sample_direction.begin());
            }
        } else {
            auto dir_vec = toml::get<std::vector<double>>(dir);
            if (dir_vec.size() >= 3) {
                std::copy_n(dir_vec.begin(), 3, options.sample_direction.begin());
            }
        }
    } else if (toml_input.contains("sample_direction")) {
        const auto dir = toml::find(toml_input, "sample_direction");
        if (dir.at(0).is(toml::value_t::integer)) {
            auto dir_vec = toml::get<std::vector<int>>(dir);
            if (dir_vec.size() >= 3) {
                std::copy_n(dir_vec.begin(), 3, options.sample_direction.begin());
            }
        } else {
            auto dir_vec = toml::get<std::vector<double>>(dir);
            if (dir_vec.size() >= 3) {
                std::copy_n(dir_vec.begin(), 3, options.sample_direction.begin());
            }
        }
    }

    if (toml_input.contains("lattice_params")) {
        auto params = toml::find<std::vector<double>>(toml_input, "lattice_params");
        if (params.size() >= 1) {
            std::copy(params.begin(), params.end(), options.lattice_parameters.begin());
        }
    } else if (toml_input.contains("lattice_parameters")) {
        auto params = toml::find<std::vector<double>>(toml_input, "lattice_parameters");
        if (params.size() >= 1) {
            std::copy(params.begin(), params.end(), options.lattice_parameters.begin());
        }
    }

    if (toml_input.contains("laue_type")) {
        auto laue_type = toml::find<std::string>(toml_input, "laue_type");
        options.lattice_type = string_to_lattice_type(laue_type);
    }

    if (toml_input.contains("lattice_basename")) {
        options.lattice_basename = toml::find<std::string>(toml_input, "lattice_basename");
    }

    return options;
}

/**
 * @brief Enhanced LightUpOptions::from_toml that handles legacy format
 */
std::vector<LightUpOptions> LightUpOptions::from_toml_with_legacy(const toml::value& toml_input) {
    std::vector<LightUpOptions> light_up_configs;

    // First check if we have legacy format in [Visualizations]
    if (has_legacy_light_up(toml_input)) {
        auto legacy_options = parse_legacy_light_up(toml_input);
        if (legacy_options.enabled) {
            // Legacy format doesn't have material_name, so assign default
            legacy_options.material_name = "default_material";
            light_up_configs.push_back(legacy_options);
            std::cout << "Info: Legacy LightUp configuration detected. "
                      << "Assigned to default_material. Consider updating to new format."
                      << std::endl;
        }
    }

    // Then check for modern format in [PostProcessing.light_up]
    // Modern format takes precedence if both exist
    if (toml_input.contains("PostProcessing")) {
        const auto post_proc = toml::find(toml_input, "PostProcessing");
        if (post_proc.contains("light_up")) {
            const auto light_up_section = toml::find(post_proc, "light_up");

            if (light_up_section.is_array()) {
                // New array format: multiple light_up configurations
                for (const auto& light_config : light_up_section.as_array()) {
                    auto light_options = LightUpOptions::from_toml(light_config);
                    if (light_options.enabled) {
                        if (light_options.material_name.empty()) {
                            WARNING_0_OPT("LightUp config in array missing material_name. "
                                          "Skipping.");
                            continue;
                        }
                        light_up_configs.push_back(light_options);
                    }
                }
            } else {
                // Single config format (legacy or modern)
                auto modern_options = LightUpOptions::from_toml(light_up_section);
                if (modern_options.enabled) {
                    // If no material_name specified in single config, assign default for backward
                    // compatibility
                    if (modern_options.material_name.empty()) {
                        modern_options.material_name = "default_material";
                        std::cout
                            << "Info: Single LightUp configuration without material_name detected. "
                            << "Assigned to default_material. Consider adding material_name field."
                            << std::endl;
                    }
                    // Only add if we don't already have a legacy config (modern takes precedence)
                    if (light_up_configs.empty() ||
                        light_up_configs[0].material_name != "default_material") {
                        light_up_configs.clear(); // Clear any legacy config
                        light_up_configs.push_back(modern_options);
                    }
                }
            }
        }
    }

    return light_up_configs;
}

bool LightUpOptions::resolve_region_id(const std::vector<MaterialOptions>& materials) {
    for (const auto& material : materials) {
        if (material.material_name == material_name) {
            region_id = material.region_id;
            return true;
        }
    }
    std::ostringstream err;
    err << "Error: LightUp configuration references unknown material: " << material_name
        << std::endl;
    WARNING_0_OPT(err.str());
    return false;
}

bool LightUpOptions::validate() const {
    if (!enabled) {
        return true;
    }

    if (material_name.empty()) {
        WARNING_0_OPT("Error: LightUp configuration must specify a material_name");
        return false;
    }

    if (hkl_directions.size() < 1) {
        WARNING_0_OPT("Error: LightUp table did not provide any values in the hkl_directions");
        return false;
    }

    if (distance_tolerance < 0) {
        WARNING_0_OPT("Error: LightUp table did not provide a positive distance_tolerance value");
        return false;
    }

    switch (lattice_type) {
    case LatticeType::CUBIC: {
        if (lattice_parameters.size() != 1) {
            WARNING_0_OPT("Error: LightUp table did not provide the right number of "
                          "lattice_parameters: 'cubic' -> a");
            return false;
        }
        break;
    }
    case LatticeType::HEXAGONAL:
    case LatticeType::TRIGONAL:
    case LatticeType::TETRAGONAL: {
        if (lattice_parameters.size() != 2) {
            WARNING_0_OPT("Error: LightUp table did not provide the right number of "
                          "lattice_parameters: 'hexagonal / trigonal / tetragonal' -> a, c");
            return false;
        }
        break;
    }
    case LatticeType::RHOMBOHEDRAL: {
        if (lattice_parameters.size() != 2) {
            WARNING_0_OPT("Error: LightUp table did not provide the right number of "
                          "lattice_parameters: 'rhombohedral' -> a, alpha (in radians)");
            return false;
        }
        break;
    }
    case LatticeType::ORTHORHOMBIC: {
        if (lattice_parameters.size() != 3) {
            WARNING_0_OPT("Error: LightUp table did not provide the right number of "
                          "lattice_parameters: 'orthorhombic' -> a, b, c");
            return false;
        }
        break;
    }
    case LatticeType::MONOCLINIC: {
        if (lattice_parameters.size() != 4) {
            WARNING_0_OPT("Error: LightUp table did not provide the right number of "
                          "lattice_parameters: 'monoclinic' -> a, b, c, beta (in radians)");
            return false;
        }
        break;
    }
    case LatticeType::TRICLINIC: {
        if (lattice_parameters.size() != 6) {
            WARNING_0_OPT(
                "Error: LightUp table did not provide the right number of lattice_parameters: "
                "'triclinic' -> a, b, c, alpha, beta, gamma (in radians)");
            return false;
        }
        break;
    }
    default:
        break;
    }

    for (const auto lp : lattice_parameters) {
        if (lp < 0) {
            WARNING_0_OPT(
                "Error: LightUp table did not provide a positive lattice_parameters value");
            return false;
        }
    }

    // Implement validation logic
    return true;
}

VisualizationOptions VisualizationOptions::from_toml(const toml::value& toml_input) {
    VisualizationOptions options;

    if (toml_input.contains("visit")) {
        options.visit = toml::find<bool>(toml_input, "visit");
    }

    if (toml_input.contains("paraview")) {
        options.paraview = toml::find<bool>(toml_input, "paraview");
    }

    if (toml_input.contains("adios2")) {
        options.adios2 = toml::find<bool>(toml_input, "adios2");
    }

    if (toml_input.contains("steps") || toml_input.contains("output_frequency")) {
        // Support both naming conventions
        const auto& freq_key = toml_input.contains("steps") ? "steps" : "output_frequency";
        options.output_frequency = toml::find<int>(toml_input, freq_key);
    }

    if (toml_input.contains("floc")) {
        options.floc = toml::find<std::string>(toml_input, "floc");
    }

    if (toml_input.contains("field_location")) {
        options.field_location = string_to_viz_field_location(
            toml::find<std::string>(toml_input, "field_location"));
    }

    return options;
}

bool VisualizationOptions::validate() const {
    if (output_frequency < 1) {
        WARNING_0_OPT("Error: Visualizations table did not provide a valid output frequency valid "
                      "as it was less than 1");
        return false;
    }
    if (field_location == VizFieldLocation::NOTYPE) {
        WARNING_0_OPT("Error: Visualizations table field_location must be \"zone\" or "
                      "\"quadrature\"");
        return false;
    }
    return true;
}

/**
 * @brief Parse legacy volume averaging options from [Visualizations] table
 */
VolumeAverageOptions parse_legacy_volume_averaging(const toml::value& toml_input) {
    VolumeAverageOptions options;

    if (!toml_input.contains("Visualizations")) {
        return options;
    }

    const auto viz_table = toml::find(toml_input, "Visualizations");

    // Check if volume averaging should be enabled
    // In legacy format, presence of avg_stress_fname means it's enabled
    // or if one of the other fields are noted, but
    options.enabled = true;
    options.stress = true; // Stress was always enabled in legacy format
    if (viz_table.contains("avg_stress_fname")) {
        options.avg_stress_fname = toml::find<std::string>(viz_table, "avg_stress_fname");
    }

    // Extract output directory from floc
    if (viz_table.contains("floc")) {
        options.output_directory = toml::find<std::string>(viz_table, "floc");
    }

    // Extract output frequency from steps
    if (viz_table.contains("steps")) {
        options.output_frequency = toml::find<int>(viz_table, "steps");
    }

    // Check for additional_avgs flag
    bool additional_avgs = false;
    if (viz_table.contains("additional_avgs")) {
        additional_avgs = toml::find<bool>(viz_table, "additional_avgs");
        options.additional_avgs = additional_avgs;
    }

    // Set deformation gradient options
    if (additional_avgs || viz_table.contains("avg_def_grad_fname")) {
        options.def_grad = true;
        if (viz_table.contains("avg_def_grad_fname")) {
            options.avg_def_grad_fname = toml::find<std::string>(viz_table, "avg_def_grad_fname");
        }
    }

    // Set plastic work options
    if (additional_avgs || viz_table.contains("avg_pl_work_fname")) {
        options.plastic_work = true;
        if (viz_table.contains("avg_pl_work_fname")) {
            options.avg_pl_work_fname = toml::find<std::string>(viz_table, "avg_pl_work_fname");
        }
    }

    if (additional_avgs || viz_table.contains("avg_eps_fname")) {
        options.eq_pl_strain = true;
        if (viz_table.contains("avg_eps_fname")) {
            options.avg_eq_pl_strain_fname = toml::find<std::string>(viz_table, "avg_eps_fname");
        }
    }

    // Set Euler strain options
    if (additional_avgs || viz_table.contains("avg_euler_strain_fname")) {
        options.euler_strain = true;
        if (viz_table.contains("avg_euler_strain_fname")) {
            options.avg_euler_strain_fname = toml::find<std::string>(viz_table,
                                                                     "avg_euler_strain_fname");
        }
    }

    if (additional_avgs || viz_table.contains("avg_elastic_strain_fname")) {
        options.elastic_strain = true;
        if (viz_table.contains("avg_elastic_strain_fname")) {
            options.avg_elastic_strain_fname = toml::find<std::string>(viz_table,
                                                                       "avg_elastic_strain_fname");
        }
    }

    return options;
}

VolumeAverageOptions VolumeAverageOptions::from_toml(const toml::value& toml_input) {
    VolumeAverageOptions options;

    if (toml_input.contains("enabled")) {
        options.enabled = toml::find<bool>(toml_input, "enabled");
    }

    if (toml_input.contains("stress")) {
        options.stress = toml::find<bool>(toml_input, "stress");
    }

    if (toml_input.contains("def_grad")) {
        options.def_grad = toml::find<bool>(toml_input, "def_grad");
    }

    if (toml_input.contains("euler_strain")) {
        options.euler_strain = toml::find<bool>(toml_input, "euler_strain");
    }

    if (toml_input.contains("plastic_work")) {
        options.plastic_work = toml::find<bool>(toml_input, "plastic_work");
    }

    if (toml_input.contains("elastic_strain")) {
        options.elastic_strain = toml::find<bool>(toml_input, "elastic_strain");
    }

    if (toml_input.contains("eq_pl_strain")) {
        options.eq_pl_strain = toml::find<bool>(toml_input, "eq_pl_strain");
    }

    if (toml_input.contains("output_directory")) {
        options.output_directory = toml::find<std::string>(toml_input, "output_directory");
    }

    if (toml_input.contains("output_frequency")) {
        options.output_frequency = toml::find<int>(toml_input, "output_frequency");
    }

    if (toml_input.contains("avg_stress_fname")) {
        options.avg_stress_fname = toml::find<std::string>(toml_input, "avg_stress_fname");
    }

    if (toml_input.contains("avg_def_grad_fname")) {
        options.avg_def_grad_fname = toml::find<std::string>(toml_input, "avg_def_grad_fname");
    }

    if (toml_input.contains("avg_euler_strain_fname")) {
        options.avg_euler_strain_fname = toml::find<std::string>(toml_input,
                                                                 "avg_euler_strain_fname");
    }

    if (toml_input.contains("avg_pl_work_fname")) {
        options.avg_pl_work_fname = toml::find<std::string>(toml_input, "avg_pl_work_fname");
    }

    if (toml_input.contains("avg_eq_pl_strain_fname")) {
        options.avg_eq_pl_strain_fname = toml::find<std::string>(toml_input,
                                                                 "avg_eq_pl_strain_fname");
    }

    if (toml_input.contains("avg_elastic_strain_fname")) {
        options.avg_elastic_strain_fname = toml::find<std::string>(toml_input,
                                                                   "avg_elastic_strain_fname");
    }

    options.statistics_valid = parse_volume_statistics(
        toml_input, "[PostProcessing.volume_averages]", options.statistics);
    if (toml_input.contains("histograms")) {
        for (const auto& [quantity, table] : toml::find<toml::table>(toml_input, "histograms")) {
            options.histograms[quantity] = HistogramOptions::from_toml(table);
        }
    }
    const bool any_average = options.stress || options.def_grad || options.euler_strain ||
                             options.plastic_work || options.eq_pl_strain || options.elastic_strain;
    if ((!options.statistics.empty() || !options.histograms.empty()) &&
        (!options.enabled || !any_average)) {
        WARNING_0_OPT("Warning: [PostProcessing.volume_averages] statistics / histograms are "
                      "given but no volume average is enabled, so none are written");
    }

    return options;
}

/**
 * @brief Enhanced VolumeAverageOptions::from_toml that handles legacy format
 */
VolumeAverageOptions VolumeAverageOptions::from_toml_with_legacy(const toml::value& toml_input) {
    VolumeAverageOptions options;

    // First check if we have legacy format in [Visualizations]
    const bool has_legacy = has_legacy_volume_averaging(toml_input);
    if (has_legacy) {
        options = parse_legacy_volume_averaging(toml_input);
    }

    // Then check for modern format in [PostProcessing.volume_averages]
    // Modern format takes precedence if both exist
    if (toml_input.contains("PostProcessing")) {
        const auto post_proc = toml::find(toml_input, "PostProcessing");
        if (post_proc.contains("volume_averages")) {
            auto modern_options = VolumeAverageOptions::from_toml(
                toml::find(post_proc, "volume_averages"));
            // Only override legacy settings if modern ones are explicitly enabled, but
            // without legacy settings the modern table is always used so that
            // enabled = false turns volume averaging off
            if (modern_options.enabled || !has_legacy) {
                options = modern_options;
            }
        }
    }

    return options;
}

bool VolumeAverageOptions::validate() const {
    // Implement validation logic
    if (!enabled) {
        return true;
    }
    if (output_frequency < 1) {
        WARNING_0_OPT("Error: VolumeAverage table did not provide a valid output frequency valid "
                      "as it was less than 1");
        return false;
    }
    if (!statistics_valid) {
        return false;
    }
    // Histograms are of enabled built-in averages, by the same names as their switches
    const std::map<std::string, bool> enabled_quantities = {{"stress", stress},
                                                            {"def_grad", def_grad},
                                                            {"euler_strain", euler_strain},
                                                            {"plastic_work", plastic_work},
                                                            {"eq_pl_strain", eq_pl_strain},
                                                            {"elastic_strain", elastic_strain}};
    for (const auto& [quantity, histogram] : histograms) {
        const auto known = enabled_quantities.find(quantity);
        if (known == enabled_quantities.end() || !known->second) {
            std::ostringstream err;
            err << "Error: [PostProcessing.volume_averages.histograms] entry '" << quantity
                << "' is not an enabled volume average (stress, def_grad, euler_strain, "
                   "plastic_work, eq_pl_strain, elastic_strain)";
            WARNING_0_OPT(err.str());
            return false;
        }
        if (!histogram.validate("[PostProcessing.volume_averages.histograms] " + quantity)) {
            return false;
        }
    }
    return true;
}

HistogramOptions HistogramOptions::from_toml(const toml::value& toml_input) {
    HistogramOptions options;
    options.requested = true;
    // Expected: { bins = N, range = [lower, upper] }; anything else is reported by validate()
    if (!toml_input.is_table() || !toml_input.contains("bins") || !toml_input.contains("range")) {
        options.well_formed = false;
        return options;
    }
    options.bins = toml::find<int>(toml_input, "bins");
    const auto range = toml::find<std::vector<double>>(toml_input, "range");
    if (range.size() == 2) {
        options.range_min = range[0];
        options.range_max = range[1];
    } else {
        options.well_formed = false;
    }
    return options;
}

bool HistogramOptions::validate(const std::string& where) const {
    std::ostringstream err;
    if (!well_formed || bins < 1) {
        err << "Error: " << where
            << " histogram needs bins >= 1 and range = [lower, upper] (two values)";
    } else if (!(range_min < range_max)) {
        err << "Error: " << where << " histogram range must be [lower, upper] with lower < upper";
    } else {
        return true;
    }
    WARNING_0_OPT(err.str());
    return false;
}

const std::vector<std::string>& volume_statistic_names() {
    static const std::vector<std::string> names = {
        "mean", "variance", "std", "skewness", "kurtosis", "min", "max"};
    return names;
}

bool parse_volume_statistics(const toml::value& toml_input,
                             const std::string& where,
                             std::vector<std::string>& statistics) {
    statistics.clear();
    if (!toml_input.contains("statistics")) {
        return true;
    }
    std::vector<std::string> requested;
    if (toml_input.at("statistics").is_string()) {
        requested = {toml::find<std::string>(toml_input, "statistics")};
    } else {
        requested = toml::find<std::vector<std::string>>(toml_input, "statistics");
    }
    const auto& known = volume_statistic_names();
    for (const auto& name : requested) {
        if (name == "all") {
            if (requested.size() > 1) {
                WARNING_0_OPT("Error: " + where +
                              " statistics = \"all\" cannot be combined with other names");
                return false;
            }
            statistics = known;
            return true;
        }
        if (std::find(known.begin(), known.end(), name) == known.end()) {
            std::ostringstream err;
            err << "Error: " << where << " statistics entry '" << name
                << "' is not a statistic. Known: all";
            for (const auto& k : known) {
                err << ", " << k;
            }
            WARNING_0_OPT(err.str());
            return false;
        }
        if (std::find(statistics.begin(), statistics.end(), name) == statistics.end()) {
            statistics.push_back(name);
        }
    }
    return true;
}

ProjectionOptions ProjectionOptions::from_toml(const toml::value& toml_input) {
    ProjectionOptions options;

    if (toml_input.contains("enabled_projections")) {
        options.enabled_projections = toml::find<std::vector<std::string>>(toml_input,
                                                                           "enabled_projections");
    }

    if (toml_input.contains("auto_enable_compatible")) {
        options.auto_enable_compatible = toml::find<bool>(toml_input, "auto_enable_compatible");
    }

    return options;
}

int ProjectionOptions::region_number(const std::string& projection) {
    constexpr std::string_view prefix = "region_";
    const std::string_view entry = projection;
    if (entry.substr(0, prefix.size()) != prefix) {
        return -1;
    }
    // The rest must be a whole number and nothing else
    const std::string_view digits = entry.substr(prefix.size());
    int number = -1;
    const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), number);
    if (digits.empty() || error != std::errc() || end != digits.data() + digits.size()) {
        return -1;
    }
    return number;
}

bool ProjectionOptions::validate() const {
    // Names are checked by PostProcessingDriver::RegisterProjection (the one list of built-in
    // projections); region_<N> numbers are checked in ExaOptions::validate, which knows the
    // number of materials.
    return true;
}

UserFunctionOptions UserFunctionOptions::from_toml(const toml::value& toml_input) {
    UserFunctionOptions options;

    if (toml_input.contains("library_path")) {
        options.library_path = toml::find<std::string>(toml_input, "library_path");
    }
    if (toml_input.contains("function_name")) {
        options.function_name = toml::find<std::string>(toml_input, "function_name");
    }
    if (toml_input.contains("type")) {
        options.type = string_to_user_function_type(toml::find<std::string>(toml_input, "type"));
    }
    if (toml_input.contains("component_names")) {
        options.component_names = toml::find<std::vector<std::string>>(toml_input,
                                                                       "component_names");
    }
    if (toml_input.contains("regions")) {
        options.regions = toml::find<std::vector<int>>(toml_input, "regions");
    }
    if (toml_input.contains("global")) {
        options.global = toml::find<bool>(toml_input, "global");
    }
    options.statistics_valid = parse_volume_statistics(
        toml_input, "[[PostProcessing.user_functions]]", options.statistics);
    if (toml_input.contains("histogram")) {
        options.histogram = HistogramOptions::from_toml(toml::find(toml_input, "histogram"));
    }

    // Defaults: name from the function, component count from the labels, labels from the name
    options.name = toml_input.contains("name") ? toml::find<std::string>(toml_input, "name")
                                               : options.function_name;
    if (toml_input.contains("num_components")) {
        options.num_components = toml::find<int>(toml_input, "num_components");
    } else if (!options.component_names.empty()) {
        options.num_components = static_cast<int>(options.component_names.size());
    }
    if (options.component_names.empty() && options.num_components >= 1) {
        if (options.num_components == 1) {
            options.component_names = {options.name};
        } else {
            for (int i = 1; i <= options.num_components; i++) {
                options.component_names.push_back(options.name + "_" + std::to_string(i));
            }
        }
    }

    return options;
}

bool UserFunctionOptions::validate(size_t num_materials) const {
    const auto fail = [this](const std::string& problem) {
        std::ostringstream err;
        err << "Error: [[PostProcessing.user_functions]] entry '"
            << (name.empty() ? function_name : name) << "' " << problem;
        WARNING_0_OPT(err.str());
        return false;
    };

    if (function_name.empty()) {
        return fail("needs a function_name");
    }
    if (type == UserFunctionType::NOTYPE) {
        return fail("needs type = \"visualization\", \"volume_average\" or \"volume_integral\"");
    }
    if (num_components < 1) {
        return fail("num_components must be at least 1");
    }
    if (static_cast<int>(component_names.size()) != num_components) {
        return fail("has " + std::to_string(component_names.size()) +
                    " component_names but num_components = " + std::to_string(num_components));
    }
    if (!statistics_valid) {
        return false;
    }
    const bool wants_distribution = !statistics.empty() || histogram.requested;
    if (wants_distribution && type != UserFunctionType::VOLUME_AVERAGE) {
        return fail("asks for statistics or a histogram, which only type = \"volume_average\" "
                    "entries have");
    }
    if (histogram.requested &&
        !histogram.validate("[[PostProcessing.user_functions]] entry '" + name + "'")) {
        return false;
    }
    // The name becomes part of file names
    if (name.find_first_of(":/\\") != std::string::npos) {
        return fail("name must not contain ':', '/' or '\\' (it is used in file names)");
    }
    std::set<int> seen_regions;
    for (const int region : regions) {
        if (region < 1 || static_cast<size_t>(region) > num_materials) {
            return fail("region " + std::to_string(region) + " is not a material number (1 to " +
                        std::to_string(num_materials) + ")");
        }
        if (!seen_regions.insert(region).second) {
            return fail("lists region " + std::to_string(region) + " more than once");
        }
    }
    return true;
}

SurfaceQuantityOptions SurfaceQuantityOptions::from_toml(const toml::value& toml_input) {
    SurfaceQuantityOptions options;

    if (toml_input.contains("enabled")) {
        options.enabled = toml::find<bool>(toml_input, "enabled");
    }

    if (toml_input.contains("boundary_ids")) {
        options.boundary_ids = toml::find<std::vector<int>>(toml_input, "boundary_ids");
    }

    if (toml_input.contains("moments")) {
        options.moments = toml::find<bool>(toml_input, "moments");
    }

    if (toml_input.contains("output_frequency")) {
        options.output_frequency = toml::find<int>(toml_input, "output_frequency");
    }

    if (toml_input.contains("fname")) {
        options.fname = toml::find<std::string>(toml_input, "fname");
    }

    return options;
}

bool SurfaceQuantityOptions::validate() const {
    if (!enabled) {
        return true;
    }
    if (boundary_ids.empty()) {
        WARNING_0_OPT("Error: SurfaceQuantities table is enabled but boundary_ids is empty. "
                      "List the mesh boundary attributes to report on, e.g. "
                      "boundary_ids = [1, 4]");
        return false;
    }
    std::vector<int> sorted_ids(boundary_ids);
    std::sort(sorted_ids.begin(), sorted_ids.end());
    if (sorted_ids.front() < 1) {
        WARNING_0_OPT("Error: SurfaceQuantities table boundary_ids must be positive mesh "
                      "boundary attributes");
        return false;
    }
    if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) != sorted_ids.end()) {
        WARNING_0_OPT("Error: SurfaceQuantities table boundary_ids contains duplicate IDs");
        return false;
    }
    if (output_frequency < 1) {
        WARNING_0_OPT("Error: SurfaceQuantities table did not provide a valid output frequency "
                      "as it was less than 1");
        return false;
    }
    if (fname.empty()) {
        WARNING_0_OPT("Error: SurfaceQuantities table fname must not be empty");
        return false;
    }
    return true;
}

PostProcessingOptions PostProcessingOptions::from_toml(const toml::value& toml_input) {
    PostProcessingOptions options;

    // Use the new legacy-aware parsing for volume averages
    options.volume_averages = VolumeAverageOptions::from_toml_with_legacy(toml_input);

    // Use the new legacy-aware parsing for light-up options
    options.light_up_configs = LightUpOptions::from_toml_with_legacy(toml_input);

    // Handle projections (existing code)
    if (toml_input.contains("PostProcessing")) {
        const auto post_proc = toml::find(toml_input, "PostProcessing");
        if (post_proc.contains("projections")) {
            options.projections = ProjectionOptions::from_toml(
                toml::find(post_proc, "projections"));
        }
        if (post_proc.contains("surface_quantities")) {
            options.surface_quantities = SurfaceQuantityOptions::from_toml(
                toml::find(post_proc, "surface_quantities"));
        }
        if (post_proc.contains("user_functions")) {
            for (const auto& entry :
                 toml::find<std::vector<toml::value>>(post_proc, "user_functions")) {
                options.user_functions.push_back(UserFunctionOptions::from_toml(entry));
            }
        }
    }

    return options;
}

bool PostProcessingOptions::validate() const {
    // Validate volume averages and projections
    if (!volume_averages.validate())
        return false;
    if (!projections.validate())
        return false;
    if (!surface_quantities.validate())
        return false;

    // Validate each light_up configuration
    for (const auto& light_config : light_up_configs) {
        if (!light_config.validate())
            return false;
    }
    return true;
}

std::vector<LightUpOptions> PostProcessingOptions::get_enabled_light_up_configs() const {
    std::vector<LightUpOptions> enabled_configs;
    for (const auto& config : light_up_configs) {
        if (config.enabled && config.region_id.has_value()) {
            enabled_configs.push_back(config);
        }
    }
    return enabled_configs;
}

LightUpOptions* PostProcessingOptions::get_light_up_config_for_region(int region_id) {
    for (auto& config : light_up_configs) {
        if (config.enabled && config.region_id == region_id) {
            return &config;
        }
    }
    return nullptr;
}

const LightUpOptions* PostProcessingOptions::get_light_up_config_for_region(int region_id) const {
    for (const auto& config : light_up_configs) {
        if (config.enabled && config.region_id == region_id) {
            return &config;
        }
    }
    return nullptr;
}