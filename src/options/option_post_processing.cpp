#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"

#include <iostream>
#include <set>

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
        WARNING_0_OPT("Warning: LightUp configuration missing 'material_name' field");
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
                            WARNING_0_OPT("Warning: LightUp config in array missing material_name. "
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

    return options;
}

bool VisualizationOptions::validate() const {
    if (output_frequency < 1) {
        WARNING_0_OPT("Error: Visualizations table did not provide a valid output frequency valid "
                      "as it was less than 1");
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

bool ProjectionOptions::validate() const {
    // Implement validation logic
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
    }

    return options;
}

bool PostProcessingOptions::validate() const {
    // Validate volume averages and projections
    if (!volume_averages.validate())
        return false;
    if (!projections.validate())
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