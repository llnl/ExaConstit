#include "options/option_parser_v2.hpp"

#include "options/option_util.hpp"

#include "mfem.hpp"

#include "ECMech_cases.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "TOML_Reader/toml.hpp"

namespace fs = std::filesystem;

namespace {

void apply_exacmech_grain_defaults(MaterialOptions& material) {
    if (material.mech_type != MechType::EXACMECH || !material.model.exacmech ||
        !material.grain_info) {
        return;
    }

    auto& grain = material.grain_info.value();
    auto index_map = ecmech::modelParamIndexMap(material.model.exacmech->shortcut);

    if (grain.ori_state_var_loc < 0) {
        grain.ori_state_var_loc = static_cast<int>(index_map["index_lattice_ori"]);
    }
    if (grain.ori_stride == 0) {
        grain.ori_stride = 4;
    }
    if (grain.ori_type == OriType::NOTYPE) {
        grain.ori_type = OriType::QUAT;
    }
}

void finalize_material_options(ExaOptions& options) {
    int max_grains = -1;
    int index = 0;

    for (auto& mat : options.materials) {
        apply_exacmech_grain_defaults(mat);

        // Grain info (if crystal plasticity)
        if (mat.grain_info.has_value()) {
            const auto& grain = mat.grain_info.value();
            if (grain.orientation_file.has_value()) {
                if (!options.orientation_file.has_value()) {
                    options.orientation_file = grain.orientation_file.value();
                }
                if (grain.orientation_file.value().compare(options.orientation_file.value()) != 0) {
                    MFEM_ABORT("Check material grain tables as orientation files in there are not "
                               "consistent between values listed elsewhere");
                }
            }

            if (grain.grain_file.has_value()) {
                if (!options.grain_file.has_value()) {
                    options.grain_file = grain.grain_file.value();
                }
                if (grain.grain_file.value().compare(options.grain_file.value()) != 0) {
                    MFEM_ABORT("Check material grain tables as grain files in there are not "
                               "consistent between values listed elsewhere");
                }
            }

            if (max_grains < grain.num_grains && index > 0) {
                MFEM_ABORT("Check material grain tables as values in there are not consistent "
                           "between multiple materials");
            }

            max_grains = grain.num_grains;
            index++;
        }
    }
}

/**
 * @brief Create the targets of any dangling symlinks along a path
 *
 * @param path Path to check component by component
 * @param depth Current recursion depth (guards against symlink cycles)
 * @return false if a dangling symlink's target could not be created
 *
 * @details create_directories() does not create the target of a dangling symlink, it
 * just fails since the link itself already exists. A dangling link in the output path
 * is a realistic setup though, e.g. `ln -s /remote/storage/location results` done before
 * the storage directory itself has been created. So walk the path and create the
 * target of every dangling link along it (which may itself contain dangling links).
 */
bool create_dangling_symlink_targets(const fs::path& path, int depth = 0) {
    constexpr int max_depth = 40;
    if (depth > max_depth) {
        return false;
    }
    fs::path prefix;
    for (const auto& part : path) {
        prefix /= part;
        std::error_code ec;
        const auto link_status = fs::symlink_status(prefix, ec);
        if (ec || !fs::is_symlink(link_status) || fs::exists(prefix, ec)) {
            continue;
        }
        fs::path target = fs::read_symlink(prefix, ec);
        if (ec) {
            return false;
        }
        if (target.is_relative()) {
            target = prefix.parent_path() / target;
        }
        if (!create_dangling_symlink_targets(target, depth + 1)) {
            return false;
        }
        fs::create_directories(target, ec);
        if (ec) {
            return false;
        }
    }
    return true;
}

} // namespace

// Implementation of the struct conversion methods
void ExaOptions::parse_options(const std::string& filename, int my_id) {
    try {
        // Parse the main TOML file
        {
            fs::path fpath(filename);
            basename = fpath.stem().string();
        }
        toml::value toml_input = toml::parse(filename);

        // Parse the full configuration
        parse_from_toml(toml_input);

        // Validate the complete configuration
        if (!validate()) {
            WARNING_0_OPT("Error: Configuration validation failed.");
            if (my_id == 0) {
                mfem::mfem_error("MFEM_ABORT: Configuration validation failed for option file");
            } else {
                mfem::mfem_error("");
            }
        }
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Error parsing options: " << e.what();
        std::string err = oss.str();
        WARNING_0_OPT(err);
        if (my_id == 0) {
            mfem::mfem_error("MFEM_ABORT: Configuration validation failed for option file");
        } else {
            mfem::mfem_error("");
        }
    }

    // Resolve the output directory exactly once, on rank 0, and broadcast it so every
    // rank and every later consumer (post-processing, logger, visualization) works from
    // the same absolute path. Previously each rank resolved symlinks in this path on its
    // own, several times over the course of a run, and paired filesystem checks with
    // later actions (exists() followed by create_directories(), etc.). With the output
    // directory being a symlink to another file system, ranks could see different
    // states of that path between those checks and actions (a TOCTOU race).
    //
    // The directory is created *before* it is resolved: create_directories() follows a
    // symlink that is already in place and otherwise creates a real directory, and
    // canonical() then returns the physical directory that now exists. Nothing after
    // this point goes back through a symlink in the user provided path, so another run
    // (e.g. a different ExaConstit run on the same node) creating or changing that
    // symlink later on cannot redirect where this run writes. If the path changes in
    // the brief window between the two calls, canonical() fails rather than silently
    // resolving to somewhere unexpected, and we report it.
    //
    // All error paths above abort on every rank, so every rank reaches this collective.
    //
    // If the output directory can't be created or resolved, every rank aborts here rather
    // than running the whole simulation only to have none of its output saved.
    auto& output_dir = post_processing.volume_averages.output_directory;
    std::string resolved_str;
    bool output_ok = true;
    if (my_id == 0) {
        fs::path dir = output_dir.empty() ? fs::path(".") : output_dir;
        std::error_code ec;
        fs::path abs_dir = fs::absolute(dir, ec);
        if (ec) {
            abs_dir = dir;
        }
        fs::path resolved;
        if (!create_dangling_symlink_targets(abs_dir)) {
            std::cerr << "Error: The output directory " << abs_dir
                      << " contains a symlink whose target does not exist and could not be "
                         "created"
                      << std::endl;
            output_ok = false;
        } else {
            fs::create_directories(abs_dir, ec);
            if (!ec) {
                resolved = fs::canonical(abs_dir, ec);
            }
            if (!ec && !fs::is_directory(resolved, ec)) {
                ec = std::make_error_code(std::errc::not_a_directory);
            }
            if (ec || resolved.empty()) {
                std::cerr << "Error: Unable to create and resolve the output directory " << abs_dir
                          << ": " << ec.message() << std::endl;
                output_ok = false;
            }
        }
        resolved_str = resolved.string();
    }
    MPI_Bcast(&output_ok, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!output_ok) {
        if (my_id == 0) {
            mfem::mfem_error("MFEM_ABORT: Unable to set up the output directory");
        } else {
            mfem::mfem_error("");
        }
    }
    int length = static_cast<int>(resolved_str.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    resolved_str.resize(static_cast<size_t>(length));
    MPI_Bcast(resolved_str.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
    output_dir = resolved_str;
}

void ExaOptions::parse_from_toml(const toml::value& toml_input) {
    // Parse basic metadata
    if (toml_input.contains("Version")) {
        version = toml::find<std::string>(toml_input, "Version");
    }

    if (toml_input.contains("basename")) {
        basename = toml::find<std::string>(toml_input, "basename");
    }

    // Check for modular configuration
    if (toml_input.contains("materials")) {
        auto mat_files = toml::find<std::vector<std::string>>(toml_input, "materials");
        for (auto& mat_file : mat_files) {
            material_files.push_back(mat_file);
        }
    }

    if (toml_input.contains("post_processing")) {
        post_processing_file = toml::find<std::string>(toml_input, "post_processing");
    }

    if (toml_input.contains("grain_file")) {
        grain_file = toml::find<std::string>(toml_input, "grain_file");
    }
    if (toml_input.contains("orientation_file")) {
        orientation_file = toml::find<std::string>(toml_input, "orientation_file");
    }

    // New fields for optional region mapping
    if (toml_input.contains("region_mapping_file")) {
        region_mapping_file = toml::find<std::string>(toml_input, "region_mapping_file");
    }

    // Parse component sections
    parse_mesh_options(toml_input);
    parse_time_options(toml_input);
    parse_solver_options(toml_input);
    parse_boundary_options(toml_input);
    parse_visualization_options(toml_input);

    // Parse materials from main file if no external files are specified
    if (material_files.empty()) {
        parse_material_options(toml_input);
    } else {
        load_material_files();
    }
    finalize_material_options(*this);

    // Parse post-processing from main file if no external file is specified
    if (!post_processing_file) {
        parse_post_processing_options(toml_input);
    } else {
        load_post_processing_file();
    }
}

void ExaOptions::parse_mesh_options(const toml::value& toml_input) {
    if (toml_input.contains("Mesh")) {
        mesh = MeshOptions::from_toml(toml::find(toml_input, "Mesh"));
    }
}

void ExaOptions::parse_time_options(const toml::value& toml_input) {
    if (!toml_input.contains("Time")) {
        return;
    }

    // Same parsing (restart options, nested sections, time_type) as TimeOptions::from_toml
    time = TimeOptions::from_toml(toml::find(toml_input, "Time"));
}

void ExaOptions::parse_solver_options(const toml::value& toml_input) {
    if (toml_input.contains("Solvers")) {
        solvers = SolverOptions::from_toml(toml::find(toml_input, "Solvers"));
    }
}

void ExaOptions::parse_material_options(const toml::value& toml_input) {
    // Check for materials array under "Materials" section
    if (toml_input.contains("Materials")) {
        auto materials_section = toml::find(toml_input, "Materials");
        materials = MaterialOptions::from_toml_array(materials_section);
    }
    // Legacy format - material properties directly in Properties section
    else if (toml_input.contains("Properties")) {
        MaterialOptions single_material;

        // Parse properties section
        if (toml_input.at("Properties").contains("Properties")) {
            single_material.properties = MaterialProperties::from_toml(
                toml::find(toml_input.at("Properties"), "Properties"));
        }
        // Parse material variables if present
        else if (toml_input.at("Properties").contains("Matl_Props")) {
            single_material.properties = MaterialProperties::from_toml(
                toml::find(toml_input.at("Properties"), "Matl_Props"));
        } else {
            single_material.properties = MaterialProperties::from_toml(
                toml::find(toml_input, "Properties"));
        }

        // Parse global temperature if present
        if (toml_input.at("Properties").contains("temperature")) {
            const auto props = toml_input.at("Properties");
            if (props.at("temperature").is_integer()) {
                single_material.temperature = static_cast<double>(
                    toml::find<int>(props, "temperature"));
            } else {
                single_material.temperature = toml::find<double>(props, "temperature");
            }
        }

        // Parse state variables if present
        if (toml_input.at("Properties").contains("State_Vars")) {
            single_material.state_vars = StateVariables::from_toml(
                toml::find(toml_input.at("Properties"), "State_Vars"));
        }

        // Parse grain info if present
        if (toml_input.at("Properties").contains("Grain")) {
            single_material.grain_info = GrainInfo::from_toml(
                toml::find(toml_input.at("Properties"), "Grain"));
        }

        // Try to determine model type and options
        if (toml_input.contains("Model")) {
            parse_model_options(toml_input, single_material);
        }

        // Add the single material
        materials.push_back(single_material);
    }
}

void ExaOptions::parse_model_options(const toml::value& toml_input, MaterialOptions& material) {
    if (!toml_input.contains("Model")) {
        return;
    }

    const auto model_section = toml::find(toml_input, "Model");

    // Parse common model properties
    if (model_section.contains("mech_type")) {
        std::string mech_type_str = toml::find<std::string>(model_section, "mech_type");
        material.mech_type = string_to_mech_type(mech_type_str);
    }

    if (model_section.contains("cp")) {
        material.model.crystal_plasticity = toml::find<bool>(model_section, "cp");
    }

    // Parse ExaCMech-specific options
    if (material.mech_type == MechType::EXACMECH && model_section.contains("ExaCMech")) {
        material.model.exacmech = ExaCMechModelOptions::from_toml(
            toml::find(model_section, "ExaCMech"));

        // Validate that we have a valid shortcut (either directly or derived)
        std::string effective_shortcut = material.model.exacmech->get_effective_shortcut();

        if (effective_shortcut.empty()) {
            WARNING_0_OPT("Error: Invalid ExaCMech model configuration. Either shortcut or both "
                          "xtal_type and slip_type must be provided.");
        }
        // When using legacy parameters, set the derived shortcut for other code to use
        if (material.model.exacmech->shortcut.empty() && !effective_shortcut.empty()) {
            material.model.exacmech->shortcut = effective_shortcut;
        }

        auto index_map = ecmech::modelParamIndexMap(material.model.exacmech->shortcut);

        // add more checks later like
        material.model.exacmech->gdot_size = index_map["num_slip_system"];
        material.model.exacmech->hard_size = index_map["num_hardening"];
    }

    // Check for legacy format where mech_type was in Model section
    if (material.mech_type == MechType::UMAT) {
        // If mech_type is "umat" and there's no UMAT subsection, create default UMAT options
        if (!model_section.contains("UMAT")) {
            // Create UMAT options with defaults for legacy format
            material.model.umat = UmatOptions{};
            // The defaults in UmatOptions should handle the rest
        }
    }

    // Parse UMAT-specific options
    else if (material.mech_type == MechType::UMAT && model_section.contains("UMAT")) {
        material.model.umat = UmatOptions::from_toml(toml::find(model_section, "UMAT"));
    }
}

void ExaOptions::parse_boundary_options(const toml::value& toml_input) {
    if (toml_input.contains("BCs")) {
        boundary_conditions = BoundaryOptions::from_toml(toml::find(toml_input, "BCs"));

        // Transform and validate
        const bool bc_check = boundary_conditions.validate();
        if (!bc_check) {
            throw std::runtime_error("BC validation failed");
        }
    }
}

void ExaOptions::parse_visualization_options(const toml::value& toml_input) {
    if (toml_input.contains("Visualizations")) {
        visualization = VisualizationOptions::from_toml(toml::find(toml_input, "Visualizations"));
    }
}

void ExaOptions::parse_post_processing_options(const toml::value& toml_input) {
    post_processing = PostProcessingOptions::from_toml(toml_input);
}

void ExaOptions::load_material_files() {
    materials.clear();

    for (const auto& file_path : material_files) {
        try {
            toml::value mat_toml = toml::parse(file_path);

            // Parse the material
            auto material = MaterialOptions::from_toml(mat_toml);

            // Parse model options separately to handle the shortcut derivation
            if (mat_toml.contains("Model")) {
                parse_model_options(mat_toml, material);
            }

            // Add the material to our list
            materials.push_back(material);

        } catch (const std::exception& e) {
            std::ostringstream err;
            err << "Error parsing material file " << file_path << ": " << e.what();
            WARNING_0_OPT(err.str());
            throw; // Re-throw to propagate the error
        }
    }
}

void ExaOptions::load_post_processing_file() {
    if (post_processing_file.has_value()) {
        try {
            toml::value pp_toml = toml::parse(post_processing_file.value());
            post_processing = PostProcessingOptions::from_toml(pp_toml);
        } catch (const std::exception& e) {
            std::ostringstream err;
            err << "Error parsing post-processing file " << post_processing_file.value() << ": "
                << e.what();
            WARNING_0_OPT(err.str());
            throw; // Re-throw to propagate the error
        }
    }
}

bool ExaOptions::validate() {
    // Basic validation - could be expanded with more comprehensive checks

    if (!mesh.validate())
        return false;
    if (!time.validate())
        return false;
    if (!solvers.validate())
        return false;
    if (!visualization.validate())
        return false;
    if (!boundary_conditions.validate())
        return false;

    // Check that we have at least one material
    if (materials.empty()) {
        WARNING_0_OPT("Error: No materials defined in configuration.");
        return false;
    }

    if (materials.size() > 1) {
        if (!region_mapping_file) {
            WARNING_0_OPT("Error: region_mapping_file was not provided even though multiple "
                          "materials were asked for.");
            return false;
        } else if (mesh.mesh_type == MeshType::AUTO && !grain_file) {
            WARNING_0_OPT("Error: region_mapping_file was provided but no grain_file was provided "
                          "when using auto mesh.");
            return false;
        }
    }

    if (materials.size() > 1) {
        if (!region_mapping_file) {
            WARNING_0_OPT("Error: region_mapping_file was not provided even though multiple "
                          "materials were asked for.");
            return false;
        } else if (mesh.mesh_type == MeshType::AUTO && !grain_file) {
            WARNING_0_OPT("Error: region_mapping_file was provided but no grain_file was provided "
                          "when using auto mesh.");
            return false;
        }
    }

    size_t index = 1;
    for (auto& mat : materials) {
        if (!mat.validate())
            return false;
        // Update the region_id value after validating
        // everything so to make it easier for users to
        // validation errors
        mat.region_id = static_cast<int>(index++);
    }

    // Handle legacy "default_material" mapping
    // If we have light_up configs with "default_material" and only one material, map them
    bool has_default_material_configs = false;
    for (const auto& light_config : post_processing.light_up_configs) {
        if (light_config.material_name == "default_material") {
            has_default_material_configs = true;
            break;
        }
    }

    if (has_default_material_configs) {
        if (materials.size() == 1) {
            // Single material case: map default_material to the actual material name
            std::string actual_material_name = materials[0].material_name;
            for (auto& light_config : post_processing.light_up_configs) {
                if (light_config.material_name == "default_material") {
                    light_config.material_name = actual_material_name;
                    std::ostringstream info;
                    info << "Info: Mapped default_material to '" << actual_material_name << "'";
                    INFO_0_OPT(info.str());
                }
            }
        } else {
            // Multiple materials: error - can't auto-resolve default_material
            WARNING_0_OPT("Error: Found default_material in light_up config but multiple materials "
                          "defined. ");
            WARNING_0_OPT("Please specify explicit material_name for each light_up configuration.");
            return false;
        }
    }

    // Resolve light_up material names to region IDs
    for (auto& light_config : post_processing.light_up_configs) {
        if (light_config.enabled) {
            if (!light_config.resolve_region_id(materials)) {
                return false;
            }
        }
    }

    // Build material-to-regions map for efficiency
    std::unordered_multimap<std::string, int> material_to_regions;
    for (const auto& region : materials) {
        material_to_regions.emplace(region.material_name, region.region_id);
    }

    // Track which light-up configurations already exist to avoid duplicates
    std::set<std::pair<std::string, int>> existing_configs;

    // First pass: collect existing configurations
    for (const auto& lightup : post_processing.light_up_configs) {
        existing_configs.insert(std::make_pair(lightup.material_name, lightup.region_id.value()));
    }

    // Create a temporary vector to hold new LightUpOptions
    std::vector<LightUpOptions> additional_lightup_options;

    // Second pass: process each light-up option
    for (auto& lightup : post_processing.light_up_configs) {
        // Mark original options as user-generated
        lightup.is_auto_generated = false;

        // Find all regions with this material
        auto range = material_to_regions.equal_range(lightup.material_name);
        std::vector<int> regions_with_material;

        for (auto it = range.first; it != range.second; ++it) {
            regions_with_material.push_back(it->second);
        }

        if (regions_with_material.empty()) {
            std::ostringstream info;
            info << "Error: PostProcessing.light_up material '" << lightup.material_name
                 << "' not found in any region";
            WARNING_0_OPT(info.str());
            return false;
        }

        // Sort for consistent ordering
        std::sort(regions_with_material.begin(), regions_with_material.end());

        // Update the original lightup with the first region ID
        lightup.region_id = regions_with_material[0];

        // Create duplicates for remaining regions if they don't already exist
        for (size_t i = 1; i < regions_with_material.size(); ++i) {
            int target_region_id = regions_with_material[i];

            // Check if this configuration already exists
            auto config_key = std::make_pair(lightup.material_name, target_region_id);
            if (existing_configs.find(config_key) == existing_configs.end()) {
                // Create duplicate
                LightUpOptions duplicate = lightup;
                duplicate.region_id = target_region_id;
                duplicate.is_auto_generated = true; // Mark as auto-generated

                additional_lightup_options.push_back(duplicate);
                existing_configs.insert(config_key); // Track that we've added this
            }
        }
    }

    // Append the duplicated options to the main vector
    if (!additional_lightup_options.empty()) {
        post_processing.light_up_configs.insert(post_processing.light_up_configs.end(),
                                                additional_lightup_options.begin(),
                                                additional_lightup_options.end());
    }

    // Validate post-processing after region resolution
    if (!post_processing.validate())
        return false;

    if (region_mapping_file) {
        if (!std::filesystem::exists(*region_mapping_file)) {
            std::ostringstream err;
            err << "Error: Region mapping file: " << *region_mapping_file << " does not exist";
            WARNING_0_OPT(err.str());
            return false;
        }
    }

    if (grain_file) {
        if (!std::filesystem::exists(*grain_file)) {
            std::ostringstream err;
            err << "Error: Grain file provided at top level: " << *grain_file << " does not exist";
            WARNING_0_OPT(err.str());
            return false;
        }
    }

    if (orientation_file) {
        if (!std::filesystem::exists(*orientation_file)) {
            std::ostringstream err;
            err << "Error: Orientation file provided at top level: " << *orientation_file
                << " does not exist";
            WARNING_0_OPT(err.str());
            return false;
        }
    }

    return true;
}

// Implementation of validation methods for component structs

void ExaOptions::print_options() const {
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid != 0)
        return; // Only print from rank 0

    std::cout << "\n==================================================\n";
    std::cout << "ExaConstit Options Summary\n";
    std::cout << "==================================================\n";

    // Basic info
    std::cout << "\nSimulation Information:\n";
    std::cout << "  Base name: " << basename << "\n";
    std::cout << "  Version: " << version << "\n";

    // Print each component
    print_mesh_options();
    print_time_options();
    print_solver_options();
    print_material_options();
    print_boundary_options();
    print_visualization_options();
    print_post_processing_options();

    // Configuration files
    if (!material_files.empty() || post_processing_file.has_value() ||
        orientation_file.has_value() || grain_file.has_value() || region_mapping_file.has_value()) {
        std::cout << "\nConfiguration Files:\n";

        if (!material_files.empty()) {
            std::cout << "  Material files:\n";
            for (const auto& file : material_files) {
                std::cout << "    - " << file << "\n";
            }
        }

        if (post_processing_file.has_value()) {
            std::cout << "  Post-processing file: " << post_processing_file.value() << "\n";
        }

        if (orientation_file.has_value()) {
            std::cout << "  Orientation file: " << orientation_file.value() << "\n";
        }

        if (grain_file.has_value()) {
            std::cout << "  Grain file: " << grain_file.value() << "\n";
        }

        if (region_mapping_file.has_value()) {
            std::cout << "  Region mapping file: " << region_mapping_file.value() << "\n";
        }
    }

    std::cout << "\n==================================================\n\n";
}

void ExaOptions::print_mesh_options() const {
    std::cout << "\nMesh Options:\n";

    if (mesh.mesh_type == MeshType::FILE) {
        std::cout << "  Type: File-based mesh\n";
        std::cout << "  Mesh file: " << mesh.mesh_file << "\n";
    } else if (mesh.mesh_type == MeshType::AUTO) {
        std::cout << "  Type: Auto-generated mesh\n";
        std::cout << "  Dimensions (nx, ny, nz): " << mesh.nxyz[0] << " x " << mesh.nxyz[1] << " x "
                  << mesh.nxyz[2] << "\n";
        std::cout << "  Physical size (x, y, z): " << mesh.mxyz[0] << " x " << mesh.mxyz[1] << " x "
                  << mesh.mxyz[2] << "\n";
    }

    std::cout << "  Polynomial order: " << mesh.order << "\n";
    std::cout << "  Serial refinement levels: " << mesh.ref_ser << "\n";
    std::cout << "  Parallel refinement levels: " << mesh.ref_par << "\n";
    std::cout << "  Periodicity: " << (mesh.periodicity ? "Enabled" : "Disabled") << "\n";
}

void ExaOptions::print_time_options() const {
    std::cout << "\nTime Stepping Options:\n";

    // Which mode was picked and how: an explicit time_type wins over the section priority
    const char* selected_by = time.time_type_explicit
                                  ? "set by time_type"
                                  : "selected by section priority Custom > Auto > Fixed";

    if (time.time_type == TimeStepType::FIXED && time.fixed_time.has_value()) {
        std::cout << "  Type: Fixed time stepping (" << selected_by << ")\n";
        std::cout << "  Time step (dt): " << time.fixed_time->dt << "\n";
        std::cout << "  Final time: " << time.fixed_time->t_final << "\n";
    } else if (time.time_type == TimeStepType::AUTO && time.auto_time.has_value()) {
        std::cout << "  Type: Automatic time stepping (" << selected_by << ")\n";
        std::cout << "  Initial dt: " << time.auto_time->dt_start << "\n";
        std::cout << "  Minimum dt: " << time.auto_time->dt_min << "\n";
        std::cout << "  Maximum dt: " << time.auto_time->dt_max << "\n";
        std::cout << "  Scaling factor: " << time.auto_time->dt_scale << "\n";
        std::cout << "  Final time: " << time.auto_time->t_final << "\n";
    } else if (time.time_type == TimeStepType::CUSTOM && time.custom_time.has_value()) {
        std::cout << "  Type: Custom time stepping (" << selected_by << ")\n";
        std::cout << "  Number of steps: " << time.custom_time->nsteps << "\n";
        std::cout << "  Custom dt file: " << time.custom_time->floc << "\n";
        if (!time.custom_time->dt_values.empty()) {
            std::cout << "  Total simulation time: "
                      << std::accumulate(time.custom_time->dt_values.begin(),
                                         time.custom_time->dt_values.end(),
                                         0.0)
                      << "\n";
        }
    } else {
        std::cout << "  Type: none (time_type has no matching [Time.*] section)\n";
    }

    if (time.restart) {
        std::cout << "  Restart enabled:\n";
        std::cout << "    Restart time: " << time.restart_time << "\n";
        std::cout << "    Restart cycle: " << time.restart_cycle << "\n";
    }
}

void ExaOptions::print_solver_options() const {
    std::cout << "\nSolver Options:\n";

    // Assembly and runtime
    std::cout << "  Assembly type: ";
    switch (solvers.assembly) {
    case AssemblyType::FULL:
        std::cout << "Full assembly\n";
        break;
    case AssemblyType::PA:
        std::cout << "Partial assembly\n";
        break;
    case AssemblyType::EA:
        std::cout << "Element assembly\n";
        break;
    default:
        std::cout << "Unknown\n";
        break;
    }

    std::cout << "  Runtime model: ";
    switch (solvers.rtmodel) {
    case RTModel::CPU:
        std::cout << "CPU\n";
        break;
    case RTModel::OPENMP:
        std::cout << "OpenMP\n";
        break;
    case RTModel::GPU:
        std::cout << "GPU\n";
        break;
    default:
        std::cout << "Unknown\n";
        break;
    }

    std::cout << "  Integration model: ";
    switch (solvers.integ_model) {
    case IntegrationModel::DEFAULT:
        std::cout << "Default\n";
        break;
    case IntegrationModel::BBAR:
        std::cout << "B-bar\n";
        break;
    default:
        std::cout << "Unknown\n";
        break;
    }

    // Linear solver
    std::cout << "\n  Linear solver:\n";
    std::cout << "    Type: ";
    switch (solvers.linear_solver.solver_type) {
    case LinearSolverType::CG:
        std::cout << "Conjugate Gradient\n";
        break;
    case LinearSolverType::GMRES:
        std::cout << "GMRES\n";
        break;
    case LinearSolverType::MINRES:
        std::cout << "MINRES\n";
        break;
    case LinearSolverType::BICGSTAB:
        std::cout << "BiCGSTAB\n";
        break;
    default:
        std::cout << "Unknown\n";
        break;
    }

    std::cout << "    Preconditioner: ";
    switch (solvers.linear_solver.preconditioner) {
    case PreconditionerType::JACOBI:
        std::cout << "Jacobi\n";
        break;
    case PreconditionerType::AMG:
        std::cout << "AMG\n";
        break;
    case PreconditionerType::ILU:
        std::cout << "ILU\n";
        break;
    case PreconditionerType::L1GS:
        std::cout << "L1GS\n";
        break;
    case PreconditionerType::CHEBYSHEV:
        std::cout << "CHEBYSHEV\n";
        break;
    default:
        std::cout << "Unknown\n";
        break;
    }

    std::cout << "    Absolute tolerance: " << solvers.linear_solver.abs_tol << "\n";
    std::cout << "    Relative tolerance: " << solvers.linear_solver.rel_tol << "\n";
    std::cout << "    Maximum iterations: " << solvers.linear_solver.max_iter << "\n";
    std::cout << "    Print level: " << solvers.linear_solver.print_level << "\n";

    // Nonlinear solver
    std::cout << "\n  Nonlinear solver:\n";
    std::cout << "    Type: ";
    switch (solvers.nonlinear_solver.nl_solver) {
    case NonlinearSolverType::NR:
        std::cout << "Newton-Raphson\n";
        break;
    case NonlinearSolverType::NRLS:
        std::cout << "Newton-Raphson with line search\n";
        break;
    case NonlinearSolverType::TRDOG:
        std::cout << "Trust-region dogleg (SNLS port)\n";
        break;
    default:
        std::cout << "Unknown\n";
        break;
    }

    std::cout << "    Maximum iterations: " << solvers.nonlinear_solver.iter << "\n";
    std::cout << "    Relative tolerance: " << solvers.nonlinear_solver.rel_tol << "\n";
    std::cout << "    Absolute tolerance: " << solvers.nonlinear_solver.abs_tol << "\n";
    std::cout << "    Print info level: " << solvers.nonlinear_solver.print_info << "\n";

    // Trust-region parameters: print if either the solver is TRDOG or the user
    // supplied a [trust_region] sub-table. The latter case is informational —
    // it lets the user spot misconfigurations where they set TR options without
    // selecting the TRDOG solver.
    const bool is_trdog = (solvers.nonlinear_solver.nl_solver == NonlinearSolverType::TRDOG);
    const bool tr_supplied = solvers.nonlinear_solver.trust_region.has_value();

    if (is_trdog || tr_supplied) {
        std::cout << "\n    Trust-region parameters";
        if (is_trdog && !tr_supplied) {
            std::cout << " (using defaults)";
        } else if (!is_trdog && tr_supplied) {
            std::cout << " (WARNING: supplied but solver is not TRDOG)";
        }
        std::cout << ":\n";

        // Use the supplied options if present, otherwise default-construct
        // a TrustRegionOptions to print the defaults
        const TrustRegionOptions tr_opts = tr_supplied
                                               ? solvers.nonlinear_solver.trust_region.value()
                                               : TrustRegionOptions{};

        std::cout << "      delta_init      = " << tr_opts.delta_init << "\n";
        std::cout << "      delta_min       = " << tr_opts.delta_min << "\n";
        std::cout << "      delta_max       = " << tr_opts.delta_max << "\n";
        std::cout << "      xi_lg           = " << tr_opts.xi_lg << "\n";
        std::cout << "      xi_ug           = " << tr_opts.xi_ug << "\n";
        std::cout << "      xi_lo           = " << tr_opts.xi_lo << "\n";
        std::cout << "      xi_uo           = " << tr_opts.xi_uo << "\n";
        std::cout << "      xi_inc          = " << tr_opts.xi_inc << "\n";
        std::cout << "      xi_dec          = " << tr_opts.xi_dec << "\n";
        std::cout << "      xi_forced_inc   = " << tr_opts.xi_forced_inc << "\n";
        std::cout << "      reject_increase = " << (tr_opts.reject_increase ? "true" : "false")
                  << "\n";
    }
}

void ExaOptions::print_material_options() const {
    std::cout << "\nMaterial Options:\n";
    std::cout << "  Number of materials: " << materials.size() << "\n";

    for (size_t i = 0; i < materials.size(); ++i) {
        const auto& mat = materials[i];
        std::cout << "\n  Material " << i + 1 << ":\n";
        std::cout << "    Name: " << mat.material_name << "\n";
        std::cout << "    Region ID: " << mat.region_id << "\n";
        std::cout << "    Temperature: " << mat.temperature << " K\n";

        std::cout << "    Mechanics type: ";
        switch (mat.mech_type) {
        case MechType::UMAT:
            std::cout << "UMAT\n";
            break;
        case MechType::EXACMECH:
            std::cout << "ExaCMech\n";
            break;
        default:
            std::cout << "Unknown\n";
            break;
        }

        std::cout << "    Crystal plasticity: "
                  << (mat.model.crystal_plasticity ? "Enabled" : "Disabled") << "\n";

        // Material properties
        std::cout << "    Properties file: " << mat.properties.properties_file << "\n";
        std::cout << "    Number of properties: " << mat.properties.num_props << "\n";

        // State variables
        std::cout << "    State variables file: " << mat.state_vars.state_file << "\n";
        std::cout << "    Number of state variables: " << mat.state_vars.num_vars << "\n";

        // Grain info (if crystal plasticity)
        if (mat.grain_info.has_value()) {
            const auto& grain = mat.grain_info.value();
            std::cout << "    Grain information:\n";
            if (grain.orientation_file.has_value()) {
                std::cout << "      Orientation file: " << grain.orientation_file.value() << "\n";
            }
            if (grain.grain_file.has_value()) {
                std::cout << "      Grain file: " << grain.grain_file.value() << "\n";
            }
            std::cout << "      Number of grains: " << grain.num_grains << "\n";
            std::cout << "      Orientation type: ";
            switch (grain.ori_type) {
            case OriType::EULER:
                std::cout << "Euler angles\n";
                break;
            case OriType::QUAT:
                std::cout << "Quaternions\n";
                break;
            case OriType::CUSTOM:
                std::cout << "Custom\n";
                break;
            default:
                std::cout << "Unknown\n";
                break;
            }
            std::cout << "      Orientation state var location: " << grain.ori_state_var_loc
                      << "\n";
            std::cout << "      Orientation stride: " << grain.ori_stride << "\n";
        }

        // Model-specific options

        if (mat.model.umat.has_value() && mat.mech_type == MechType::UMAT) {
            const auto& umat = mat.model.umat.value();
            std::cout << "    UMAT options:\n";
            std::cout << "      Library: " << umat.library_path << "\n";
            std::cout << "      Function: " << umat.function_name << "\n";
            std::cout << "      Thermal: " << (umat.thermal ? "Enabled" : "Disabled") << "\n";
            std::cout << "      Dynamic loading: "
                      << (umat.enable_dynamic_loading ? "Enabled" : "Disabled") << "\n";
            std::cout << "      Load strategy: " << umat.load_strategy << "\n";
            if (!umat.search_paths.empty()) {
                std::cout << "      Search paths: ";
                for (size_t j = 0; j < umat.search_paths.size(); ++j) {
                    std::cout << umat.search_paths[j];
                    if (j < umat.search_paths.size() - 1)
                        std::cout << ", ";
                }
                std::cout << "\n";
            }
        }

        if (mat.model.exacmech.has_value() && mat.mech_type == MechType::EXACMECH) {
            const auto& ecmech = mat.model.exacmech.value();
            std::cout << "    ExaCMech options:\n";
            std::cout << "      Model: " << ecmech.get_effective_shortcut() << "\n";
            if (!ecmech.shortcut.empty()) {
                std::cout << "      Shortcut: " << ecmech.shortcut << "\n";
            } else {
                std::cout << "      Crystal type: " << ecmech.xtal_type << "\n";
                std::cout << "      Slip type: " << ecmech.slip_type << "\n";
            }
        }
    }
}

void ExaOptions::print_boundary_options() const {
    std::cout << "\nBoundary Conditions:\n";

    // Modern velocity BCs
    if (!boundary_conditions.velocity_bcs.empty()) {
        std::cout << "  Velocity boundary conditions: " << boundary_conditions.velocity_bcs.size()
                  << "\n";
        for (size_t i = 0; i < boundary_conditions.velocity_bcs.size(); ++i) {
            const auto& bc = boundary_conditions.velocity_bcs[i];
            std::cout << "    BC " << i + 1 << ":\n";

            // Print essential IDs
            std::cout << "      Essential IDs: ";
            for (const auto& id : bc.essential_ids) {
                std::cout << id << " ";
            }
            std::cout << "\n";

            // Print essential components
            std::cout << "      Essential components: ";
            for (const auto& comp : bc.essential_comps) {
                std::cout << comp << " ";
            }
            std::cout << "\n";

            // Print essential values - these are the actual velocity values
            std::cout << "      Essential values: ";
            for (const auto& val : bc.essential_vals) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    }

    // Velocity gradient BCs
    if (!boundary_conditions.vgrad_bcs.empty()) {
        std::cout << "  Velocity gradient boundary conditions: "
                  << boundary_conditions.vgrad_bcs.size() << "\n";
        for (size_t i = 0; i < boundary_conditions.vgrad_bcs.size(); ++i) {
            const auto& bc = boundary_conditions.vgrad_bcs[i];
            std::cout << "    VGrad BC " << i + 1 << ":\n";

            // Print essential IDs
            std::cout << "      Essential IDs: ";
            for (const auto& id : bc.essential_ids) {
                std::cout << id << " ";
            }
            std::cout << "\n";

            // Print the velocity gradient tensor (3x3 matrix stored as 9 values)
            std::cout << "      Velocity gradient tensor:\n";
            if (bc.velocity_gradient.size() >= 9) {
                // Print as a 3x3 matrix for clarity
                std::cout << "        | " << std::setw(12) << bc.velocity_gradient[0] << " "
                          << std::setw(12) << bc.velocity_gradient[1] << " " << std::setw(12)
                          << bc.velocity_gradient[2] << " |\n";
                std::cout << "        | " << std::setw(12) << bc.velocity_gradient[3] << " "
                          << std::setw(12) << bc.velocity_gradient[4] << " " << std::setw(12)
                          << bc.velocity_gradient[5] << " |\n";
                std::cout << "        | " << std::setw(12) << bc.velocity_gradient[6] << " "
                          << std::setw(12) << bc.velocity_gradient[7] << " " << std::setw(12)
                          << bc.velocity_gradient[8] << " |\n";
            } else {
                // Fallback if not exactly 9 values
                std::cout << "        Values: ";
                for (const auto& val : bc.velocity_gradient) {
                    std::cout << val << " ";
                }
                std::cout << "\n";
            }

            // Print origin if specified
            if (bc.origin.has_value()) {
                std::cout << "      Origin: (" << bc.origin->at(0) << ", " << bc.origin->at(1)
                          << ", " << bc.origin->at(2) << ")\n";
            }

            // Print time info if this BC is time-dependent
            if (bc.time_info.time_dependent || bc.time_info.cycle_dependent) {
                std::cout << "      Time-dependent: "
                          << (bc.time_info.time_dependent ? "Yes" : "No") << "\n";
                std::cout << "      Cycle-dependent: "
                          << (bc.time_info.cycle_dependent ? "Yes" : "No") << "\n";
            }
        }
    }

    // Time-dependent info (general)
    if (boundary_conditions.time_info.time_dependent ||
        boundary_conditions.time_info.cycle_dependent) {
        std::cout << "\n  General time-dependent BC settings:\n";
        std::cout << "    Time-dependent: "
                  << (boundary_conditions.time_info.time_dependent ? "Yes" : "No") << "\n";
        std::cout << "    Cycle-dependent: "
                  << (boundary_conditions.time_info.cycle_dependent ? "Yes" : "No") << "\n";
        if (!boundary_conditions.update_steps.empty()) {
            std::cout << "    Update steps: ";
            for (const auto& step : boundary_conditions.update_steps) {
                std::cout << step << " ";
            }
            std::cout << "\n";
        }
    }

    if (boundary_conditions.mono_def_bcs) {
        std::cout
            << "\n  Experimental Feature: monotonic loading BCs in the Z-direction being applied\n";
        std::cout << "                        all other defined BC constraints will be ignored\n";
    }

    // Print the internal BCManager maps if they're populated
    // These show how the BCs are organized by time step
    if (!boundary_conditions.map_ess_vel.empty() || !boundary_conditions.map_ess_vgrad.empty() ||
        !boundary_conditions.map_ess_comp.empty() || !boundary_conditions.map_ess_id.empty()) {
        std::cout << "\n  BCManager internal mappings:\n";

        // Print essential velocity map
        if (!boundary_conditions.map_ess_vel.empty()) {
            std::cout << "    Essential velocities by step:\n";
            for (const auto& [step, values] : boundary_conditions.map_ess_vel) {
                std::cout << "      Step " << step << ": ";
                // Print first few values to avoid overwhelming output
                size_t count = 0;
                for (const auto& val : values) {
                    if (count++ < 6) { // Show first 6 values
                        std::cout << val << " ";
                    }
                }
                if (values.size() > 6) {
                    std::cout << "... (" << values.size() << " total values)";
                }
                std::cout << "\n";
            }
        }

        // Print essential velocity gradient map
        if (!boundary_conditions.map_ess_vgrad.empty()) {
            std::cout << "    Essential velocity gradients by step:\n";
            for (const auto& [step, values] : boundary_conditions.map_ess_vgrad) {
                std::cout << "      Step " << step << ": ";
                if (values.size() >= 9) {
                    std::cout << "(3x3 tensor with " << values.size() / 9 << " tensors)\n";
                } else {
                    std::cout << values.size() << " values\n";
                }
            }
        }

        // Print essential components map
        if (!boundary_conditions.map_ess_comp.empty()) {
            std::cout << "    Essential components mapping:\n";
            for (const auto& [type, step_map] : boundary_conditions.map_ess_comp) {
                std::cout << "      Type '" << type << "':\n";
                for (const auto& [step, comp_ids] : step_map) {
                    std::cout << "        Step " << step << ": ";
                    size_t count = 0;
                    for (const auto& id : comp_ids) {
                        if (count++ < 10) { // Show first 10 component IDs
                            std::cout << id << " ";
                        }
                    }
                    if (comp_ids.size() > 10) {
                        std::cout << "... (" << comp_ids.size() << " total)";
                    }
                    std::cout << "\n";
                }
            }
        }

        // Print essential IDs map
        if (!boundary_conditions.map_ess_id.empty()) {
            std::cout << "    Essential IDs mapping:\n";
            for (const auto& [type, step_map] : boundary_conditions.map_ess_id) {
                std::cout << "      Type '" << type << "':\n";
                for (const auto& [step, ids] : step_map) {
                    std::cout << "        Step " << step << ": ";
                    for (const auto& id : ids) {
                        std::cout << id << " ";
                    }
                    std::cout << "\n";
                }
            }
        }
    }

    // Legacy format information if present
    if (boundary_conditions.legacy_bcs.changing_ess_bcs) {
        std::cout << "\n  Legacy BC format detected:\n";
        std::cout << "    Changing essential BCs: Yes\n";
        std::cout << "    Update steps: ";
        for (const auto& step : boundary_conditions.legacy_bcs.update_steps) {
            std::cout << step << " ";
        }
        std::cout << "\n";
    }
}

void ExaOptions::print_visualization_options() const {
    std::cout << "\nVisualization Options:\n";
    std::cout << "  VisIt: " << (visualization.visit ? "Enabled" : "Disabled") << "\n";
    std::cout << "  ParaView: " << (visualization.paraview ? "Enabled" : "Disabled") << "\n";
    std::cout << "  Conduit: " << (visualization.conduit ? "Enabled" : "Disabled") << "\n";
    std::cout << "  ADIOS2: " << (visualization.adios2 ? "Enabled" : "Disabled") << "\n";
    std::cout << "  Output frequency: " << visualization.output_frequency << "\n";
    std::cout << "  Output location: " << visualization.floc << "\n";
}

void ExaOptions::print_post_processing_options() const {
    std::cout << "\nPost-Processing Options:\n";

    // Volume averages
    const auto& vol_avg = post_processing.volume_averages;
    std::cout << "  Volume averages: " << (vol_avg.enabled ? "Enabled" : "Disabled") << "\n";
    if (vol_avg.enabled) {
        std::cout << "    Output directory: " << vol_avg.output_directory << "\n";
        std::cout << "    Output frequency: " << vol_avg.output_frequency << "\n";
        std::cout << "    Stress: " << (vol_avg.stress ? "Yes" : "No");
        if (vol_avg.stress)
            std::cout << " (" << vol_avg.avg_stress_fname << ")";
        std::cout << "\n";

        std::cout << "    Deformation gradient: " << (vol_avg.def_grad ? "Yes" : "No");
        if (vol_avg.def_grad)
            std::cout << " (" << vol_avg.avg_def_grad_fname << ")";
        std::cout << "\n";

        std::cout << "    Euler strain: " << (vol_avg.euler_strain ? "Yes" : "No");
        if (vol_avg.euler_strain)
            std::cout << " (" << vol_avg.avg_euler_strain_fname << ")";
        std::cout << "\n";

        std::cout << "    Plastic work: " << (vol_avg.plastic_work ? "Yes" : "No");
        if (vol_avg.plastic_work)
            std::cout << " (" << vol_avg.avg_pl_work_fname << ")";
        std::cout << "\n";

        std::cout << "    Equivalent plastic strain: " << (vol_avg.eq_pl_strain ? "Yes" : "No");
        if (vol_avg.eq_pl_strain)
            std::cout << " (" << vol_avg.avg_eq_pl_strain_fname << ")";
        std::cout << "\n";

        std::cout << "    Elastic strain: " << (vol_avg.elastic_strain ? "Yes" : "No");
        if (vol_avg.elastic_strain)
            std::cout << " (" << vol_avg.avg_elastic_strain_fname << ")";
        std::cout << "\n";

        std::cout << "    Additional averages: " << (vol_avg.additional_avgs ? "Yes" : "No")
                  << "\n";
    }

    // Projections
    const auto& proj = post_processing.projections;
    std::cout << "  Projections:\n";
    std::cout << "    Auto-enable compatible: " << (proj.auto_enable_compatible ? "Yes" : "No")
              << "\n";
    if (!proj.enabled_projections.empty()) {
        std::cout << "    Enabled projections:\n";
        for (const auto& p : proj.enabled_projections) {
            std::cout << "      - " << p << "\n";
        }
    }

    // Light-up options
    const auto& light_configs = post_processing.light_up_configs;
    if (light_configs.empty()) {
        std::cout << "  Light-up analysis: Disabled\n";
    } else {
        std::cout << "\n=== LightUp Configuration Summary ===\n";

        // Group by material for cleaner output
        std::map<std::string, std::vector<const LightUpOptions*>> by_material;
        for (const auto& lightup : light_configs) {
            by_material[lightup.material_name].push_back(&lightup);
        }

        for (const auto& [material, options] : by_material) {
            std::cout << " Material: " << material << "\n";
            for (const auto* opt : options) {
                auto& light = *opt;
                std::cout << "   Region ";
                if (light.region_id.has_value()) {
                    std::cout << light.region_id.value();
                } else {
                    std::cout << "(unassigned)";
                }
                std::cout << " (" << (light.is_auto_generated ? "auto" : "user") << ")\n";
                std::cout << "\n";
                std::cout << "   Enabled: " << (light.enabled ? "Yes" : "No") << "\n";
                if (light.enabled) {
                    std::cout << "     Laue Group: ";
                    switch (light.lattice_type) {
                    case LatticeType::CUBIC: {
                        std::cout << "cubic\n";
                        break;
                    }
                    case LatticeType::HEXAGONAL: {
                        std::cout << "hexagonal\n";
                        break;
                    }
                    case LatticeType::TRIGONAL: {
                        std::cout << "trigonal\n";
                        break;
                    }
                    case LatticeType::RHOMBOHEDRAL: {
                        std::cout << "rhombohedral\n";
                        break;
                    }
                    case LatticeType::TETRAGONAL: {
                        std::cout << "tetragonal\n";
                        break;
                    }
                    case LatticeType::ORTHORHOMBIC: {
                        std::cout << "orthorhombic\n";
                        break;
                    }
                    case LatticeType::MONOCLINIC: {
                        std::cout << "monoclinic\n";
                        break;
                    }
                    case LatticeType::TRICLINIC: {
                        std::cout << "triclinic\n";
                        break;
                    }
                    default: {
                        std::cout << "unknown\n";
                    }
                    }

                    std::cout << "     Lattice parameters: ( ";
                    for (const auto& lp : light.lattice_parameters) {
                        std::cout << lp << " ";
                    }
                    std::cout << ")\n";

                    std::cout << "     Distance tolerance: " << light.distance_tolerance << "\n";
                    std::cout << "     Sample direction: (" << light.sample_direction[0] << ", "
                              << light.sample_direction[1] << ", " << light.sample_direction[2]
                              << ")\n";
                    std::cout << "     Output basename: " << light.lattice_basename << "\n";

                    if (!light.hkl_directions.empty()) {
                        std::cout << "     HKL directions:\n";
                        for (const auto& hkl : light.hkl_directions) {
                            std::cout << "        - [" << hkl[0] << ", " << hkl[1] << ", " << hkl[2]
                                      << "]\n";
                        }
                    }
                }
            }
        }
        std::cout << "=====================================\n\n";
    }
}
