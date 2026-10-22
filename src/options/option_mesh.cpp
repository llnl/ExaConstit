#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"

#include <iostream>

namespace fs = std::filesystem;

MeshOptions MeshOptions::from_toml(const toml::value& toml_input) {
    MeshOptions options;

    if (toml_input.contains("type")) {
        options.mesh_type = string_to_mesh_type(toml::find<std::string>(toml_input, "type"));
    }

    // For a parallel decomposition floc is the file prefix (see MeshOptions::mesh_file)
    if (options.mesh_type == MeshType::FILE || options.mesh_type == MeshType::PARALLEL) {
        if (toml_input.contains("floc")) {
            options.mesh_file = toml::find<std::string>(toml_input, "floc");
        }
    }

    if (toml_input.contains("refine_serial") || toml_input.contains("ref_ser")) {
        const auto& key = toml_input.contains("refine_serial") ? "refine_serial" : "ref_ser";
        options.ref_ser = toml::find<int>(toml_input, key);
    }

    if (toml_input.contains("refine_parallel") || toml_input.contains("ref_par")) {
        const auto& key = toml_input.contains("refine_parallel") ? "refine_parallel" : "ref_par";
        options.ref_par = toml::find<int>(toml_input, key);
    }

    if (toml_input.contains("order") || toml_input.contains("p_refinement")) {
        // Support both "order" and "p_refinement" for backward compatibility
        const auto& key = toml_input.contains("order") ? "order" : "p_refinement";
        options.order = toml::find<int>(toml_input, key);
    }

    if (toml_input.contains("periodicity")) {
        options.periodicity = toml::find<bool>(toml_input, "periodicity");
    }

    // Handle Auto mesh section
    if (options.mesh_type == MeshType::AUTO) {
        auto auto_section = toml::find(toml_input, "Auto");
        if (auto_section.contains("length") || auto_section.contains("mxyz")) {
            const auto& key = auto_section.contains("length") ? "length" : "mxyz";
            auto length_array = toml::find<std::vector<double>>(auto_section, key);
            if (length_array.size() >= 3) {
                std::copy_n(length_array.begin(), 3, options.mxyz.begin());
            }
        }

        if (auto_section.contains("ncuts") || auto_section.contains("nxyz")) {
            const auto& key = auto_section.contains("ncuts") ? "ncuts" : "nxyz";
            auto ncuts_array = toml::find<std::vector<int>>(auto_section, key);
            if (ncuts_array.size() >= 3) {
                std::copy_n(ncuts_array.begin(), 3, options.nxyz.begin());
            }
        }
    }

    // [Mesh.Microstructure]: generate the grains inside mechanics. Only stored here; it is
    // parsed and checked by the hook code from ExaOptions::validate (see
    // MeshOptions::microstructure), so the options code includes no generator header and an
    // ExaConstit built without the generator can still name the table in its error message.
    if (toml_input.contains("Microstructure")) {
        options.microstructure = toml::find(toml_input, "Microstructure");
    }

    return options;
}

bool MeshOptions::validate(bool check_input_files) const {
    if (mesh_type == MeshType::NOTYPE) {
        WARNING_0_OPT("Error: Mesh table was not provided an appropriate mesh type");
        return false;
    }

    // For auto mesh generation, check that nxyz and mxyz are valid
    if (mesh_type == MeshType::AUTO) {
        for (size_t i = 0; i < 3; ++i) {
            if (nxyz[i] <= 0) {
                std::ostringstream err;
                err << "Error: Invalid mesh discretization: nxyz[" << i << "] = " << nxyz[i]
                    << std::endl;
                WARNING_0_OPT(err.str());
                return false;
            }
            if (mxyz[i] <= 0.0) {
                std::ostringstream err;
                err << "Error: Invalid mesh dimensions: mxyz[" << i << "] = " << mxyz[i]
                    << std::endl;
                WARNING_0_OPT(err.str());
                return false;
            }
        }
    }

    // Check that mesh file exists for CUBIT or OTHER mesh types. A restart rebuilds the mesh
    // from the checkpoint, so the file is not needed (or checked) then.
    if (check_input_files && (mesh_type == MeshType::FILE) && !mesh_file.empty()) {
        if (!fs::exists(mesh_file)) {
            std::ostringstream err;
            err << "Error: Mesh file '" << mesh_file << "' does not exist." << std::endl;
            WARNING_0_OPT(err.str());
            return false;
        }
    }

    // A parallel decomposition (one mesh file per MPI rank) needs a prefix, since every part's
    // file name is built from it. Only part 0 is checked here, because validation runs before
    // the mesh is built and a missing part 0 is the common mistake (a wrong prefix). Whether the
    // number of parts equals the number of ranks is checked later in makeMesh
    // (verifyParallelMeshParts in simulation_state.cpp), where every rank looks for its own part
    // and the ranks agree on the result. As for FILE meshes, a restart (check_input_files =
    // false) rebuilds the mesh from the checkpoint and does not need the parts.
    if (mesh_type == MeshType::PARALLEL) {
        if (mesh_file.empty()) {
            WARNING_0_OPT("Error: Mesh table with type = \"parallel\" needs `floc`, the file "
                          "prefix of the parallel mesh decomposition.");
            return false;
        }
        if (check_input_files) {
            // Part r is named prefix + r zero-padded to six digits (mfem::MakeParFilename)
            const std::string first_part = mfem::MakeParFilename(mesh_file.string(), 0);
            if (!fs::exists(first_part)) {
                std::ostringstream err;
                err << "Error: Parallel mesh decomposition '" << mesh_file.string()
                    << "' not found: part 0 should be the file '" << first_part
                    << "' (the prefix followed by the six-digit rank)." << std::endl;
                WARNING_0_OPT(err.str());
                return false;
            }
        }
    }

    if (ref_ser < 0) {
        WARNING_0_OPT(
            "Error: Mesh table has `ref_ser` / `refine_serial` set to value less than 0.");
        return false;
    }

    if (ref_par < 0) {
        WARNING_0_OPT(
            "Error: Mesh table has `ref_par` / `refine_parallel` set to value less than 0.");
        return false;
    }

    if (order < 1) {
        WARNING_0_OPT("Error: Mesh table has `p_refinement` /  `order` set to value less than 1.");
        return false;
    }

    // Implement validation logic
    return true;
}