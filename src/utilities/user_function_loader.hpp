#pragma once

/**
 * @file user_function_loader.hpp
 * @brief Loading user functions from shared libraries, with a check of their signature
 *
 * Users can hand ExaConstit plain C++ functions in a shared library, listed in the options file
 * like a UMAT (library_path + function name):
 *
 * - state variable functions ([Materials.State_Vars] register_function / initialize_function):
 *       void f(std::shared_ptr<SimulationState> sim_state, int region);
 * - post-processing user functions ([[PostProcessing.user_functions]]):
 *       void f(std::shared_ptr<SimulationState> sim_state, int region,
 *              std::shared_ptr<mfem::expt::PartialQuadratureFunction> values);
 *
 * The functions must be normal C++ functions at global scope (not extern "C", not in a
 * namespace). A C++ compiler stores the argument types in the symbol name it writes to the
 * library ("name mangling"), so ExaConstit can ask for exactly the symbol of the required
 * signature: a function with different arguments has a different symbol and is not found. The
 * expected symbol is built from the function name and the C++ type of the required function
 * (Itanium C++ ABI, used by GCC, Clang and the vendor compilers on Linux and macOS).
 *
 * CheckUserFunctions() loads every function listed in the options right after the options are
 * read, on every rank, and stops the run with the required signature if one is missing or
 * different. Later calls to LoadUserFunction() for the same function come from the loader's cache.
 *
 * Each of these keys can instead name a factory of user classes (see exaconstit_user.hpp and
 * user_classes.hpp): key = { factory = "...", variant = "..." }.
 */

#include "mfem_expt/partial_qfunc.hpp"
#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"
#include "utilities/dynamic_function_loader.hpp"
#include "utilities/unified_logger.hpp"
#include "utilities/user_classes.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

class SimulationState;

namespace exaconstit {

/**
 * @brief Pointer to a state variable user function (register_function / initialize_function)
 */
using StateVariableFunction = void (*)(std::shared_ptr<SimulationState>, int);

/**
 * @brief Pointer to a post-processing user function ([[PostProcessing.user_functions]])
 */
using PostProcessingFunction = void (*)(std::shared_ptr<SimulationState>,
                                        int,
                                        std::shared_ptr<mfem::expt::PartialQuadratureFunction>);

/// Readable form of the required signatures, used in error messages and docs
inline std::string StateVariableFunctionSignature(const std::string& name) {
    return "void " + name + "(std::shared_ptr<SimulationState> sim_state, int region)";
}

inline std::string PostProcessingFunctionSignature(const std::string& name) {
    return "void " + name +
           "(std::shared_ptr<SimulationState> sim_state, int region, "
           "std::shared_ptr<mfem::expt::PartialQuadratureFunction> values)";
}

/**
 * @brief Resolve a user library path
 *
 * @param library_path Relative or absolute path to the library
 * @param search_paths Directories to search for a relative path
 * @return Resolved path (unchanged if nothing was found; loading then reports the error)
 *
 * Order: an absolute path as is, then each search path, then the current directory.
 */
inline std::filesystem::path
ResolveLibraryPath(const std::filesystem::path& library_path,
                   const std::vector<std::filesystem::path>& search_paths = {}) {
    if (library_path.empty() || library_path.is_absolute()) {
        return library_path;
    }
    for (const auto& search_path : search_paths) {
        const auto full_path = search_path / library_path;
        if (std::filesystem::exists(full_path)) {
            return full_path;
        }
    }
    if (std::filesystem::exists(library_path)) {
        return std::filesystem::absolute(library_path);
    }
    MFEM_WARNING_0("Library not found: " << library_path);
    return library_path;
}

/**
 * @brief Symbol name a C++ compiler gives a global function of type FuncPtr named function_name
 *
 * Itanium C++ ABI: "_Z" + length of the name + name + encoded parameter types. The parameter
 * encoding is taken from typeid of the function type ("F" + return type + parameters + "E");
 * all user functions return void ("v").
 */
template <typename FuncPtr>
std::string MangledFunctionName(const std::string& function_name) {
    using FuncType = std::remove_pointer_t<FuncPtr>;
    const std::string type_name = typeid(FuncType).name();
    MFEM_VERIFY(type_name.size() > 3 && type_name.compare(0, 2, "Fv") == 0 &&
                    type_name.back() == 'E',
                "Unexpected C++ type name '" << type_name
                                             << "' for a user function; this compiler does not "
                                                "follow the Itanium C++ ABI");
    return "_Z" + std::to_string(function_name.size()) + function_name +
           type_name.substr(2, type_name.size() - 3);
}

/**
 * @brief Load a user function, stopping the run if it is missing or has the wrong signature
 *
 * @param library_path Library holding the function (empty: search the ExaConstit executable)
 * @param function_name Function name as written in the user's source
 * @param signature Readable required signature, for the error message
 * @param purpose Where the function is used (e.g. "register_function of material steel_1")
 * @return Pointer to the function
 */
template <typename FuncPtr>
FuncPtr LoadUserFunction(const std::filesystem::path& library_path,
                         const std::string& function_name,
                         const std::string& signature,
                         const std::string& purpose) {
    const auto resolved = ResolveLibraryPath(library_path);
    const std::string where = resolved.empty() ? std::string("the ExaConstit executable")
                                               : resolved.string();

    SymbolConfig config;
    config.search_names = {MangledFunctionName<FuncPtr>(function_name)};
    config.enable_fortran_mangling = false;
    config.enable_builtin_search = resolved.empty();
    config.case_sensitive = true;

    const auto result = DynamicFunctionLoader<FuncPtr>::load(
        resolved.string(), config, LoadStrategy::PERSISTENT);
    if (result.success) {
        return result.function;
    }

    if (result.error_message.rfind("Failed to load library", 0) == 0) {
        // Every rank stops (the message is printed once): all ranks run the same check
        ABORT_0_OPT("Could not open the library of the " << purpose << " ('" << function_name
                                                         << "'): " << where << ": "
                                                         << result.error_message);
    }

    // The library opened but the exact symbol is not there. Say whether a plain extern "C"
    // function of that name exists, which is the usual mistake.
    bool has_c_symbol = false;
#if !defined(_WIN32)
    void* handle = resolved.empty() ? RTLD_DEFAULT
                                    : ::dlopen(resolved.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle) {
        has_c_symbol = (::dlsym(handle, function_name.c_str()) != nullptr);
        if (!resolved.empty()) {
            ::dlclose(handle);
        }
    }
#endif
    ABORT_0_OPT("The " << purpose << " '" << function_name << "' was not found in " << where
                       << " with the required signature\n    " << signature
                       << "\nDeclare it at global scope as a normal C++ function (not extern "
                          "\"C\", not in a namespace) with exactly these argument types."
                       << (has_c_symbol ? "\nA function of that name exists but is declared "
                                          "extern \"C\" (or has different arguments)."
                                        : ""));
    return nullptr;
}

/**
 * @brief Label of a material in messages about its user functions, e.g. "steel_1"
 *
 * @param material Material options
 * @return Material name and 1-based material number
 */
inline std::string MaterialLabel(const MaterialOptions& material) {
    return material.material_name + "_" + std::to_string(material.region_id);
}

/**
 * @brief Load and check every user function listed in the options
 *
 * Call right after the options are read, on every rank, before the mesh is built. Stops the run
 * with the required signature for the first function that is missing or different.
 */
inline void CheckUserFunctions(const ExaOptions& options) {
    for (const auto& material : options.materials) {
        const auto& vars = material.state_vars;
        const std::string material_label = MaterialLabel(material);
        if (!vars.register_function.empty()) {
            LoadUserFunction<StateVariableFunction>(
                vars.library_path,
                vars.register_function,
                StateVariableFunctionSignature(vars.register_function),
                "register_function of material " + material_label);
        }
        if (!vars.initialize_function.empty()) {
            LoadUserFunction<StateVariableFunction>(
                vars.library_path,
                vars.initialize_function,
                StateVariableFunctionSignature(vars.initialize_function),
                "initialize_function of material " + material_label);
        }
    }
    for (const auto& user_function : options.post_processing.user_functions) {
        if (user_function.function_name.empty()) {
            continue; // Factory table, checked below
        }
        LoadUserFunction<PostProcessingFunction>(
            user_function.library_path,
            user_function.function_name,
            PostProcessingFunctionSignature(user_function.function_name),
            "post-processing user function of entry '" + user_function.name + "'");
    }
    // Factory tables: create every object now, checking variant and class
    CheckUserObjects(options);
}

} // namespace exaconstit
