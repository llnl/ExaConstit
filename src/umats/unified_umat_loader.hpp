#pragma once

#include "umats/userumat.h"
#include "utilities/dynamic_function_loader.hpp"
#include "utilities/unified_logger.hpp"

#include <iostream>
#include <sstream>

namespace exaconstit {

/**
 * @brief Unified UMAT loader using the generic dynamic function loader framework
 *
 * This replaces both DynamicUmatLoader and UmatResolver with a single unified interface
 * that leverages the templated DynamicFunctionLoader.
 */
class UnifiedUmatLoader {
public:
    using Loader = DynamicFunctionLoader<UmatFunction>;

    /**
     * @brief Load a UMAT function with automatic symbol resolution
     *
     * @param library_path Path to library (empty for built-in)
     * @param function_name Specific function name to search for
     * @param strategy Loading strategy
     * @return UMAT function pointer or nullptr on failure
     */
    static UmatFunction LoadUmat(const std::string& library_path,
                                 LoadStrategy strategy = LoadStrategy::PERSISTENT,
                                 const std::string& function_name = "umat_call") {
        // Configure symbol search
        SymbolConfig config;

        // Primary search name
        if (!function_name.empty() && function_name != "auto") {
            config.search_names.push_back(function_name);
        }

        // Add default UMAT symbol variants
        config.search_names.insert(config.search_names.end(),
                                   {"umat_call", // Common C wrapper
                                    "umat",      // Standard Fortran name
                                    "userumat",  // Alternative name
                                    "UMAT",      // Sometimes uppercase
                                    "USERUMAT"});

        // Enable all search features for UMATs
        config.enable_fortran_mangling = true;
        config.enable_builtin_search = library_path.empty();
        config.case_sensitive = false; // Be flexible with case

        // Perform the load
        auto result = Loader::load(library_path, config, strategy);

        if (result.success) {
            // Log success if needed
            if (std::getenv("EXACONSTIT_DEBUG_UMAT")) {
                std::cout << "Successfully loaded UMAT"
                          << (library_path.empty() ? " (built-in)" : " from: " + library_path)
                          << " using symbol: " << result.resolved_symbol << std::endl;
            }

            // Compatibility: warn if different symbol was used
            if (!function_name.empty() && function_name != "auto" &&
                result.resolved_symbol != function_name) {
                MFEM_WARNING_0("Requested function '" << function_name << "' not found, using '"
                                                      << result.resolved_symbol << "' instead");
            }
        }
        // On failure, callers report the error through GetLastError()

        return result.function;
    }

    /**
     * @brief Unload a UMAT library
     */
    static bool UnloadUmat(const std::string& library_path) {
        // Use same config as load for consistency
        SymbolConfig config;
        config.search_names = {"umat_call", "umat", "userumat", "UMAT", "USERUMAT"};
        config.enable_fortran_mangling = true;
        config.enable_builtin_search = library_path.empty();

        return Loader::unload(library_path, config);
    }

    /**
     * @brief Get already-loaded UMAT without loading
     */
    static UmatFunction GetUmat(const std::string& library_path) {
        // Try to load with LAZY_LOAD strategy - if already loaded, just returns it
        return LoadUmat(library_path, LoadStrategy::LAZY_LOAD);
    }

    /**
     * @brief Check if a UMAT library is loaded
     */
    static bool IsLoaded(const std::string& library_path) {
        SymbolConfig config;
        config.search_names = {"umat_call", "umat", "userumat"};
        return Loader::validate(library_path, config);
    }

    /**
     * @brief Validate a UMAT library
     */
    static bool ValidateLibrary(const std::string& library_path,
                                const std::string& function_name = "") {
        SymbolConfig config;
        if (!function_name.empty()) {
            config.search_names.push_back(function_name);
        }
        config.search_names.insert(config.search_names.end(), {"umat_call", "umat", "userumat"});
        config.enable_fortran_mangling = true;

        return Loader::validate(library_path, config);
    }

    /**
     * @brief Get last error message
     */
    static std::string GetLastError() {
        return Loader::get_last_error();
    }

    /**
     * @brief Clear all cached UMATs
     */
    static void ClearCache() {
        Loader::clear_cache();
    }
};
} // namespace exaconstit