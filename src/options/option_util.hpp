#pragma once

#include "mfem.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "mpi.h"

/**
 * Macro to simplify warning's so it only does it on Rank 0 and nowhere else
 *
 * Options are parsed before the UnifiedLogger is initialized, so this checks the rank
 * directly rather than going through MFEM_WARNING_0.
 */
#define WARNING_0_OPT(...)                        \
    {                                             \
        int mpi_rank;                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); \
        if (mpi_rank == 0) {                      \
            MFEM_WARNING((__VA_ARGS__));          \
        }                                         \
    }

/**
 * Macro to abort on every rank while only printing the message on Rank 0
 *
 * Like WARNING_0_OPT, this is the options parsing counterpart to MFEM_ABORT_0 for use
 * before the UnifiedLogger is initialized.
 */
#define ABORT_0_OPT(...)                          \
    {                                             \
        int mpi_rank;                             \
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); \
        if (mpi_rank == 0) {                      \
            MFEM_ABORT(__VA_ARGS__);              \
        } else {                                  \
            mfem::mfem_error("");                 \
        }                                         \
    }

/**
 * Options parsing counterpart to MFEM_VERIFY_0, see ABORT_0_OPT
 */
#define VERIFY_0_OPT(condition, ...)  \
    do {                              \
        if (!(condition)) {           \
            ABORT_0_OPT(__VA_ARGS__); \
        }                             \
    } while (0)

#define INFO_0_OPT(...)                              \
    {                                                \
        int mpi_rank;                                \
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);    \
        if (mpi_rank == 0) {                         \
            std::cout << (__VA_ARGS__) << std::endl; \
        }                                            \
    }

/**
 * @brief Convert string to enum with validation and error handling
 *
 * @tparam EnumType The enum type to convert to
 * @param str String value to convert from configuration file
 * @param mapping Map from string values to corresponding enum values
 * @param default_value Default enum value to return if string not found in mapping
 * @param enum_name Descriptive name of enum type for error message reporting
 *
 * @return EnumType value corresponding to input string, or default_value if not found
 *
 * @details This template function provides a unified way to convert string values
 * from TOML configuration files to strongly-typed enum values. If the input string
 * is not found in the mapping, a warning is printed and the default value is returned.
 */
template <typename EnumType>
inline EnumType string_to_enum(const std::string& str,
                               const std::map<std::string, EnumType>& mapping,
                               EnumType default_value,
                               const std::string& enum_name) {
    auto it = mapping.find(str);
    if (it != mapping.end()) {
        return it->second;
    }
    std::ostringstream err;
    err << "Unknown " << enum_name << " type '" << str << "', using default.";
    WARNING_0_OPT(err.str());
    return default_value;
}

/**
 * @brief Load vector of double values from a text file
 *
 * @param filename Path to file containing whitespace-separated numeric values
 * @param expected_size Expected number of values to read (0 = no size check)
 *
 * @return Vector of double values loaded from file
 *
 * @throws std::runtime_error if file cannot be opened for reading
 *
 * @details This function reads numeric values from a text file where values are
 * separated by whitespace (spaces, tabs, newlines). If expected_size > 0 and the
 * number of values read doesn't match, a warning is printed but execution continues.
 */
inline std::vector<double> load_vector_from_file(const std::filesystem::path& filename,
                                                 int expected_size) {
    std::vector<double> result;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename.string());
    }

    double value;
    while (file >> value) {
        result.push_back(value);
    }

    if (expected_size > 0 && result.size() != static_cast<size_t>(expected_size)) {
        std::ostringstream err;
        err << "File " << filename << " contains " << result.size() << " values, but "
            << expected_size << " were expected.";
        WARNING_0_OPT(err.str());
    }

    return result;
}
