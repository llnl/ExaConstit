#pragma once

#include "options/option_parser_v2.hpp"
#include "postprocessing/postprocessing_hdf5_file.hpp"

#include "mfem.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/**
 * @brief Utility class for managing file paths and directories in PostProcessingDriver
 *
 * This class handles:
 * 1. Proper file naming according to ExaOptions conventions
 * 2. Directory creation when needed
 * 3. Path resolution (relative vs absolute)
 * 4. Error handling for file operations
 */
class PostProcessingFileManager {
public:
    /**
     * @brief Initialize file manager with ExaOptions
     */
    PostProcessingFileManager(const ExaOptions& options);

    /**
     * @brief Get the full file path for a volume average output
     *
     * @param calc_type Type of calculation (e.g., "stress", "def_grad")
     * @param region Region index (-1 for global)
     * @param region_name Optional region name (if available)
     * @return Full file path
     */
    std::string GetVolumeAverageFilePath(const std::string& calc_type,
                                         int region = -1,
                                         const std::string& region_name = "") const;

    /**
     * @brief Get the output directory path
     */
    fs::path GetOutputDirectory() const {
        return m_output_directory;
    }
    /**
     * @brief Get the visualization directory path
     */
    fs::path GetVizDirectory() const {
        return m_output_viz;
    }

    /**
     * @brief Get the base filename (without extension)
     */
    std::string GetBaseFilename() const {
        return m_base_filename;
    }

    /**
     * @brief Create output directory if it doesn't exist
     *
     * @return true if directory exists or was created successfully
     *
     * Ensures the main output directory exists before file operations.
     * Creates the directory structure using filesystem operations with
     * proper error handling. Only MPI rank 0 performs directory creation
     * to avoid race conditions in parallel execution.
     */
    bool EnsureOutputDirectoryExists();

    /**
     * @brief Create directory if it doesn't exist
     *
     * @param output_dir Directory path to create
     * @param comm MPI communicator associated with a given region
     * @return true if directory exists or was created successfully
     *
     * Generic directory creation utility with filesystem error handling.
     * Used for both main output directory and subdirectory creation
     * such as visualization output folders.
     */
    bool EnsureDirectoryExists(fs::path& output_dir, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Create and open an output file with proper error handling
     *
     * @param filepath Full path to the file
     * @param append Whether to append to existing file
     * @param comm MPI communicator associated with a given region
     * @return Unique pointer to opened file stream
     *
     * Only called by the rank that writes the file (rank 0 of comm), so any problems are
     * reported from that rank.
     */
    std::unique_ptr<std::ofstream>
    CreateOutputFile(const fs::path& filepath, bool append = true, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Get column header string for volume average output files
     *
     * @param calc_type Type of calculation
     * @return Header string with column descriptions
     *
     * Provides standardized column headers for volume average output files.
     * Headers include time, volume, and appropriate component labels for
     * each calculation type (tensor components, scalar values, etc.).
     *
     * Ensures consistent output format for post-processing tools and
     * provides clear documentation of data organization in output files.
     */
    std::string GetVolumeAverageHeader(const std::string& calc_type) const;

    /**
     * @brief Set the column names of a volume average file (written after Time and Volume)
     *
     * @param calc_type Calculation name, as passed to WriteVolumeAverage()
     * @param column_names One name per value
     *
     * Used for quantities the file manager does not know (post-processing user functions);
     * the built-in quantities have fixed column names.
     */
    void SetVolumeAverageColumnNames(const std::string& calc_type,
                                     const std::vector<std::string>& column_names) {
        m_column_names[calc_type] = column_names;
    }

    /**
     * @brief Put a calculation's files under another name and in a subfolder
     *
     * @param calc_type Calculation name, as passed to WriteVolumeAverage()
     * @param file_stem File name stem used instead of avg_<calc_type> (region / global suffix
     *                  and .txt are added as usual)
     * @param subdirectory Folder inside the output directory (created by the caller with
     *                     EnsureDirectoryExists(), which is collective)
     *
     * Used for the volume statistics (stats/<name>_<statistic>_*.txt, stats/<name>_hist_*.txt).
     */
    void SetVolumeAverageFileLocation(const std::string& calc_type,
                                      const std::string& file_stem,
                                      const fs::path& subdirectory) {
        m_file_locations[calc_type] = {file_stem, subdirectory};
    }

    /**
     * @brief Comment line written above a calculation's column header (e.g. histogram edges)
     */
    void SetVolumeAverageHeaderComment(const std::string& calc_type, const std::string& comment) {
        m_header_comments[calc_type] = comment;
    }

    /**
     * @brief Histogram bin edges of a calculation, stored as its bin_edges dataset in the HDF5
     *        output (the text files list them in their header comment instead)
     */
    void SetVolumeAverageBinEdges(const std::string& calc_type, const std::vector<double>& edges) {
        m_bin_edges[calc_type] = edges;
    }

    /// True if the post-processing output goes to the single HDF5 file (output_format = "hdf5")
    bool WritesHDF5() const {
        return m_hdf5 != nullptr;
    }

    /**
     * @brief End of a post-processing update: the HDF5 output gathers the rows queued on every
     *        rank to rank 0 and appends them to the file (collective; nothing to do for text)
     */
    void FinishOutputStep() {
        if (m_hdf5) {
            m_hdf5->FinishOutputStep();
        }
    }

    /**
     * @brief Restarted run: the HDF5 output drops the rows after the checkpoint time
     *        (collective; nothing to do for text)
     */
    void ContinueAfterRestart(const double restart_time) {
        if (m_hdf5) {
            m_hdf5->ContinueAfterRestart(restart_time);
        }
    }

    /**
     * @brief Column names (after Time and Volume) of a volume average file
     *
     * @param calc_type Calculation name
     * @return The names set with SetVolumeAverageColumnNames(), or the built-in names of the
     *         built-in quantities (e.g. Sxx ... for stress), or {calc_type} for anything else
     */
    std::vector<std::string> GetVolumeAverageColumnNames(const std::string& calc_type) const;

    /**
     * @brief Check if output should occur at the current step
     *
     * @param step Current time step number
     * @return true if output should occur, false otherwise
     *
     * Implements output frequency control based on ExaOptions configuration.
     * Uses modulo operation to determine if current step matches the
     * configured output frequency for volume averaging operations.
     */
    bool ShouldOutputAtStep(int step) const;

    /**
     * @brief Write time and volume with consistent formatting
     *
     * @param stream Output file stream
     * @param time Current simulation time
     * @param volume Total volume
     *
     * Provides consistent formatting for the first two columns
     * that appear in all volume average output files.
     */
    void WriteTimeAndVolume(std::ofstream& stream, double time, double volume) const {
        stream << std::setw(COLUMN_WIDTH) << time << std::setw(COLUMN_WIDTH) << volume;
    }

    /**
     * @brief Write vector data with consistent column formatting
     *
     * @param stream Output file stream
     * @param data Vector or array containing the data values
     * @param size Number of elements to write
     *
     * Writes each element with proper column width alignment.
     * Template allows use with mfem::Vector, std::vector, or C arrays.
     */
    template <typename T>
    void WriteVectorData(std::ofstream& stream, const T& data, int size) const {
        for (int i = 0; i < size; ++i) {
            stream << std::setw(COLUMN_WIDTH) << data[i];
        }
    }

    /**
     * @brief Write single scalar value with consistent formatting
     *
     * @param stream Output file stream
     * @param value Scalar value to write
     *
     * Writes a single value with proper column width alignment.
     */
    template <typename T>
    void WriteScalarData(std::ofstream& stream, const T& value) const {
        stream << std::setw(COLUMN_WIDTH) << value;
    }

    /**
     * @brief Safe template version that avoids deprecated conversions
     */
    template <typename T>
    void WriteVolumeAverage(const std::string& calc_type,
                            int region,
                            const std::string& region_name,
                            double time,
                            double volume,
                            const T& data,
                            int data_size = -1,
                            MPI_Comm comm = MPI_COMM_WORLD) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank != 0)
            return;

        if (m_hdf5) {
            QueueVolumeAverageRow(
                calc_type, region, region_name, time, volume, ValuesOf(data, data_size));
            return;
        }

        auto filepath = GetVolumeAverageFilePath(calc_type, region, region_name);

        auto file = CreateOutputFile(filepath, true, comm);

        if (file && file->is_open()) {
            // Write the header if the file is empty. Checking the opened file itself
            // rather than calling exists() beforehand avoids another check-then-act gap.
            file->seekp(0, std::ios_base::end);
            if (file->tellp() == 0) {
                *file << GetVolumeAverageHeader(calc_type);
            }

            WriteTimeAndVolume(*file, time, volume);
            WriteDataSafe(*file, data, data_size);
            *file << "\n" << std::flush;
        }
    }

    /**
     * @brief Get the full file path for a surface quantity output
     *
     * @param surface_id Mesh boundary attribute of the surface
     * @return Full file path, [stem]_surface_[id][extension] in the output directory
     */
    std::string GetSurfaceQuantityFilePath(int surface_id) const;

    /**
     * @brief Get the column header line for surface quantity output files
     *
     * @param moments Whether the moment columns are included
     * @return Header string with column descriptions
     */
    std::string GetSurfaceQuantityHeader(bool moments) const;

    /**
     * @brief Column names of a surface quantity file after Time, Surface_ID and Area
     *
     * @param moments Whether the moment columns are included
     */
    std::vector<std::string> GetSurfaceQuantityColumnNames(bool moments) const;

    /**
     * @brief Write one row of surface quantity output (rank 0 of MPI_COMM_WORLD only)
     *
     * @param surface_id Mesh boundary attribute of the surface
     * @param preamble Comment lines written once, before the column header, when the file is
     *        empty
     * @param moments Whether the moment columns are included in values
     * @param time Current simulation time
     * @param area Current surface area
     * @param values Displacement (3), force (3), normal (3) and, if moments, moment (3)
     */
    void WriteSurfaceQuantity(int surface_id,
                              const std::string& preamble,
                              bool moments,
                              double time,
                              double area,
                              const std::vector<double>& values) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0)
            return;

        if (m_hdf5) {
            const std::string group = "surface/surface_" + std::to_string(surface_id);
            std::vector<std::string> names = {"time", "area"};
            std::vector<double> row = {time, area};
            const auto columns = GetSurfaceQuantityColumnNames(moments);
            names.insert(names.end(), columns.begin(), columns.end());
            row.insert(row.end(), values.begin(), values.end());
            m_hdf5->AddRow(group, names, row);
            if (!preamble.empty()) {
                m_hdf5->AddFixed(group + "/description", preamble);
            }
            return;
        }

        auto file = CreateOutputFile(GetSurfaceQuantityFilePath(surface_id), true);

        if (file && file->is_open()) {
            // Same header-if-empty check as WriteVolumeAverage(), so a restart continuation
            // appends to the existing file.
            file->seekp(0, std::ios_base::end);
            if (file->tellp() == 0) {
                *file << preamble;
                *file << GetSurfaceQuantityHeader(moments);
            }

            *file << std::setw(COLUMN_WIDTH) << time << std::setw(COLUMN_WIDTH) << surface_id
                  << std::setw(COLUMN_WIDTH) << area;
            WriteDataSafe(*file, values, static_cast<int>(values.size()));
            *file << "\n" << std::flush;
        }
    }

private:
    // Column width for scientific notation with 12 digits: "-1.234567890123e-05"
    static constexpr int COLUMN_WIDTH = 18;
    /**
     * @brief Get specific filename for a calculation type
     *
     * @param calc_type Type of calculation (e.g., "stress", "def_grad")
     * @return Filename with extension from ExaOptions configuration
     *
     * Maps calculation type strings to configured filenames from ExaOptions.
     * Supports standard calculation types (stress, deformation gradient,
     * plastic work, strains) with fallback to default naming for custom types.
     *
     * Enables user customization of output filenames through configuration
     * while maintaining consistent internal calculation type naming.
     */
    std::string GetSpecificFilename(const std::string& calc_type) const;

    /**
     * @brief Construct region-specific filename with proper formatting
     *
     * @param base_name Base filename without extension
     * @param extension File extension (including dot)
     * @param region Region index
     * @param region_name Optional region name for descriptive filenames
     * @return Formatted filename with region identifier
     *
     * Creates region-specific filenames using either region index or
     * descriptive region name when available. Handles special formatting
     * requirements and ensures consistent naming across all output files.
     *
     * Format examples:
     * - "stress_region_0.txt" (index-based)
     * - "stress_grain_austenite.txt" (name-based)
     */
    std::string ConstructRegionFilename(const std::string& base_filename,
                                        const std::string& extension,
                                        int region,
                                        const std::string& region_name) const;

private:
    /**
     * @brief Configure stream for high-precision output
     *
     * @param stream Reference to output stream to configure
     * @param precision Number of digits of precision (default 15)
     *
     * Centralizes precision configuration for all output streams.
     * Uses scientific notation to ensure consistent formatting for
     * small values that might be missed with default precision.
     */
    void ConfigureStreamPrecision(std::ofstream& stream, int precision = 8) const {
        stream.precision(precision);
        stream.setf(std::ios::scientific, std::ios::floatfield);
        // Optional: Set width for consistent column alignment
        // stream.width(22); // Adjust based on your needs
    }

    /**
     * @brief Safe data writing that avoids deprecated conversions
     */
    void WriteDataSafe(std::ofstream& stream, const mfem::Vector& data, int size) const {
        int actual_size = (size > 0) ? size : data.Size();
        for (int i = 0; i < actual_size; ++i) {
            stream << std::setw(COLUMN_WIDTH) << data[i]; // Use operator[] instead of conversion
        }
    }

    void WriteDataSafe(std::ofstream& stream, double data, int /*size*/) const {
        stream << std::setw(COLUMN_WIDTH) << data; // Direct scalar value
    }

    void WriteDataSafe(std::ofstream& stream, const double* data, int size) const {
        for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
            stream << std::setw(COLUMN_WIDTH) << data[i]; // Array access, no pointer dereferencing
        }
    }

    void WriteDataSafe(std::ofstream& stream, const std::vector<double>& data, int size) const {
        const size_t actual_size = (size > 0) ? static_cast<size_t>(size) : data.size();
        for (size_t i = 0; i < actual_size; ++i) {
            stream << std::setw(COLUMN_WIDTH) << data[i];
        }
    }

    /// The values WriteDataSafe() writes, as a vector (for the HDF5 output)
    std::vector<double> ValuesOf(const mfem::Vector& data, int size) const {
        const int actual_size = (size > 0) ? size : data.Size();
        return std::vector<double>(data.GetData(), data.GetData() + actual_size);
    }
    std::vector<double> ValuesOf(double data, int /*size*/) const {
        return {data};
    }
    std::vector<double> ValuesOf(const double* data, int size) const {
        return std::vector<double>(data, data + size);
    }
    std::vector<double> ValuesOf(const std::vector<double>& data, int size) const {
        const size_t actual_size = (size > 0) ? static_cast<size_t>(size) : data.size();
        return std::vector<double>(data.begin(), data.begin() + static_cast<long>(actual_size));
    }

    /**
     * @brief Queue one volume average / statistics / histogram row for the HDF5 output
     *
     * The group follows the text file the row would go to: a calculation's file becomes
     * volume/<calc>/<region> with time, volume and its columns; its statistics files
     * (key "stats:<name>:<statistic>") volume/<name>/<region>/statistics/<statistic> and its
     * histogram file (key "hist:<name>") volume/<name>/<region>/histogram, both with only their
     * columns (they share the average's time and volume). <region> is the region name used in
     * the text file names, or "global".
     */
    void QueueVolumeAverageRow(const std::string& calc_type,
                               int region,
                               const std::string& region_name,
                               double time,
                               double volume,
                               const std::vector<double>& values);

    /**
     * @brief Create a centered string within a fixed column width
     *
     * @param text Text to center
     * @param width Total column width
     * @return Centered string with padding
     *
     * Centers text within the specified width using spaces for padding. Text that does not fit
     * is written whole with a space on each side (never truncated).
     */
    std::string CenterText(const std::string& text, int width) const {
        const size_t width_z = static_cast<size_t>(width);
        if (text.length() >= width_z) {
            // Too long for the column: keep the whole name, separated by spaces, so every header
            // name stays readable and splits into one word per column (the header is then wider
            // than the data columns)
            return " " + text + " ";
        }

        const size_t padding = width_z - text.length();
        const size_t left_pad = padding / 2;
        const size_t right_pad = padding - left_pad; // Handle odd padding

        return std::string(left_pad, ' ') + text + std::string(right_pad, ' ');
    }

    /**
     * @brief Reference to ExaOptions configuration
     *
     * Provides access to user-specified configuration including output
     * directories, filenames, and frequency settings. Used throughout
     * the file manager for consistent configuration-driven behavior.
     */
    const ExaOptions& m_options;
    /**
     * @brief Main output directory path
     *
     * Base directory for all postprocessing output files. Constructed
     * from ExaOptions basename and output directory settings with
     * proper path formatting and trailing slash handling.
     */
    fs::path m_output_directory;
    /**
     * @brief Visualization output directory path
     *
     * Subdirectory for visualization files (VisIt, ParaView, ADIOS2).
     * Created only when visualization output is enabled in ExaOptions.
     * Provides organized separation of data files and visualization files.
     */
    fs::path m_output_viz;
    /**
     * @brief Base filename without extension
     *
     * Core filename component used for all output files. Derived from
     * ExaOptions basename setting and used as the foundation for
     * region-specific and calculation-specific filename construction.
     */
    std::string m_base_filename;
    /**
     * @brief Output frequency for volume averaging
     *
     * Timestep interval for volume average output. Copied from ExaOptions
     * volume averaging configuration and used by ShouldOutputAtStep()
     * for consistent output timing control.
     */
    int m_output_frequency;

    /**
     * @brief Cache of opened files to avoid reopening
     *
     * Weak pointer cache that tracks opened ofstream objects to prevent
     * repeated file opening/closing operations. Uses weak_ptr to allow
     * automatic cleanup when files are no longer referenced elsewhere.
     * Improves performance for frequent output operations to the same files.
     */
    mutable std::map<std::string, std::weak_ptr<std::ofstream>> m_file_cache;

    /**
     * @brief Column names of volume average files of quantities set by
     * SetVolumeAverageColumnNames(), by calculation name
     */
    std::map<std::string, std::vector<std::string>> m_column_names;

    /// File name stem and subfolder of calculations set by SetVolumeAverageFileLocation()
    std::map<std::string, std::pair<std::string, fs::path>> m_file_locations;

    /// Comment lines set by SetVolumeAverageHeaderComment()
    std::map<std::string, std::string> m_header_comments;

    /// Histogram bin edges set by SetVolumeAverageBinEdges()
    std::map<std::string, std::vector<double>> m_bin_edges;

    /// The single output file with output_format = "hdf5" (null for text output)
    std::unique_ptr<PostProcessingHDF5File> m_hdf5;
};

// Implementation

inline PostProcessingFileManager::PostProcessingFileManager(const ExaOptions& options)
    : m_options(options) {
    // Use the basename from ExaOptions
    m_base_filename = options.basename;

    // Get output directory from volume averages options
    m_output_directory = options.post_processing.volume_averages.output_directory;

    // Get output frequency
    m_output_frequency = options.post_processing.volume_averages.output_frequency;

    // If output directory is empty, use current directory
    if (m_output_directory.empty()) {
        m_output_directory = ".";
    }
    // The output directory has already been resolved to an absolute path once on rank 0
    // and broadcast to all ranks in ExaOptions::parse_options. We purposefully do not
    // resolve it again here, as resolving it independently on every rank (and for every
    // file manager instance) is what opened the door to ranks seeing different states of
    // a symlinked output directory.
    m_output_directory = m_output_directory / m_base_filename;

    if (options.visualization.visit || options.visualization.paraview ||
        options.visualization.adios2) {
        m_output_viz = m_output_directory / "visualizations";
    }

    if (options.post_processing.output_format == PostProcessingFormat::HDF5) {
        m_hdf5 = std::make_unique<PostProcessingHDF5File>(m_output_directory /
                                                          "post_processing.hdf5");
    }
}

inline void PostProcessingFileManager::QueueVolumeAverageRow(const std::string& calc_type,
                                                             int region,
                                                             const std::string& region_name,
                                                             double time,
                                                             double volume,
                                                             const std::vector<double>& values) {
    const std::string region_label = (region == -1)        ? std::string("global")
                                     : region_name.empty() ? std::to_string(region)
                                                           : region_name;
    const auto columns = GetVolumeAverageColumnNames(calc_type);
    MFEM_VERIFY(columns.size() == values.size(),
                "HDF5 output of " << calc_type << ": " << columns.size() << " column names but "
                                  << values.size() << " values");

    // Statistics and histogram keys: "stats:<name>:<statistic>", "hist:<name>"
    const auto colon = calc_type.find(':');
    const std::string kind = (colon == std::string::npos) ? "" : calc_type.substr(0, colon);
    if (kind == "stats") {
        const auto second = calc_type.find(':', colon + 1);
        const std::string name = calc_type.substr(colon + 1, second - colon - 1);
        const std::string statistic = calc_type.substr(second + 1);
        m_hdf5->AddRow(
            "volume/" + name + "/" + region_label + "/statistics/" + statistic, columns, values);
    } else if (kind == "hist") {
        const std::string group = "volume/" + calc_type.substr(colon + 1) + "/" + region_label +
                                  "/histogram";
        m_hdf5->AddRow(group, columns, values);
        const auto edges = m_bin_edges.find(calc_type);
        if (edges != m_bin_edges.end()) {
            m_hdf5->AddFixed(group + "/bin_edges", edges->second);
        }
    } else {
        std::vector<std::string> names = {"time", "volume"};
        names.insert(names.end(), columns.begin(), columns.end());
        std::vector<double> row = {time, volume};
        row.insert(row.end(), values.begin(), values.end());
        m_hdf5->AddRow("volume/" + calc_type + "/" + region_label, names, row);
    }
}

inline std::string PostProcessingFileManager::GetVolumeAverageFilePath(
    const std::string& calc_type, int region, const std::string& region_name) const {
    // Get base filename with extension for this calculation type (or the stem and subfolder set
    // by SetVolumeAverageFileLocation)
    const auto location = m_file_locations.find(calc_type);
    const bool relocated = (location != m_file_locations.end());
    fs::path specific_filename = relocated ? fs::path(location->second.first + ".txt")
                                           : fs::path(GetSpecificFilename(calc_type));

    // Split into base and extension
    fs::path extension = specific_filename.extension();
    fs::path base_name = specific_filename;
    if (extension.string().empty()) {
        extension = ".txt";
    } else {
        base_name = base_name.stem();
    }

    fs::path filename;
    if (region == -1) {
        // Global file
        filename = base_name.string() + "_global" + extension.string();
    } else {
        // Region-specific file
        filename = ConstructRegionFilename(
            base_name.string(), extension.string(), region, region_name);
    }

    return relocated ? m_output_directory / location->second.second / filename
                     : m_output_directory / filename;
}

inline std::string
PostProcessingFileManager::GetSpecificFilename(const std::string& calc_type) const {
    const auto& vol_opts = m_options.post_processing.volume_averages;
    // Map calculation types to specific filenames from ExaOptions
    if (calc_type == "stress") {
        return vol_opts.avg_stress_fname;
    } else if (calc_type == "def_grad") {
        return vol_opts.avg_def_grad_fname;
    } else if (calc_type == "plastic_work" || calc_type == "pl_work") {
        return vol_opts.avg_pl_work_fname;
    } else if (calc_type == "euler_strain") {
        return vol_opts.avg_euler_strain_fname;
    } else if (calc_type == "eps" || calc_type == "eq_pl_strain") {
        return vol_opts.avg_eq_pl_strain_fname;
    } else if (calc_type == "elastic_strain" || calc_type == "estrain") {
        return vol_opts.avg_elastic_strain_fname;
    } else {
        // Default naming for custom calculation types
        return "avg_" + calc_type + ".txt";
    }
}

inline std::string
PostProcessingFileManager::ConstructRegionFilename(const std::string& base_filename,
                                                   const std::string& extension,
                                                   int region,
                                                   const std::string& region_name) const {
    if (!region_name.empty()) {
        // Use region name if available
        return base_filename + "_region_" + region_name + extension;
    } else {
        // Use region index
        return base_filename + "_region_" + std::to_string(region) + extension;
    }
}

inline bool PostProcessingFileManager::EnsureDirectoryExists(fs::path& output_dir, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    bool success = false;
    if (rank == 0) {
        // Only rank 0 touches the file system, and it does so without first checking
        // whether the directory exists. create_directories() is a no-op for directories
        // that already exist, so a separate exists() check would only add a window where
        // the path could change between the check and the creation (TOCTOU).
        // output_dir is used as given; it is built from the output directory that was
        // already resolved once in ExaOptions::parse_options.
        std::error_code ec;
        const bool created = fs::create_directories(output_dir, ec);
        if (ec) {
            MFEM_WARNING("Failed to create output directory " << output_dir << ": "
                                                              << ec.message());
        } else if (!fs::is_directory(output_dir, ec)) {
            MFEM_WARNING("Path exists but is not a directory: " << output_dir);
        } else {
            std::cout << (created ? "Creating output directory: " : "Using existing directory: ")
                      << output_dir << std::endl;
            success = true;
        }

        // Make sure we can actually write to the directory
        if (success) {
            fs::path test_file = output_dir / "test_write.tmp";
            std::ofstream test_stream(test_file);
            if (!test_stream.is_open()) {
                success = false;
                MFEM_WARNING("Output directory is not writable: " << output_dir);
            } else {
                test_stream.close();
                fs::remove(test_file, ec);
            }
        }
    }

    // All ranks wait until rank 0 is done so the directory exists before anyone uses it
    MPI_Bcast(&success, 1, MPI_C_BOOL, 0, comm);
    return success;
}

inline bool PostProcessingFileManager::EnsureOutputDirectoryExists() {
    bool success = EnsureDirectoryExists(m_output_directory);
    if (!m_output_viz.empty()) {
        bool viz_success = EnsureDirectoryExists(m_output_viz);
        success &= viz_success;
    }

    return success;
}

inline std::unique_ptr<std::ofstream> PostProcessingFileManager::CreateOutputFile(
    const fs::path& filepath, bool append, [[maybe_unused]] MPI_Comm comm) {
    try {
        // Note, this is only called by the rank that writes the file (e.g. rank 0 of a
        // region's communicator), so nothing in here may be collective. This previously
        // called MPI_Barrier(comm) whenever the directory appeared to be missing, which
        // would hang since the other ranks never call this function.
        //
        // The path is used as given rather than re-resolving symlinks in it, and the
        // parent directory is created without a separate exists() check beforehand
        // (create_directories() is a no-op if it already exists).
        const fs::path& resolved_path = filepath;
        fs::path dir_path = resolved_path.parent_path();
        if (!dir_path.empty()) {
            std::error_code ec;
            fs::create_directories(dir_path, ec);
            if (ec) {
                MFEM_WARNING("Failed to create directory " << dir_path << ": " << ec.message());
            }
        }

        // Open file
        std::ios_base::openmode mode = std::ios_base::out;
        if (append) {
            mode |= std::ios_base::app;
        }

        auto file = std::make_unique<std::ofstream>(resolved_path, mode);

        if (!file->is_open()) {
            MFEM_WARNING("Failed to open output file: " << resolved_path);
            return nullptr;
        }
        // Apply precision configuration
        ConfigureStreamPrecision(*file);
        return file;

    } catch (const fs::filesystem_error& ex) {
        MFEM_WARNING("Filesystem error when creating file " << filepath << ": " << ex.what());
        return nullptr;
    } catch (const std::exception& ex) {
        MFEM_WARNING("Error when creating file " << filepath << ": " << ex.what());
        return nullptr;
    }
}

inline std::string PostProcessingFileManager::GetSurfaceQuantityFilePath(int surface_id) const {
    fs::path specific_filename = m_options.post_processing.surface_quantities.fname;

    fs::path extension = specific_filename.extension();
    fs::path base_name = specific_filename;
    if (extension.string().empty()) {
        extension = ".txt";
    } else {
        base_name = base_name.stem();
    }

    fs::path filename = base_name.string() + "_surface_" + std::to_string(surface_id) +
                        extension.string();
    return m_output_directory / filename;
}

inline std::string PostProcessingFileManager::GetSurfaceQuantityHeader(bool moments) const {
    std::ostringstream header;

    header << CenterText("# Time", COLUMN_WIDTH);
    header << CenterText("Surface_ID", COLUMN_WIDTH);
    header << CenterText("Area", COLUMN_WIDTH);
    for (const auto& column : GetSurfaceQuantityColumnNames(moments)) {
        header << CenterText(column, COLUMN_WIDTH);
    }

    header << "\n";
    return header.str();
}

inline std::vector<std::string>
PostProcessingFileManager::GetSurfaceQuantityColumnNames(bool moments) const {
    std::vector<std::string> names = {"Disp_x",
                                      "Disp_y",
                                      "Disp_z",
                                      "Force_x",
                                      "Force_y",
                                      "Force_z",
                                      "Normal_x",
                                      "Normal_y",
                                      "Normal_z"};
    if (moments) {
        names.insert(names.end(), {"Moment_x", "Moment_y", "Moment_z"});
    }
    return names;
}

inline std::vector<std::string>
PostProcessingFileManager::GetVolumeAverageColumnNames(const std::string& calc_type) const {
    const auto custom = m_column_names.find(calc_type);
    if (custom != m_column_names.end()) {
        return custom->second;
    }
    if (calc_type == "stress") {
        // Voigt order of the stress quadrature function: 11, 22, 33, 23, 13, 12
        return {"Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"};
    } else if (calc_type == "def_grad") {
        // The deformation gradient is stored column by column (column-major 3x3)
        return {"F11", "F21", "F31", "F12", "F22", "F32", "F13", "F23", "F33"};
    } else if (calc_type == "euler_strain") {
        return {"E11", "E22", "E33", "E23", "E13", "E12"};
    } else if (calc_type == "plastic_work" || calc_type == "pl_work") {
        return {"Plastic_Work"};
    } else if (calc_type == "elastic_strain") {
        return {"Ee11", "Ee22", "Ee33", "Ee23", "Ee13", "Ee12"};
    } else if (calc_type == "eps" || calc_type == "eq_pl_strain") {
        return {"Equiv_Plastic_Strain"}; // Shortened to fit better
    }
    return {calc_type};
}

// Updated GetVolumeAverageHeader method with proper alignment:
inline std::string
PostProcessingFileManager::GetVolumeAverageHeader(const std::string& calc_type) const {
    std::ostringstream header;

    const auto comment = m_header_comments.find(calc_type);
    if (comment != m_header_comments.end()) {
        header << comment->second << "\n";
    }

    // Set formatting for header to match data columns
    header << CenterText("# Time", COLUMN_WIDTH);
    header << CenterText("Volume", COLUMN_WIDTH);
    for (const auto& column : GetVolumeAverageColumnNames(calc_type)) {
        header << CenterText(column, COLUMN_WIDTH);
    }

    header << "\n";
    return header.str();
}

inline bool PostProcessingFileManager::ShouldOutputAtStep(int step) const {
    return (step % m_output_frequency == 0);
}