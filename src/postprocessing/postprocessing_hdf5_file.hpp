#pragma once

/**
 * @file postprocessing_hdf5_file.hpp
 * @brief The single HDF5 file of the post-processing output ([PostProcessing] output_format =
 *        "hdf5")
 *
 * With output_format = "hdf5", every row that would go to a post-processing text file (volume
 * averages, volume statistics and histograms, user function averages / integrals, surface
 * quantities) goes into one file, <output_directory>/<basename>/post_processing.hdf5, written
 * with Conduit relay (HDF5 underneath). Each text file becomes a group and each text column a
 * dataset of that name holding one value per output step, e.g. /volume/stress/global/Szz. The
 * full layout is documented in src/options.toml (Output Format).
 *
 * Who writes: the rows of a material are produced on the first rank owning that material (the
 * root of its region communicator), which is not always rank 0, but only one rank may write an
 * HDF5 file. So every rank queues its rows (AddRow(), AddFixed()), and FinishOutputStep(), called
 * by all ranks at the end of each post-processing update, gathers them to rank 0, which appends
 * them to the file and flushes it, so the file is complete and readable after every step.
 *
 * Restarts: ContinueAfterRestart() drops the rows written after the checkpoint (steps that were
 * computed before the run stopped and are now computed again), so no step appears twice.
 *
 * Conduit comes with Axom; without Axom (no EXACONSTIT_USE_AXOM) the option checks stop a run
 * asking for HDF5 output before anything here is constructed.
 */

#include <filesystem>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

/**
 * @brief Queues post-processing rows on every rank and appends them to one HDF5 file on rank 0
 *
 * Constructed on every rank of the communicator (all of its rank-0 work happens in the
 * collective FinishOutputStep() and ContinueAfterRestart()).
 */
class PostProcessingHDF5File {
public:
    /**
     * @brief Set up the file (nothing is opened until the first FinishOutputStep())
     *
     * @param path File to write; a fresh run replaces an existing file, a restart continues it
     *             (ContinueAfterRestart())
     * @param comm Communicator of all ranks taking part (rank 0 writes)
     */
    PostProcessingHDF5File(const std::filesystem::path& path, MPI_Comm comm = MPI_COMM_WORLD);

    /// Closes the file on rank 0
    ~PostProcessingHDF5File();

    PostProcessingHDF5File(const PostProcessingHDF5File&) = delete;
    PostProcessingHDF5File& operator=(const PostProcessingHDF5File&) = delete;

    /**
     * @brief Queue one row of a group (any rank, not collective)
     *
     * @param group Group path, e.g. "volume/stress/material_A_1" or
     *              "volume/stress/global/statistics/std"
     * @param names Dataset names, one per value (a group's main row starts with "time")
     * @param values One value per name
     *
     * A group gets at most one row per output step; its row number is the number of rows the
     * group already has in the file.
     */
    void AddRow(const std::string& group,
                const std::vector<std::string>& names,
                const std::vector<double>& values);

    /**
     * @brief Queue a dataset that is not a time series (e.g. histogram bin edges), written once
     *
     * @param path Dataset path, e.g. "volume/stress/global/histogram/bin_edges"
     * @param values Its values (ignored if the dataset is already in the file)
     */
    void AddFixed(const std::string& path, const std::vector<double>& values);

    /// Text version of AddFixed() (e.g. the comment lines of a surface quantity file)
    void AddFixed(const std::string& path, const std::string& text);

    /**
     * @brief Gather the queued rows to rank 0 and append them to the file (collective)
     *
     * Called by every rank at the end of each post-processing update, whether or not it queued
     * anything. The first call of a fresh run creates the file, replacing an old one.
     */
    void FinishOutputStep();

    /**
     * @brief Continue the file of an earlier run from its checkpoint (collective)
     *
     * @param restart_time Simulation time of the checkpoint; rows with a later time are removed
     *                     (the times are written as exact doubles, so the comparison is exact)
     *
     * Called once, before the first step of a restarted run. If there is no file yet, the run
     * starts a new one.
     */
    void ContinueAfterRestart(double restart_time);

private:
    /// Conduit nodes and the open file live here, so this header does not need Conduit
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
