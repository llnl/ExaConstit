#include "postprocessing/postprocessing_hdf5_file.hpp"

#include "mfem.hpp"

#if defined(EXACONSTIT_USE_AXOM)

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <system_error>

#include "conduit.hpp"
#include "conduit_relay_io_hdf5.hpp"
#include "conduit_relay_mpi.hpp"

namespace {

/// Datasets that are not time series (written once, never cut back after a restart)
bool IsFixedDataset(const std::string& name) {
    return name == "bin_edges" || name == "description";
}

/**
 * @brief Conduit's HDF5 settings for this file, for the lifetime of the object
 *
 * Conduit's HDF5 settings are global (the restart checkpoints use them too), so they are set
 * only around this file's reads and writes and restored afterwards. The defaults chunk growing
 * datasets in 1 MB gzip blocks; with hundreds of datasets that each grow by one value per step,
 * every write would decompress and recompress 1 MB per dataset. Small uncompressed chunks
 * (64 values) keep each write cheap and the file small.
 */
class ScopedHDF5Settings {
public:
    ScopedHDF5Settings() {
        conduit::relay::io::hdf5_options(m_saved);
        conduit::Node settings;
        settings["chunking/enabled"] = "true"; // growing datasets need chunks
        settings["chunking/chunk_size"] = 64 * static_cast<int>(sizeof(double));
        settings["chunking/compression/method"] = "none";
        conduit::relay::io::hdf5_set_options(settings);
    }
    ~ScopedHDF5Settings() {
        conduit::relay::io::hdf5_set_options(m_saved);
    }
    ScopedHDF5Settings(const ScopedHDF5Settings&) = delete;
    ScopedHDF5Settings& operator=(const ScopedHDF5Settings&) = delete;

private:
    conduit::Node m_saved;
};

std::string JoinPath(const std::string& parent, const std::string& name) {
    return parent.empty() ? name : parent + "/" + name;
}

} // namespace

struct PostProcessingHDF5File::Impl {
    std::filesystem::path path;
    MPI_Comm comm;
    int rank = 0;

    /// Rows queued on this rank since the last FinishOutputStep(): group path -> name -> value
    conduit::Node pending_rows;
    /// Fixed datasets queued on this rank: dataset path -> values
    conduit::Node pending_fixed;
    /// Fixed datasets this rank has already queued (each is sent once per run)
    std::set<std::string> fixed_queued;

    // Rank 0 only
    hid_t file_id = -1;
    /// Rows each group has in the file (the offset of its next row)
    std::map<std::string, conduit::index_t> rows;
    /// Fixed datasets already in the file
    std::set<std::string> fixed_in_file;

    /// Rank 0: write the leaf children of every group in tree as rows at the group's next row,
    /// then recurse into its subgroups
    void WriteRows(const conduit::Node& tree, const std::string& group) {
        conduit::Node row;
        conduit::index_t num_values = 0;
        for (conduit::index_t i = 0; i < tree.number_of_children(); i++) {
            const conduit::Node& child = tree.child(i);
            if (child.dtype().is_empty()) {
                continue; // a group left without rows by CutBack()
            }
            if (child.dtype().is_object()) {
                WriteRows(child, JoinPath(group, child.name()));
            } else {
                row[child.name()].set_external(const_cast<conduit::Node&>(child));
                num_values = child.dtype().number_of_elements();
            }
        }
        if (row.number_of_children() == 0 || num_values == 0) {
            return;
        }
        conduit::Node opts;
        opts["offset"] = rows[group];
        conduit::relay::io::hdf5_write(row, file_id, group, opts);
        rows[group] += num_values;
    }

    /// Rank 0: write every leaf of tree that is not yet in the file
    void WriteFixed(const conduit::Node& tree, const std::string& parent) {
        for (conduit::index_t i = 0; i < tree.number_of_children(); i++) {
            const conduit::Node& child = tree.child(i);
            const std::string child_path = JoinPath(parent, child.name());
            if (child.dtype().is_empty()) {
                continue;
            }
            if (child.dtype().is_object()) {
                WriteFixed(child, child_path);
            } else if (fixed_in_file.insert(child_path).second) {
                conduit::Node wrapper;
                wrapper[child_path].set_external(const_cast<conduit::Node&>(child));
                conduit::relay::io::hdf5_write(wrapper, file_id);
            }
        }
    }

    /**
     * @brief Rank 0: cut a tree read from the file back to the rows up to restart_time
     *
     * @param tree Group as read from the file
     * @param restart_time Last time to keep
     * @param kept Rows kept in the nearest enclosing group with a time dataset (statistics and
     *             histogram groups share their average's rows), -1 if there is none yet
     * @param series Output: the kept time-series datasets
     * @param fixed Output: the fixed datasets (copied as they are)
     */
    static void CutBack(const conduit::Node& tree,
                        const double restart_time,
                        conduit::index_t kept,
                        conduit::Node& series,
                        conduit::Node& fixed) {
        if (tree.has_child("time")) {
            const conduit::Node& time = tree["time"];
            const conduit::index_t num_rows = time.dtype().number_of_elements();
            const conduit::float64_array times = time.value();
            kept = 0;
            while (kept < num_rows && times[kept] <= restart_time) {
                kept++;
            }
        }
        for (conduit::index_t i = 0; i < tree.number_of_children(); i++) {
            const conduit::Node& child = tree.child(i);
            const std::string& name = child.name();
            if (child.dtype().is_object()) {
                CutBack(child, restart_time, kept, series[name], fixed[name]);
            } else if (IsFixedDataset(name)) {
                fixed[name] = child;
            } else {
                const conduit::index_t num_rows = child.dtype().number_of_elements();
                const conduit::index_t keep = (kept < 0) ? num_rows : std::min(kept, num_rows);
                if (keep > 0) {
                    const conduit::float64_array values = child.value();
                    std::vector<double> kept_values(static_cast<size_t>(keep));
                    for (conduit::index_t k = 0; k < keep; k++) {
                        kept_values[static_cast<size_t>(k)] = values[k];
                    }
                    series[name].set(kept_values);
                }
            }
        }
    }

    /// Rank 0: open the file of a fresh run (replacing an old one) if it is not open yet
    void OpenNew() {
        if (file_id >= 0) {
            return;
        }
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            std::cout << "Replacing the post-processing output of an earlier run: " << path
                      << std::endl;
        }
        file_id = conduit::relay::io::hdf5_create_file(path.string());
    }
};

PostProcessingHDF5File::PostProcessingHDF5File(const std::filesystem::path& path, MPI_Comm comm)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->path = path;
    m_impl->comm = comm;
    MPI_Comm_rank(comm, &m_impl->rank);
}

PostProcessingHDF5File::~PostProcessingHDF5File() {
    if (m_impl && m_impl->file_id >= 0) {
        conduit::relay::io::hdf5_close_file(m_impl->file_id);
    }
}

void PostProcessingHDF5File::AddRow(const std::string& group,
                                    const std::vector<std::string>& names,
                                    const std::vector<double>& values) {
    MFEM_VERIFY(names.size() == values.size(),
                "HDF5 output row of " << group << ": " << names.size() << " names but "
                                      << values.size() << " values");
    MFEM_VERIFY(!names.empty() && !m_impl->pending_rows.has_path(JoinPath(group, names[0])),
                "HDF5 output: two rows of " << group << " in one output step");
    for (size_t i = 0; i < names.size(); i++) {
        m_impl->pending_rows[JoinPath(group, names[i])] = values[i];
    }
}

void PostProcessingHDF5File::AddFixed(const std::string& path, const std::vector<double>& values) {
    if (m_impl->fixed_queued.insert(path).second) {
        m_impl->pending_fixed[path].set(values);
    }
}

void PostProcessingHDF5File::AddFixed(const std::string& path, const std::string& text) {
    if (m_impl->fixed_queued.insert(path).second) {
        m_impl->pending_fixed[path].set(text);
    }
}

void PostProcessingHDF5File::FinishOutputStep() {
    // Most steps write nothing (output every N steps); skip the gather then
    int queued = (m_impl->pending_rows.number_of_children() > 0 ||
                  m_impl->pending_fixed.number_of_children() > 0)
                     ? 1
                     : 0;
    MPI_Allreduce(MPI_IN_PLACE, &queued, 1, MPI_INT, MPI_MAX, m_impl->comm);
    if (queued == 0) {
        return;
    }

    // Everything this rank queued, gathered to rank 0 (one entry per rank, schemas may differ)
    conduit::Node send;
    send["rank"] = m_impl->rank; // never empty, whatever was queued
    if (m_impl->pending_rows.number_of_children() > 0) {
        send["rows"].set_external(m_impl->pending_rows);
    }
    if (m_impl->pending_fixed.number_of_children() > 0) {
        send["fixed"].set_external(m_impl->pending_fixed);
    }
    conduit::Node received;
    conduit::relay::mpi::gather_using_schema(send, received, 0, m_impl->comm);
    m_impl->pending_rows.reset();
    m_impl->pending_fixed.reset();

    if (m_impl->rank != 0) {
        return;
    }
    const ScopedHDF5Settings settings;
    m_impl->OpenNew();
    for (conduit::index_t r = 0; r < received.number_of_children(); r++) {
        const conduit::Node& from_rank = received.child(r);
        if (from_rank.has_child("fixed")) {
            m_impl->WriteFixed(from_rank["fixed"], "");
        }
        if (from_rank.has_child("rows")) {
            m_impl->WriteRows(from_rank["rows"], "");
        }
    }
    conduit::relay::io::hdf5_flush_file(m_impl->file_id);
}

void PostProcessingHDF5File::ContinueAfterRestart(const double restart_time) {
    if (m_impl->rank == 0) {
        const ScopedHDF5Settings settings;
        std::error_code ec;
        if (m_impl->file_id < 0 && std::filesystem::exists(m_impl->path, ec)) {
            // Read the earlier run's file. A job killed in the middle of a write can leave it
            // unreadable; it is then kept under another name and this run starts a new file.
            conduit::Node earlier;
            bool readable = true;
            try {
                conduit::relay::io::hdf5_read(m_impl->path.string(), earlier);
            } catch (const conduit::Error& error) {
                readable = false;
                auto kept = m_impl->path;
                kept += ".unreadable";
                std::filesystem::rename(m_impl->path, kept, ec);
                std::cout << "Warning: could not read the post-processing output of the earlier "
                             "run ("
                          << error.message() << "); it is kept as " << kept
                          << " and this run starts a new file" << std::endl;
            }
            if (readable) {
                // Keep its rows up to the checkpoint and write them to a new file (datasets
                // growable, so this run's rows are appended), then replace the old file with it,
                // so the earlier history is never lost half-way
                conduit::Node series, fixed;
                Impl::CutBack(earlier, restart_time, -1, series, fixed);
                auto cut = m_impl->path;
                cut += ".restart";
                m_impl->file_id = conduit::relay::io::hdf5_create_file(cut.string());
                m_impl->WriteFixed(fixed, "");
                m_impl->WriteRows(series, "");
                conduit::relay::io::hdf5_close_file(m_impl->file_id);
                std::filesystem::rename(cut, m_impl->path, ec);
                MFEM_VERIFY(!ec,
                            "Could not replace " << m_impl->path << " with " << cut << ": "
                                                 << ec.message());
                m_impl->file_id =
                    conduit::relay::io::hdf5_open_file_for_read_write(m_impl->path.string());
                std::cout << "Continuing the post-processing output in " << m_impl->path
                          << " after time " << restart_time << std::endl;
            }
        }
    }
    MPI_Barrier(m_impl->comm);
}

#else // !EXACONSTIT_USE_AXOM

// Without Axom there is no Conduit; the option checks stop a run asking for HDF5 output before
// this class is constructed.
struct PostProcessingHDF5File::Impl {};

PostProcessingHDF5File::PostProcessingHDF5File(const std::filesystem::path&, MPI_Comm) {
    MFEM_ABORT("HDF5 post-processing output needs ExaConstit built with Axom (ENABLE_AXOM)");
}
PostProcessingHDF5File::~PostProcessingHDF5File() = default;
void PostProcessingHDF5File::AddRow(const std::string&,
                                    const std::vector<std::string>&,
                                    const std::vector<double>&) {}
void PostProcessingHDF5File::AddFixed(const std::string&, const std::vector<double>&) {}
void PostProcessingHDF5File::AddFixed(const std::string&, const std::string&) {}
void PostProcessingHDF5File::FinishOutputStep() {}
void PostProcessingHDF5File::ContinueAfterRestart(double) {}

#endif
