#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "axom/sidre.hpp"

// Diagnostic and test tool: compares two ExaConstit checkpoint directories
// (<directory>/cycle_<N>/, see doc/restart.md) group by group and view by view.
//
// Usage: compare_restart_checkpoints <checkpoint_dir_a> <checkpoint_dir_b> [--rtol <tol>]
//
// Both directories must hold the same set of rank files. The commit file and every rank file
// are loaded in full, including the arrays stored as external views, and every group and view
// is compared: names, types, sizes and contents. A quadrature function stored as `copy_of`
// another one (see RestartCheckpoint::WriteQuadratureFunctions()) is resolved to that
// function's values first, so a begin/end pair stored as a copy in one checkpoint and as full
// values in the other still compares equal, as it should, since both restore the same state.
//
// Without --rtol the comparison is bitwise. With --rtol, double arrays may differ by
// rtol * max(|a|, |b|) per entry; everything else must still match exactly.
//
// Exit status: 0 = the checkpoints match, 1 = they differ, 2 = usage or I/O error.
// test/test_restart_runs.py relies on the exit status and on the last line of the output,
// which is always a one-line summary.

namespace fs = std::filesystem;

namespace {

using axom::sidre::DataStore;
using axom::sidre::Group;
using axom::sidre::View;

/**
 * @brief Running totals of a comparison
 */
struct ComparisonTally {
    /** @brief Relative tolerance for double arrays (0 = bitwise) */
    double rtol = 0.0;
    /** @brief Number of views (or resolved quadrature functions) compared */
    long views = 0;
    /** @brief Number of mismatches found */
    long mismatches = 0;
    /** @brief Largest absolute difference seen between double entries */
    double max_abs = 0.0;
    /** @brief Largest relative difference seen between double entries */
    double max_rel = 0.0;
};

/**
 * @brief Encode a registry name the same way restart_checkpoint.cpp does
 *
 * @param name Registry name
 * @return The name with '%' encoded as "%25" and '/' as "%2F"
 *
 * @details Needed to find the group a `copy_of` entry refers to, since `copy_of` holds the
 * plain registry name while the groups are named with the encoded form.
 */
std::string SidreSafeName(const std::string& name) {
    std::string encoded;
    for (const char character : name) {
        if (character == '%') {
            encoded += "%25";
        } else if (character == '/') {
            encoded += "%2F";
        } else {
            encoded += character;
        }
    }
    return encoded;
}

/**
 * @brief A Sidre file loaded in full, including the data of its external views
 *
 * @details Group::load() restores an external view as a description only, so this allocates
 * a buffer for every external view, points the view at it, and then reads the data with
 * Group::loadExternalData(), the same two-step read restart_checkpoint.cpp uses. Zero-length
 * external views are described but have no data, so they are removed before the read and
 * remembered by path, so the sets of empty arrays in two files can still be compared.
 */
class LoadedFile {
public:
    /**
     * @brief Load a Sidre file and all of its external data
     *
     * @param path File to load
     * @param protocol Sidre protocol of the file ("sidre_hdf5" or "sidre_json")
     * @return True if the file and its external data were read
     */
    bool Load(const fs::path& path, const char* protocol) {
        Group* root = m_store.getRoot();
        if (!root->load(path.string(), protocol)) {
            return false;
        }
        std::vector<std::pair<Group*, std::string>> empty_views;
        BindExternalViews(root, empty_views);
        for (auto& [group, name] : empty_views) {
            group->destroyView(name);
        }
        return m_external_buffers.empty() || root->loadExternalData(path.string());
    }

    /**
     * @brief Root group of the loaded file
     *
     * @return The root group
     */
    Group* Root() {
        return m_store.getRoot();
    }

    /**
     * @brief Paths of the zero-length external views removed during loading
     *
     * @return The set of paths
     */
    const std::set<std::string>& EmptyExternalViews() const {
        return m_empty_paths;
    }

private:
    /**
     * @brief Point every non-empty external view at a newly allocated buffer
     *
     * @param group Group to process, recursively
     * @param empty_views Receives the zero-length external views, to be removed by the caller
     */
    void BindExternalViews(Group* group, std::vector<std::pair<Group*, std::string>>& empty_views) {
        for (View& view : group->views()) {
            if (!view.isExternal()) {
                continue;
            }
            const std::size_t bytes = static_cast<std::size_t>(view.getTotalBytes());
            if (bytes == 0) {
                empty_views.emplace_back(group, view.getName());
                m_empty_paths.insert(view.getPathName());
                continue;
            }
            m_external_buffers.emplace_back(bytes);
            view.setExternalDataPtr(m_external_buffers.back().data());
        }
        for (Group& child : group->groups()) {
            BindExternalViews(&child, empty_views);
        }
    }

    /** @brief DataStore holding the loaded file */
    DataStore m_store;
    /** @brief Buffers the external views were pointed at */
    std::vector<std::vector<unsigned char>> m_external_buffers;
    /** @brief Paths of the zero-length external views */
    std::set<std::string> m_empty_paths;
};

/**
 * @brief Print one mismatch and count it
 *
 * @param tally Running totals
 * @param file File the mismatch was found in
 * @param path Path of the group or view within the file
 * @param message Description of the mismatch
 */
void ReportMismatch(ComparisonTally& tally,
                    const std::string& file,
                    const std::string& path,
                    const std::string& message) {
    std::cout << "  MISMATCH [" << file << "] " << path << ": " << message << std::endl;
    ++tally.mismatches;
}

/**
 * @brief Compare two double arrays
 *
 * @param a First array
 * @param b Second array
 * @param count Number of entries in each
 * @param tally Running totals (tolerance in, largest differences updated)
 * @param message Receives a description of the differences when the arrays do not match
 * @return True if the arrays match: bitwise when tally.rtol is 0, otherwise every entry within
 * rtol * max(|a|, |b|)
 *
 * @details Entries with identical bits always match, which also makes two identical NaNs
 * match. The description gives the number of differing entries, the largest absolute and
 * relative differences, and the first differing entry, printed with 17 significant digits so
 * the two values can be told apart.
 */
bool CompareDoubles(const double* a,
                    const double* b,
                    const long count,
                    ComparisonTally& tally,
                    std::string& message) {
    if (count == 0) {
        return true;
    }
    if (tally.rtol == 0.0 &&
        std::memcmp(a, b, static_cast<std::size_t>(count) * sizeof(double)) == 0) {
        return true;
    }
    long differing = 0;
    long first_differing = -1;
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (long i = 0; i < count; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(double)) == 0) {
            continue;
        }
        const double difference = std::abs(a[i] - b[i]);
        const double scale = std::max({std::abs(a[i]), std::abs(b[i]), 1e-300});
        max_abs = std::max(max_abs, difference);
        max_rel = std::max(max_rel, difference / scale);
        if (tally.rtol == 0.0 || !(difference <= tally.rtol * scale)) {
            if (first_differing < 0) {
                first_differing = i;
            }
            ++differing;
        }
    }
    tally.max_abs = std::max(tally.max_abs, max_abs);
    tally.max_rel = std::max(tally.max_rel, max_rel);
    if (differing == 0) {
        return true;
    }
    std::ostringstream description;
    description << differing << " of " << count << " entries differ (max abs " << std::scientific
                << std::setprecision(3) << max_abs << ", max rel " << max_rel << "; first at "
                << first_differing << ": " << std::defaultfloat << std::setprecision(17)
                << a[first_differing] << " vs " << b[first_differing] << ")";
    message = description.str();
    return false;
}

/**
 * @brief Compare two views with the same path
 *
 * @param a View from the first file
 * @param b View from the second file
 * @param file File name, for the mismatch messages
 * @param tally Running totals
 *
 * @details Strings must be equal, other views must have the same type and number of entries,
 * double arrays are compared with CompareDoubles(), and everything else byte for byte.
 */
void CompareViews(View* a, View* b, const std::string& file, ComparisonTally& tally) {
    ++tally.views;
    const std::string path = a->getPathName();
    if (a->getTypeID() != b->getTypeID()) {
        ReportMismatch(tally, file, path, "the types differ");
        return;
    }
    if (a->isString() || b->isString()) {
        if (!a->isString() || !b->isString() || std::string(a->getString()) != b->getString()) {
            ReportMismatch(tally,
                           file,
                           path,
                           std::string("the strings differ: '") +
                               (a->isString() ? a->getString() : "?") + "' vs '" +
                               (b->isString() ? b->getString() : "?") + "'");
        }
        return;
    }
    if (a->getNumElements() != b->getNumElements()) {
        ReportMismatch(tally,
                       file,
                       path,
                       "the entry counts differ: " + std::to_string(a->getNumElements()) + " vs " +
                           std::to_string(b->getNumElements()));
        return;
    }
    const long count = static_cast<long>(a->getNumElements());
    if (count == 0) {
        return;
    }
    const void* data_a = a->getVoidPtr();
    const void* data_b = b->getVoidPtr();
    if (data_a == nullptr || data_b == nullptr) {
        ReportMismatch(tally, file, path, "no data was loaded");
        return;
    }
    if (a->getTypeID() == axom::sidre::FLOAT64_ID) {
        std::string message;
        if (!CompareDoubles(static_cast<const double*>(data_a),
                            static_cast<const double*>(data_b),
                            count,
                            tally,
                            message)) {
            ReportMismatch(tally, file, path, message);
        }
        return;
    }
    if (std::memcmp(data_a, data_b, static_cast<std::size_t>(a->getTotalBytes())) != 0) {
        ReportMismatch(tally, file, path, "the contents differ");
    }
}

/**
 * @brief The values view of a quadrature function entry, following `copy_of` if present
 *
 * @param qfuncs The `qfuncs/` group
 * @param entry One quadrature function's group
 * @return The view holding the function's values, or nullptr if it cannot be found
 */
View* ResolveQuadratureFunctionValues(Group* qfuncs, Group* entry) {
    if (entry->hasView("copy_of")) {
        const std::string source = SidreSafeName(entry->getView("copy_of")->getString());
        if (!qfuncs->hasGroup(source)) {
            return nullptr;
        }
        Group* source_entry = qfuncs->getGroup(source);
        return source_entry->hasView("values") ? source_entry->getView("values") : nullptr;
    }
    return entry->hasView("values") ? entry->getView("values") : nullptr;
}

/**
 * @brief Compare the `qfuncs/` groups of two rank files
 *
 * @param a `qfuncs/` group of the first file
 * @param b `qfuncs/` group of the second file
 * @param file File name, for the mismatch messages
 * @param tally Running totals
 *
 * @details Both files must have the same quadrature functions with the same space name, vdim
 * and value count. The values are compared after resolving `copy_of`, so how each file chose
 * to store a function does not matter, only what it restores. A function with no values and
 * no `copy_of` in both files is an empty function whose zero-length view was removed while
 * loading, and has nothing to compare.
 */
void CompareQuadratureFunctions(Group* a,
                                Group* b,
                                const std::string& file,
                                ComparisonTally& tally) {
    std::set<std::string> names;
    for (Group& group : a->groups()) {
        names.insert(group.getName());
    }
    for (Group& group : b->groups()) {
        names.insert(group.getName());
    }
    for (const std::string& name : names) {
        const std::string path = a->getPathName() + "/" + name;
        if (!a->hasGroup(name) || !b->hasGroup(name)) {
            ReportMismatch(tally,
                           file,
                           path,
                           std::string("only in the ") + (a->hasGroup(name) ? "first" : "second") +
                               " checkpoint");
            continue;
        }
        Group* entry_a = a->getGroup(name);
        Group* entry_b = b->getGroup(name);
        for (const char* metadata : {"qspace_name", "vdim", "value_count"}) {
            if (entry_a->hasView(metadata) != entry_b->hasView(metadata)) {
                ReportMismatch(
                    tally, file, path + "/" + metadata, "present in only one checkpoint");
            } else if (entry_a->hasView(metadata)) {
                CompareViews(entry_a->getView(metadata), entry_b->getView(metadata), file, tally);
            }
        }

        const bool empty_a = !entry_a->hasView("copy_of") && !entry_a->hasView("values");
        const bool empty_b = !entry_b->hasView("copy_of") && !entry_b->hasView("values");
        if (empty_a && empty_b) {
            continue;
        }
        View* values_a = ResolveQuadratureFunctionValues(a, entry_a);
        View* values_b = ResolveQuadratureFunctionValues(b, entry_b);
        if (values_a == nullptr || values_b == nullptr) {
            ReportMismatch(tally, file, path, "the values could not be resolved");
            continue;
        }
        ++tally.views;
        if (values_a->getNumElements() != values_b->getNumElements()) {
            ReportMismatch(tally, file, path, "the value counts differ");
            continue;
        }
        std::string message;
        if (!CompareDoubles(static_cast<const double*>(values_a->getVoidPtr()),
                            static_cast<const double*>(values_b->getVoidPtr()),
                            static_cast<long>(values_a->getNumElements()),
                            tally,
                            message)) {
            ReportMismatch(tally, file, path, message);
        }
    }
}

/**
 * @brief Compare two groups recursively
 *
 * @param a Group from the first file
 * @param b Group from the second file
 * @param file File name, for the mismatch messages
 * @param tally Running totals
 *
 * @details Both groups must have the same views and child groups. The top-level `qfuncs/`
 * group is compared with CompareQuadratureFunctions() so that `copy_of` entries are resolved.
 */
void CompareGroups(Group* a, Group* b, const std::string& file, ComparisonTally& tally) {
    std::set<std::string> views;
    for (View& view : a->views()) {
        views.insert(view.getName());
    }
    for (View& view : b->views()) {
        views.insert(view.getName());
    }
    for (const std::string& name : views) {
        if (!a->hasView(name) || !b->hasView(name)) {
            ReportMismatch(tally,
                           file,
                           a->getPathName() + "/" + name,
                           std::string("view only in the ") +
                               (a->hasView(name) ? "first" : "second") + " checkpoint");
            continue;
        }
        CompareViews(a->getView(name), b->getView(name), file, tally);
    }

    std::set<std::string> groups;
    for (Group& group : a->groups()) {
        groups.insert(group.getName());
    }
    for (Group& group : b->groups()) {
        groups.insert(group.getName());
    }
    for (const std::string& name : groups) {
        if (!a->hasGroup(name) || !b->hasGroup(name)) {
            ReportMismatch(tally,
                           file,
                           a->getPathName() + "/" + name,
                           std::string("group only in the ") +
                               (a->hasGroup(name) ? "first" : "second") + " checkpoint");
            continue;
        }
        if (name == "qfuncs" && a->isRoot()) {
            CompareQuadratureFunctions(a->getGroup(name), b->getGroup(name), file, tally);
        } else {
            CompareGroups(a->getGroup(name), b->getGroup(name), file, tally);
        }
    }
}

/**
 * @brief Rank files in a checkpoint directory
 *
 * @param directory Checkpoint directory
 * @return Names of the rank_*.hdf5 files, sorted
 */
std::vector<std::string> RankFiles(const fs::path& directory) {
    std::vector<std::string> files;
    std::error_code error;
    for (const auto& item : fs::directory_iterator(directory, error)) {
        const std::string name = item.path().filename().string();
        if (name.rfind("rank_", 0) == 0 && item.path().extension() == ".hdf5") {
            files.push_back(name);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace

int main(int argc, char** argv) {
    ComparisonTally tally;
    std::vector<std::string> directories;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--rtol" && i + 1 < argc) {
            tally.rtol = std::atof(argv[++i]);
        } else {
            directories.push_back(argument);
        }
    }
    if (directories.size() != 2) {
        std::cerr << "usage: " << argv[0] << " <checkpoint_dir_a> <checkpoint_dir_b> [--rtol <tol>]"
                  << std::endl;
        return 2;
    }
    const fs::path directory_a = directories[0];
    const fs::path directory_b = directories[1];

    const std::vector<std::string> files_a = RankFiles(directory_a);
    const std::vector<std::string> files_b = RankFiles(directory_b);
    if (files_a.empty() || files_a != files_b) {
        std::cout << "ERROR: the rank files differ or are missing (" << files_a.size() << " in "
                  << directory_a.string() << ", " << files_b.size() << " in "
                  << directory_b.string() << ")" << std::endl;
        return 2;
    }

    {
        LoadedFile commit_a;
        LoadedFile commit_b;
        if (!commit_a.Load(directory_a / "commit.json", "sidre_json") ||
            !commit_b.Load(directory_b / "commit.json", "sidre_json")) {
            std::cout << "ERROR: unable to load commit.json in both directories" << std::endl;
            return 2;
        }
        CompareGroups(commit_a.Root(), commit_b.Root(), "commit.json", tally);
    }
    for (const std::string& file : files_a) {
        LoadedFile file_a;
        LoadedFile file_b;
        if (!file_a.Load(directory_a / file, "sidre_hdf5") ||
            !file_b.Load(directory_b / file, "sidre_hdf5")) {
            std::cout << "ERROR: unable to load " << file << " in both directories" << std::endl;
            return 2;
        }
        if (file_a.EmptyExternalViews() != file_b.EmptyExternalViews()) {
            ReportMismatch(tally, file, "/", "the sets of empty arrays differ");
        }
        CompareGroups(file_a.Root(), file_b.Root(), file, tally);
    }

    std::cout << (tally.mismatches == 0 ? "MATCH" : "DIFFER") << ": " << files_a.size()
              << " rank files, " << tally.views << " views compared, " << tally.mismatches
              << " mismatches (max abs " << std::scientific << std::setprecision(3) << tally.max_abs
              << ", max rel " << tally.max_rel << ")" << (tally.rtol == 0.0 ? ", bitwise" : "")
              << std::endl;
    return tally.mismatches == 0 ? 0 : 1;
}
